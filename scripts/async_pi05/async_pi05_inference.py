import asyncio
import logging
import os
import time
from typing import Any

import cv2
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.models.model import Observation
from openpi.models.tokenizer import PaligemmaTokenizer
import openpi.shared.nnx_utils as nnx_utils
from openpi.training.config import get_config

# GPU memory optimization settings
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

PALIGEMMA_EOS_TOKEN = 1
max_decoding_steps = 25
temperature = 0.1

logger = logging.getLogger(__name__)


class AsyncPi05Inference:
    """异步 Pi0.5 推理服务器,支持 subtask generation"""

    def __init__(self, config_name: str = "right_pi05_20", gpu_id: int = 1):
        self.config_name = config_name
        self.gpu_id = gpu_id
        self.model = None
        self.tokenizer = None
        self.jit_sample_low_level_task = None
        self.jit_sample_actions = None
        self._initialized = False

        # 共享状态
        self.current_low_prompt = None
        self.low_prompt_lock = asyncio.Lock()

        # 设置 GPU
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["OPENPI_DATA_HOME"] = "/root/.cache/openpi"

    async def initialize(self):
        """异步初始化模型"""
        if self._initialized:
            return

        logger.info("开始初始化 Pi0.5 模型...")

        # 初始化模型配置
        config = get_config(self.config_name)
        model_rng = jax.random.key(0)

        # 创建模型
        self.model = config.model.create(model_rng)

        # 加载预训练参数
        graphdef, state = nnx.split(self.model)
        loader = config.weight_loader
        params = nnx.state(self.model)

        # 转换参数为 bfloat16
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        # 加载参数
        params_shape = params.to_pure_dict()
        loaded_params = loader.load(params_shape)
        state.replace_by_pure_dict(loaded_params)
        self.model = nnx.merge(graphdef, state)

        # 初始化 tokenizer
        self.tokenizer = PaligemmaTokenizer(max_len=256)

        # JIT 编译关键函数
        self.jit_sample_low_level_task = nnx_utils.module_jit(self.model.sample_low_level_task, static_argnums=(3,))
        self.jit_sample_actions = nnx_utils.module_jit(self.model.sample_actions)

        self._initialized = True
        logger.info("Pi0.5 模型初始化完成")

    def create_random_image(self, height: int = 224, width: int = 224) -> np.ndarray:
        """创建随机图像作为 fallback"""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    def load_image_with_fallback(self, img_path: str, img_name: str) -> np.ndarray:
        """加载图像,支持 fallback 到随机图像"""
        if not os.path.exists(img_path):
            logger.warning(f"图像文件不存在: {img_path},使用随机图像: {img_name}")
            return self.create_random_image()

        img = cv2.imread(img_path)
        if img is not None:
            logger.info(f"成功加载图像: {img_name}, 形状: {img.shape}")
            return img
        logger.warning(f"无法读取图像: {img_name},使用随机图像")
        return self.create_random_image()

    def prepare_observation(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str = "",
        state: np.ndarray | None = None,
    ) -> Observation:
        """准备观察数据"""

        # 转换图像到模型期望的格式 [-1, 1]
        img_dict = {}
        for key, img in images.items():
            img_dict[key] = jnp.array(img[np.newaxis, :, :, :]).astype(jnp.float32)

        # 准备状态数据
        state = jnp.zeros((1, 32), dtype=jnp.float32) if state is None else jnp.array(state)[np.newaxis, :]

        # Tokenize prompts
        tokenized_prompt, tokenized_prompt_mask, token_ar_mask, token_loss_mask = (
            self.tokenizer.tokenize_high_low_prompt(high_level_prompt, low_level_prompt, state)
        )
        # 构建观察数据
        data = {
            "image": img_dict,
            "image_mask": {key: jnp.ones(1, dtype=jnp.bool) for key in img_dict},
            "state": state,
            "tokenized_prompt": jnp.stack([tokenized_prompt], axis=0),
            "tokenized_prompt_mask": jnp.stack([tokenized_prompt_mask], axis=0),
            "token_ar_mask": jnp.stack([token_ar_mask], axis=0),
            "token_loss_mask": jnp.stack([token_loss_mask], axis=0),
        }

        observation = Observation.from_dict(data)
        rng = jax.random.key(42)
        observation = _model.preprocess_observation(
            rng, observation, train=False, image_keys=list(observation.images.keys())
        )

        # 根据 loss mask 设置低级别任务 tokens
        loss_mask = jnp.array(observation.token_loss_mask)
        new_tokenized_prompt = observation.tokenized_prompt.at[loss_mask].set(0)
        new_tokenized_prompt_mask = observation.tokenized_prompt_mask.at[loss_mask].set(False)

        new_observation = _model.Observation(
            images=observation.images,
            image_masks=observation.image_masks,
            state=observation.state,
            tokenized_prompt=new_tokenized_prompt,
            tokenized_prompt_mask=new_tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )

        observation = _model.preprocess_observation(
            None, new_observation, train=False, image_keys=list(observation.images.keys())
        )
        return jax.tree.map(jax.device_put, observation)

    async def generate_subtask(
        self, observation: Observation, rng: jax.Array, max_decoding_steps: int = 200, temperature: float = 0.1
    ) -> tuple[jnp.ndarray, str]:
        """生成子任务"""
        start_time = time.time()

        # 生成子任务 tokens
        predicted_token, _kv_cache, _mask, _ar_mask = self.jit_sample_low_level_task(
            rng, observation, max_decoding_steps, PALIGEMMA_EOS_TOKEN, temperature
        )

        # 解码生成的子任务
        subtask_text = self.tokenizer.detokenize(np.array(predicted_token[0], dtype=np.int32))

        generation_time = time.time() - start_time
        logger.info(f"子任务生成耗时: {generation_time:.3f}s")
        logger.info(f"生成的子任务: {subtask_text}")

        return predicted_token, subtask_text

    async def infer(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str = "ABCDEFG",
        state: np.ndarray | None = None,
        *,
        generate_subtask: bool = True,
        max_decoding_steps: int = 25,
        temperature: float = 0.1,
        noise: np.ndarray | None = None,
        subtask_refresh_interval: float | None = None,
    ) -> dict[str, Any]:
        """
        异步推理函数,支持子任务生成和定期刷新

        Args:
            images: 图像字典,键为图像类型,值为图像数组
            high_level_prompt: 高级别任务描述
            low_level_prompt: 低级别任务描述(可选)
            state: 机器人状态
            generate_subtask: 是否生成子任务(如果为True,则不生成动作,动作由持续生成功能处理)
            max_decoding_steps: 最大解码步数
            temperature: 采样温度
            noise: 动作噪声(可选,仅在generate_subtask=False时使用)
            subtask_refresh_interval: 子任务刷新间隔(秒),None表示不刷新

        Returns:
            包含子任务和时序信息的字典(动作由持续生成功能处理)
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        rng = jax.random.key(int(time.time() * 1000) % 2**32)

        # 准备观察数据
        observation = self.prepare_observation(images, high_level_prompt, low_level_prompt, state)

        results = {
            "state": np.array(observation.state[0]) if observation.state is not None else None,
            "actions": None,
            "subtask": None,
            "subtask_tokens": None,
            "timing": {},
        }

        # 生成子任务(如果需要)
        if generate_subtask:
            subtask_tokens, subtask_text = await self.generate_subtask(
                observation, rng, max_decoding_steps, temperature
            )
            results["subtask"] = subtask_text
            results["subtask_tokens"] = np.array(subtask_tokens[0])

        # 如果不需要生成动作,则跳过动作生成
        if not generate_subtask:
            # 只生成动作,不生成子任务
            action_start_time = time.time()
            if noise is not None:
                noise = jnp.array(noise)[np.newaxis, ...] if noise.ndim == 2 else jnp.array(noise)
                sampled_actions = self.jit_sample_actions(rng, observation, noise=noise)
            else:
                sampled_actions = self.jit_sample_actions(rng, observation)

            action_time = time.time() - action_start_time
            results["actions"] = np.array(sampled_actions[0])

            total_time = time.time() - start_time
            results["timing"] = {"total_ms": total_time * 1000, "action_ms": action_time * 1000, "subtask_ms": 0}
        else:
            # 只生成子任务,不生成动作(动作由持续生成功能处理)
            total_time = time.time() - start_time
            results["timing"] = {"total_ms": total_time * 1000, "action_ms": 0, "subtask_ms": total_time * 1000}

        # 如果设置了刷新间隔,启动定期刷新任务
        if subtask_refresh_interval is not None and subtask_refresh_interval > 0:
            results["subtask_refresh_interval"] = subtask_refresh_interval
            results["subtask_refresh_task"] = asyncio.create_task(
                self._periodic_subtask_refresh(
                    images,
                    high_level_prompt,
                    low_level_prompt,
                    state,
                    subtask_refresh_interval,
                    max_decoding_steps,
                    temperature,
                )
            )

        logger.info(f"推理完成,总耗时: {total_time:.3f}s")
        return results

    async def _periodic_subtask_refresh(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str,
        state: np.ndarray | None,
        refresh_interval: float,
        max_decoding_steps: int,
        temperature: float,
    ):
        """定期刷新子任务的后台任务"""
        refresh_count = 0

        # 初始化共享状态
        async with self.low_prompt_lock:
            self.current_low_prompt = low_level_prompt

        while True:
            try:
                await asyncio.sleep(refresh_interval)
                refresh_count += 1

                # 获取当前的 low_level_prompt
                async with self.low_prompt_lock:
                    current_low_prompt = self.current_low_prompt

                logger.info(f"开始第 {refresh_count} 次子任务刷新...")
                logger.info(f"当前 low_level_prompt: {current_low_prompt}")

                # 准备新的观察数据,使用当前的 low_level_prompt
                observation = self.prepare_observation(images, high_level_prompt, current_low_prompt, state)

                # 生成新的子任务
                rng = jax.random.key(int(time.time() * 1000) % 2**32)
                subtask_tokens, subtask_text = await self.generate_subtask(
                    observation, rng, max_decoding_steps, temperature
                )

                logger.info(f"第 {refresh_count} 次刷新完成,新子任务: {subtask_text}")

                # 更新共享的 low_level_prompt
                async with self.low_prompt_lock:
                    self.current_low_prompt = subtask_text

                logger.info(f"更新 low_level_prompt: {subtask_text}")

                # 回调函数处理新的子任务(不生成动作)
                await self._on_subtask_refresh(
                    subtask_text,
                    subtask_tokens,
                    refresh_count,
                    subtask_text,
                    None,  # 不传递动作
                )

            except asyncio.CancelledError:
                logger.info("子任务刷新任务被取消")
                break
            except Exception as e:
                logger.error(f"子任务刷新出错: {e}")
                await asyncio.sleep(1)  # 出错后等待1秒再重试

    async def _on_subtask_refresh(
        self,
        subtask_text: str,
        subtask_tokens: jnp.ndarray,
        refresh_count: int,
        updated_low_prompt: str,
        updated_actions: jnp.ndarray,
    ):
        """子任务刷新回调函数,可以被子类重写"""
        # 默认实现:只记录日志
        logger.info(f"子任务已刷新 (第{refresh_count}次): {subtask_text}")
        logger.info(f"更新后的 low_level_prompt: {updated_low_prompt}")
        if updated_actions is not None:
            logger.info(f"基于新 subtask 的动作形状: {np.array(updated_actions[0]).shape}")

        # 子类可以重写此方法来处理新的子任务和动作
        # 例如:发送WebSocket消息、更新数据库、执行动作等

    async def start_continuous_action_generation(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str,
        state: np.ndarray | None = None,
        action_interval: float = 0.5,
        max_actions: int = 100,
    ):
        """持续生成动作序列"""
        action_count = 0

        logger.info(f"开始持续动作生成,间隔: {action_interval}s,最大动作数: {max_actions}")

        while action_count < max_actions:
            try:
                await asyncio.sleep(action_interval)
                action_count += 1

                # 获取当前的 low_level_prompt(线程安全)
                async with self.low_prompt_lock:
                    current_low_prompt = self.current_low_prompt

                # 准备观察数据(使用当前的low_level_prompt)
                observation = self.prepare_observation(images, high_level_prompt, current_low_prompt, state)

                # 生成动作
                rng = jax.random.key(int(time.time() * 1000) % 2**32)
                actions = self.jit_sample_actions(rng, observation)

                logger.info(f"生成第 {action_count} 个动作序列,形状: {np.array(actions[0]).shape}")
                logger.info(f"当前 low_level_prompt: {current_low_prompt}")

                # 回调处理动作
                await self._on_action_generated(actions, action_count, current_low_prompt)

            except asyncio.CancelledError:
                logger.info("持续动作生成被取消")
                break
            except Exception as e:
                logger.error(f"动作生成出错: {e}")
                await asyncio.sleep(1)

    async def _on_action_generated(self, actions: jnp.ndarray, action_count: int, current_low_prompt: str):
        """动作生成回调函数,可以被子类重写"""
        logger.info(f"第 {action_count} 个动作序列已生成")
        # 子类可以重写此方法来处理生成的动作
        # 例如:发送到机器人执行、保存到文件等


async def main():
    """测试异步推理服务器"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 确保模块日志记录器也显示日志
    logger.setLevel(logging.INFO)

    # 创建推理服务器
    inference_server = AsyncPi05Inference(config_name="right_pi05_20", gpu_id=1)

    # 准备测试图像
    img_name_list = ["faceImg.png", "leftImg.png", "rightImg.png"]
    images = {}

    for i, img_name in enumerate(img_name_list):
        img_path = img_name
        img = inference_server.load_image_with_fallback(img_path, img_name)
        key = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"][i]
        images[key] = img

    # 测试推理
    high_level_prompt = "Pick up the flashcard on the table"
    low_level_prompt = ""

    print("开始异步推理测试...")
    results = await inference_server.infer(
        images=images,
        high_level_prompt=high_level_prompt,
        low_level_prompt=low_level_prompt,
        generate_subtask=True,
        max_decoding_steps=200,
        temperature=0.1,
        subtask_refresh_interval=2.0,  # 每2秒刷新一次
    )

    print("推理结果:")
    if results["actions"] is not None:
        print(f"生成的动作形状: {results['actions'].shape}")
    print(f"生成的子任务: {results['subtask']}")
    print(f"时序信息: {results['timing']}")

    # 启动持续动作生成(与subtask刷新并行)
    print("启动持续动作生成...")
    action_task = asyncio.create_task(
        inference_server.start_continuous_action_generation(
            images=images,
            high_level_prompt=high_level_prompt,
            low_level_prompt=low_level_prompt,
            action_interval=0.5,  # 每0.5秒生成一个动作
            max_actions=20,
        )
    )

    # 等待两个任务运行一段时间
    print("等待子任务刷新和动作生成...")
    await asyncio.sleep(10000000000000000000000000)  # 等待10秒,观察两个过程

    # 取消所有任务
    if "subtask_refresh_task" in results:
        results["subtask_refresh_task"].cancel()
        print("已取消子任务刷新任务")

    action_task.cancel()
    print("已取消持续动作生成任务")


if __name__ == "__main__":
    asyncio.run(main())

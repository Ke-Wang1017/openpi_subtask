#!/usr/bin/env python3
"""
LeRobot 数据格式的异步推理观察器
直接使用 inference engine,无需 WebSocket 服务器
"""

import asyncio
from collections.abc import Callable
import json
import logging
from pathlib import Path
import time
import traceback
from typing import Any

from async_pi05_inference import AsyncPi05Inference
import h5py
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LeRobotInferenceObserver:
    """LeRobot 数据格式的异步推理观察器"""

    def __init__(self, config_name: str = "right_pi05_20", gpu_id: int = 1, output_dir: str = "./inference_outputs"):
        self.inference_engine = AsyncPi05Inference(config_name=config_name, gpu_id=gpu_id)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.observation_callbacks: list[Callable] = []
        self._initialized = False

    async def initialize(self):
        """初始化推理引擎"""
        if not self._initialized:
            await self.inference_engine.initialize()
            self._initialized = True
            logger.info("推理引擎初始化完成")

    def load_lerobot_episode(self, episode_path: str, frame_idx: int = 0) -> dict[str, Any]:
        """加载 LeRobot 格式的 episode 数据"""
        episode_data = {}

        try:
            # 尝试加载 HDF5 文件
            if episode_path.endswith((".hdf5", ".h5")):
                with h5py.File(episode_path, "r") as f:
                    # 加载图像数据
                    image_keys = ["base", "left_wrist", "right_wrist"]
                    for key in image_keys:
                        if key in f:
                            episode_data[key] = np.array(f[key])

                    # 加载状态数据
                    if "state" in f:
                        episode_data["state"] = np.array(f["state"])

                    # 加载动作数据
                    if "actions" in f:
                        episode_data["actions"] = np.array(f["actions"])

                    # 加载元数据
                    if "meta" in f:
                        episode_data["meta"] = dict(f["meta"].attrs)

            # 尝试加载 JSONL 文件
            elif episode_path.endswith(".jsonl"):
                with open(episode_path) as f:
                    for line in f:
                        data = json.loads(line.strip())
                        episode_data.update(data)

            logger.info(f"成功加载 LeRobot episode: {episode_path}")
            logger.info(f"数据键: {list(episode_data.keys())}")

        except Exception as e:
            logger.error(f"加载 LeRobot episode 失败: {e}")
            # 返回模拟数据
            episode_data = self._create_fake_lerobot_data()

        return episode_data

    def create_fake_lerobot_data(self) -> dict[str, Any]:
        """创建模拟的 LeRobot 数据"""
        return {
            "base": np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8),
            "left_wrist": np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8),
            "right_wrist": np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8),
            "state": np.random.randn(10, 32).astype(np.float32),
            "actions": np.random.randn(10, 50, 32).astype(np.float32),
            "high_level_prompt": "Pick up the red block and place it in the box",
            "low_level_prompt": "Move to the red block, grasp it, lift it up, move to the box, place it down",
        }

    def prepare_images_from_lerobot(self, episode_data: dict[str, Any], frame_idx: int = 0) -> dict[str, np.ndarray]:
        """从 LeRobot 数据准备图像"""
        images = {}
        lerobot_to_inference_keys = {
            "base": "base_0_rgb",
            "left_wrist": "left_wrist_0_rgb",
            "right_wrist": "right_wrist_0_rgb",
        }

        for lerobot_key, inference_key in lerobot_to_inference_keys.items():
            if lerobot_key in episode_data:
                img_data = episode_data[lerobot_key]

                # 处理不同形状的图像数据
                if len(img_data.shape) == 4:  # (T, H, W, C)
                    img = img_data[frame_idx]
                elif len(img_data.shape) == 3:  # (H, W, C)
                    img = img_data
                else:
                    logger.warning(f"不支持的图像形状: {img_data.shape}")
                    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

                # 确保图像格式正确
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

                images[inference_key] = img
                logger.info(f"准备图像: {inference_key}, 形状: {img.shape}")
            else:
                # 如果缺少图像,使用随机图像
                images[inference_key] = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                logger.warning(f"缺少 {lerobot_key} 图像,使用随机图像")

        return images

    def prepare_state_from_lerobot(self, episode_data: dict[str, Any], frame_idx: int = 0) -> np.ndarray | None:
        """从 LeRobot 数据准备状态"""
        if "state" in episode_data:
            state_data = episode_data["state"]
            if len(state_data.shape) == 2:  # (T, state_dim)
                return state_data[frame_idx]
            if len(state_data.shape) == 1:  # (state_dim,)
                return state_data
            logger.warning(f"不支持的状态形状: {state_data.shape}")
            return np.random.randn(32).astype(np.float32)
        logger.warning("LeRobot 数据中缺少状态信息,使用随机状态")
        return np.random.randn(32).astype(np.float32)

    def add_observation_callback(self, callback: Callable[[dict[str, Any]], None]):
        """添加观察回调函数"""
        self.observation_callbacks.append(callback)

    async def _notify_observers(self, data: dict[str, Any]):
        """通知所有观察者"""
        for callback in self.observation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"观察回调出错: {e}")

    async def observe_single_inference(
        self,
        episode_data: dict[str, Any],
        frame_idx: int = 0,
        high_level_prompt: str | None = None,
        low_level_prompt: str | None = None,
        *,
        generate_subtask: bool = True,
        max_decoding_steps: int = 25,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """观察单次推理"""
        await self.initialize()

        # 准备数据
        images = self.prepare_images_from_lerobot(episode_data, frame_idx)
        state = self.prepare_state_from_lerobot(episode_data, frame_idx)

        # 使用 LeRobot 数据中的 prompt 或提供的 prompt
        if high_level_prompt is None:
            high_level_prompt = episode_data.get("high_level_prompt", "Complete the manipulation task")
        if low_level_prompt is None:
            low_level_prompt = episode_data.get("low_level_prompt", "Execute the planned sequence")

        logger.info(f"开始单次推理观察 (frame {frame_idx}):")
        logger.info(f"  高级任务: {high_level_prompt}")
        logger.info(f"  低级任务: {low_level_prompt}")

        # 执行推理
        start_time = time.time()
        result = await self.inference_engine.infer(
            images=images,
            high_level_prompt=high_level_prompt,
            low_level_prompt=low_level_prompt,
            state=state,
            generate_subtask=generate_subtask,
            max_decoding_steps=max_decoding_steps,
            temperature=temperature,
        )
        inference_time = time.time() - start_time

        # 构建观察数据
        observation_data = {
            "timestamp": time.time(),
            "frame_idx": frame_idx,
            "inference_time": inference_time,
            "result": result,
            "images_shape": {k: v.shape for k, v in images.items()},
            "state_shape": state.shape if state is not None else None,
            "high_level_prompt": high_level_prompt,
            "low_level_prompt": low_level_prompt,
        }

        # 通知观察者
        await self._notify_observers(observation_data)

        # 保存结果
        await self._save_observation(observation_data)

        logger.info(f"推理完成,耗时: {inference_time:.3f}s")
        if result.get("subtask"):
            logger.info(f"生成的子任务: {result['subtask']}")

        return observation_data

    async def observe_continuous_inference(
        self,
        episode_data: dict[str, Any],
        start_frame: int = 0,
        max_frames: int = 10,
        frame_interval: float = 1.0,
        high_level_prompt: str | None = None,
        low_level_prompt: str | None = None,
        subtask_refresh_interval: float | None = None,
    ) -> list[dict[str, Any]]:
        """持续观察推理过程"""
        await self.initialize()

        observations = []
        current_frame = start_frame

        logger.info("开始持续推理观察:")
        logger.info(f"  起始帧: {start_frame}")
        logger.info(f"  最大帧数: {max_frames}")
        logger.info(f"  帧间隔: {frame_interval}s")
        logger.info(f"  子任务刷新间隔: {subtask_refresh_interval}s")

        # 准备初始数据
        images = self.prepare_images_from_lerobot(episode_data, current_frame)
        state = self.prepare_state_from_lerobot(episode_data, current_frame)

        if high_level_prompt is None:
            high_level_prompt = episode_data.get("high_level_prompt", "Complete the manipulation task")
        if low_level_prompt is None:
            low_level_prompt = episode_data.get("low_level_prompt", "Execute the planned sequence")

        # 执行初始推理
        logger.info("执行初始推理...")
        initial_result = await self.inference_engine.infer(
            images=images,
            high_level_prompt=high_level_prompt,
            low_level_prompt=low_level_prompt,
            state=state,
            generate_subtask=True,
            subtask_refresh_interval=subtask_refresh_interval,
        )

        initial_observation = {
            "timestamp": time.time(),
            "frame_idx": current_frame,
            "inference_time": 0,
            "result": initial_result,
            "images_shape": {k: v.shape for k, v in images.items()},
            "state_shape": state.shape if state is not None else None,
            "high_level_prompt": high_level_prompt,
            "low_level_prompt": low_level_prompt,
            "is_initial": True,
        }

        observations.append(initial_observation)
        await self._notify_observers(initial_observation)
        await self._save_observation(initial_observation)

        # 持续观察
        for frame_idx in range(start_frame + 1, min(start_frame + max_frames, len(episode_data.get("base", [])))):
            try:
                await asyncio.sleep(frame_interval)

                # 准备当前帧数据
                current_images = self.prepare_images_from_lerobot(episode_data, frame_idx)
                current_state = self.prepare_state_from_lerobot(episode_data, frame_idx)

                # 执行推理
                start_time = time.time()
                result = await self.inference_engine.infer(
                    images=current_images,
                    high_level_prompt=high_level_prompt,
                    low_level_prompt=low_level_prompt,
                    state=current_state,
                    generate_subtask=True,
                )
                inference_time = time.time() - start_time

                # 构建观察数据
                observation = {
                    "timestamp": time.time(),
                    "frame_idx": frame_idx,
                    "inference_time": inference_time,
                    "result": result,
                    "images_shape": {k: v.shape for k, v in current_images.items()},
                    "state_shape": current_state.shape if current_state is not None else None,
                    "high_level_prompt": high_level_prompt,
                    "low_level_prompt": low_level_prompt,
                    "is_initial": False,
                }

                observations.append(observation)
                await self._notify_observers(observation)
                await self._save_observation(observation)

                logger.info(f"帧 {frame_idx} 推理完成,耗时: {inference_time:.3f}s")
                if result.get("subtask"):
                    logger.info(f"生成的子任务: {result['subtask']}")

            except Exception as e:
                logger.error(f"帧 {frame_idx} 推理失败: {e}")
                continue

        logger.info(f"持续观察完成,共处理 {len(observations)} 帧")
        return observations

    async def _save_observation(self, observation: dict[str, Any]):
        """保存观察数据"""
        timestamp = int(observation["timestamp"])
        frame_idx = observation["frame_idx"]

        # 保存为 JSON 文件
        output_file = self.output_dir / f"observation_{timestamp}_{frame_idx}.json"

        # 准备可序列化的数据
        serializable_data = {
            "timestamp": observation["timestamp"],
            "frame_idx": observation["frame_idx"],
            "inference_time": observation["inference_time"],
            "result": {
                "actions": observation["result"]["actions"].tolist()
                if observation["result"]["actions"] is not None
                else None,
                "subtask": observation["result"]["subtask"],
                "subtask_tokens": observation["result"]["subtask_tokens"].tolist()
                if observation["result"]["subtask_tokens"] is not None
                else None,
                "state": observation["result"]["state"].tolist()
                if observation["result"]["state"] is not None
                else None,
                "timing": observation["result"]["timing"],
            },
            "images_shape": observation["images_shape"],
            "state_shape": observation["state_shape"],
            "high_level_prompt": observation["high_level_prompt"],
            "low_level_prompt": observation["low_level_prompt"],
            "is_initial": observation.get("is_initial", False),
        }

        with open(output_file, "w") as f:
            json.dump(serializable_data, f, indent=2)

        logger.debug(f"观察数据已保存: {output_file}")


async def main():
    """测试 LeRobot 推理观察器"""
    # 创建观察器
    observer = LeRobotInferenceObserver(config_name="right_pi05_20", gpu_id=1, output_dir="./lerobot_inference_outputs")

    # 添加观察回调
    async def on_observation(data):
        logger.info("📊 收到观察数据:")
        logger.info(f"   帧索引: {data['frame_idx']}")
        logger.info(f"   推理耗时: {data['inference_time']:.3f}s")
        if data["result"].get("subtask"):
            logger.info(f"   子任务: {data['result']['subtask']}")

    observer.add_observation_callback(on_observation)

    # 创建模拟的 LeRobot 数据
    fake_episode = observer.create_fake_lerobot_data()

    try:
        # 测试单次推理
        logger.info("=" * 60)
        logger.info("测试单次推理观察")
        logger.info("=" * 60)

        await observer.observe_single_inference(
            episode_data=fake_episode,
            frame_idx=0,
            high_level_prompt="Pick up the red block and place it in the box",
            low_level_prompt="Move to the block, grasp it, lift it up",
        )

        # 测试持续推理
        logger.info("\n" + "=" * 60)
        logger.info("测试持续推理观察")
        logger.info("=" * 60)

        continuous_obs = await observer.observe_continuous_inference(
            episode_data=fake_episode,
            start_frame=0,
            max_frames=5,
            frame_interval=1.0,
            high_level_prompt="Organize the workspace",
            low_level_prompt="Sort items by category",
        )

        logger.info("\n📈 观察总结:")
        logger.info("   单次推理: 成功")
        logger.info(f"   持续推理: {len(continuous_obs)} 帧")
        logger.info(f"   输出目录: {observer.output_dir}")

    except Exception as e:
        logger.error(f"测试失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

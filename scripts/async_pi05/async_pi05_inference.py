import asyncio
import dataclasses
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
from openpi.training import weight_loaders

# GPU memory optimization settings
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_ENABLE_X64"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

PALIGEMMA_EOS_TOKEN = 1

logger = logging.getLogger(__name__)


class AsyncPi05Inference:
    """Asynchronous Pi0.5 inference engine， support subtask generation 
    and action prediction."""

    def __init__(self, config_name: str = "right_pi05_20", gpu_id: int = 1, checkpoint_path: str | None = None):
        self.config_name = config_name
        self.gpu_id = gpu_id
        self.checkpoint_path = checkpoint_path

        self.model = None
        self.tokenizer = None
        self.jit_sample_low_level_task = None
        self.jit_sample_actions = None

        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()

        # Shared prompt state for periodic refresh/action loops.
        self.current_low_prompt = ""
        self.low_prompt_lock = asyncio.Lock()

        # Setup GPU env
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.setdefault("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))

    async def _run_blocking(self, fn, *, use_model_lock: bool = False):
        loop = asyncio.get_running_loop()
        if use_model_lock:
            async with self._model_lock:
                return await loop.run_in_executor(None, fn)
        return await loop.run_in_executor(None, fn)

    def _initialize_blocking(self):
        logger.info("Starting Pi0.5 model initialization...")

        config = get_config(self.config_name)
        if self.checkpoint_path is not None:
            checkpoint_path = self.checkpoint_path
            if os.path.isdir(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_path, "params")
            logger.info("Overriding checkpoint path: %s", checkpoint_path)
            config = dataclasses.replace(
                config,
                weight_loader=weight_loaders.CheckpointWeightLoader(checkpoint_path),
            )

        model_rng = jax.random.key(0)
        model = config.model.create(model_rng)

        graphdef, state = nnx.split(model)
        loader = config.weight_loader
        params = nnx.state(model)

        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        params_shape = params.to_pure_dict()
        loaded_params = loader.load(params_shape)
        state.replace_by_pure_dict(loaded_params)
        self.model = nnx.merge(graphdef, state)

        self.tokenizer = PaligemmaTokenizer(max_len=256)
        self.jit_sample_low_level_task = nnx_utils.module_jit(self.model.sample_low_level_task, static_argnums=(3,))
        self.jit_sample_actions = nnx_utils.module_jit(self.model.sample_actions)

        logger.info("Pi0.5 model initialization completed")

    async def initialize(self):
        """Asynchronously initialize model."""
        if self._initialized:
            return

        async with self._initialize_lock:
            if self._initialized:
                return
            await self._run_blocking(self._initialize_blocking, use_model_lock=True)
            self._initialized = True

    def create_random_image(self, height: int = 224, width: int = 224) -> np.ndarray:
        """Create a random image as fallback."""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    def load_image_with_fallback(self, img_path: str, img_name: str) -> np.ndarray:
        """Load image, fallback to random image when loading fails."""
        if not os.path.exists(img_path):
            logger.warning("Image file does not exist: %s, using random image: %s", img_path, img_name)
            return self.create_random_image()

        img = cv2.imread(img_path)
        if img is not None:
            logger.info("Successfully loaded image: %s, shape: %s", img_name, img.shape)
            return img

        logger.warning("Unable to read image: %s, using random image", img_name)
        return self.create_random_image()

    def prepare_observation(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str = "",
        state: np.ndarray | None = None,
        *,
        mask_subtask_tokens: bool = True,
    ) -> Observation:
        """Prepare model-ready observation data."""
        img_dict = {}
        image_mask_dict = {}

        for key, img in images.items():
            img_array = np.asarray(img, dtype=np.uint8)
            img_dict[key] = jnp.array(img_array[np.newaxis, :, :, :])
            image_mask_dict[key] = jnp.array([not (key == "right_wrist_0_rgb" and not np.any(img_array))], dtype=jnp.bool_)

        if state is None:
            state_vec = np.zeros((32,), dtype=np.float32)
        else:
            state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
            if state_vec.shape[0] < 32:
                state_vec = np.pad(state_vec, ((0, 32 - state_vec.shape[0])), constant_values=0.0)
            elif state_vec.shape[0] > 32:
                state_vec = state_vec[:32]

        state_batch = jnp.asarray(state_vec, dtype=jnp.float32)[np.newaxis, :]

        (
            tokenized_prompt,
            tokenized_prompt_mask,
            token_ar_mask,
            token_loss_mask,
            _subtask_region_mask,
            _action_region_mask,
        ) = self.tokenizer.tokenize_high_low_prompt(high_level_prompt, low_level_prompt, state_vec)

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
            rng,
            observation,
            train=False,
            image_keys=list(observation.images.keys()),
        )

        if mask_subtask_tokens and observation.token_loss_mask is not None:
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
                None,
                new_observation,
                train=False,
                image_keys=list(observation.images.keys()),
            )

        return jax.tree.map(jax.device_put, observation)

    async def generate_subtask(
        self,
        observation: Observation,
        rng: jax.Array,
        max_decoding_steps: int = 25,
        temperature: float = 0.1,
    ) -> tuple[jnp.ndarray, str]:
        """Generate low-level subtask text from observation."""
        start_time = time.time()

        def _sample():
            return self.jit_sample_low_level_task(
                rng,
                observation,
                max_decoding_steps,
                PALIGEMMA_EOS_TOKEN,
                temperature,
            )

        predicted_token, _kv_cache, _mask, _ar_mask = await self._run_blocking(_sample, use_model_lock=True)
        subtask_text = self.tokenizer.detokenize(np.array(predicted_token[0], dtype=np.int32)).strip()

        generation_time = time.time() - start_time
        logger.info("Subtask generation time: %.3fs", generation_time)
        logger.info("Generated subtask: %s", subtask_text)

        return predicted_token, subtask_text

    async def generate_actions(
        self,
        observation: Observation,
        rng: jax.Array,
        noise: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate action trajectory from observation."""

        def _sample():
            if noise is not None:
                noise_tensor = jnp.asarray(noise)
                if noise_tensor.ndim == 2:
                    noise_tensor = noise_tensor[np.newaxis, ...]
                return self.jit_sample_actions(rng, observation, noise=noise_tensor)
            return self.jit_sample_actions(rng, observation)

        sampled_actions = await self._run_blocking(_sample, use_model_lock=True)
        return np.array(sampled_actions[0])

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
        """Asynchronous inference API compatible with existing call sites."""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        rng = jax.random.key(int(time.time() * 1000) % 2**32)

        observation = await self._run_blocking(
            lambda: self.prepare_observation(
                images,
                high_level_prompt,
                low_level_prompt,
                state,
                mask_subtask_tokens=generate_subtask,
            ),
            use_model_lock=True,
        )

        results = {
            "state": np.array(observation.state[0]) if observation.state is not None else None,
            "actions": None,
            "subtask": None,
            "subtask_tokens": None,
            "timing": {},
        }

        if generate_subtask:
            subtask_tokens, subtask_text = await self.generate_subtask(
                observation,
                rng,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
            )
            results["subtask"] = subtask_text
            results["subtask_tokens"] = np.array(subtask_tokens[0])
            action_ms = 0.0
            subtask_ms = (time.time() - start_time) * 1000
        else:
            action_start_time = time.time()
            actions = await self.generate_actions(observation, rng, noise=noise)
            action_ms = (time.time() - action_start_time) * 1000
            subtask_ms = 0.0
            results["actions"] = actions

        total_ms = (time.time() - start_time) * 1000
        results["timing"] = {
            "total_ms": total_ms,
            "action_ms": action_ms,
            "subtask_ms": subtask_ms,
        }

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

        logger.info("Inference completed, total time: %.3fs", total_ms / 1000.0)
        return results

    async def infer_subtask_then_actions(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        state: np.ndarray | None = None,
        *,
        low_level_prompt_for_subtask: str = "",
        max_decoding_steps: int = 25,
        temperature: float = 0.1,
        noise: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run a full pipeline: subtask generation followed by action prediction."""
        start_time = time.time()

        subtask_results = await self.infer(
            images=images,
            high_level_prompt=high_level_prompt,
            low_level_prompt=low_level_prompt_for_subtask,
            state=state,
            generate_subtask=True,
            max_decoding_steps=max_decoding_steps,
            temperature=temperature,
        )

        subtask = subtask_results.get("subtask", "")

        action_results = await self.infer(
            images=images,
            high_level_prompt=high_level_prompt,
            low_level_prompt=subtask,
            state=state,
            generate_subtask=False,
            max_decoding_steps=max_decoding_steps,
            temperature=temperature,
            noise=noise,
        )

        total_ms = (time.time() - start_time) * 1000
        return {
            "state": action_results.get("state"),
            "actions": action_results.get("actions"),
            "subtask": subtask,
            "subtask_tokens": subtask_results.get("subtask_tokens"),
            "timing": {
                "total_ms": total_ms,
                "subtask_ms": subtask_results["timing"].get("total_ms", 0.0),
                "action_ms": action_results["timing"].get("total_ms", 0.0),
            },
        }

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
        """Background task for periodic subtask refresh."""
        refresh_count = 0

        async with self.low_prompt_lock:
            self.current_low_prompt = low_level_prompt

        while True:
            try:
                await asyncio.sleep(refresh_interval)
                refresh_count += 1

                async with self.low_prompt_lock:
                    current_low_prompt = self.current_low_prompt

                result = await self.infer(
                    images=images,
                    high_level_prompt=high_level_prompt,
                    low_level_prompt=current_low_prompt,
                    state=state,
                    generate_subtask=True,
                    max_decoding_steps=max_decoding_steps,
                    temperature=temperature,
                )

                new_subtask = result.get("subtask") or current_low_prompt
                subtask_tokens = result.get("subtask_tokens")

                async with self.low_prompt_lock:
                    self.current_low_prompt = new_subtask

                await self._on_subtask_refresh(
                    new_subtask,
                    subtask_tokens,
                    refresh_count,
                    new_subtask,
                    None,
                )

            except asyncio.CancelledError:
                logger.info("Subtask refresh task cancelled")
                break
            except Exception as e:
                logger.error("Error in subtask refresh: %s", e)
                await asyncio.sleep(1)

    async def _on_subtask_refresh(
        self,
        subtask_text: str,
        subtask_tokens: np.ndarray | None,
        refresh_count: int,
        updated_low_prompt: str,
        updated_actions: np.ndarray | None,
    ):
        """Subtask refresh callback function, can be overridden by subclasses."""
        logger.info("Subtask refreshed (count: %d): %s", refresh_count, subtask_text)
        logger.info("Updated low_level_prompt: %s", updated_low_prompt)
        if updated_actions is not None:
            logger.info("Action shape based on new subtask: %s", np.array(updated_actions).shape)

    async def start_continuous_action_generation(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str,
        state: np.ndarray | None = None,
        action_interval: float = 0.5,
        max_actions: int = 100,
    ):
        """Continuously generate action sequences."""
        action_count = 0

        async with self.low_prompt_lock:
            self.current_low_prompt = low_level_prompt

        logger.info("Starting continuous action generation, interval: %ss, max actions: %d", action_interval, max_actions)

        while action_count < max_actions:
            try:
                await asyncio.sleep(action_interval)
                action_count += 1

                async with self.low_prompt_lock:
                    current_low_prompt = self.current_low_prompt

                result = await self.infer(
                    images=images,
                    high_level_prompt=high_level_prompt,
                    low_level_prompt=current_low_prompt,
                    state=state,
                    generate_subtask=False,
                )
                actions = result.get("actions")

                await self._on_action_generated(actions, action_count, current_low_prompt)

            except asyncio.CancelledError:
                logger.info("Continuous action generation cancelled")
                break
            except Exception as e:
                logger.error("Error in action generation: %s", e)
                await asyncio.sleep(1)

    async def _on_action_generated(self, actions: np.ndarray | None, action_count: int, current_low_prompt: str):
        """Action generation callback function, can be overridden by subclasses."""
        logger.info("%dth action sequence generated with prompt: %s", action_count, current_low_prompt)
        if actions is not None:
            logger.info("Action shape: %s", np.array(actions).shape)


async def main():
    """Quick smoke test entrypoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    inference_server = AsyncPi05Inference(config_name="libero_pi05_subtask_hybrid", gpu_id=1)
    await inference_server.initialize()


if __name__ == "__main__":
    asyncio.run(main())

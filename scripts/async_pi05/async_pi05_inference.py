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
max_decoding_steps = 25
temperature = 0.1

logger = logging.getLogger(__name__)


class AsyncPi05Inference:
    """Asynchronous Pi0.5 inference server, supports subtask generation"""

    def __init__(self, config_name: str = "right_pi05_20", gpu_id: int = 1, checkpoint_path: str | None = None):
        self.config_name = config_name
        self.gpu_id = gpu_id
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.jit_sample_low_level_task = None
        self.jit_sample_actions = None
        self._initialized = False

        # Shared state
        self.current_low_prompt = None
        self.low_prompt_lock = asyncio.Lock()

        # Setup GPU
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.setdefault("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))

    async def initialize(self):
        """Asynchronously initialize model"""
        if self._initialized:
            return

        logger.info("Starting Pi0.5 model initialization...")

        # Initialize model config
        config = get_config(self.config_name)
        if self.checkpoint_path is not None:
            checkpoint_path = self.checkpoint_path
            if os.path.isdir(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_path, "params")
            logging.info("Overriding checkpoint path: %s", checkpoint_path)
            config = dataclasses.replace(
                config,
                weight_loader=weight_loaders.CheckpointWeightLoader(checkpoint_path),
            )
        model_rng = jax.random.key(0)

        # Create model
        self.model = config.model.create(model_rng)

        # Load pretrained parameters
        graphdef, state = nnx.split(self.model)
        loader = config.weight_loader
        params = nnx.state(self.model)

        # Convert parameters to bfloat16
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        # Load parameters
        params_shape = params.to_pure_dict()
        loaded_params = loader.load(params_shape)
        state.replace_by_pure_dict(loaded_params)
        self.model = nnx.merge(graphdef, state)

        # Initialize tokenizer
        self.tokenizer = PaligemmaTokenizer(max_len=256)

        # JIT compile key functions
        self.jit_sample_low_level_task = nnx_utils.module_jit(self.model.sample_low_level_task, static_argnums=(3,))
        self.jit_sample_actions = nnx_utils.module_jit(self.model.sample_actions)

        self._initialized = True
        logger.info("Pi0.5 model initialization completed")

    def create_random_image(self, height: int = 224, width: int = 224) -> np.ndarray:
        """创建随机图像作为 fallback"""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    def load_image_with_fallback(self, img_path: str, img_name: str) -> np.ndarray:
        """Load image, supports fallback to random image"""
        if not os.path.exists(img_path):
            logger.warning(f"Image file does not exist: {img_path}, using random image: {img_name}")
            return self.create_random_image()

        img = cv2.imread(img_path)
        if img is not None:
            logger.info(f"Successfully loaded image: {img_name}, shape: {img.shape}")
            return img
        logger.warning(f"Unable to read image: {img_name}, using random image")
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
        """Prepare observation data"""

        # Convert images to model expected format [-1, 1]
        img_dict = {}
        for key, img in images.items():
            img_dict[key] = jnp.array(img[np.newaxis, :, :, :]).astype(jnp.float32)

        # Prepare state data
        state = jnp.zeros((1, 32), dtype=jnp.float32) if state is None else jnp.array(state)[np.newaxis, :]

        # Tokenize prompts
        (
            tokenized_prompt,
            tokenized_prompt_mask,
            token_ar_mask,
            token_loss_mask,
            _subtask_region_mask,
            _action_region_mask,
        ) = self.tokenizer.tokenize_high_low_prompt(high_level_prompt, low_level_prompt, state)
        # Build observation data
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

        if mask_subtask_tokens and observation.token_loss_mask is not None:
            # Set low-level task tokens based on loss mask
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
        """Generate subtask"""
        start_time = time.time()

        # Generate subtask tokens
        predicted_token, _kv_cache, _mask, _ar_mask = self.jit_sample_low_level_task(
            rng, observation, max_decoding_steps, PALIGEMMA_EOS_TOKEN, temperature
        )

        # Decode generated subtask
        subtask_text = self.tokenizer.detokenize(np.array(predicted_token[0], dtype=np.int32))

        generation_time = time.time() - start_time
        logger.info(f"Subtask generation time: {generation_time:.3f}s")
        logger.info(f"Generated subtask: {subtask_text}")

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
        Asynchronous inference function, supports subtask generation and periodic refresh

        Args:
            images: Image dictionary, keys are image types, values are image arrays
            high_level_prompt: High-level task description
            low_level_prompt: Low-level task description (optional)
            state: Robot state
            generate_subtask: Whether to generate subtask (if True, actions are not generated, actions are handled by continuous generation)
            max_decoding_steps: Maximum decoding steps
            temperature: Sampling temperature
            noise: Action noise (optional, only used when generate_subtask=False)
            subtask_refresh_interval: Subtask refresh interval (seconds), None means no refresh

        Returns:
            Dictionary containing subtask and timing information (actions are handled by continuous generation)
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        rng = jax.random.key(int(time.time() * 1000) % 2**32)

        # Prepare observation data
        observation = self.prepare_observation(
            images, high_level_prompt, low_level_prompt, state, mask_subtask_tokens=generate_subtask
        )

        results = {
            "state": np.array(observation.state[0]) if observation.state is not None else None,
            "actions": None,
            "subtask": None,
            "subtask_tokens": None,
            "timing": {},
        }

        # Generate subtask (if needed)
        if generate_subtask:
            subtask_tokens, subtask_text = await self.generate_subtask(
                observation, rng, max_decoding_steps, temperature
            )
            results["subtask"] = subtask_text
            results["subtask_tokens"] = np.array(subtask_tokens[0])

        # If action generation is not needed, skip action generation
        if not generate_subtask:
            # Only generate actions, do not generate subtask
            action_start_time = time.time()
            if noise is not None:
                noise = jnp.array(noise)[np.newaxis, ...] if noise.ndim == 2 else jnp.array(noise)
                sampled_actions = self.jit_sample_actions(rng, observation, noise=noise)
            else:
                sampled_actions = self.jit_sample_actions(rng, observation)

            action_time = time.time() - action_start_time
            # sampled_actions is (x_0, output_tokens) where x_0 has shape (batch, horizon, dim)
            # Extract first batch: x_0[0] gives shape (horizon, dim)
            results["actions"] = np.array(sampled_actions[0][0])

            total_time = time.time() - start_time
            results["timing"] = {"total_ms": total_time * 1000, "action_ms": action_time * 1000, "subtask_ms": 0}
        else:
            # Only generate subtask, do not generate actions (actions are handled by continuous generation)
            total_time = time.time() - start_time
            results["timing"] = {"total_ms": total_time * 1000, "action_ms": 0, "subtask_ms": total_time * 1000}

        # If refresh interval is set, start periodic refresh task
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

        logger.info(f"Inference completed, total time: {total_time:.3f}s")
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
        """Background task for periodic subtask refresh"""
        refresh_count = 0

        # Initialize shared state
        async with self.low_prompt_lock:
            self.current_low_prompt = low_level_prompt

        while True:
            try:
                await asyncio.sleep(refresh_interval)
                refresh_count += 1

                # Get current low_level_prompt
                async with self.low_prompt_lock:
                    current_low_prompt = self.current_low_prompt

                logger.info(f"Starting {refresh_count}th subtask refresh...")
                logger.info(f"Current low_level_prompt: {current_low_prompt}")

                # Prepare new observation data, using current low_level_prompt
                observation = self.prepare_observation(
                    images, high_level_prompt, current_low_prompt, state, mask_subtask_tokens=True
                )

                # Generate new subtask
                rng = jax.random.key(int(time.time() * 1000) % 2**32)
                subtask_tokens, subtask_text = await self.generate_subtask(
                    observation, rng, max_decoding_steps, temperature
                )

                logger.info(f"{refresh_count}th refresh completed, new subtask: {subtask_text}")

                # Update shared low_level_prompt
                async with self.low_prompt_lock:
                    self.current_low_prompt = subtask_text

                logger.info(f"Updated low_level_prompt: {subtask_text}")

                # Callback function to handle new subtask (do not generate actions)
                await self._on_subtask_refresh(
                    subtask_text,
                    subtask_tokens,
                    refresh_count,
                    subtask_text,
                    None,  # Do not pass actions
                )

            except asyncio.CancelledError:
                logger.info("Subtask refresh task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in subtask refresh: {e}")
                await asyncio.sleep(1)  # Wait 1 second before retrying after error

    async def _on_subtask_refresh(
        self,
        subtask_text: str,
        subtask_tokens: jnp.ndarray,
        refresh_count: int,
        updated_low_prompt: str,
        updated_actions: jnp.ndarray,
    ):
        """Subtask refresh callback function, can be overridden by subclasses"""
        # Default implementation: only log
        logger.info(f"Subtask refreshed (count: {refresh_count}): {subtask_text}")
        logger.info(f"Updated low_level_prompt: {updated_low_prompt}")
        if updated_actions is not None:
            logger.info(f"Action shape based on new subtask: {np.array(updated_actions[0]).shape}")

        # Subclasses can override this method to handle new subtasks and actions
        # For example: send WebSocket messages, update database, execute actions, etc.

    async def start_continuous_action_generation(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str,
        state: np.ndarray | None = None,
        action_interval: float = 0.5,
        max_actions: int = 100,
    ):
        """Continuously generate action sequences"""
        action_count = 0

        logger.info(f"Starting continuous action generation, interval: {action_interval}s, max actions: {max_actions}")

        while action_count < max_actions:
            try:
                await asyncio.sleep(action_interval)
                action_count += 1

                # Get current low_level_prompt (thread-safe)
                async with self.low_prompt_lock:
                    current_low_prompt = self.current_low_prompt

                # Prepare observation data (using current low_level_prompt)
                observation = self.prepare_observation(
                    images, high_level_prompt, current_low_prompt, state, mask_subtask_tokens=False
                )

                # Generate actions
                rng = jax.random.key(int(time.time() * 1000) % 2**32)
                actions = self.jit_sample_actions(rng, observation)

                logger.info(f"Generated {action_count}th action sequence, shape: {np.array(actions[0]).shape}")
                logger.info(f"Current low_level_prompt: {current_low_prompt}")

                # Callback to handle actions
                await self._on_action_generated(actions, action_count, current_low_prompt)

            except asyncio.CancelledError:
                logger.info("Continuous action generation cancelled")
                break
            except Exception as e:
                logger.error(f"Error in action generation: {e}")
                await asyncio.sleep(1)

    async def _on_action_generated(self, actions: jnp.ndarray, action_count: int, current_low_prompt: str):
        """Action generation callback function, can be overridden by subclasses"""
        logger.info(f"{action_count}th action sequence generated")
        # Subclasses can override this method to handle generated actions
        # For example: send to robot for execution, save to file, etc.


async def main():
    """Test asynchronous inference server"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Ensure module logger also displays logs
    logger.setLevel(logging.INFO)

    # Create inference server
    inference_server = AsyncPi05Inference(config_name="right_pi05_20", gpu_id=1)

    # Prepare test images
    img_name_list = ["faceImg.png", "leftImg.png", "rightImg.png"]
    images = {}

    for i, img_name in enumerate(img_name_list):
        img_path = img_name
        img = inference_server.load_image_with_fallback(img_path, img_name)
        key = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"][i]
        images[key] = img

    # Test inference
    high_level_prompt = "Pick up the flashcard on the table"
    low_level_prompt = ""

    print("Starting asynchronous inference test...")
    results = await inference_server.infer(
        images=images,
        high_level_prompt=high_level_prompt,
        low_level_prompt=low_level_prompt,
        generate_subtask=True,
        max_decoding_steps=200,
        temperature=0.1,
        subtask_refresh_interval=2.0,  # Refresh every 2 seconds
    )

    print("Inference results:")
    if results["actions"] is not None:
        print(f"Generated action shape: {results['actions'].shape}")
    print(f"Generated subtask: {results['subtask']}")
    print(f"Timing info: {results['timing']}")

    # Start continuous action generation (parallel with subtask refresh)
    print("Starting continuous action generation...")
    action_task = asyncio.create_task(
        inference_server.start_continuous_action_generation(
            images=images,
            high_level_prompt=high_level_prompt,
            low_level_prompt=low_level_prompt,
            action_interval=0.5,  # Generate one action every 0.5 seconds
            max_actions=20,
        )
    )

    # Wait for both tasks to run for a period
    print("Waiting for subtask refresh and action generation...")
    await asyncio.sleep(10000000000000000000000000)  # Wait 10 seconds, observe both processes

    # Cancel all tasks
    if "subtask_refresh_task" in results:
        results["subtask_refresh_task"].cancel()
        print("Cancelled subtask refresh task")

    action_task.cancel()
    print("Cancelled continuous action generation task")


if __name__ == "__main__":
    asyncio.run(main())

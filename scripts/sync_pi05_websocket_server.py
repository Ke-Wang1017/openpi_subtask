from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
import threading
import numpy as np
import tyro
import websockets
import websockets.sync.server

PALIGEMMA_EOS_TOKEN = 1

# GPU memory optimization settings (must be set before importing JAX)
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_ENABLE_X64", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

# Lazy-import heavy deps to avoid blocking before logging is configured.
jax = None
jnp = None
nnx = None
_model = None
Observation = None
PaligemmaTokenizer = None
nnx_utils = None
get_config = None
weight_loaders = None


def _lazy_imports() -> None:
    global jax, jnp, nnx, _model, Observation, PaligemmaTokenizer, nnx_utils, get_config, weight_loaders
    if jax is not None:
        return
    from flax import nnx as _nnx
    import jax as _jax
    import jax.numpy as _jnp

    from openpi.models import model as _openpi_model
    from openpi.models.model import Observation as _Observation
    from openpi.models.tokenizer import PaligemmaTokenizer as _PaligemmaTokenizer
    import openpi.shared.nnx_utils as _nnx_utils
    from openpi.training.config import get_config as _get_config
    from openpi.training import weight_loaders as _weight_loaders

    nnx = _nnx
    jax = _jax
    jnp = _jnp
    _model = _openpi_model
    Observation = _Observation
    PaligemmaTokenizer = _PaligemmaTokenizer
    nnx_utils = _nnx_utils
    get_config = _get_config
    weight_loaders = _weight_loaders


class SyncPi05Inference:
    """Synchronous Pi0.5 inference with optional subtask generation."""

    def __init__(self, config_name: str, gpu_id: int, checkpoint_path: str | None):
        self.config_name = config_name
        self.gpu_id = gpu_id
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.jit_sample_low_level_task = None
        self.jit_sample_actions = None
        self._initialized = False
        self._init_lock = threading.Lock()

        os.environ.setdefault("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return

            logging.info("Initializing Pi0.5 model (may take a while)...")
            _lazy_imports()

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
            self.model = config.model.create(model_rng)

            graphdef, state = nnx.split(self.model)
            loader = config.weight_loader
            params = nnx.state(self.model)

            params = nnx_utils.state_map(
                params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
            )

            params_shape = params.to_pure_dict()
            loaded_params = loader.load(params_shape)
            state.replace_by_pure_dict(loaded_params)
            self.model = nnx.merge(graphdef, state)

            self.tokenizer = PaligemmaTokenizer(max_len=256)
            self.jit_sample_low_level_task = nnx_utils.module_jit(self.model.sample_low_level_task, static_argnums=(3,))
            self.jit_sample_actions = nnx_utils.module_jit(self.model.sample_actions)

            self._initialized = True
            logging.info("Pi0.5 model initialization completed")

    def prepare_observation(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str,
        state: np.ndarray | None,
        *,
        mask_subtask_tokens: bool,
    ) -> Observation:
        _lazy_imports()
        img_dict = {}
        for key, img in images.items():
            img_dict[key] = jnp.array(img[np.newaxis, :, :, :]).astype(jnp.float32)

        state = jnp.zeros((1, 32), dtype=jnp.float32) if state is None else jnp.array(state)[np.newaxis, :]

        (
            tokenized_prompt,
            tokenized_prompt_mask,
            token_ar_mask,
            token_loss_mask,
            _subtask_region_mask,
            _action_region_mask,
        ) = self.tokenizer.tokenize_high_low_prompt(high_level_prompt, low_level_prompt, state)

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

    def generate_subtask(
        self, observation: Observation, rng: jax.Array, max_decoding_steps: int, temperature: float
    ) -> tuple[jnp.ndarray, str]:
        start_time = time.time()

        predicted_token, _kv_cache, _mask, _ar_mask = self.jit_sample_low_level_task(
            rng, observation, max_decoding_steps, PALIGEMMA_EOS_TOKEN, temperature
        )
        subtask_text = self.tokenizer.detokenize(np.array(predicted_token[0], dtype=np.int32))

        generation_time = time.time() - start_time
        logging.info("Subtask generation time: %.3fs", generation_time)
        logging.info("Generated subtask: %s", subtask_text)

        return predicted_token, subtask_text

    def infer(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str,
        state: np.ndarray | None,
        *,
        generate_subtask: bool,
        max_decoding_steps: int,
        temperature: float,
        noise: np.ndarray | None = None,
    ) -> dict:
        if not self._initialized:
            self.initialize()

        start_time = time.time()
        rng = jax.random.key(int(time.time() * 1000) % 2**32)

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

        if generate_subtask:
            subtask_tokens, subtask_text = self.generate_subtask(
                observation, rng, max_decoding_steps, temperature
            )
            results["subtask"] = subtask_text
            results["subtask_tokens"] = np.array(subtask_tokens[0])
            total_time = time.time() - start_time
            results["timing"] = {"total_ms": total_time * 1000, "action_ms": 0, "subtask_ms": total_time * 1000}
            logging.info("Inference completed, total time: %.3fs", total_time)
            return results

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

        logging.info("Inference completed, total time: %.3fs", total_time)
        return results


class SyncPi05WebSocketServer:
    """Synchronous Pi0.5 inference server (JSON over websockets)."""

    def __init__(
        self,
        host: str,
        port: int,
        config_name: str,
        gpu_id: int,
        checkpoint_path: str | None,
    ) -> None:
        self.host = host
        self.port = port
        self.inference = SyncPi05Inference(config_name, gpu_id, checkpoint_path)
        self._latest_observations = {}

    def _handle(self, websocket) -> None:
        metadata = {
            "server_type": "SyncPi05Inference",
            "version": "1.0.0",
            "capabilities": ["subtask_generation", "action_prediction"],
            "max_decoding_steps": 25,
            "supported_image_types": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
            "refresh_mode": "client_pull",
        }
        websocket.send(json.dumps(metadata))

        try:
            for message in websocket:
                try:
                    request = json.loads(message)
                except json.JSONDecodeError:
                    websocket.send(json.dumps({"error": "Invalid JSON format", "status": "error"}))
                    continue

                if request.get("type") == "update_observation":
                    self._latest_observations[id(websocket)] = request
                    continue

                try:
                    response = self._process_request(request)
                except Exception as exc:
                    logging.exception("Error processing request")
                    response = {"error": str(exc), "status": "error"}

                websocket.send(json.dumps(response))
        except websockets.ConnectionClosed:
            logging.info("Client disconnected")
        finally:
            self._latest_observations.pop(id(websocket), None)

    def _process_request(self, request: dict) -> dict:
        if "images" not in request or "high_level_prompt" not in request:
            return {"error": "Missing required fields: images, high_level_prompt", "status": "error"}

        images_data = request["images"]
        high_level_prompt = request["high_level_prompt"]
        low_level_prompt = request.get("low_level_prompt", "")
        state = request.get("state")
        generate_subtask = request.get("generate_subtask", True)
        max_decoding_steps = request.get("max_decoding_steps", 25)
        temperature = request.get("temperature", 0.1)
        noise = request.get("noise")

        images = {key: np.array(img, dtype=np.uint8) for key, img in images_data.items()}
        state_array = np.array(state, dtype=np.float32) if state is not None else None
        noise_array = np.array(noise, dtype=np.float32) if noise is not None else None

        start_time = time.time()
        results = self.inference.infer(
            images=images,
            high_level_prompt=high_level_prompt,
            low_level_prompt=low_level_prompt,
            state=state_array,
            generate_subtask=generate_subtask,
            max_decoding_steps=max_decoding_steps,
            temperature=temperature,
            noise=noise_array,
        )
        total_time = time.time() - start_time

        actions = results["actions"]
        if actions is not None and actions.ndim == 2:
            actions = actions[None, ...]

        return {
            "status": "success",
            "actions": actions.tolist() if actions is not None else None,
            "subtask": results["subtask"],
            "subtask_tokens": results["subtask_tokens"].tolist() if results["subtask_tokens"] is not None else None,
            "state": results["state"].tolist() if results["state"] is not None else None,
            "timing": results["timing"],
            "server_timing": {"total_ms": total_time * 1000},
            "subtask_refresh_enabled": False,
        }

    def serve_forever(self, *, skip_init: bool = False) -> None:
        logging.info("Starting sync Pi0.5 WebSocket server: %s:%s", self.host, self.port)

        with websockets.sync.server.serve(
            self._handle,
            self.host,
            self.port,
            max_size=10 * 1024 * 1024,
            compression=None,
        ):
            logging.info("Server started, listening on %s:%s", self.host, self.port)
            if skip_init:
                logging.warning("Skipping model initialization (--skip-init). Inference unavailable.")
            else:
                threading.Thread(target=self._init_background, daemon=True).start()
            try:
                while True:
                    time.sleep(3600)
            except KeyboardInterrupt:
                logging.info("Shutting down server")

    def _init_background(self) -> None:
        try:
            self.inference.initialize()
        except Exception:
            logging.exception("Model initialization failed")


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8765
    config: str = "libero_pi05_action_expert"
    gpu_id: int = 0
    checkpoint: str | None = None
    skip_init: bool = False
    log_level: str = "INFO"


def main() -> None:
    args = tyro.cli(Args)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    server = SyncPi05WebSocketServer(
        host=args.host,
        port=args.port,
        config_name=args.config,
        gpu_id=args.gpu_id,
        checkpoint_path=args.checkpoint,
    )
    server.serve_forever(skip_init=args.skip_init)


if __name__ == "__main__":
    main()

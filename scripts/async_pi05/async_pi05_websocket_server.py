import argparse
import asyncio
import json
import logging
import os
import pathlib
import time
from typing import Any

from async_pi05_inference import AsyncPi05Inference
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)

# Image key mapping: Various input keys -> Model keys.
# Model expects: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
INPUT_TO_MODEL_IMAGE_KEYS = {
    # LIBERO-style keys -> Model keys
    "agentview_rgb": "base_0_rgb",
    "wrist_rgb_left": "left_wrist_0_rgb",
    "wrist_rgb": "left_wrist_0_rgb",
    # Model-style keys (pass-through)
    "base_0_rgb": "base_0_rgb",
    "left_wrist_0_rgb": "left_wrist_0_rgb",
    "right_wrist_0_rgb": "right_wrist_0_rgb",
}


def map_image_keys_to_model(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map input image keys to model keys and inject a right wrist placeholder when missing."""
    mapped = {}
    for key, value in images.items():
        mapped[INPUT_TO_MODEL_IMAGE_KEYS.get(key, key)] = value

    # LIBERO normally has only two cameras.
    if "right_wrist_0_rgb" not in mapped and mapped:
        template = next(iter(mapped.values()))
        mapped["right_wrist_0_rgb"] = np.zeros_like(template)

    return mapped


<<<<<<< Updated upstream
def load_norm_stats(checkpoint_path: str | None, config_name: str) -> dict | None:
    """Load normalization statistics from checkpoint or config assets."""
    if checkpoint_path is None:
        return None
    
    # Try to find norm_stats.json in checkpoint assets
=======
def _log_norm_values(norm_stats: dict) -> None:
    """Log normalization values used at inference time."""
    for key in ("state", "actions"):
        stats = norm_stats.get(key)
        if stats is None:
            logger.info("Normalization stats missing key: %s", key)
            continue

        for stat_name in ("mean", "std", "q01", "q99"):
            value = stats.get(stat_name)
            if value is None:
                logger.info("norm_stats[%s][%s] = None", key, stat_name)
            else:
                logger.info("norm_stats[%s][%s] = %s", key, stat_name, np.array(value, dtype=np.float32))


def load_norm_stats(checkpoint_path: str | None, config_name: str) -> tuple[dict | None, str | None]:
    """Load normalization statistics from checkpoint assets."""
    if checkpoint_path is None:
        raise ValueError(
            f"Normalization stats are required for inference, but checkpoint_path is None (config={config_name})."
        )

>>>>>>> Stashed changes
    checkpoint_dir = pathlib.Path(checkpoint_path)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent

    search_paths = [
        checkpoint_dir / "assets" / "KeWangRobotics" / "libero_10_subtasks" / "norm_stats.json",
    ]

    for path in search_paths:
        if path.exists():
            logger.info("Loading norm_stats from: %s", path)
            with open(path) as f:
                data = json.load(f)
                return data.get("norm_stats", data)
    
    logger.warning("No norm_stats found, skipping normalization")
    return None


<<<<<<< Updated upstream
def normalize_state(state: np.ndarray, norm_stats: dict, pad_to_dim: int = 32, use_quantiles: bool = True) -> np.ndarray:
    """Normalize state using quantile stats from norm_stats and pad to expected dimension.
    
    Args:
        state: Raw state array (e.g., 8D for LIBERO)
        norm_stats: Dictionary with 'state' key containing 'q01', 'q99' (and 'mean', 'std')
        pad_to_dim: Dimension to pad state to (default 8 for Pi05 model)
        use_quantiles: If True, use quantile normalization. Otherwise, use z-score.
    
    Returns:
        Normalized and padded state array
    """
=======
def normalize_state(state: np.ndarray, norm_stats: dict | None, pad_to_dim: int = 32, use_quantiles: bool = True) -> np.ndarray:
    """Normalize state and pad to model dimension."""
    state = np.asarray(state, dtype=np.float32)

>>>>>>> Stashed changes
    if norm_stats is None or "state" not in norm_stats:
        if state.shape[-1] < pad_to_dim:
            pad_width = [(0, 0)] * (state.ndim - 1) + [(0, pad_to_dim - state.shape[-1])]
            state = np.pad(state, pad_width, constant_values=0.0)
        return state

    stats = norm_stats["state"]
    normalized = state.copy().astype(np.float32)

    if use_quantiles:
        q01 = np.array(stats["q01"], dtype=np.float32)
        q99 = np.array(stats["q99"], dtype=np.float32)
        state_dim = min(state.shape[-1], len(q01))
        normalized[..., :state_dim] = (
            (state[..., :state_dim] - q01[:state_dim]) / (q99[:state_dim] - q01[:state_dim] + 1e-6) * 2.0 - 1.0
        )
    else:
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        state_dim = min(state.shape[-1], len(mean))
        normalized[..., :state_dim] = (state[..., :state_dim] - mean[:state_dim]) / (std[:state_dim] + 1e-6)

    if normalized.shape[-1] < pad_to_dim:
        pad_width = [(0, 0)] * (normalized.ndim - 1) + [(0, pad_to_dim - normalized.shape[-1])]
        normalized = np.pad(normalized, pad_width, constant_values=0.0)

    return normalized


def unnormalize_actions(actions: np.ndarray, norm_stats: dict | None, use_quantiles: bool = True) -> np.ndarray:
    """Unnormalize model actions to action space."""
    if norm_stats is None or "actions" not in norm_stats:
        return actions

    stats = norm_stats["actions"]
    unnormalized = actions.copy()

    if use_quantiles:
        q01 = np.array(stats["q01"], dtype=np.float32)
        q99 = np.array(stats["q99"], dtype=np.float32)
        action_dim = min(actions.shape[-1], len(q01))
        unnormalized[..., :action_dim] = (
            (actions[..., :action_dim] + 1.0) / 2.0 * (q99[:action_dim] - q01[:action_dim] + 1e-6)
            + q01[:action_dim]
        )
    else:
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        action_dim = min(actions.shape[-1], len(mean))
        unnormalized[..., :action_dim] = actions[..., :action_dim] * (std[:action_dim] + 1e-6) + mean[:action_dim]

    return unnormalized


def _to_list(value: Any) -> Any:
    if value is None:
        return None
    return np.asarray(value).tolist()


class AsyncPi05WebSocketServer:
    """Truly asynchronous Pi0.5 inference WebSocket server for LIBERO."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        config_name: str = "right_pi05_20",
        gpu_id: int = 1,
        checkpoint_path: str | None = None,
    ):
        self.host = host
        self.port = port
        self.config_name = config_name
        self.checkpoint_path = checkpoint_path

        self.inference_engine = AsyncPi05Inference(
            config_name=config_name,
            gpu_id=gpu_id,
            checkpoint_path=checkpoint_path,
        )
<<<<<<< Updated upstream
        self.clients = set()
        self.norm_stats = None  # Will be loaded during initialization
=======

        self.clients: set[WebSocketServerProtocol] = set()
        self.norm_stats: dict[str, Any] | None = None
        self.norm_stats_path: str | None = None

        self.active_refresh_tasks: dict[WebSocketServerProtocol, asyncio.Task] = {}
        self.send_locks: dict[WebSocketServerProtocol, asyncio.Lock] = {}

    async def _send_json(self, websocket: WebSocketServerProtocol, payload: dict[str, Any]) -> None:
        lock = self.send_locks.setdefault(websocket, asyncio.Lock())
        async with lock:
            await websocket.send(json.dumps(payload))
>>>>>>> Stashed changes

    async def register_client(self, websocket: WebSocketServerProtocol):
        self.clients.add(websocket)
        self.send_locks.setdefault(websocket, asyncio.Lock())
        logger.info("Client connected: %s", websocket.remote_address)

    async def _cancel_refresh_task(self, websocket: WebSocketServerProtocol):
        task = self.active_refresh_tasks.pop(websocket, None)
        if task is None:
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def unregister_client(self, websocket: WebSocketServerProtocol):
        self.clients.discard(websocket)
        self.send_locks.pop(websocket, None)
        await self._cancel_refresh_task(websocket)
        logger.info("Client disconnected: %s", websocket.remote_address)

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str | None = None):
        await self.register_client(websocket)

        try:
            metadata = {
                "server_type": "AsyncPi05Inference",
                "version": "3.0.0",
                "capabilities": ["subtask_generation", "action_prediction", "periodic_refresh"],
                "max_decoding_steps": 25,
                "supported_image_types": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
                "alternative_image_types": ["agentview_rgb", "wrist_rgb_left"],
                "normalization_enabled": self.norm_stats is not None,
                "request_id_supported": True,
            }
            await self._send_json(websocket, metadata)

            async for message in websocket:
                try:
                    request = json.loads(message)
                    response = await self.process_request(websocket, request)
                    await self._send_json(websocket, response)
                except json.JSONDecodeError:
                    await self._send_json(websocket, {"error": "Invalid JSON format", "status": "error"})
                except Exception as e:
                    logger.exception("Error processing request")
                    await self._send_json(websocket, {"error": str(e), "status": "error"})

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed: %s", websocket.remote_address)
        finally:
            await self.unregister_client(websocket)

    async def process_request(self, websocket: WebSocketServerProtocol, request: dict[str, Any]) -> dict[str, Any]:
        """Process one request; supports subtask/action generation and refresh scheduling."""
        if "images" not in request or "high_level_prompt" not in request:
            return {"error": "Missing required fields: images, high_level_prompt", "status": "error"}

        request_id = request.get("request_id")

        images_data = request["images"]
        high_level_prompt = request["high_level_prompt"]
        low_level_prompt = request.get("low_level_prompt", "")
        state = request.get("state")

<<<<<<< Updated upstream
            # Convert state data, normalize (quantile), and pad to 8D
            state_array = None
            if state is not None:
                raw_state = np.array(state, dtype=np.float32)
                # Log raw gripper state (last 2 dims of 8D state)
                if len(raw_state) >= 8:
                    print(f"[GRIPPER DEBUG] Raw gripper state (dims 6-7): {raw_state[6:8]}")
                # Apply quantile normalization and padding
                state_array = normalize_state(raw_state, self.norm_stats, pad_to_dim=8, use_quantiles=True)
                if len(raw_state) >= 8:
                    print(f"[GRIPPER DEBUG] Quantile-normalized gripper state (dims 6-7): {state_array[6:8]}")
                print(f"[GRIPPER DEBUG] Full normalized state shape: {state_array.shape}")
=======
        generate_subtask = request.get("generate_subtask", True)
        generate_actions = request.get("generate_actions", True)
>>>>>>> Stashed changes

        max_decoding_steps = int(request.get("max_decoding_steps", 25))
        temperature = float(request.get("temperature", 0.1))
        noise = request.get("noise")
        subtask_refresh_interval = request.get("subtask_refresh_interval")

        images = {key: np.array(img_data, dtype=np.uint8) for key, img_data in images_data.items()}
        images = map_image_keys_to_model(images)

        state_array = None
        if state is not None:
            raw_state = np.array(state, dtype=np.float32)
            state_array = normalize_state(raw_state, self.norm_stats, pad_to_dim=32, use_quantiles=True)

        noise_array = np.array(noise, dtype=np.float32) if noise is not None else None

        start_time = time.time()
        subtask = None
        subtask_tokens = None
        state_result = None
        actions = None
        subtask_ms = 0.0
        action_ms = 0.0

        if generate_subtask:
            subtask_start = time.time()
            subtask_result = await self.inference_engine.infer(
                images=images,
                high_level_prompt=high_level_prompt,
                low_level_prompt=low_level_prompt,
                state=state_array,
                generate_subtask=True,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
                noise=None,
            )
            subtask_ms = (time.time() - subtask_start) * 1000
            subtask = subtask_result.get("subtask", "")
            subtask_tokens = subtask_result.get("subtask_tokens")
            state_result = subtask_result.get("state")
        else:
            subtask = low_level_prompt

        if generate_actions:
            prompt_for_actions = subtask if subtask is not None else low_level_prompt
            action_start = time.time()
            action_result = await self.inference_engine.infer(
                images=images,
                high_level_prompt=high_level_prompt,
                low_level_prompt=prompt_for_actions,
                state=state_array,
                generate_subtask=False,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
                noise=noise_array,
            )
            action_ms = (time.time() - action_start) * 1000
            state_result = action_result.get("state", state_result)
            actions = action_result.get("actions")
            if actions is not None:
                actions = unnormalize_actions(np.asarray(actions), self.norm_stats, use_quantiles=True)

        total_ms = (time.time() - start_time) * 1000

        await self._cancel_refresh_task(websocket)
        subtask_refresh_enabled = False
        if subtask_refresh_interval is not None and float(subtask_refresh_interval) > 0:
            subtask_refresh_enabled = True
            refresh_low_prompt = subtask if subtask else low_level_prompt
            refresh_task = asyncio.create_task(
                self._handle_periodic_refresh(
                    websocket=websocket,
                    images=images,
                    high_level_prompt=high_level_prompt,
                    low_level_prompt=refresh_low_prompt,
                    state=state_array,
                    refresh_interval=float(subtask_refresh_interval),
                    max_decoding_steps=max_decoding_steps,
                    temperature=temperature,
                    source_request_id=request_id,
                )
            )
            self.active_refresh_tasks[websocket] = refresh_task

        response = {
            "status": "success",
            "subtask": subtask,
            "subtask_tokens": _to_list(subtask_tokens),
            "actions": _to_list(actions),
            "state": _to_list(state_result),
            "timing": {
                "subtask_ms": subtask_ms,
                "action_ms": action_ms,
                "total_ms": total_ms,
            },
            "subtask_refresh_enabled": subtask_refresh_enabled,
            "subtask_refresh_interval": float(subtask_refresh_interval) if subtask_refresh_enabled else None,
        }

        if request_id is not None:
            response["request_id"] = request_id

        return response

    async def _handle_periodic_refresh(
        self,
        websocket: WebSocketServerProtocol,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str,
        state: np.ndarray | None,
        refresh_interval: float,
        max_decoding_steps: int,
        temperature: float,
        source_request_id: str | None,
    ):
        """Periodic subtask refresh loop for a connected client."""
        refresh_count = 0
        current_low_prompt = low_level_prompt

        while True:
            try:
                await asyncio.sleep(refresh_interval)
                refresh_count += 1

                result = await self.inference_engine.infer(
                    images=images,
                    high_level_prompt=high_level_prompt,
                    low_level_prompt=current_low_prompt,
                    state=state,
                    generate_subtask=True,
                    max_decoding_steps=max_decoding_steps,
                    temperature=temperature,
                    noise=None,
                )

                current_low_prompt = result.get("subtask") or current_low_prompt
                refresh_message = {
                    "type": "subtask_refresh",
                    "subtask": current_low_prompt,
                    "subtask_tokens": _to_list(result.get("subtask_tokens")),
                    "refresh_count": refresh_count,
                    "timestamp": time.time(),
                }
                if source_request_id is not None:
                    refresh_message["source_request_id"] = source_request_id

                await self._send_json(websocket, refresh_message)

            except asyncio.CancelledError:
                logger.info("Refresh task cancelled: %s", websocket.remote_address)
                break
            except websockets.exceptions.ConnectionClosed:
                logger.info("Connection closed while refreshing: %s", websocket.remote_address)
                break
            except Exception as e:
                logger.error("Error in periodic refresh: %s", e)
                await asyncio.sleep(1)

    async def start_server(self, *, skip_init: bool = False):
        """Start websocket server."""
        logger.info("Starting async Pi0.5 WebSocket server: %s:%s", self.host, self.port)

        if skip_init:
            logger.warning("Skipping model initialization (--skip-init). Inference unavailable.")
        else:
            logger.info("Initializing inference engine (this may take a while)...")
            await self.inference_engine.initialize()
            logger.info("Inference engine initialization completed")
<<<<<<< Updated upstream
            
            # Load normalization stats
            self.norm_stats = load_norm_stats(self.checkpoint_path, self.config_name)
            if self.norm_stats:
                logger.info(f"Loaded normalization stats for state ({len(self.norm_stats.get('state', {}).get('mean', []))}D) and actions ({len(self.norm_stats.get('actions', {}).get('mean', []))}D)")
            else:
                logger.warning("No normalization stats loaded - state and actions will not be normalized")
=======

            self.norm_stats, self.norm_stats_path = load_norm_stats(self.checkpoint_path, self.config_name)
            logger.info("Normalization file loaded from: %s", self.norm_stats_path)
            logger.info(
                "Loaded normalization stats for state (%dD) and actions (%dD)",
                len(self.norm_stats.get("state", {}).get("mean", [])),
                len(self.norm_stats.get("actions", {}).get("mean", [])),
            )
            _log_norm_values(self.norm_stats)
>>>>>>> Stashed changes

        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=60,
            ping_timeout=60,
            max_size=10 * 1024 * 1024,
        )

        logger.info("Server started, listening on %s:%s", self.host, self.port)

        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down server...")
            server.close()
            await server.wait_closed()
            logger.info("Server closed")


async def main():
    parser = argparse.ArgumentParser(description="Async Pi0.5 WebSocket Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen address")
    parser.add_argument("--port", type=int, default=8765, help="Listen port")
    parser.add_argument("--config", type=str, default="libero_pi05_action_expert", help="Model config name")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID, use -1 for CPU")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (directory or params file).",
    )
    parser.add_argument("--skip-init", action="store_true", help="Skip model initialization")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level: DEBUG/INFO/WARN/ERROR")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    os.environ.setdefault("OPENPI_DATA_HOME", os.path.expanduser("~/.cache/openpi"))

    server = AsyncPi05WebSocketServer(
        host=args.host,
        port=args.port,
        config_name=args.config,
        gpu_id=args.gpu_id,
        checkpoint_path=args.checkpoint,
    )

    await server.start_server(skip_init=args.skip_init)


if __name__ == "__main__":
    asyncio.run(main())

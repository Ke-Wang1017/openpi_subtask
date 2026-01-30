import argparse
import asyncio
import json
import logging
import os
import pathlib
import time
from typing import Any

from async_pi05_inference import AsyncPi05Inference
import jax
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)

# Image key mapping: LIBERO-style keys -> Model-expected keys
# The Pi05 model expects these specific keys: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
LIBERO_TO_MODEL_IMAGE_KEYS = {
    # LIBERO-style keys (from training data transforms)
    "agentview_rgb": "base_0_rgb",
    "wrist_rgb_left": "left_wrist_0_rgb",
    "wrist_rgb": "left_wrist_0_rgb",  # Alternative name
    # Model-style keys (pass through)
    "base_0_rgb": "base_0_rgb",
    "left_wrist_0_rgb": "left_wrist_0_rgb",
    "right_wrist_0_rgb": "right_wrist_0_rgb",
}


def map_image_keys_to_model(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Map LIBERO image keys to model expected keys.
    
    The Pi05 model expects: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
    LIBERO datasets use: agentview_rgb, wrist_rgb_left
    
    This function maps LIBERO keys to model keys and adds a dummy right_wrist if needed.
    """
    mapped = {}
    for key, value in images.items():
        new_key = LIBERO_TO_MODEL_IMAGE_KEYS.get(key, key)
        mapped[new_key] = value
    
    # Add dummy right_wrist_0_rgb if not present (LIBERO only has 2 cameras)
    if "right_wrist_0_rgb" not in mapped and len(mapped) > 0:
        # Use the first image as template for creating a zero image
        template = next(iter(mapped.values()))
        mapped["right_wrist_0_rgb"] = np.zeros_like(template)
        logger.debug("Added dummy right_wrist_0_rgb image")
    
    return mapped


def load_norm_stats(checkpoint_path: str | None, config_name: str) -> dict | None:
    """Load normalization statistics from checkpoint or config assets."""
    if checkpoint_path is None:
        return None
    
    # Try to find norm_stats.json in checkpoint assets
    checkpoint_dir = pathlib.Path(checkpoint_path)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent
    
    # Common locations for norm_stats
    search_paths = [
        checkpoint_dir / "assets" / "KeWangRobotics" / "libero_10_subtasks" / "norm_stats.json",
        checkpoint_dir / "assets" / "libero_subtask" / "norm_stats.json",
        checkpoint_dir / "norm_stats.json",
        pathlib.Path(os.path.expanduser("~/.cache/openpi/openpi-assets")) / config_name / "KeWangRobotics" / "libero_10_subtasks" / "norm_stats.json",
    ]
    
    for path in search_paths:
        if path.exists():
            logger.info(f"Loading norm_stats from: {path}")
            with open(path) as f:
                data = json.load(f)
                return data.get("norm_stats", data)
    
    logger.warning("No norm_stats found, skipping normalization")
    return None


def normalize_state(state: np.ndarray, norm_stats: dict, pad_to_dim: int = 32) -> np.ndarray:
    """Normalize state using mean/std from norm_stats and pad to expected dimension.
    
    Args:
        state: Raw state array (e.g., 8D for LIBERO)
        norm_stats: Dictionary with 'state' key containing 'mean' and 'std'
        pad_to_dim: Dimension to pad state to (default 32 for Pi05 model)
    
    Returns:
        Normalized and padded state array
    """
    if norm_stats is None or "state" not in norm_stats:
        # Still need to pad even without normalization
        if state.shape[-1] < pad_to_dim:
            pad_width = [(0, 0)] * (state.ndim - 1) + [(0, pad_to_dim - state.shape[-1])]
            state = np.pad(state, pad_width, constant_values=0.0)
        return state
    
    stats = norm_stats["state"]
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    
    # Normalize only the dimensions we have stats for
    state_dim = min(state.shape[-1], len(mean))
    normalized = state.copy().astype(np.float32)
    normalized[..., :state_dim] = (state[..., :state_dim] - mean[:state_dim]) / (std[:state_dim] + 1e-6)
    
    # Pad to expected dimension (zeros for extra dimensions)
    if normalized.shape[-1] < pad_to_dim:
        pad_width = [(0, 0)] * (normalized.ndim - 1) + [(0, pad_to_dim - normalized.shape[-1])]
        normalized = np.pad(normalized, pad_width, constant_values=0.0)
    
    return normalized


def unnormalize_actions(actions: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Unnormalize actions using mean/std from norm_stats."""
    if norm_stats is None or "actions" not in norm_stats:
        return actions
    
    stats = norm_stats["actions"]
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    
    # Handle dimension mismatch - only unnormalize the dimensions we have stats for
    action_dim = min(actions.shape[-1], len(mean))
    unnormalized = actions.copy()
    unnormalized[..., :action_dim] = actions[..., :action_dim] * (std[:action_dim] + 1e-6) + mean[:action_dim]
    return unnormalized


class AsyncPi05WebSocketServer:
    """Pi0.5 inference server based on WebSocket.
    
    Simplified synchronous approach:
    - Each request generates subtask first, then actions
    - Returns both subtask and actions in one response
    - Includes proper state normalization and action unnormalization
    """

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
        self.clients = set()
        self.norm_stats = None  # Will be loaded during initialization

    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")

    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Unregister client"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected: {websocket.remote_address}")

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str | None = None):
        """Handle client connection"""
        await self.register_client(websocket)

        try:
            # Send server metadata
            metadata = {
                "server_type": "Pi05Inference",
                "version": "2.0.0",
                "capabilities": ["subtask_generation", "action_prediction"],
                "max_decoding_steps": 25,
                # Server accepts both LIBERO-style and model-style image keys
                "supported_image_types": {
                    "libero_style": ["agentview_rgb", "wrist_rgb_left"],
                    "model_style": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
                },
                "normalization_enabled": self.norm_stats is not None,
            }
            await websocket.send(json.dumps(metadata))

            async for message in websocket:
                try:
                    # Parse request
                    request = json.loads(message)
                    response = await self.process_request(request)

                    # Send response
                    await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    error_response = {"error": "Invalid JSON format", "status": "error"}
                    await websocket.send(json.dumps(error_response))

                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    import traceback
                    traceback.print_exc()
                    error_response = {"error": str(e), "status": "error"}
                    await websocket.send(json.dumps(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client connection closed: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self.unregister_client(websocket)

    async def process_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process inference request - generates subtask first, then actions.
        
        This is a simplified synchronous approach where:
        1. Subtask is generated from high-level prompt
        2. Actions are generated using the subtask as low-level prompt
        3. Both subtask and actions are returned in one response
        """
        try:
            # Validate request format
            if "images" not in request or "high_level_prompt" not in request:
                return {"error": "Missing required fields: images, high_level_prompt", "status": "error"}

            # Extract request parameters
            images_data = request["images"]
            high_level_prompt = request["high_level_prompt"]
            state = request.get("state")
            max_decoding_steps = request.get("max_decoding_steps", 25)
            temperature = request.get("temperature", 0.1)
            noise = request.get("noise")

            # Convert image data
            images = {}
            for key, img_data in images_data.items():
                if isinstance(img_data, list):
                    img_array = np.array(img_data, dtype=np.uint8)
                else:
                    img_array = np.array(img_data, dtype=np.uint8)
                images[key] = img_array
            
            # Map LIBERO-style keys to model-expected keys
            images = map_image_keys_to_model(images)
            logger.debug(f"Mapped image keys: {list(images.keys())}")

            # Convert state data, normalize, and pad to 32D
            state_array = None
            if state is not None:
                raw_state = np.array(state, dtype=np.float32)
                # Log raw gripper state (last 2 dims of 8D state)
                if len(raw_state) >= 8:
                    print(f"[GRIPPER DEBUG] Raw gripper state (dims 6-7): {raw_state[6:8]}")
                # Apply normalization and padding
                state_array = normalize_state(raw_state, self.norm_stats, pad_to_dim=32)
                if len(raw_state) >= 8:
                    print(f"[GRIPPER DEBUG] Normalized gripper state (dims 6-7): {state_array[6:8]}")
                print(f"[GRIPPER DEBUG] Full normalized state shape: {state_array.shape}")

            # Convert noise data
            noise_array = None
            if noise is not None:
                noise_array = np.array(noise, dtype=np.float32)

            start_time = time.time()

            # Step 1: Generate subtask from high-level prompt
            logger.info(f"Generating subtask for: {high_level_prompt}")
            subtask_start = time.time()
            subtask_results = await self.inference_engine.infer(
                images=images,
                high_level_prompt=high_level_prompt,
                low_level_prompt="",  # Empty for subtask generation
                state=state_array,
                generate_subtask=True,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
                noise=None,
            )
            subtask_time = time.time() - subtask_start
            
            subtask = subtask_results.get("subtask", "")
            logger.info(f"Generated subtask: {subtask}")

            # Step 2: Generate actions using subtask as low-level prompt
            logger.info(f"Generating actions with subtask: {subtask}")
            action_start = time.time()
            action_results = await self.inference_engine.infer(
                images=images,
                high_level_prompt=high_level_prompt,
                low_level_prompt=subtask,  # Use generated subtask
                state=state_array,
                generate_subtask=False,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
                noise=noise_array,
            )
            action_time = time.time() - action_start

            total_time = time.time() - start_time

            # Build response with both subtask and actions
            actions = action_results["actions"]
            
            # Unnormalize actions before returning
            if actions is not None:
                # Log raw model output gripper (dim 6) before unnormalization
                raw_gripper = actions[0, 0, 6] if actions.ndim == 3 else actions[0, 6]
                print(f"[GRIPPER DEBUG] Raw model gripper output (normalized, first action): {raw_gripper:.4f}")
                actions = unnormalize_actions(actions, self.norm_stats)
                # Log unnormalized gripper action
                unnorm_gripper = actions[0, 0, 6] if actions.ndim == 3 else actions[0, 6]
                print(f"[GRIPPER DEBUG] Unnormalized gripper action (first action): {unnorm_gripper:.4f}")
                print(f"[GRIPPER DEBUG] Actions shape: {actions.shape}")
            
            response = {
                "status": "success",
                "subtask": subtask,
                "actions": actions.tolist() if actions is not None else None,
                "state": action_results["state"].tolist() if action_results["state"] is not None else None,
                "timing": {
                    "subtask_ms": subtask_time * 1000,
                    "action_ms": action_time * 1000,
                    "total_ms": total_time * 1000,
                },
            }

            logger.info(f"Request completed in {total_time:.3f}s (subtask: {subtask_time:.3f}s, action: {action_time:.3f}s)")
            return response

        except Exception as e:
            logger.error(f"Error processing inference request: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e), "status": "error"}

    async def start_server(self, *, skip_init: bool = False):
        """Start WebSocket server

        skip_init: If True, skip model initialization, only for connectivity testing.
        """
        logger.info(f"Starting Pi0.5 WebSocket server: {self.host}:{self.port}")

        if skip_init:
            logger.warning("Skipping model initialization (--skip-init). Only for connectivity testing, inference unavailable.")
        else:
            # Initialize inference engine
            logger.info("Initializing inference engine (may take a long time, please wait)...")
            await self.inference_engine.initialize()
            logger.info("Inference engine initialization completed")
            
            # Load normalization stats
            self.norm_stats = load_norm_stats(self.checkpoint_path, self.config_name)
            if self.norm_stats:
                logger.info(f"Loaded normalization stats for state ({len(self.norm_stats.get('state', {}).get('mean', []))}D) and actions ({len(self.norm_stats.get('actions', {}).get('mean', []))}D)")
            else:
                logger.warning("No normalization stats loaded - state and actions will not be normalized")

        # Start WebSocket server
        logger.info("Starting WebSocket listener...")
        # Increase max_size to handle large messages (e.g., action arrays, images)
        server = await websockets.serve(
            self.handle_client, self.host, self.port, ping_interval=60, ping_timeout=60, max_size=10 * 1024 * 1024
        )

        logger.info(f"Server started, listening on {self.host}:{self.port}")

        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down server...")
            server.close()
            await server.wait_closed()
            logger.info("Server closed")


async def main():
    """Start server"""
    parser = argparse.ArgumentParser(description="Pi0.5 WebSocket Server")
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
    parser.add_argument("--skip-init", action="store_true", help="Skip model initialization, only for connectivity testing")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level: DEBUG/INFO/WARN/ERROR")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create and start server
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

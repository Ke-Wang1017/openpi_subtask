import argparse
import asyncio
import json
import logging
import time
from typing import Any

from async_pi05_inference import AsyncPi05Inference
import jax
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class AsyncPi05WebSocketServer:
    """Asynchronous Pi0.5 inference server based on WebSocket"""

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
        self.inference_engine = AsyncPi05Inference(
            config_name=config_name,
            gpu_id=gpu_id,
            checkpoint_path=checkpoint_path,
        )
        self.clients = set()
        self.active_refresh_tasks = {}  # store active refresh tasks
        self.latest_observations = {}  # store latest observation per client

    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register new client"""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")

    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Unregister client"""
        self.clients.discard(websocket)

        # Cancel the refresh task for this client
        if websocket in self.active_refresh_tasks:
            task = self.active_refresh_tasks[websocket]
            task.cancel()
            del self.active_refresh_tasks[websocket]
            logger.info(f"Cancelled refresh task for client {websocket.remote_address}")

        if websocket in self.latest_observations:
            del self.latest_observations[websocket]

        logger.info(f"Client disconnected: {websocket.remote_address}")

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str | None = None):
        """Handle client connection"""
        await self.register_client(websocket)

        try:
            # Send server metadata
            metadata = {
                "server_type": "AsyncPi05Inference",
                "version": "1.0.0",
                "capabilities": ["subtask_generation", "action_prediction"],
                "max_decoding_steps": 25,
                "supported_image_types": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
            }
            await websocket.send(json.dumps(metadata))

            async for message in websocket:
                try:
                    # Parse request
                    request = json.loads(message)
                    if request.get("type") == "update_observation":
                        try:
                            images_data = request.get("images")
                            high_level_prompt = request.get("high_level_prompt")
                            low_level_prompt = request.get("low_level_prompt", "ABCDEFG")
                            state = request.get("state")

                            if images_data is None or high_level_prompt is None:
                                logger.warning("update_observation missing required fields")
                                continue

                            images = {}
                            for key, img_data in images_data.items():
                                if isinstance(img_data, list):
                                    img_array = np.array(img_data, dtype=np.uint8)
                                else:
                                    img_array = np.array(img_data, dtype=np.uint8)
                                images[key] = img_array

                            state_array = None
                            if state is not None:
                                state_array = np.array(state, dtype=np.float32)

                            self.latest_observations[websocket] = {
                                "images": images,
                                "high_level_prompt": high_level_prompt,
                                "low_level_prompt": low_level_prompt,
                                "state": state_array,
                            }
                        except Exception as e:
                            logger.error(f"Failed to update observation: {e}")
                        continue

                    response = await self.process_request(request, websocket)

                    # Send response
                    await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    error_response = {"error": "Invalid JSON format", "status": "error"}
                    await websocket.send(json.dumps(error_response))

                except Exception as e:
                    logger.error(f"Error processing request: {e}")
                    error_response = {"error": str(e), "status": "error"}
                    await websocket.send(json.dumps(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client connection closed: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            await self.unregister_client(websocket)

    async def process_request(self, request: dict[str, Any], websocket: WebSocketServerProtocol) -> dict[str, Any]:
        """Process inference request"""
        try:
            # Validate request format
            if "images" not in request or "high_level_prompt" not in request:
                return {"error": "Missing required fields: images, high_level_prompt", "status": "error"}

            # Extract request parameters
            images_data = request["images"]
            high_level_prompt = request["high_level_prompt"]
            low_level_prompt = request.get("low_level_prompt", "ABCDEFG")
            state = request.get("state")
            generate_subtask = request.get("generate_subtask", True)
            max_decoding_steps = request.get("max_decoding_steps", 25)
            temperature = request.get("temperature", 0.1)
            noise = request.get("noise")
            subtask_refresh_interval = request.get("subtask_refresh_interval")  # New: subtask refresh interval

            # Convert image data
            images = {}
            for key, img_data in images_data.items():
                if isinstance(img_data, list):
                    # Assume [height, width, channels] format
                    img_array = np.array(img_data, dtype=np.uint8)
                else:
                    # Assume already a numpy array
                    img_array = np.array(img_data, dtype=np.uint8)
                images[key] = img_array

            # Convert state data
            state_array = None
            if state is not None:
                state_array = np.array(state, dtype=np.float32)

            # Convert noise data
            noise_array = None
            if noise is not None:
                noise_array = np.array(noise, dtype=np.float32)

            # Store latest observation for refresh
            self.latest_observations[websocket] = {
                "images": images,
                "high_level_prompt": high_level_prompt,
                "low_level_prompt": low_level_prompt,
                "state": state_array,
            }

            # Execute inference
            start_time = time.time()
            results = await self.inference_engine.infer(
                images=images,
                high_level_prompt=high_level_prompt,
                low_level_prompt=low_level_prompt,
                state=state_array,
                generate_subtask=generate_subtask,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
                noise=noise_array,
                subtask_refresh_interval=subtask_refresh_interval,
            )
            total_time = time.time() - start_time

            # Build response
            actions = results["actions"]
            response = {
                "status": "success",
                "actions": actions.tolist() if actions is not None else None,
                "subtask": results["subtask"],
                "subtask_tokens": results["subtask_tokens"].tolist() if results["subtask_tokens"] is not None else None,
                "state": results["state"].tolist() if results["state"] is not None else None,
                "timing": results["timing"],
                "server_timing": {"total_ms": total_time * 1000},
            }

            # If periodic refresh is enabled, start background task
            if subtask_refresh_interval is not None and subtask_refresh_interval > 0:
                response["subtask_refresh_interval"] = subtask_refresh_interval
                response["subtask_refresh_enabled"] = True

                # Start periodic refresh task
                refresh_task = asyncio.create_task(
                    self._handle_periodic_refresh(
                        websocket,
                        images,
                        high_level_prompt,
                        low_level_prompt,
                        state_array,
                        subtask_refresh_interval,
                        max_decoding_steps,
                        temperature,
                    )
                )
                self.active_refresh_tasks[websocket] = refresh_task
                logger.info(f"Started periodic refresh task for client {websocket.remote_address}, interval: {subtask_refresh_interval}s")
            else:
                response["subtask_refresh_enabled"] = False

            return response

        except Exception as e:
            logger.error(f"Error processing inference request: {e}")
            return {"error": str(e), "status": "error"}

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
    ):
        """Handle periodic subtask refresh"""
        refresh_count = 0

        while True:
            try:
                await asyncio.sleep(refresh_interval)
                refresh_count += 1
                latest = self.latest_observations.get(websocket)
                if latest:
                    images = latest["images"]
                    high_level_prompt = latest["high_level_prompt"]
                    low_level_prompt = latest["low_level_prompt"]
                    state = latest["state"]

                logger.info(f"Starting {refresh_count}th subtask refresh (client: {websocket.remote_address})")

                # Prepare new observation data
                observation = self.inference_engine.prepare_observation(
                    images, high_level_prompt, low_level_prompt, state, mask_subtask_tokens=True
                )

                # Generate new subtask
                rng = jax.random.key(int(time.time() * 1000) % 2**32)
                subtask_tokens, subtask_text = await self.inference_engine.generate_subtask(
                    observation, rng, max_decoding_steps, temperature
                )

                # Send refresh message to client
                refresh_message = {
                    "type": "subtask_refresh",
                    "subtask": subtask_text,
                    "subtask_tokens": subtask_tokens.tolist(),
                    "refresh_count": refresh_count,
                    "timestamp": time.time(),
                }

                await websocket.send(json.dumps(refresh_message))
                logger.info(f"{refresh_count}th refresh completed, new subtask: {subtask_text}")
                if latest:
                    latest["low_level_prompt"] = subtask_text

            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client connection closed, stopping refresh task: {websocket.remote_address}")
                break
            except asyncio.CancelledError:
                logger.info(f"Refresh task cancelled: {websocket.remote_address}")
                break
            except Exception as e:
                logger.error(f"Error in periodic refresh: {e}")
                await asyncio.sleep(1)  # Wait 1 second before retrying after error

    async def start_server(self, *, skip_init: bool = False):
        """Start WebSocket server

        skip_init: If True, skip model initialization, only for connectivity testing.
        """
        logger.info(f"Starting asynchronous Pi0.5 WebSocket server: {self.host}:{self.port}")

        if skip_init:
            logger.warning("Skipping model initialization (--skip-init). Only for connectivity testing, inference unavailable.")
        else:
            # Initialize inference engine
            logger.info("Initializing inference engine (may take a long time, please wait)...")
            await self.inference_engine.initialize()
            logger.info("Inference engine initialization completed")

        # Start WebSocket server
        logger.info("Starting WebSocket listener...")
        # Increase max_size to handle large messages (e.g., action arrays, images)
        # Default is 1MB, set to 10MB to handle large inference responses
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

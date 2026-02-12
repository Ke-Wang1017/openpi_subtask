import asyncio
import argparse
from collections.abc import Awaitable, Callable
import json
import logging
import pathlib
import time
import uuid

import cv2
import numpy as np
import websockets

logger = logging.getLogger(__name__)


class AsyncPi05Client:
    """Asynchronous Pi0.5 websocket client with request-id routing."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port

        self.websocket = None
        self.server_metadata = None

        self._send_lock = asyncio.Lock()
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._receiver_task: asyncio.Task | None = None

        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._refresh_callbacks: list[Callable[[dict], Awaitable[None] | None]] = []

    async def connect(self):
        """Connect to server and start background receiver."""
        uri = f"ws://{self.host}:{self.port}"
        logger.info("Connecting to server: %s", uri)

        self.websocket = await websockets.connect(uri)

        # Metadata is sent first.
        metadata_message = await self.websocket.recv()
        self.server_metadata = json.loads(metadata_message)
        logger.info("Server metadata: %s", self.server_metadata)

        self._receiver_task = asyncio.create_task(self._receiver_loop())

    async def disconnect(self):
        """Disconnect and cleanup pending requests."""
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None

        if self.websocket is not None:
            await self.websocket.close()
            self.websocket = None

        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(RuntimeError("Client disconnected"))
        self._pending_requests.clear()

    def load_image(self, img_path: str) -> np.ndarray:
        """Load image from path with random fallback."""
        if not img_path:
            return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        img = cv2.imread(img_path)
        if img is None:
            logger.warning("Unable to load image: %s, using random image", img_path)
            return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        return img

    def add_refresh_callback(self, callback: Callable[[dict], Awaitable[None] | None]) -> None:
        self._refresh_callbacks.append(callback)

    def remove_refresh_callback(self, callback: Callable[[dict], Awaitable[None] | None]) -> None:
        self._refresh_callbacks = [cb for cb in self._refresh_callbacks if cb is not callback]

    async def _receiver_loop(self):
        """Background receiver that routes messages by request_id and refresh type."""
        try:
            while self.websocket is not None:
                raw_message = await self.websocket.recv()
                data = json.loads(raw_message)

                if isinstance(data, dict) and data.get("type") == "subtask_refresh":
                    await self._refresh_queue.put(data)
                    for callback in list(self._refresh_callbacks):
                        try:
                            result = callback(data)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception as e:
                            logger.error("Refresh callback failed: %s", e)
                    continue

                request_id = data.get("request_id") if isinstance(data, dict) else None
                if request_id and request_id in self._pending_requests:
                    future = self._pending_requests[request_id]
                    if not future.done():
                        future.set_result(data)
                    continue

                # Backward-compatible fallback: if there is exactly one pending request and no request_id,
                # route this message to it.
                if len(self._pending_requests) == 1:
                    only_request_id = next(iter(self._pending_requests.keys()))
                    future = self._pending_requests[only_request_id]
                    if not future.done():
                        future.set_result(data)
                else:
                    logger.debug("Unmatched message from server: %s", data)

        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed")
        except Exception as e:
            logger.error("Receiver loop failed: %s", e)
        finally:
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(RuntimeError("Connection closed while waiting for response"))

    async def infer(
        self,
        images: dict[str, str],
        high_level_prompt: str,
        low_level_prompt: str = "",
        state: np.ndarray | None = None,
        *,
        generate_subtask: bool = True,
        generate_actions: bool = True,
        max_decoding_steps: int = 25,
        temperature: float = 0.1,
        noise: np.ndarray | None = None,
        subtask_refresh_interval: float | None = None,
        request_timeout: float | None = 30.0,
    ) -> dict:
        """Send an inference request and await matched response."""
        if self.websocket is None:
            raise RuntimeError("Not connected to server")

        images_data = {}
        for key, img_path in images.items():
            img = self.load_image(img_path)
            images_data[key] = img.tolist()

        request_id = uuid.uuid4().hex
        request = {
            "request_id": request_id,
            "images": images_data,
            "high_level_prompt": high_level_prompt,
            "low_level_prompt": low_level_prompt,
            "generate_subtask": generate_subtask,
            "generate_actions": generate_actions,
            "max_decoding_steps": max_decoding_steps,
            "temperature": temperature,
        }

        if state is not None:
            request["state"] = np.asarray(state).tolist()

        if noise is not None:
            request["noise"] = np.asarray(noise).tolist()

        if subtask_refresh_interval is not None:
            request["subtask_refresh_interval"] = subtask_refresh_interval

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending_requests[request_id] = future

        start_time = time.time()
        try:
            async with self._send_lock:
                await self.websocket.send(json.dumps(request))

            if request_timeout is None:
                response = await future
            else:
                response = await asyncio.wait_for(future, timeout=request_timeout)

            total_time = time.time() - start_time
            if isinstance(response, dict):
                response["client_timing"] = {"total_ms": total_time * 1000}

            if response.get("status") == "error":
                raise RuntimeError(f"Server error: {response.get('error')}")

            return response
        finally:
            pending = self._pending_requests.pop(request_id, None)
            if pending is not None and not pending.done():
                pending.cancel()

    async def batch_infer(self, requests: list[dict], delay_between_requests: float = 0.1) -> list[dict]:
        """Execute batch inference requests sequentially."""
        results: list[dict] = []

        for i, request in enumerate(requests):
            logger.info("Processing request %d/%d", i + 1, len(requests))
            try:
                result = await self.infer(**request)
                results.append(result)
                if i < len(requests) - 1:
                    await asyncio.sleep(delay_between_requests)
            except Exception as e:
                logger.error("Request %d failed: %s", i + 1, e)
                results.append({"error": str(e)})

        return results

    async def listen_for_refresh_messages(self, callback=None):
        """Listen for periodic refresh messages from internal queue."""
        if self.websocket is None:
            raise RuntimeError("Not connected to server")

        while True:
            data = await self._refresh_queue.get()
            logger.info(
                "Received subtask refresh: %s (count: %s)",
                data.get("subtask"),
                data.get("refresh_count"),
            )
            if callback is not None:
                result = callback(data)
                if asyncio.iscoroutine(result):
                    await result


async def test_single_inference():
    client = AsyncPi05Client(host="localhost", port=8765)

    try:
        await client.connect()

        images = {
            "agentview_rgb": "faceImg.png",
            "wrist_rgb_left": "leftImg.png",
        }

        high_level_prompt = "Pick up the flashcard on the table"

        logger.info("Starting inference...")
        result = await client.infer(
            images=images,
            high_level_prompt=high_level_prompt,
            generate_subtask=True,
            generate_actions=True,
            max_decoding_steps=25,
            temperature=0.1,
            subtask_refresh_interval=2.0,
        )

        print("Inference results:")
        print(f"Status: {result.get('status')}")
        if result.get("actions") is not None:
            print(f"Action shape: {np.array(result['actions']).shape}")
        else:
            print("Action shape: None")
        print(f"Generated subtask: {result.get('subtask')}")
        print(f"Timing info: {result.get('timing')}")
        print(f"Client timing: {result.get('client_timing')}")

    finally:
        await client.disconnect()


def _parse_state_csv(state_csv: str | None) -> np.ndarray | None:
    if not state_csv:
        return None
    values = [x.strip() for x in state_csv.split(",") if x.strip()]
    if not values:
        return None
    return np.asarray([float(x) for x in values], dtype=np.float32)


def _default_image_path(filename: str) -> str:
    return str(pathlib.Path(__file__).with_name(filename))


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Async Pi0.5 websocket client")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument(
        "--high-level-prompt",
        type=str,
        default="Pick up the flashcard on the table",
        help="High-level instruction",
    )
    parser.add_argument("--low-level-prompt", type=str, default="", help="Optional low-level prompt seed")
    parser.add_argument(
        "--agentview-image",
        type=str,
        default=_default_image_path("faceImg.png"),
        help="Path for agentview/base camera image",
    )
    parser.add_argument(
        "--wrist-left-image",
        type=str,
        default=_default_image_path("leftImg.png"),
        help="Path for left wrist camera image",
    )
    parser.add_argument(
        "--right-wrist-image",
        type=str,
        default="",
        help="Optional path for right wrist image",
    )
    parser.add_argument(
        "--image-key-format",
        type=str,
        choices=["libero", "model"],
        default="libero",
        help="Use LIBERO keys (agentview/wrist) or model keys (base_0/wrist_0)",
    )
    parser.add_argument(
        "--state",
        type=str,
        default="",
        help="Optional comma-separated state vector (e.g. '0,0,0,0,0,0,0,0')",
    )
    parser.add_argument("--max-decoding-steps", type=int, default=25, help="Max decoding steps")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument(
        "--subtask-refresh-interval",
        type=float,
        default=None,
        help="Optional refresh interval in seconds",
    )
    parser.add_argument("--request-timeout", type=float, default=30.0, help="Request timeout in seconds")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser


async def run_from_cli(args: argparse.Namespace) -> None:
    client = AsyncPi05Client(host=args.host, port=args.port)
    state = _parse_state_csv(args.state)

    if args.image_key_format == "libero":
        images = {
            "agentview_rgb": args.agentview_image,
            "wrist_rgb_left": args.wrist_left_image,
        }
        if args.right_wrist_image:
            # Kept for compatibility even though LIBERO typically has 2 cameras.
            images["right_wrist_0_rgb"] = args.right_wrist_image
    else:
        images = {
            "base_0_rgb": args.agentview_image,
            "left_wrist_0_rgb": args.wrist_left_image,
        }
        if args.right_wrist_image:
            images["right_wrist_0_rgb"] = args.right_wrist_image

    try:
        await client.connect()
        result = await client.infer(
            images=images,
            high_level_prompt=args.high_level_prompt,
            low_level_prompt=args.low_level_prompt,
            state=state,
            generate_subtask=True,
            generate_actions=True,
            max_decoding_steps=args.max_decoding_steps,
            temperature=args.temperature,
            subtask_refresh_interval=args.subtask_refresh_interval,
            request_timeout=args.request_timeout,
        )
        print(json.dumps(result, indent=2))
    finally:
        await client.disconnect()


async def main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    await run_from_cli(args)


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import json
import logging
import time
from typing import Any

import cv2
import numpy as np
import websockets

logger = logging.getLogger(__name__)


class SyncPi05Client:
    """Synchronous Pi0.5 inference client"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.websocket = None
        self.server_metadata = None

    async def connect(self):
        """Connect to server"""
        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"Connecting to server: {uri}")

        self.websocket = await websockets.connect(uri)

        # Receive server metadata
        metadata_message = await self.websocket.recv()
        self.server_metadata = json.loads(metadata_message)
        logger.info(f"Server metadata: {self.server_metadata}")

    async def disconnect(self):
        """Disconnect"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    def load_image(self, img_path: str) -> np.ndarray:
        """Load image"""
        if not img_path:
            # Create random image as fallback
            return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Unable to load image: {img_path}, using random image")
            return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        return img

    async def infer(
        self,
        images: dict[str, str],  # 图像路径字典
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
        Send inference request

        Args:
            images: Image path dictionary, keys are image types, values are image file paths
            high_level_prompt: High-level task description
            low_level_prompt: Low-level task description
            state: Robot state
            generate_subtask: Whether to generate subtask
            max_decoding_steps: Maximum decoding steps
            temperature: Sampling temperature
            noise: Action noise
            subtask_refresh_interval: Subtask refresh interval (seconds), None means no refresh

        Returns:
            Inference result dictionary
        """
        if not self.websocket:
            raise RuntimeError("Not connected to server")

        # Load images
        images_data = {}
        for key, img_path in images.items():
            img = self.load_image(img_path)
            images_data[key] = img.tolist()  # Convert to list for JSON serialization

        # Build request
        request = {
            "images": images_data,
            "high_level_prompt": high_level_prompt,
            "low_level_prompt": low_level_prompt,
            "generate_subtask": generate_subtask,
            "max_decoding_steps": max_decoding_steps,
            "temperature": temperature,
        }

        if state is not None:
            request["state"] = state.tolist()

        if noise is not None:
            request["noise"] = noise.tolist()

        if subtask_refresh_interval is not None:
            request["subtask_refresh_interval"] = subtask_refresh_interval

        # Send request
        start_time = time.time()
        await self.websocket.send(json.dumps(request))

        # Receive response
        response_message = await self.websocket.recv()
        response = json.loads(response_message)

        total_time = time.time() - start_time

        if response.get("status") == "error":
            raise RuntimeError(f"Server error: {response.get('error')}")

        # Add client timing information
        response["client_timing"] = {"total_ms": total_time * 1000}

        return response

    async def batch_infer(self, requests: list, delay_between_requests: float = 0.1) -> list:
        """Batch inference requests"""
        results = []

        for i, request in enumerate(requests):
            logger.info(f"Processing request {i + 1}/{len(requests)}")

            try:
                result = await self.infer(**request)
                results.append(result)

                if i < len(requests) - 1:  # Not the last request
                    await asyncio.sleep(delay_between_requests)

            except Exception as e:
                logger.error(f"Request {i + 1} failed: {e}")
                results.append({"error": str(e)})

        return results

    async def listen_for_refresh_messages(self, callback=None):
        """Listen for periodic refresh messages"""
        if not self.websocket:
            raise RuntimeError("Not connected to server")

        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)

                if data.get("type") == "subtask_refresh":
                    logger.info(f"Received subtask refresh: {data['subtask']} (count: {data['refresh_count']})")

                    if callback:
                        await callback(data)
                else:
                    # Handle other types of messages
                    logger.info(f"Received message: {data}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed, stopping refresh message listener")
        except Exception as e:
            logger.error(f"Error listening for refresh messages: {e}")


async def test_single_inference():
    """Test single inference request"""
    client = SyncPi05Client(host="localhost", port=8765)

    try:
        await client.connect()

        # Prepare test data
        images = {"base_0_rgb": "faceImg.png", "left_wrist_0_rgb": "leftImg.png", "right_wrist_0_rgb": "rightImg.png"}

        high_level_prompt = "Pick up the flashcard on the table"

        # Execute inference
        logger.info("Starting inference...")
        result = await client.infer(
            images=images,
            high_level_prompt=high_level_prompt,
            generate_subtask=True,
            max_decoding_steps=25,
            temperature=0.1,
            subtask_refresh_interval=2.0,  # Refresh subtask every 2 seconds
        )

        # Print results
        print("Inference results:")
        print(f"Status: {result.get('status')}")
        if result.get("actions") is not None:
            print(f"Action shape: {np.array(result['actions']).shape}")
        else:
            print("Action shape: None")
        print(f"Generated subtask: {result.get('subtask')}")
        print(f"Timing info: {result.get('timing')}")
        print(f"Client timing: {result.get('client_timing')}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        await client.disconnect()


async def test_batch_inference():
    """Test batch inference requests"""
    client = SyncPi05Client(host="localhost", port=8765)

    try:
        await client.connect()

        # Prepare batch requests
        requests = [
            {
                "images": {
                    "base_0_rgb": "faceImg.png",
                    "left_wrist_0_rgb": "leftImg.png",
                    "right_wrist_0_rgb": "rightImg.png",
                },
                "high_level_prompt": "Pick up the flashcard on the table",
                "generate_subtask": True,
            },
            {
                "images": {
                    "base_0_rgb": "faceImg.png",
                    "left_wrist_0_rgb": "leftImg.png",
                    "right_wrist_0_rgb": "rightImg.png",
                },
                "high_level_prompt": "Move the pen to the box",
                "generate_subtask": True,
            },
        ]

        # Execute batch inference
        logger.info("Starting batch inference...")
        results = await client.batch_infer(requests, delay_between_requests=0.5)

        # Print results
        print(f"Batch inference completed, processed {len(results)} requests")
        for i, result in enumerate(results):
            if "error" in result:
                print(f"Request {i + 1} failed: {result['error']}")
            else:
                print(f"Request {i + 1} succeeded:")
                print(f"  Subtask: {result.get('subtask')}")
                if result.get("actions") is not None:
                    print(f"  Action shape: {np.array(result['actions']).shape}")
                else:
                    print("  Action shape: None")

    except Exception as e:
        logger.error(f"Batch test failed: {e}")
    finally:
        await client.disconnect()


async def test_periodic_refresh():
    """Test periodic refresh functionality"""
    client = SyncPi05Client(host="localhost", port=8765)

    try:
        await client.connect()

        # Prepare test data
        images = {"base_0_rgb": "faceImg.png", "left_wrist_0_rgb": "leftImg.png", "right_wrist_0_rgb": "rightImg.png"}

        high_level_prompt = "Pick up the flashcard on the table"

        # Define refresh callback function
        async def on_refresh(data):
            print(f"\n🔄 Subtask refresh (count: {data['refresh_count']}):")
            print(f"   New subtask: {data['subtask']}")
            print(f"   Timestamp: {data['timestamp']}")
            print("-" * 50)

        # Start listening task
        listen_task = asyncio.create_task(client.listen_for_refresh_messages(callback=on_refresh))

        # Execute inference and enable periodic refresh
        logger.info("Starting inference and enabling periodic refresh...")
        result = await client.infer(
            images=images,
            high_level_prompt=high_level_prompt,
            generate_subtask=True,
            subtask_refresh_interval=2.0,  # Refresh every 2 seconds
        )

        print("Initial inference results:")
        print(f"Status: {result.get('status')}")
        if result.get("actions") is not None:
            print(f"Action shape: {np.array(result['actions']).shape}")
        else:
            print("Action shape: None")
        print(f"Initial subtask: {result.get('subtask')}")
        print(f"Periodic refresh enabled: {result.get('subtask_refresh_enabled')}")
        print(f"Refresh interval: {result.get('subtask_refresh_interval')} seconds")
        print("\nWaiting for periodic refresh messages... (Press Ctrl+C to stop)")

        # Wait for a period to observe refresh
        try:
            await asyncio.wait_for(listen_task, timeout=10.0)  # Wait 10 seconds
        except TimeoutError:
            print("Test completed, observed 10 seconds of refresh process")

    except KeyboardInterrupt:
        print("\nUser interrupted test")
    except Exception as e:
        logger.error(f"Periodic refresh test failed: {e}")
    finally:
        listen_task.cancel()
        await client.disconnect()


async def main():
    """Main function"""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    print("Synchronous Pi0.5 inference client test")
    print("=" * 50)

    # Test single inference
    print("\n1. Test single inference request")
    await test_single_inference()

    # Wait a bit
    await asyncio.sleep(2)

    # Test periodic refresh
    print("\n2. Test periodic refresh functionality")
    await test_periodic_refresh()

    # Wait a bit
    await asyncio.sleep(2)

    # Test batch inference
    print("\n3. Test batch inference requests")
    await test_batch_inference()


if __name__ == "__main__":
    asyncio.run(main())

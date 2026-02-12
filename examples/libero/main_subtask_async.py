from __future__ import annotations

import asyncio
import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
from typing import Any, Dict
import uuid

# Default to headless software rendering unless the user sets MUJOCO_GL.
os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import tqdm
import tyro
import websockets

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class SubtaskWebsocketAsyncClient:
    """Simple asynchronous JSON websocket client for Pi0.5 server.
    
    Each request sends images, high-level prompt, and state.
    Server generates subtask first, then actions, and returns both.
    If periodic subtask refresh is enabled, refresh messages are routed
    in a background receiver task so they do not break request/response flow.
    """

    def __init__(self, ws, metadata: dict, timeout: float = 120.0):
        self._ws = ws
        self._metadata = metadata
        self._timeout = timeout
        self._send_lock = asyncio.Lock()
        self._pending_requests: dict[str, asyncio.Future] = {}
        self._refresh_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._receiver_task = asyncio.create_task(self._receiver_loop())

    @classmethod
    async def create(cls, host: str = "0.0.0.0", port: int = 8765, timeout: float = 120.0) -> "SubtaskWebsocketAsyncClient":
        uri = f"ws://{host}:{port}"
        logging.info("Connecting to subtask server: %s", uri)
        ws = await websockets.connect(
            uri,
            compression=None, 
            max_size=10 * 1024 * 1024,
            close_timeout=timeout,
        )
        metadata_raw = await ws.recv()
        metadata = json.loads(metadata_raw)
        logging.info("Server metadata: %s", metadata)
        return cls(ws=ws, metadata=metadata, timeout=timeout)

    async def infer(
        self,
        *,
        images: Dict[str, np.ndarray],
        high_level_prompt: str,
        state: np.ndarray,
        max_decoding_steps: int = 25,
        temperature: float = 0.1,
        subtask_refresh_interval: float | None = None,
    ) -> dict:
        """Send inference request and receive subtask + actions.
        
        Args:
            images: Dictionary of image arrays
            high_level_prompt: The high-level task instruction
            state: Robot state array
            max_decoding_steps: Max steps for subtask generation
            temperature: Sampling temperature
            subtask_refresh_interval: Refresh interval in seconds for periodic subtask updates
            
        Returns:
            Dictionary with 'subtask' and 'actions' keys
        """
        request_id = uuid.uuid4().hex
        request = {
            "request_id": request_id,
            "images": {key: value.tolist() for key, value in images.items()},
            "high_level_prompt": high_level_prompt,
            "state": state.tolist(),
            "max_decoding_steps": max_decoding_steps,
            "temperature": temperature,
        }
        if subtask_refresh_interval is not None:
            request["subtask_refresh_interval"] = float(subtask_refresh_interval)

        loop = asyncio.get_running_loop()
        response_future = loop.create_future()
        self._pending_requests[request_id] = response_future

        try:
            async with self._send_lock:
                await self._ws.send(json.dumps(request))
            response = await asyncio.wait_for(response_future, timeout=self._timeout)
        finally:
            pending = self._pending_requests.pop(request_id, None)
            if pending is not None and not pending.done():
                pending.cancel()
        
        if response.get("status") == "error":
            raise RuntimeError(f"Server error: {response.get('error')}")
        
        return response

    def drain_refresh_messages(self) -> list[dict[str, Any]]:
        """Drain queued periodic subtask refresh messages."""
        messages: list[dict[str, Any]] = []
        while True:
            try:
                messages.append(self._refresh_queue.get_nowait())
            except asyncio.QueueEmpty:
                return messages

    async def _receiver_loop(self) -> None:
        """Route websocket messages to request futures or refresh queue."""
        try:
            while True:
                raw_message = await self._ws.recv()
                data = json.loads(raw_message)

                if isinstance(data, dict) and data.get("type") == "subtask_refresh":
                    await self._refresh_queue.put(data)
                    continue

                request_id = data.get("request_id") if isinstance(data, dict) else None
                if request_id and request_id in self._pending_requests:
                    future = self._pending_requests[request_id]
                    if not future.done():
                        future.set_result(data)
                    continue

                # Backward compatibility with servers that do not echo request_id.
                if len(self._pending_requests) == 1:
                    only_request_id = next(iter(self._pending_requests.keys()))
                    future = self._pending_requests[only_request_id]
                    if not future.done():
                        future.set_result(data)
                else:
                    logging.debug("Unmatched websocket message: %s", data)

        except asyncio.CancelledError:
            pass
        except websockets.exceptions.ConnectionClosed:
            logging.info("Websocket connection closed")
        except Exception:
            logging.exception("Background websocket receiver failed")
        finally:
            for future in self._pending_requests.values():
                if not future.done():
                    future.set_exception(RuntimeError("Websocket connection closed"))

    async def close(self) -> None:
        """Close the websocket connection."""
        try:
            if self._receiver_task is not None:
                self._receiver_task.cancel()
                try:
                    await self._receiver_task
                except asyncio.CancelledError:
                    pass
            await self._ws.close()
        except Exception:
            pass


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8765
    resize_size: int = 224
    replan_steps: int = 10
    max_decoding_steps: int = 25
    temperature: float = 0.1
    subtask_refresh_interval: float | None = None  # Enable periodic subtask refresh when > 0

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 5  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)
    save_debug_images: bool = False  # Save raw/rotated/resized images for orientation checks
    debug_image_out_path: str = "data/libero/debug_images"  # Path to save debug images
    debug_max_images_per_episode: int = 10  # Max debug frames saved per episode


async def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    video_out_dir = pathlib.Path(args.video_out_path).expanduser().resolve()
    video_out_dir.mkdir(parents=True, exist_ok=True)
    debug_out_dir = pathlib.Path(args.debug_image_out_path).expanduser().resolve()
    if args.save_debug_images:
        debug_out_dir.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = await SubtaskWebsocketAsyncClient.create(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_segment = "".join(
            c if (c.isalnum() or c in ("_", "-")) else "_" for c in task_description.replace(" ", "_")
        )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info("\nTask: %s", task_description)

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            debug_saved_count = 0
            refresh_state_logged = False

            logging.info("Starting episode %d...", task_episodes + 1)
            while t < max_steps + args.num_steps_wait:
                try:
                    for refresh_message in client.drain_refresh_messages():
                        logging.info(
                            "Subtask refresh #%s: %s",
                            refresh_message.get("refresh_count"),
                            refresh_message.get("subtask"),
                        )

                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image (no 180-degree rotation)
                    raw_img = np.ascontiguousarray(obs["agentview_image"])
                    raw_wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])

                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(raw_img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(raw_wrist_img, args.resize_size, args.resize_size)
                    )

                    if args.save_debug_images and debug_saved_count < args.debug_max_images_per_episode:
                        _save_image_debug_bundle(
                            debug_out_dir=debug_out_dir,
                            task_segment=task_segment,
                            episode_idx=episode_idx,
                            step_idx=t,
                            raw_img=raw_img,
                            raw_wrist_img=raw_wrist_img,
                            model_img=img,
                            model_wrist_img=wrist_img,
                        )
                        debug_saved_count += 1

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        state = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        )
                        # Use model-style image keys (matching LiberoInputs/LiberoSubtaskInputs)
                        # Server will add zeroed right_wrist_0_rgb if not provided
                        images = {
                            "base_0_rgb": img,
                            "left_wrist_0_rgb": wrist_img,
                        }

                        # Single request: server generates subtask first, then actions
                        response = await client.infer(
                            images=images,
                            high_level_prompt=str(task_description),
                            state=state,
                            max_decoding_steps=args.max_decoding_steps,
                            temperature=args.temperature,
                            subtask_refresh_interval=1.0,
                        )
                        if response.get("subtask_refresh_enabled") and not refresh_state_logged:
                            logging.info(
                                "Periodic subtask refresh enabled (interval: %.2fs)",
                                float(response.get("subtask_refresh_interval", 0.0)),
                            )
                            refresh_state_logged = True
                        
                        # Log the generated subtask
                        subtask = response.get("subtask", "")
                        logging.info("Generated subtask: %s", subtask)
                        print(f"Subtask: {subtask}")
                        
                        # Get action chunk
                        action_chunk = response.get("actions")
                        if action_chunk is None:
                            raise RuntimeError("No actions returned from server.")
                        action_chunk = np.asarray(action_chunk)
                        if action_chunk.ndim == 3:
                            action_chunk = action_chunk[0]
                        print(f"Action chunk shape: {action_chunk.shape}")
                        
                        # Log timing info
                        timing = response.get("timing", {})
                        logging.info("Timing - subtask: %.1fms, action: %.1fms, total: %.1fms",
                                   timing.get("subtask_ms", 0),
                                   timing.get("action_ms", 0),
                                   timing.get("total_ms", 0))
                        
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    action = action[:7]

                    # Execute action in environment
                    obs, _reward, done, _info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    import traceback
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            print(f"Length of Replay images: {len(replay_images)}")
            if replay_images:
                video_path = (
                    video_out_dir
                    / f"rollout_{task_segment}_episode_{episode_idx + 1:04d}_{suffix}.mp4"
                )
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                logging.info("Saved replay video: %s", video_path)
            else:
                logging.warning("No frames captured; skipping video save for episode %d.", episode_idx + 1)

            # Log current results
            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %d", total_episodes)
            logging.info("# successes: %d (%.1f%%)", total_successes, total_successes / total_episodes * 100.0)

        # Log final results
        logging.info("Current task success rate: %f", float(task_successes) / float(task_episodes))
        logging.info("Current total success rate: %f", float(total_successes) / float(total_episodes))

    logging.info("Total success rate: %f", float(total_successes) / float(total_episodes))
    logging.info("Total episodes: %d", total_episodes)
    await client.close()


def _save_image_debug_bundle(
    *,
    debug_out_dir: pathlib.Path,
    task_segment: str,
    episode_idx: int,
    step_idx: int,
    raw_img: np.ndarray,
    raw_wrist_img: np.ndarray,
    model_img: np.ndarray,
    model_wrist_img: np.ndarray,
) -> None:
    """Save raw + model-input images to verify preprocessing alignment."""
    episode_dir = debug_out_dir / task_segment / f"episode_{episode_idx + 1:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"step_{step_idx:04d}"

    imageio.imwrite(episode_dir / f"{prefix}_agent_raw.png", np.asarray(raw_img))
    imageio.imwrite(episode_dir / f"{prefix}_wrist_raw.png", np.asarray(raw_wrist_img))
    imageio.imwrite(episode_dir / f"{prefix}_agent_model_input.png", np.asarray(model_img))
    imageio.imwrite(episode_dir / f"{prefix}_wrist_model_input.png", np.asarray(model_wrist_img))


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    asyncio.run(eval_libero(args))

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import queue
import threading
from typing import Dict

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
import websockets.sync.client

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class SubtaskWebsocketClient:
    """JSON websocket client for async_pi05_websocket_server.py."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, on_refresh=None):
        uri = f"ws://{host}:{port}"
        logging.info("Connecting to subtask server: %s", uri)
        # Increase max_size to handle large messages (e.g., action arrays, images)
        # Default is 1MB, set to 10MB to match server limit
        self._ws = websockets.sync.client.connect(uri, compression=None, max_size=10 * 1024 * 1024)
        metadata = self._ws.recv()
        self._metadata = json.loads(metadata)
        logging.info("Server metadata: %s", self._metadata)
        self._on_refresh = on_refresh
        self._closed = False
        self._response_queue = queue.Queue()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

    def infer(
        self,
        *,
        images: Dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str,
        state: np.ndarray,
        generate_subtask: bool,
        max_decoding_steps: int,
        temperature: float,
        subtask_refresh_interval: float | None = None,
    ) -> dict:
        request = {
            "images": {key: value.tolist() for key, value in images.items()},
            "high_level_prompt": high_level_prompt,
            "low_level_prompt": low_level_prompt,
            "state": state.tolist(),
            "generate_subtask": generate_subtask,
            "max_decoding_steps": max_decoding_steps,
            "temperature": temperature,
        }
        if subtask_refresh_interval is not None:
            request["subtask_refresh_interval"] = subtask_refresh_interval
        self._ws.send(json.dumps(request))
        response = self._response_queue.get()
        if response.get("status") == "error":
            raise RuntimeError(f"Server error: {response.get('error')}")
        return response

    def _recv_loop(self) -> None:
        while not self._closed:
            try:
                message = self._ws.recv()
            except Exception as e:
                if not self._closed:
                    logging.info("Receiver thread exiting: %s", e)
                break
            try:
                data = json.loads(message)
            except Exception as e:
                logging.warning("Failed to decode message: %s", e)
                continue

            if data.get("type") == "subtask_refresh":
                if self._on_refresh:
                    try:
                        self._on_refresh(data)
                    except Exception as e:
                        logging.error("Refresh handler error: %s", e)
            else:
                self._response_queue.put(data)

    def close(self) -> None:
        self._closed = True
        self._ws.close()
        self._recv_thread.join(timeout=1)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8765
    resize_size: int = 224
    replan_steps: int = 5
    max_decoding_steps: int = 25
    temperature: float = 0.1
    subtask_refresh_interval: float = 0.5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

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

    current_subtask = {"text": ""}
    current_subtask_lock = threading.Lock()

    def _set_current_subtask(text: str | None) -> None:
        if text is None:
            return
        with current_subtask_lock:
            current_subtask["text"] = text

    def _on_refresh(data: dict) -> None:
        refreshed = data.get("subtask")
        if refreshed:
            _set_current_subtask(refreshed)
            logging.info("Subtask refresh: %s", refreshed)
            print(f"Subtask refresh: {refreshed}")

    client = SubtaskWebsocketClient(args.host, args.port, on_refresh=_on_refresh)
    refresh_started = False

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info("\nTask: %s", task_description)

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            _set_current_subtask("")

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info("Starting episode %d...", task_episodes + 1)
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

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
                        # Map LIBERO image keys to model expected keys
                        # Model expects: base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb
                        images = {
                            "base_0_rgb": img,
                            "left_wrist_0_rgb": wrist_img,
                            "right_wrist_0_rgb": np.zeros_like(img),
                        }

                        with current_subtask_lock:
                            low_level_prompt = current_subtask["text"]

                        # Generate subtask from high-level instruction.
                        refresh_interval = None
                        if not refresh_started and args.subtask_refresh_interval > 0:
                            refresh_interval = args.subtask_refresh_interval
                        subtask_response = client.infer(
                            images=images,
                            high_level_prompt=str(task_description),
                            low_level_prompt=low_level_prompt,
                            state=state,
                            generate_subtask=True,
                            max_decoding_steps=args.max_decoding_steps,
                            temperature=args.temperature,
                            subtask_refresh_interval=refresh_interval,
                        )
                        if subtask_response.get("subtask"):
                            _set_current_subtask(subtask_response["subtask"])
                        if refresh_interval is not None:
                            refresh_started = True
                        with current_subtask_lock:
                            logging.info("Subtask: %s", current_subtask["text"])
                            print(f"Subtask: {current_subtask['text']}")

                        # Query model to get action chunk.
                        with current_subtask_lock:
                            low_level_prompt = current_subtask["text"]
                        action_response = client.infer(
                            images=images,
                            high_level_prompt=str(task_description),
                            low_level_prompt=low_level_prompt,
                            state=state,
                            generate_subtask=False,
                            max_decoding_steps=args.max_decoding_steps,
                            temperature=args.temperature,
                        )
                        action_chunk = action_response.get("actions")
                        if action_chunk is None:
                            raise RuntimeError("No actions returned from server.")
                        action_chunk = np.asarray(action_chunk)
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, _reward, done, _info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error("Caught exception: %s", e)
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info("Success: %s", done)
            logging.info("# episodes completed so far: %d", total_episodes)
            logging.info("# successes: %d (%.1f%%)", total_successes, total_successes / total_episodes * 100.0)

        # Log final results
        logging.info("Current task success rate: %f", float(task_successes) / float(task_episodes))
        logging.info("Current total success rate: %f", float(total_successes) / float(total_episodes))

    logging.info("Total success rate: %f", float(total_successes) / float(total_episodes))
    logging.info("Total episodes: %d", total_episodes)
    client.close()


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
    eval_libero(args)

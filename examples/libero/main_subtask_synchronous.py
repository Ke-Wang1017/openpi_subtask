from __future__ import annotations

import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import time
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
LIBERO_ENV_RESOLUTION = 128  # resolution used to render training data


class SubtaskSyncClient:
    """Synchronous JSON websocket client for sync_pi05_websocket_server.py."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        *,
        connect_timeout_s: float = 60.0,
        connect_retries: int = 5,
        connect_retry_delay_s: float = 2.0,
    ):
        uri = f"ws://{host}:{port}"
        logging.info("Connecting to sync subtask server: %s", uri)
        last_error = None
        for attempt in range(1, connect_retries + 1):
            try:
                self._ws = websockets.sync.client.connect(
                    uri,
                    compression=None,
                    max_size=10 * 1024 * 1024,
                    open_timeout=connect_timeout_s,
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= connect_retries:
                    raise
                logging.warning(
                    "WebSocket connect attempt %d/%d failed: %s. Retrying in %.1fs...",
                    attempt,
                    connect_retries,
                    exc,
                    connect_retry_delay_s,
                )
                time.sleep(connect_retry_delay_s)
        if last_error is not None:
            logging.info("WebSocket connection established after retries.")
        metadata = self._ws.recv()
        self._metadata = json.loads(metadata)
        logging.info("Server metadata: %s", self._metadata)

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
        self._ws.send(json.dumps(request))
        response = json.loads(self._ws.recv())
        if response.get("status") == "error":
            raise RuntimeError(f"Server error: {response.get('error')}")
        return response

    def close(self) -> None:
        self._ws.close()


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8765
    resize_size: int = 128
    replan_steps: int = 10
    max_decoding_steps: int = 25
    temperature: float = 0.1
    subtask_refresh_interval: float = 1.0

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

    client = SubtaskSyncClient(args.host, args.port)

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
            current_subtask = ""
            last_subtask_refresh = 0.0
            refresh_started = False

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

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            obs["robot0_eef_quat"],
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

                    if (
                        refresh_started
                        and args.subtask_refresh_interval > 0
                        and (time.time() - last_subtask_refresh) >= args.subtask_refresh_interval
                    ):
                        subtask_response = client.infer(
                            images=images,
                            high_level_prompt=str(task_description),
                            low_level_prompt=current_subtask,
                            state=state,
                            generate_subtask=True,
                            max_decoding_steps=args.max_decoding_steps,
                            temperature=args.temperature,
                        )
                        if subtask_response.get("subtask"):
                            current_subtask = subtask_response["subtask"]
                            logging.info("Subtask refresh: %s", current_subtask)
                            print(f"Subtask refresh: {current_subtask}")
                        last_subtask_refresh = time.time()

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        subtask_response = client.infer(
                            images=images,
                            high_level_prompt=str(task_description),
                            low_level_prompt=current_subtask,
                            state=state,
                            generate_subtask=True,
                            max_decoding_steps=args.max_decoding_steps,
                            temperature=args.temperature,
                        )
                        if subtask_response.get("subtask"):
                            current_subtask = subtask_response["subtask"]
                        if args.subtask_refresh_interval > 0:
                            refresh_started = True
                            last_subtask_refresh = time.time()
                        logging.info("Subtask: %s", current_subtask)
                        print(f"Subtask: {current_subtask}")

                        # Query model to get action chunk.
                        action_response = client.infer(
                            images=images,
                            high_level_prompt=str(task_description),
                            low_level_prompt=current_subtask,
                            state=state,
                            generate_subtask=False,
                            max_decoding_steps=args.max_decoding_steps,
                            temperature=args.temperature,
                        )
                        action_chunk = action_response.get("actions")[0]
                        logging.info("action_chunk shape: %s", np.array(action_chunk).shape)
                        if action_chunk is None:
                            raise RuntimeError("No actions returned from server.")
                        action_chunk = np.asarray(action_chunk)

                        logging.info("action_chunk length: %d", len(action_chunk))
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()[:7]  # only take the first 7 actions

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
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_episode_{episode_idx + 1:04d}_{suffix}.mp4",
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

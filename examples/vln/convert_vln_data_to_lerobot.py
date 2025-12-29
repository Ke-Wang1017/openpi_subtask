"""
Convert processed VLN dataset (our generated per-trajectory folders) into a LeRobot dataset repo.

Usage:
uv run examples/vln/convert_vln_data_to_lerobot.py \
  --input_root /root/workspace/chenyj36@xiaopeng.com/data/navigation/processed_vln_n1_final \
  --repo_id vln_n1_nav
"""

import json
from pathlib import Path
import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import tyro


def parse_image(path: Path):
    arr = np.array(Image.open(path))
    # LeRobot expects dtype 'image' and will handle storage; keep uint8
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def parse_depth_image(path: Path):
    """Parse depth image and convert to RGB format"""
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        # Convert depth to RGB by normalizing and repeating across channels
        # Normalize to 0-255 range
        if arr.dtype == np.uint16:
            arr = (arr / 65535.0 * 255).astype(np.uint8)
        elif arr.dtype == np.uint8:
            pass  # Already in correct range
        else:
            arr = (arr / arr.max() * 255).astype(np.uint8)
        # Stack to create RGB
        arr = np.stack([arr] * 3, axis=-1)
    return arr


def resize_image(image, size=(224, 224)):
    """Resize image to target size"""
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.dtype == np.uint16:
            image = (image / 65535.0 * 255).astype(np.uint8)
        else:
            image = (image / image.max() * 255).astype(np.uint8)

    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def main(input_root: str, repo_id: str = "vln_n1_nav", *, push_to_hub: bool = False):
    input_root = Path(input_root)

    # 设置本地数据集路径
    local_dataset_path = Path("/workspace/chenyj36@xiaopeng.com/lerobot_datasets") / repo_id
    if local_dataset_path.exists():
        print(f"删除已存在的数据集目录: {local_dataset_path}")
        shutil.rmtree(local_dataset_path)

    # 确保父目录存在
    local_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # Define features for VLN
    dataset = LeRobotDataset.create(
        repo_id=str(local_dataset_path),  # 使用本地路径
        robot_type="vln",
        fps=10,
        features={
            "rgb": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "depth_rgb": {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
            "point_goal": {"dtype": "float32", "shape": (3,), "names": ["point_goal"]},
            "actions": {"dtype": "float32", "shape": (3,), "names": ["actions"]},
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    # Iterate episodes: one per trajectory (episode_000000.parquet)
    parquets = list(input_root.glob("**/data/chunk-000/episode_000000.parquet"))
    print(f"Found {len(parquets)} episodes")

    for pq in tqdm(parquets, desc="Converting VLN episodes"):
        # 正确的路径构建:parquet文件在 data/chunk-000/episode_xxx.parquet
        # 所以轨迹目录应该是 parquet文件的父目录的父目录的父目录
        traj_dir = pq.parents[2]  # 从 episode_xxx.parquet -> chunk-000 -> data -> trajectory_xx
        videos = traj_dir / "videos" / "chunk-000"
        rgb_dir = videos / "observation.images.rgb"
        depth_dir = videos / "observation.images.depth"
        meta_dir = traj_dir / "meta"

        print(f"处理轨迹: {traj_dir}")
        print(f"RGB目录: {rgb_dir}")
        print(f"Depth目录: {depth_dir}")
        print(f"RGB目录存在: {rgb_dir.exists()}")
        print(f"Depth目录存在: {depth_dir.exists()}")

        # task/prompt - 从tasks.jsonl读取原始task描述
        task_str = "Navigate to the goal point"
        ep_file = meta_dir / "tasks.jsonl"
        if ep_file.exists():
            try:
                t = json.loads(ep_file.read_text())
                if isinstance(t, dict) and "task_description" in t:
                    task_str = str(t["task_description"])
                    print(f"使用原始task描述: {task_str[:100]}...")
            except Exception as e:
                print(f"读取task描述失败: {e}")

        # Load images list (sorted by filename index)
        rgb_files = sorted(rgb_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        depth_files = sorted(depth_dir.glob("*.png"), key=lambda p: int(p.stem))

        if len(rgb_files) != len(depth_files) or len(rgb_files) == 0:
            print(f"跳过轨迹 {traj_dir}: RGB文件数={len(rgb_files)}, Depth文件数={len(depth_files)}")
            continue

        # Load point_goal
        point_goal = [0.0, 0.0, 0.0]
        ep_meta = meta_dir / "episodes.jsonl"
        if ep_meta.exists():
            try:
                m = json.loads(ep_meta.read_text())
                if isinstance(m, dict) and "point_goal" in m:
                    pg = m["point_goal"]
                    if isinstance(pg, list | tuple) and len(pg) == 3:
                        point_goal = [float(x) for x in pg]
            except Exception:
                pass

        # Load actions from parquet
        actions_df = pd.read_parquet(pq)
        actions = actions_df["action"].tolist()

        # Write frames
        for i, (rf, dfp) in enumerate(zip(rgb_files, depth_files, strict=True)):
            rgb = resize_image(parse_image(rf))
            depth_rgb = resize_image(parse_depth_image(dfp))
            act = actions[i] if i < len(actions) else [0.0, 0.0, 0.0]
            dataset.add_frame(
                {
                    "rgb": rgb,
                    "depth_rgb": depth_rgb,
                    "point_goal": np.asarray(point_goal, dtype=np.float32),
                    "actions": np.asarray(act, dtype=np.float32),
                    "task": task_str,
                }
            )
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["vln", "navigation"],
            private=False,
            push_videos=False,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

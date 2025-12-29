import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class VLNInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # state: optional 3D point goal (x, y, theta)
        point_goal = data.get("observation/point_goal")
        state = np.zeros(3, dtype=np.float32) if point_goal is None else np.asarray(point_goal, dtype=np.float32)

        # images: rgb + depth (converted to 3-channel)
        rgb = (
            _parse_image(data["observation/rgb"])
            if "observation/rgb" in data
            else np.zeros((224, 224, 3), dtype=np.uint8)
        )
        depth_src = data.get("observation/depth_rgb")
        depth = _parse_image(depth_src) if depth_src is not None else np.zeros_like(rgb)

        # map to model image names
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (rgb, depth, np.zeros_like(rgb))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (rgb, np.zeros_like(rgb), depth)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        # Use 'task' if present; otherwise use default
        if "task" in data:
            if isinstance(data["task"], bytes):
                data["task"] = data["task"].decode("utf-8")
            inputs["prompt"] = data["task"]
        else:
            # Use default prompt for VLN navigation
            inputs["prompt"] = "Navigate to the goal point"

        return inputs


@dataclasses.dataclass(frozen=True)
class VLNOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Ensure we only return the first 3 dims (x, y, theta)
        return {"actions": np.asarray(data["actions"])[:, :3]}

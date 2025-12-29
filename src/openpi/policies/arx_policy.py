import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_arx_example() -> dict:
    """Creates a random input example for the ARX policy."""
    return {
        "state": np.random.rand(14),
        "image": {
            "left_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "face_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Inputs for the ARX policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI0

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist_view", "face_view", "right_wrist_view")

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel].
            output_image = einops.rearrange(img, "c h w -> h w c") if img.shape[-1] != 3 else img
            assert output_image.shape[-1] == 3, f"Image must have 3 channels, got {output_image.shape}."
            # print(f"Output image shape: {output_image.shape}")
            return output_image

        # Convert images to uint8 and rearrange to (H,W,C) format
        for key in self.EXPECTED_CAMERAS:
            assert key in data["images"], f"Images must contain {key}."
            data["images"][key] = convert_image(data["images"][key])

        inputs = {
            "image": {
                "base_0_rgb": data["images"]["face_view"],
                "left_wrist_0_rgb": data["images"]["left_wrist_view"],
                "right_wrist_0_rgb": data["images"]["right_wrist_view"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }

        if "actions" in data:
            # print(f"Input actions shape 1: {data['actions'].shape}")
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
            # print(f"Input actions shape 2: {data['actions'].shape}")
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Outputs for the ARX policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class ArxMoveInputs(transforms.DataTransformFn):
    """Inputs for the ARX policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [14]
    - actions: [action_horizon, 14]
    """

    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI0

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist_view", "face_view", "right_wrist_view")

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel].
            output_image = einops.rearrange(img, "c h w -> h w c") if img.shape[-1] != 3 else img
            assert output_image.shape[-1] == 3, f"Image must have 3 channels, got {output_image.shape}."
            # print(f"Output image shape: {output_image.shape}")
            return output_image

        # Convert images to uint8 and rearrange to (H,W,C) format
        for key in self.EXPECTED_CAMERAS:
            assert key in data["images"], f"Images must contain {key}."
            data["images"][key] = convert_image(data["images"][key])

        inputs = {
            "image": {
                "base_0_rgb": data["images"]["face_view"],
                "left_wrist_0_rgb": data["images"]["left_wrist_view"],
                "right_wrist_0_rgb": data["images"]["right_wrist_view"],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }

        if "actions" in data:
            # print(f"Input actions shape 1: {data['actions'].shape}")
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
            # print(f"Input actions shape 2: {data['actions'].shape}")
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class ArxMoveOutputs(transforms.DataTransformFn):
    """Outputs for the ARX policy."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :20])
        return {"actions": actions}

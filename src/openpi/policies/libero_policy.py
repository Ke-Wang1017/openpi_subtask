import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _get_prompt(data: dict) -> str | None:
    for key in ("prompt", "task", "high_level_prompt"):
        if key in data:
            prompt = data[key]
            return prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
    return None


def _get_state(data: dict) -> np.ndarray:
    if "observation/state" in data:
        return data["observation/state"]
    if "state" in data:
        return data["state"]
    observation = data.get("observation")
    if isinstance(observation, dict) and "state" in observation:
        return observation["state"]
    raise KeyError("Missing state: expected 'observation/state' or 'state'.")


def _get_image(
    data: dict,
    *,
    keys: tuple[str, ...],
    observation_key: str,
    images_keys: tuple[str, ...],
    label: str,
) -> np.ndarray:
    for key in keys:
        if key in data:
            return _parse_image(data[key])
    observation = data.get("observation")
    if isinstance(observation, dict) and observation_key in observation:
        return _parse_image(observation[observation_key])
    images = data.get("images")
    if isinstance(images, dict):
        for key in images_keys:
            if key in images:
                return _parse_image(images[key])
    raise KeyError(
        f"Missing {label} image: expected one of {keys} or images.{images_keys} or observation.{observation_key}."
    )


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _get_image(
            data,
            keys=("observation/image", "image", "images.agentview_rgb", "images.base_0_rgb"),
            observation_key="image",
            images_keys=("base_0_rgb", "agentview_rgb"),
            label="base",
        )
        try:
            wrist_image = _get_image(
                data,
                keys=("observation/wrist_image", "wrist_image", "images.wrist_rgb_left", "images.wrist_rgb"),
                observation_key="wrist_image",
                images_keys=("left_wrist_0_rgb", "wrist_rgb_left", "wrist_rgb"),
                label="wrist",
            )
        except KeyError:
            wrist_image = np.zeros_like(base_image)

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": _get_state(data),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        prompt = _get_prompt(data)
        if prompt is not None:
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_subtask_lerobot_example() -> dict:
    """generate example data for Libero Subtask dataset"""
    return {
        # 外部相机图像:LeRobot格式为通道优先(3,224,224),uint8
        "images.agentview_rgb": np.random.randint(256, size=(3, 128, 128), dtype=np.uint8),
        # 手腕相机图像:与外部图像格式一致
        "images.wrist_rgb": np.random.randint(256, size=(3, 128, 128), dtype=np.uint8),
        # 8维状态:由7维(3位置+3轴角+1夹爪)转换而来
        "state": np.random.rand(8).astype(np.float32),
        # 高层任务指令
        "task": "pick up the red block and place it on the blue tray",
        # 低层子任务指令
        "subtask": "move arm to block position",
        # 7维动作:与动作维度匹配
        "actions": np.random.rand(30, 7).astype(np.float32),
    }


def _parse_lerobot_image(image: np.ndarray) -> np.ndarray:
    """parse LeRobot image to model input format, strictly corresponding to image conversion logic"""
    image = np.asarray(image)

    # 处理数据类型:LeRobot图像为uint8,若存在异常则强制转换
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    # 处理通道与维度:LeRobot为通道优先(3,H,W),转为模型需要的(H,W,3)
    if image.ndim == 3:
        if image.shape[0] == 3:  # 标准LeRobot图像格式(3,224,224)
            image = einops.rearrange(image, "c h w -> h w c")
        elif image.shape[-1] == 3:  # 异常的通道在后格式,直接使用
            pass
        elif image.shape[0] == 1:  # 灰度图扩展为RGB(兼容潜在异常)
            image = einops.repeat(image, "1 h w -> h w 3")
        elif image.shape[0] == 4:  # RGBA转RGB(兼容潜在异常)
            image = einops.rearrange(image[:3, ...], "c h w -> h w c")
    elif image.ndim == 2:  # 2D灰度图转RGB
        image = einops.repeat(image, "h w -> h w 3")

    return image


@dataclasses.dataclass(frozen=True)
class LiberoSubtaskInputs(transforms.DataTransformFn):
    """adapt to support Subtask Libero dataset"""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 1. process state: directly read LeRobot's 8-dimensional state
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(f"Libero state dimension error, expected 8, got {state.shape[0]}")

        # 2. process image: parse LeRobot's exterior_image and wrist_image
        exterior_image = _parse_lerobot_image(data["images.agentview_rgb"])
        if "images.wrist_rgb_left" in data:
            wrist_image_left = _parse_lerobot_image(data["images.wrist_rgb_left"])
        elif "images.wrist_rgb" in data:
            wrist_image_left = _parse_lerobot_image(data["images.wrist_rgb"])
        else:
            raise KeyError("Missing wrist image key: expected 'images.wrist_rgb_left' or 'images.wrist_rgb'")

        # 3. organize image input based on model type
        # Match LiberoInputs: include right_wrist_0_rgb as zeroed placeholder
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # external camera + left wrist camera + right wrist camera (zeroed placeholder)
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (exterior_image, wrist_image_left, np.zeros_like(exterior_image))
                image_masks = (np.True_, np.True_, np.False_)  # Mask the zeroed right wrist
            case _model.ModelType.PI0_FAST:
                # external camera + left wrist camera + right wrist camera (zeroed placeholder)
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (exterior_image, wrist_image_left, np.zeros_like(exterior_image))
                image_masks = (np.True_, np.True_, np.True_)  # PI0_FAST doesn't mask padding
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # 4. build model input dictionary
        inputs = {
            "state": state,
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }

        # 5. add action data (used during training)
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 7:
                raise ValueError(f"Libero actions dimension error, expected 7, got {actions.shape[0]}")
            inputs["actions"] = actions

        # 6. ⭐ core: add high-level and low-level task instructions
        if "task" in data:
            high_prompt = data["task"]
            if isinstance(high_prompt, bytes):
                high_prompt = high_prompt.decode("utf-8")
            inputs["high_prompt"] = high_prompt

        if "subtask" in data:
            low_prompt = data["subtask"]
            if isinstance(low_prompt, bytes):
                low_prompt = low_prompt.decode("utf-8")
            inputs["low_prompt"] = low_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoSubtaskOutputs(transforms.DataTransformFn):
    """convert outputs from model to dataset format"""

    def __call__(self, data: dict) -> dict:
        # only return the first 7 actions
        return {"actions": np.asarray(data["actions"])[:, :7]}

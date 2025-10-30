import dataclasses
import numpy as np
import einops

from openpi import transforms
from openpi.models import model as _model


def make_flexiv_subtask_lerobot_example() -> dict:
    """生成符合 Subtask LeRobot 数据集格式的示例数据"""
    return {
        # 外部相机图像：LeRobot格式为通道优先(3,224,224)，uint8
        "exterior_image": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        # 左手腕相机图像：与外部图像格式一致
        "wrist_image_left": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        # 10维状态：由7维(3位置+3轴角+1夹爪)转换而来
        "state": np.random.rand(10).astype(np.float32),
        # 高层任务指令
        "task": "pick up the red block and place it on the blue tray",
        # 低层子任务指令
        "subtask": "move arm to block position",
        # 10维动作：与状态维度匹配
        "actions": np.random.rand(30, 10).astype(np.float32)
    }


def _parse_lerobot_image(image: np.ndarray) -> np.ndarray:
    """解析LeRobot格式图像为模型输入格式，严格对应图像转换逻辑"""
    image = np.asarray(image)
    
    # 处理数据类型：LeRobot图像为uint8，若存在异常则强制转换
    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    
    # 处理通道与维度：LeRobot为通道优先(3,H,W)，转为模型需要的(H,W,3)
    if image.ndim == 3:
        if image.shape[0] == 3:  # 标准LeRobot图像格式(3,224,224)
            image = einops.rearrange(image, "c h w -> h w c")
        elif image.shape[-1] == 3:  # 异常的通道在后格式，直接使用
            pass
        elif image.shape[0] == 1:  # 灰度图扩展为RGB（兼容潜在异常）
            image = einops.repeat(image, "1 h w -> h w 3")
        elif image.shape[0] == 4:  # RGBA转RGB（兼容潜在异常）
            image = einops.rearrange(image[:3, ...], "c h w -> h w c")
    elif image.ndim == 2:  # 2D灰度图转RGB
        image = einops.repeat(image, "h w -> h w 3")
    
    return image


@dataclasses.dataclass(frozen=True)
class FlexivSubtaskInputs(transforms.DataTransformFn):
    """适配支持 Subtask 的 Flexiv LeRobot 数据集的输入转换类"""
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 1. 处理状态：直接读取LeRobot的10维state
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (10,):
            raise ValueError(f"Flexiv state维度错误，预期10维，实际{state.shape[0]}维")

        # 2. 处理图像：解析LeRobot的exterior_image和wrist_image_left
        exterior_image = _parse_lerobot_image(data["exterior_image"])
        wrist_image_left = _parse_lerobot_image(data["wrist_image_left"])

        # 3. 根据模型类型组织图像输入
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # 外部相机 + 左手腕相机 + 右手腕填充（无数据）
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (exterior_image, wrist_image_left, np.zeros_like(exterior_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                # 外部相机 + 填充图 + 左手腕相机
                image_names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (exterior_image, np.zeros_like(exterior_image), wrist_image_left)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"不支持的模型类型: {self.model_type}")

        # 4. 构建模型输入字典
        inputs = {
            "state": state,
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }

        # 5. 添加动作数据（训练时使用）
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 10:
                raise ValueError(f"Flexiv actions维度错误，预期10维，实际{actions.shape[-1]}维")
            inputs["actions"] = actions

        # 6. ⭐ 核心：添加高层和低层任务指令
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
class FlexivSubtaskOutputs(transforms.DataTransformFn):
    """10维状态的输出转换类，与 FlexivOutputs 保持一致"""
    def __call__(self, data: dict) -> dict:
        # 仅返回前10维动作
        return {"actions": np.asarray(data["actions"])[:, :10]}


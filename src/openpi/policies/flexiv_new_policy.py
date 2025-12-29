import dataclasses
import numpy as np
import einops

from openpi import transforms
from openpi.models import model as _model


def make_flexiv_lerobot_example() -> dict:
    """生成符合当前LeRobot数据集格式的示例数据，匹配转换代码输出"""
    return {
        # 外部相机图像：LeRobot格式为通道优先(3,224,224)，uint8
        "exterior_image": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        # 左手腕相机图像：与外部图像格式一致
        "wrist_image_left": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        # 10维状态：由7维(3位置+3轴角+1夹爪)转换而来
        "state": np.random.rand(10).astype(np.float32),
        # 任务指令：对应HDF5的language_instruction字段
        "task": "pick up the red block and place it on the blue tray",
        # 10维动作：与状态维度匹配，同样由7维转换而来
        "actions": np.random.rand(30, 10).astype(np.float32)  # 多帧动作示例
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
class FlexivLerobotInputs(transforms.DataTransformFn):
    """适配当前LeRobot数据集的输入转换类，键值完全对应"""
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 1. 处理状态：直接读取LeRobot的10维state（无需额外转换）
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (10,):
            raise ValueError(f"LeRobot state维度错误，预期10维，实际{state.shape[0]}维")

        # 2. 处理图像：解析LeRobot的exterior_image和wrist_image_left
        exterior_image = _parse_lerobot_image(data["exterior_image"])  # 对应camera_3
        wrist_image_left = _parse_lerobot_image(data["wrist_image_left"])  # 对应camera_2

        # 3. 根据模型类型组织图像输入（与转换代码的相机映射一致）
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                # 外部相机(camera_3) + 左手腕相机(camera_2) + 右手腕填充（无数据）
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (exterior_image, wrist_image_left, np.zeros_like(exterior_image))
                image_masks = (np.True_, np.True_, np.False_)  # 标记填充的右手腕图像
            case _model.ModelType.PI0_FAST:
                # 外部相机(camera_3) + 填充图 + 左手腕相机(camera_2)
                image_names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (exterior_image, np.zeros_like(exterior_image), wrist_image_left)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"不支持的模型类型: {self.model_type}")

        # 4. 构建模型输入字典（严格对应LeRobot的键值）
        inputs = {
            "state": state,
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }

        # 5. 添加动作数据（训练时使用，与LeRobot的actions字段匹配）
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != 10:
                raise ValueError(f"LeRobot actions维度错误，预期10维，实际{actions.shape[-1]}维")
            inputs["actions"] = actions

        # 6. 添加任务指令（对应LeRobot的task字段，源自HDF5的language_instruction）
        if "task" in data:
            prompt = data["task"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")  # 处理字节格式的指令
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class FlexivOutputs(transforms.DataTransformFn):
    """10维状态的输出转换类，完全匹配案例写法"""
    def __call__(self, data: dict) -> dict:
        # 仅返回前10维动作，与案例中截取前8维的写法保持一致
        return {"actions": np.asarray(data["actions"])[:, :10]}

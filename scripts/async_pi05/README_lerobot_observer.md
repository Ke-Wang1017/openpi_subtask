# LeRobot 数据格式异步推理观察器

这个工具允许你直接使用 inference engine 来观察 LeRobot 格式数据的异步推理输出，无需启动 WebSocket 服务器。

## 主要功能

- 🎯 **直接使用 inference engine**：无需 WebSocket 服务器
- 📊 **LeRobot 数据格式支持**：自动处理 LeRobot 的 HDF5 和 JSONL 格式
- 🔄 **持续观察**：支持多帧连续推理观察
- 📝 **自动保存**：推理结果自动保存为 JSON 文件
- 🔔 **回调机制**：支持自定义观察回调函数
- ⚡ **异步处理**：完全异步的推理和观察过程

## 文件说明

- `lerobot_inference_observer.py` - 主要的观察器类
- `example_lerobot_usage.py` - 使用示例
- `test_lerobot_observer.py` - 测试脚本

## 快速开始

### 1. 基本使用

```python
import asyncio
from lerobot_inference_observer import LeRobotInferenceObserver

async def main():
    # 创建观察器
    observer = LeRobotInferenceObserver(
        config_name="right_pi05_20",
        gpu_id=1,
        output_dir="./inference_outputs"
    )
    
    # 添加观察回调
    async def on_observation(data):
        print(f"帧 {data['frame_idx']}: {data['result']['subtask']}")
    
    observer.add_observation_callback(on_observation)
    
    # 准备 LeRobot 数据
    episode_data = {
        "base": your_base_images,  # (T, H, W, C) 或 (H, W, C)
        "left_wrist": your_left_images,
        "right_wrist": your_right_images,
        "state": your_state_data,  # (T, state_dim) 或 (state_dim,)
        "high_level_prompt": "Your high level task",
        "low_level_prompt": "Your low level task"
    }
    
    # 单次推理观察
    result = await observer.observe_single_inference(
        episode_data=episode_data,
        frame_idx=0,
        high_level_prompt="Pick up the red block",
        low_level_prompt="Move to the block and grasp it"
    )
    
    # 持续推理观察
    results = await observer.observe_continuous_inference(
        episode_data=episode_data,
        start_frame=0,
        max_frames=10,
        frame_interval=1.0,
        subtask_refresh_interval=2.0
    )

asyncio.run(main())
```

### 2. 加载真实 LeRobot 数据

```python
# 从 HDF5 文件加载
episode_data = observer.load_lerobot_episode("/path/to/episode.hdf5")

# 从 JSONL 文件加载
episode_data = observer.load_lerobot_episode("/path/to/episode.jsonl")
```

### 3. 运行测试

```bash
# 运行基本测试
python test_lerobot_observer.py

# 运行使用示例
python example_lerobot_usage.py
```

## 数据格式支持

### 输入数据格式

观察器支持以下 LeRobot 数据格式：

- **图像数据**：
  - `base`: 基础视角图像
  - `left_wrist`: 左手腕视角图像  
  - `right_wrist`: 右手腕视角图像
  - 支持形状：(T, H, W, C) 或 (H, W, C)

- **状态数据**：
  - `state`: 机器人状态向量
  - 支持形状：(T, state_dim) 或 (state_dim,)

- **任务描述**：
  - `high_level_prompt`: 高级任务描述
  - `low_level_prompt`: 低级任务描述

### 输出数据格式

每次推理观察会生成包含以下信息的 JSON 文件：

```json
{
  "timestamp": 1234567890.123,
  "frame_idx": 0,
  "inference_time": 0.456,
  "result": {
    "actions": [[...]],  // 动作序列
    "subtask": "Move to the block and grasp it",
    "subtask_tokens": [...],
    "state": [...],
    "timing": {...}
  },
  "images_shape": {
    "base_0_rgb": [224, 224, 3],
    "left_wrist_0_rgb": [224, 224, 3],
    "right_wrist_0_rgb": [224, 224, 3]
  },
  "high_level_prompt": "Pick up the red block",
  "low_level_prompt": "Move to the block and grasp it"
}
```

## 高级功能

### 1. 自定义观察回调

```python
async def custom_callback(data):
    # 处理推理结果
    subtask = data['result']['subtask']
    actions = data['result']['actions']
    
    # 发送到其他系统
    await send_to_robot(actions)
    await log_to_database(subtask)

observer.add_observation_callback(custom_callback)
```

### 2. 子任务定期刷新

```python
# 启用子任务定期刷新
results = await observer.observe_continuous_inference(
    episode_data=episode_data,
    subtask_refresh_interval=2.0  # 每2秒刷新子任务
)
```

### 3. 批量处理多个 episode

```python
episode_paths = ["/path/to/episode1.hdf5", "/path/to/episode2.hdf5"]

for episode_path in episode_paths:
    episode_data = observer.load_lerobot_episode(episode_path)
    results = await observer.observe_continuous_inference(
        episode_data=episode_data,
        start_frame=0,
        max_frames=5
    )
```

## 注意事项

1. **内存使用**：长时间持续观察可能消耗大量内存，建议定期清理
2. **GPU 资源**：确保有足够的 GPU 内存用于推理
3. **数据格式**：确保 LeRobot 数据格式正确，缺少的数据会使用随机数据填充
4. **异步处理**：所有操作都是异步的，需要使用 `await` 关键字

## 故障排除

### 常见问题

1. **模型初始化失败**：
   - 检查 GPU 可用性
   - 确认模型配置文件存在
   - 检查依赖包是否正确安装

2. **数据加载失败**：
   - 检查文件路径是否正确
   - 确认数据格式是否符合 LeRobot 标准
   - 查看日志中的详细错误信息

3. **推理速度慢**：
   - 检查 GPU 使用情况
   - 考虑减少 `max_decoding_steps` 参数
   - 调整 `frame_interval` 参数

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 启用详细日志
observer = LeRobotInferenceObserver(...)
```

# 异步 Pi0.5 推理系统

基于 `test_pi05_subtask_generation.py` 重构的异步推理系统，支持子任务生成和动作预测。

## 文件说明

### 1. `async_pi05_inference.py`
核心异步推理引擎，包含：
- `AsyncPi05Inference` 类：主要的推理引擎
- 支持子任务生成和动作预测
- 异步模型初始化和推理
- 图像加载和预处理
- JIT 编译优化

### 2. `async_pi05_websocket_server.py`
WebSocket 服务器，提供：
- 异步 WebSocket 通信
- 客户端连接管理
- JSON 格式的请求/响应
- 错误处理和日志记录

### 3. `async_pi05_client.py`
WebSocket 客户端，支持：
- 异步连接和通信
- 单个和批量推理请求
- 图像加载和预处理
- 结果解析和时序统计

## 使用方法

### 启动服务器
```bash
cd /root/workspace/chenyj36@xiaopeng.com/openpi_pure_subtask/scripts
python async_pi05_websocket_server.py
```

### 运行客户端测试
```bash
python async_pi05_client.py
```

### 直接使用推理引擎
```python
import asyncio
from async_pi05_inference import AsyncPi05Inference

async def main():
    # 创建推理引擎
    inference = AsyncPi05Inference(config_name="right_pi05_20", gpu_id=1)
    
    # 准备图像数据
    images = {
        "base_0_rgb": your_base_image,
        "left_wrist_0_rgb": your_left_image,
        "right_wrist_0_rgb": your_right_image
    }
    
    # 执行推理
    result = await inference.infer(
        images=images,
        high_level_prompt="Pick up the flashcard on the table",
        generate_subtask=True
    )
    
    print(f"生成的动作: {result['actions']}")
    print(f"生成的子任务: {result['subtask']}")

asyncio.run(main())
```

## API 接口

### 推理请求格式
```json
{
    "images": {
        "base_0_rgb": [[[r,g,b], ...], ...],  // 图像数据
        "left_wrist_0_rgb": [[[r,g,b], ...], ...],
        "right_wrist_0_rgb": [[[r,g,b], ...], ...]
    },
    "high_level_prompt": "Pick up the flashcard on the table",
    "low_level_prompt": "ABCDEFG",  // 可选
    "state": [0.0, 0.0, ...],  // 可选，机器人状态
    "generate_subtask": true,  // 是否生成子任务
    "max_decoding_steps": 25,  // 最大解码步数
    "temperature": 0.1,  // 采样温度
    "subtask_refresh_interval": 2.0  // 可选，子任务刷新间隔（秒）
}
```

### 推理响应格式
```json
{
    "status": "success",
    "actions": [[[x,y,z,rx,ry,rz,gripper], ...], ...],  // 动作序列
    "subtask": "move to table and then pick up black pen",  // 生成的子任务
    "subtask_tokens": [1, 2, 3, ...],  // 子任务 tokens
    "state": [0.0, 0.0, ...],  // 机器人状态
    "timing": {
        "total_ms": 1500.0,
        "action_ms": 800.0,
        "subtask_ms": 700.0
    },
    "server_timing": {
        "total_ms": 1600.0
    },
    "subtask_refresh_enabled": true,  // 是否启用了定期刷新
    "subtask_refresh_interval": 2.0  // 刷新间隔（秒）
}
```

### 定期刷新消息格式
当启用定期刷新时，服务器会定期发送刷新消息：
```json
{
    "type": "subtask_refresh",
    "subtask": "updated subtask based on current state",
    "subtask_tokens": [1, 2, 3, ...],
    "refresh_count": 3,  // 刷新次数
    "timestamp": 1703123456.789
}
```

## 定期刷新功能

系统支持定期刷新子任务，让机器人能够根据当前状态动态调整任务计划：

### 启用定期刷新
```python
# 在推理请求中设置刷新间隔
result = await client.infer(
    images=images,
    high_level_prompt="Pick up the flashcard on the table",
    subtask_refresh_interval=2.0  # 每2秒刷新一次子任务
)
```

### 监听刷新消息
```python
async def on_refresh(data):
    print(f"新子任务: {data['subtask']}")
    print(f"刷新次数: {data['refresh_count']}")

# 启动监听
listen_task = asyncio.create_task(
    client.listen_for_refresh_messages(callback=on_refresh)
)
```

### 刷新机制
- 服务器会根据设定的间隔定期生成新的子任务
- 每次刷新都会基于当前的图像和状态重新生成
- 客户端可以实时接收并处理新的子任务
- 支持动态调整刷新间隔

## 性能优化

1. **JIT 编译**: 使用 `nnx_utils.module_jit` 编译关键函数
2. **内存优化**: 设置 GPU 内存限制和分配策略
3. **异步处理**: 支持并发推理请求
4. **批处理**: 支持批量推理请求
5. **定期刷新**: 支持动态子任务更新

## 配置参数

- `config_name`: 模型配置名称，默认 "right_pi05_20"
- `gpu_id`: GPU 设备 ID，默认 1
- `max_decoding_steps`: 子任务生成最大步数，默认 25
- `temperature`: 采样温度，默认 0.1
- `subtask_refresh_interval`: 子任务刷新间隔（秒），None 表示不刷新

## 错误处理

- 图像加载失败时自动使用随机图像
- 网络连接异常时自动重连
- 推理错误时返回详细错误信息
- 支持超时和重试机制

## 日志记录

所有组件都支持详细的日志记录，包括：
- 连接状态
- 推理时序
- 错误信息
- 性能统计

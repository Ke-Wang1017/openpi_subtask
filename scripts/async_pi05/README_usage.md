# Pi0.5 异步推理使用指南

## 功能概述

这个异步推理系统支持：
- **子任务生成**: 根据高级任务和当前状态生成具体子任务
- **动作预测**: 基于子任务生成机器人动作序列
- **定期刷新**: 每2秒自动更新子任务，实现动态任务规划
- **实时通信**: WebSocket 异步通信，支持并发请求

## 快速开始

### 1. 启动服务器

```bash
cd /root/workspace/chenyj36@xiaopeng.com/openpi_pure_subtask/scripts/async_pi05
python async_pi05_websocket_server.py
```

### 2. 运行快速测试

```bash
# 简单测试
python quick_test.py

# 完整测试
python test_subtask_refresh.py --test-mode all
```

## 核心功能

### 子任务刷新机制

1. **初始推理**: 使用高级任务和初始低级任务生成第一个子任务
2. **定期刷新**: 每2秒自动生成新的子任务
3. **动态更新**: 新生成的子任务自动成为下一轮的 low_level_prompt
4. **动作生成**: 基于最新的子任务生成对应的动作序列

### 工作流程

```
高级任务: "Pick up the red block and organize the workspace"
    ↓
初始低级任务: "Move to the block, grasp it"
    ↓
生成子任务1: "Move to the red block, grasp it, lift it up"
    ↓ (2秒后)
生成子任务2: "Place the block in the designated area, organize nearby items"
    ↓ (2秒后)
生成子任务3: "Clean up the workspace, arrange tools properly"
    ↓ ...
```

## 使用方法

### 1. 基础推理

```python
from async_pi05_client import AsyncPi05Client

client = AsyncPi05Client(host="localhost", port=8765)
await client.connect()

# 准备数据
images = {
    "base_0_rgb": your_base_image,
    "left_wrist_0_rgb": your_left_image,
    "right_wrist_0_rgb": your_right_image
}

# 执行推理
result = await client.infer(
    images=images,
    high_level_prompt="Pick up the red block",
    low_level_prompt="Move to block, grasp it",
    generate_subtask=True
)

print(f"生成的子任务: {result['subtask']}")
print(f"动作序列: {result['actions']}")
```

### 2. 定期刷新

```python
# 启用定期刷新
result = await client.infer(
    images=images,
    high_level_prompt="Organize the workspace",
    low_level_prompt="Start organizing",
    generate_subtask=True,
    subtask_refresh_interval=2.0  # 每2秒刷新
)

# 监听刷新消息
async def on_refresh(data):
    print(f"新子任务: {data['subtask']}")

listen_task = asyncio.create_task(
    client.listen_for_refresh_messages(callback=on_refresh)
)
```

### 3. 批量测试

```python
# 测试多个任务
test_cases = [
    {"high_level_prompt": "Task 1", "low_level_prompt": "Subtask 1"},
    {"high_level_prompt": "Task 2", "low_level_prompt": "Subtask 2"},
]

results = await client.batch_infer(test_cases)
```

## 测试脚本

### 1. 快速测试
```bash
python quick_test.py
```

### 2. 子任务刷新测试
```bash
# 测试刷新循环
python test_subtask_refresh.py --test-mode refresh

# 测试子任务演化
python test_subtask_refresh.py --test-mode evolution

# 测试一致性
python test_subtask_refresh.py --test-mode consistency

# 完整测试
python test_subtask_refresh.py --test-mode all
```

### 3. 参数配置
```bash
python test_subtask_refresh.py \
    --high-prompt "Pick up the red block and organize the workspace" \
    --low-prompt "Move to the block, grasp it" \
    --refresh-interval 3.0 \
    --test-duration 15.0 \
    --num-cycles 5
```

## 配置参数

### 服务器配置
- `host`: 服务器地址 (默认: localhost)
- `port`: 服务器端口 (默认: 8765)
- `config_name`: 模型配置 (默认: right_pi05_20)
- `gpu_id`: GPU 设备 ID (默认: 1)

### 推理参数
- `high_level_prompt`: 高级任务描述
- `low_level_prompt`: 初始低级任务 (默认: "ABCDEFG")
- `generate_subtask`: 是否生成子任务 (默认: True)
- `subtask_refresh_interval`: 刷新间隔秒数 (默认: None，不刷新)
- `max_decoding_steps`: 最大解码步数 (默认: 25)
- `temperature`: 采样温度 (默认: 0.1)

## 输出格式

### 推理结果
```json
{
    "status": "success",
    "actions": [[[x,y,z,rx,ry,rz,gripper], ...], ...],
    "subtask": "Move to the red block, grasp it, lift it up",
    "subtask_tokens": [1, 2, 3, ...],
    "state": [0.0, 0.0, ...],
    "timing": {
        "total_ms": 1500.0,
        "action_ms": 800.0,
        "subtask_ms": 700.0
    },
    "subtask_refresh_enabled": true,
    "subtask_refresh_interval": 2.0
}
```

### 刷新消息
```json
{
    "type": "subtask_refresh",
    "subtask": "Place the block in the designated area",
    "subtask_tokens": [1, 2, 3, ...],
    "refresh_count": 2,
    "timestamp": 1703123456.789
}
```

## 故障排除

### 1. 连接问题
```bash
# 检查服务器是否运行
curl http://localhost:8765/healthz

# 启动服务器
python async_pi05_websocket_server.py
```

### 2. 模型加载问题
- 确保 GPU 内存充足
- 检查模型配置文件
- 验证权重文件路径

### 3. 性能优化
- 调整 `subtask_refresh_interval` 减少刷新频率
- 使用较小的 `max_decoding_steps`
- 降低 `temperature` 提高确定性

## 扩展功能

### 自定义回调
```python
class CustomInference(AsyncPi05Inference):
    async def _on_subtask_refresh(self, subtask_text, subtask_tokens, refresh_count, 
                                 updated_low_prompt, updated_actions):
        # 自定义处理逻辑
        await self.send_to_robot(updated_actions)
        await self.update_database(subtask_text)
```

### 集成到机器人系统
```python
# 将生成的子任务和动作发送给机器人执行
async def execute_subtask(subtask, actions):
    # 解析子任务
    # 执行动作序列
    # 更新机器人状态
    pass
```

# Async Pi0.5 Inference System

An asynchronous inference system refactored from `test_pi05_subtask_generation.py`, supporting subtask generation and action prediction.

## File Description

### 1. `async_pi05_inference.py`
Core asynchronous inference engine, including:
- `AsyncPi05Inference` class: Main inference engine
- Support for subtask generation and action prediction
- Asynchronous model initialization and inference
- Image loading and preprocessing
- JIT compilation optimization

### 2. `async_pi05_websocket_server.py`
WebSocket server, providing:
- Asynchronous WebSocket communication
- Client connection management
- JSON format request/response
- Error handling and logging

### 3. `async_pi05_client.py`
WebSocket client, supporting:
- Asynchronous connection and communication
- Single and batch inference requests
- Image loading and preprocessing
- Result parsing and timing statistics

## Usage

### Start the Server
```bash
cd /root/workspace/chenyj36@xiaopeng.com/openpi_pure_subtask/scripts
python async_pi05_websocket_server.py
```

### Run Client Test
```bash
python async_pi05_client.py
```

### Direct Use of Inference Engine
```python
import asyncio
from async_pi05_inference import AsyncPi05Inference

async def main():
    # Create inference engine
    inference = AsyncPi05Inference(config_name="right_pi05_20", gpu_id=1)
    
    # Prepare image data
    images = {
        "base_0_rgb": your_base_image,
        "left_wrist_0_rgb": your_left_image,
        "right_wrist_0_rgb": your_right_image
    }
    
    # Execute inference
    result = await inference.infer(
        images=images,
        high_level_prompt="Pick up the flashcard on the table",
        generate_subtask=True
    )
    
    print(f"Generated actions: {result['actions']}")
    print(f"Generated subtask: {result['subtask']}")

asyncio.run(main())
```

## API Interface

### Inference Request Format
```json
{
    "images": {
        "base_0_rgb": [[[r,g,b], ...], ...],  // Image data
        "left_wrist_0_rgb": [[[r,g,b], ...], ...],
        "right_wrist_0_rgb": [[[r,g,b], ...], ...]
    },
    "high_level_prompt": "Pick up the flashcard on the table",
    "low_level_prompt": "ABCDEFG",  // Optional
    "state": [0.0, 0.0, ...],  // Optional, robot state
    "generate_subtask": true,  // Whether to generate subtask
    "max_decoding_steps": 25,  // Maximum decoding steps
    "temperature": 0.1,  // Sampling temperature
    "subtask_refresh_interval": 2.0  // Optional, subtask refresh interval (seconds)
}
```

### Inference Response Format
```json
{
    "status": "success",
    "actions": [[[x,y,z,rx,ry,rz,gripper], ...], ...],  // Action sequence
    "subtask": "move to table and then pick up black pen",  // Generated subtask
    "subtask_tokens": [1, 2, 3, ...],  // Subtask tokens
    "state": [0.0, 0.0, ...],  // Robot state
    "timing": {
        "total_ms": 1500.0,
        "action_ms": 800.0,
        "subtask_ms": 700.0
    },
    "server_timing": {
        "total_ms": 1600.0
    },
    "subtask_refresh_enabled": true,  // Whether periodic refresh is enabled
    "subtask_refresh_interval": 2.0  // Refresh interval (seconds)
}
```

### Periodic Refresh Message Format
When periodic refresh is enabled, the server will periodically send refresh messages:
```json
{
    "type": "subtask_refresh",
    "subtask": "updated subtask based on current state",
    "subtask_tokens": [1, 2, 3, ...],
    "refresh_count": 3,  // Refresh count
    "timestamp": 1703123456.789
}
```

## Periodic Refresh Feature

The system supports periodic subtask refresh, allowing the robot to dynamically adjust task planning based on current state:

### Enable Periodic Refresh
```python
# Set refresh interval in inference request
result = await client.infer(
    images=images,
    high_level_prompt="Pick up the flashcard on the table",
    subtask_refresh_interval=2.0  # Refresh subtask every 2 seconds
)
```

### Listen for Refresh Messages
```python
async def on_refresh(data):
    print(f"New subtask: {data['subtask']}")
    print(f"Refresh count: {data['refresh_count']}")

# Start listening
listen_task = asyncio.create_task(
    client.listen_for_refresh_messages(callback=on_refresh)
)
```

### Refresh Mechanism
- Server periodically generates new subtasks based on the set interval
- Each refresh regenerates based on current images and state
- Client can receive and process new subtasks in real-time
- Supports dynamic adjustment of refresh interval

## Performance Optimization

1. **JIT Compilation**: Uses `nnx_utils.module_jit` to compile critical functions
2. **Memory Optimization**: Sets GPU memory limits and allocation strategies
3. **Async Processing**: Supports concurrent inference requests
4. **Batch Processing**: Supports batch inference requests
5. **Periodic Refresh**: Supports dynamic subtask updates

## Configuration Parameters

- `config_name`: Model configuration name, default "right_pi05_20"
- `gpu_id`: GPU device ID, default 1
- `max_decoding_steps`: Maximum decoding steps for subtask generation, default 25
- `temperature`: Sampling temperature, default 0.1
- `subtask_refresh_interval`: Subtask refresh interval (seconds), None means no refresh

## Error Handling

- Automatically uses random images when image loading fails
- Auto-reconnects on network connection exceptions
- Returns detailed error information on inference errors
- Supports timeout and retry mechanisms

## Logging

All components support detailed logging, including:
- Connection status
- Inference timing
- Error information
- Performance statistics

# LeRobot Data Format Async Inference Observer

This tool allows you to directly use the inference engine to observe async inference outputs on LeRobot format data, without starting a WebSocket server.

## Main Features

- 🎯 **Direct Inference Engine Usage**: No WebSocket server required
- 📊 **LeRobot Data Format Support**: Automatically handles LeRobot's HDF5 and JSONL formats
- 🔄 **Continuous Observation**: Supports multi-frame continuous inference observation
- 📝 **Auto Save**: Inference results automatically saved as JSON files
- 🔔 **Callback Mechanism**: Supports custom observation callback functions
- ⚡ **Async Processing**: Fully asynchronous inference and observation process

## File Description

- `lerobot_inference_observer.py` - Main observer class
- `example_lerobot_usage.py` - Usage examples
- `test_lerobot_observer.py` - Test script

## Quick Start

### 1. Basic Usage

```python
import asyncio
from lerobot_inference_observer import LeRobotInferenceObserver

async def main():
    # Create observer
    observer = LeRobotInferenceObserver(
        config_name="right_pi05_20",
        gpu_id=1,
        output_dir="./inference_outputs"
    )
    
    # Add observation callback
    async def on_observation(data):
        print(f"Frame {data['frame_idx']}: {data['result']['subtask']}")
    
    observer.add_observation_callback(on_observation)
    
    # Prepare LeRobot data
    episode_data = {
        "base": your_base_images,  # (T, H, W, C) or (H, W, C)
        "left_wrist": your_left_images,
        "right_wrist": your_right_images,
        "state": your_state_data,  # (T, state_dim) or (state_dim,)
        "high_level_prompt": "Your high level task",
        "low_level_prompt": "Your low level task"
    }
    
    # Single inference observation
    result = await observer.observe_single_inference(
        episode_data=episode_data,
        frame_idx=0,
        high_level_prompt="Pick up the red block",
        low_level_prompt="Move to the block and grasp it"
    )
    
    # Continuous inference observation
    results = await observer.observe_continuous_inference(
        episode_data=episode_data,
        start_frame=0,
        max_frames=10,
        frame_interval=1.0,
        subtask_refresh_interval=2.0
    )

asyncio.run(main())
```

### 2. Load Real LeRobot Data

```python
# Load from HDF5 file
episode_data = observer.load_lerobot_episode("/path/to/episode.hdf5")

# Load from JSONL file
episode_data = observer.load_lerobot_episode("/path/to/episode.jsonl")
```

### 3. Run Tests

```bash
# Run basic tests
python test_lerobot_observer.py

# Run usage examples
python example_lerobot_usage.py
```

## Data Format Support

### Input Data Format

The observer supports the following LeRobot data formats:

- **Image Data**:
  - `base`: Base view images
  - `left_wrist`: Left wrist view images  
  - `right_wrist`: Right wrist view images
  - Supported shapes: (T, H, W, C) or (H, W, C)

- **State Data**:
  - `state`: Robot state vector
  - Supported shapes: (T, state_dim) or (state_dim,)

- **Task Description**:
  - `high_level_prompt`: High-level task description
  - `low_level_prompt`: Low-level task description

### Output Data Format

Each inference observation generates a JSON file containing the following information:

```json
{
  "timestamp": 1234567890.123,
  "frame_idx": 0,
  "inference_time": 0.456,
  "result": {
    "actions": [[...]],  // Action sequence
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

## Advanced Features

### 1. Custom Observation Callback

```python
async def custom_callback(data):
    # Process inference results
    subtask = data['result']['subtask']
    actions = data['result']['actions']
    
    # Send to other systems
    await send_to_robot(actions)
    await log_to_database(subtask)

observer.add_observation_callback(custom_callback)
```

### 2. Periodic Subtask Refresh

```python
# Enable periodic subtask refresh
results = await observer.observe_continuous_inference(
    episode_data=episode_data,
    subtask_refresh_interval=2.0  # Refresh subtask every 2 seconds
)
```

### 3. Batch Processing Multiple Episodes

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

## Notes

1. **Memory Usage**: Long-term continuous observation may consume significant memory, periodic cleanup is recommended
2. **GPU Resources**: Ensure sufficient GPU memory is available for inference
3. **Data Format**: Ensure LeRobot data format is correct, missing data will be filled with random data
4. **Async Processing**: All operations are asynchronous, use the `await` keyword

## Troubleshooting

### Common Issues

1. **Model Initialization Failure**:
   - Check GPU availability
   - Confirm model configuration file exists
   - Check if dependencies are correctly installed

2. **Data Loading Failure**:
   - Check if file path is correct
   - Confirm data format meets LeRobot standards
   - Check detailed error information in logs

3. **Slow Inference Speed**:
   - Check GPU usage
   - Consider reducing `max_decoding_steps` parameter
   - Adjust `frame_interval` parameter

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
observer = LeRobotInferenceObserver(...)
```

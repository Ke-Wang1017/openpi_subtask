#!/usr/bin/env python3
"""
LeRobot 数据格式推理观察器使用示例
"""

import asyncio
import logging
import numpy as np
from lerobot_inference_observer import LeRobotInferenceObserver

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_usage():
    """使用示例"""
    
    # 1. 创建观察器
    observer = LeRobotInferenceObserver(
        config_name="right_pi05_20",
        gpu_id=1,
        output_dir="./lerobot_outputs"
    )
    
    # 2. 添加自定义观察回调
    async def my_observation_callback(data):
        """自定义观察回调函数"""
        print(f"🔍 观察回调 - 帧 {data['frame_idx']}:")
        print(f"   推理耗时: {data['inference_time']:.3f}s")
        if data['result'].get('subtask'):
            print(f"   子任务: {data['result']['subtask']}")
        if data['result'].get('actions') is not None:
            print(f"   动作形状: {data['result']['actions'].shape}")
        print()
    
    observer.add_observation_callback(my_observation_callback)
    
    # 3. 创建或加载 LeRobot 格式数据
    # 这里使用模拟数据，实际使用时可以加载真实的 LeRobot 数据
    lerobot_episode = {
        "base": np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8),
        "left_wrist": np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8),
        "right_wrist": np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8),
        "state": np.random.randn(10, 32).astype(np.float32),
        "actions": np.random.randn(10, 50, 32).astype(np.float32),
        "high_level_prompt": "Pick up the red block and place it in the box",
        "low_level_prompt": "Move to the red block, grasp it, lift it up, move to the box, place it down"
    }
    
    try:
        # 4. 单次推理观察
        print("=" * 50)
        print("单次推理观察")
        print("=" * 50)
        
        single_result = await observer.observe_single_inference(
            episode_data=lerobot_episode,
            frame_idx=0,
            high_level_prompt="Pick up the red block",
            low_level_prompt="Move to the block and grasp it"
        )
        
        # 5. 持续推理观察
        print("\n" + "=" * 50)
        print("持续推理观察")
        print("=" * 50)
        
        continuous_results = await observer.observe_continuous_inference(
            episode_data=lerobot_episode,
            start_frame=0,
            max_frames=3,  # 观察3帧
            frame_interval=1.0,  # 每帧间隔1秒
            high_level_prompt="Organize the workspace",
            low_level_prompt="Sort items by category"
        )
        
        print(f"\n📊 观察完成:")
        print(f"   单次推理: 成功")
        print(f"   持续推理: {len(continuous_results)} 帧")
        print(f"   输出目录: {observer.output_dir}")
        
    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


async def example_with_real_lerobot_data():
    """使用真实 LeRobot 数据的示例"""
    
    observer = LeRobotInferenceObserver(
        config_name="right_pi05_20",
        gpu_id=1,
        output_dir="./real_lerobot_outputs"
    )
    
    # 加载真实的 LeRobot 数据
    # 假设你有 LeRobot 数据文件
    episode_path = "/path/to/your/lerobot/episode.hdf5"  # 替换为实际路径
    
    try:
        # 加载 episode 数据
        episode_data = observer.load_lerobot_episode(episode_path)
        
        # 持续观察推理
        results = await observer.observe_continuous_inference(
            episode_data=episode_data,
            start_frame=0,
            max_frames=10,
            frame_interval=0.5,
            subtask_refresh_interval=2.0  # 每2秒刷新子任务
        )
        
        print(f"真实数据观察完成: {len(results)} 帧")
        
    except FileNotFoundError:
        print(f"LeRobot 数据文件不存在: {episode_path}")
        print("请提供正确的 LeRobot 数据文件路径")
    except Exception as e:
        logger.error(f"真实数据示例失败: {e}")


if __name__ == "__main__":
    print("LeRobot 推理观察器使用示例")
    print("=" * 60)
    
    # 运行模拟数据示例
    asyncio.run(example_usage())
    
    # 如果需要测试真实数据，取消注释下面的行
    # asyncio.run(example_with_real_lerobot_data())

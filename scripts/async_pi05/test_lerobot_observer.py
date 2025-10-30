#!/usr/bin/env python3
"""
测试 LeRobot 推理观察器
"""

import asyncio
import logging
import sys
import os

# 添加当前目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lerobot_inference_observer import LeRobotInferenceObserver

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试 LeRobot 推理观察器基本功能")
    print("=" * 60)
    
    # 创建观察器
    observer = LeRobotInferenceObserver(
        config_name="right_pi05_20",
        gpu_id=1,
        output_dir="./test_outputs"
    )
    
    # 添加简单的观察回调
    async def simple_callback(data):
        print(f"✅ 收到观察数据 - 帧 {data['frame_idx']}, 耗时 {data['inference_time']:.3f}s")
        if data['result'].get('subtask'):
            print(f"   子任务: {data['result']['subtask']}")
    
    observer.add_observation_callback(simple_callback)
    
    # 创建测试数据
    import numpy as np
    test_episode = {
        "base": np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8),
        "left_wrist": np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8),
        "right_wrist": np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8),
        "state": np.random.randn(5, 32).astype(np.float32),
        "high_level_prompt": "Pick up the red block",
        "low_level_prompt": "Move to the block and grasp it"
    }
    
    try:
        print("🚀 开始测试...")
        
        # 测试单次推理
        print("\n1. 测试单次推理")
        result = await observer.observe_single_inference(
            episode_data=test_episode,
            frame_idx=0,
            high_level_prompt="Test task",
            low_level_prompt="Test subtask"
        )
        
        print(f"✅ 单次推理完成")
        print(f"   输出文件: {observer.output_dir}")
        
        # 测试持续推理（只测试2帧）
        print("\n2. 测试持续推理")
        continuous_results = await observer.observe_continuous_inference(
            episode_data=test_episode,
            start_frame=0,
            max_frames=2,
            frame_interval=0.5,
            high_level_prompt="Continuous task",
            low_level_prompt="Continuous subtask"
        )
        
        print(f"✅ 持续推理完成，处理了 {len(continuous_results)} 帧")
        
        print("\n🎉 所有测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("LeRobot 推理观察器测试")
    print("=" * 60)
    
    try:
        asyncio.run(test_basic_functionality())
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试运行失败: {e}")

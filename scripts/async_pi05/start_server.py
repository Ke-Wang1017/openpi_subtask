#!/usr/bin/env python3
"""
启动 Pi0.5 异步推理服务器
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# 添加项目路径到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OPENPI_DATA_HOME"] = "/root/.cache/openpi"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def start_server():
    """启动服务器"""
    try:
        logger.info("🚀 启动 Pi0.5 异步推理服务器...")
        
        # 导入服务器
        from async_pi05_websocket_server import AsyncPi05WebSocketServer
        
        # 创建服务器实例
        server = AsyncPi05WebSocketServer(
            host="0.0.0.0",
            port=8765,
            config_name="right_pi05_20",
            gpu_id=1
        )
        
        logger.info("✅ 服务器配置完成")
        logger.info("🌐 服务器地址: ws://0.0.0.0:8765")
        logger.info("💡 健康检查: http://localhost:8765/healthz")
        
        # 启动服务器
        await server.start_server()
        
    except ImportError as e:
        logger.error(f"❌ 导入错误: {e}")
        logger.info("💡 请确保所有依赖已安装")
        logger.info("💡 检查 Python 路径设置")
    except Exception as e:
        logger.error(f"❌ 服务器启动失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    logger.info("🎯 Pi0.5 异步推理服务器启动脚本")
    logger.info("=" * 60)
    
    try:
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("\n⏹️ 用户中断服务器")
    except Exception as e:
        logger.error(f"\n❌ 服务器运行失败: {e}")


if __name__ == "__main__":
    main()


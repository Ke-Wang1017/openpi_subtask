import asyncio
import json
import logging
import time
from typing import Any

import cv2
import numpy as np
import websockets

logger = logging.getLogger(__name__)


class AsyncPi05Client:
    """异步 Pi0.5 推理客户端"""

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.websocket = None
        self.server_metadata = None

    async def connect(self):
        """连接到服务器"""
        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"连接到服务器: {uri}")

        self.websocket = await websockets.connect(uri)

        # 接收服务器元数据
        metadata_message = await self.websocket.recv()
        self.server_metadata = json.loads(metadata_message)
        logger.info(f"服务器元数据: {self.server_metadata}")

    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None

    def load_image(self, img_path: str) -> np.ndarray:
        """加载图像"""
        if not img_path:
            # 创建随机图像作为 fallback
            return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"无法加载图像: {img_path},使用随机图像")
            return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        return img

    async def infer(
        self,
        images: dict[str, str],  # 图像路径字典
        high_level_prompt: str,
        low_level_prompt: str = "ABCDEFG",
        state: np.ndarray | None = None,
        *,
        generate_subtask: bool = True,
        max_decoding_steps: int = 25,
        temperature: float = 0.1,
        noise: np.ndarray | None = None,
        subtask_refresh_interval: float | None = None,
    ) -> dict[str, Any]:
        """
        发送推理请求

        Args:
            images: 图像路径字典,键为图像类型,值为图像文件路径
            high_level_prompt: 高级别任务描述
            low_level_prompt: 低级别任务描述
            state: 机器人状态
            generate_subtask: 是否生成子任务
            max_decoding_steps: 最大解码步数
            temperature: 采样温度
            noise: 动作噪声
            subtask_refresh_interval: 子任务刷新间隔(秒),None表示不刷新

        Returns:
            推理结果字典
        """
        if not self.websocket:
            raise RuntimeError("未连接到服务器")

        # 加载图像
        images_data = {}
        for key, img_path in images.items():
            img = self.load_image(img_path)
            images_data[key] = img.tolist()  # 转换为列表以便 JSON 序列化

        # 构建请求
        request = {
            "images": images_data,
            "high_level_prompt": high_level_prompt,
            "low_level_prompt": low_level_prompt,
            "generate_subtask": generate_subtask,
            "max_decoding_steps": max_decoding_steps,
            "temperature": temperature,
        }

        if state is not None:
            request["state"] = state.tolist()

        if noise is not None:
            request["noise"] = noise.tolist()

        if subtask_refresh_interval is not None:
            request["subtask_refresh_interval"] = subtask_refresh_interval

        # 发送请求
        start_time = time.time()
        await self.websocket.send(json.dumps(request))

        # 接收响应
        response_message = await self.websocket.recv()
        response = json.loads(response_message)

        total_time = time.time() - start_time

        if response.get("status") == "error":
            raise RuntimeError(f"服务器错误: {response.get('error')}")

        # 添加客户端时序信息
        response["client_timing"] = {"total_ms": total_time * 1000}

        return response

    async def batch_infer(self, requests: list, delay_between_requests: float = 0.1) -> list:
        """批量推理请求"""
        results = []

        for i, request in enumerate(requests):
            logger.info(f"处理请求 {i + 1}/{len(requests)}")

            try:
                result = await self.infer(**request)
                results.append(result)

                if i < len(requests) - 1:  # 不是最后一个请求
                    await asyncio.sleep(delay_between_requests)

            except Exception as e:
                logger.error(f"请求 {i + 1} 失败: {e}")
                results.append({"error": str(e)})

        return results

    async def listen_for_refresh_messages(self, callback=None):
        """监听定期刷新消息"""
        if not self.websocket:
            raise RuntimeError("未连接到服务器")

        try:
            while True:
                message = await self.websocket.recv()
                data = json.loads(message)

                if data.get("type") == "subtask_refresh":
                    logger.info(f"收到子任务刷新: {data['subtask']} (第{data['refresh_count']}次)")

                    if callback:
                        await callback(data)
                else:
                    # 处理其他类型的消息
                    logger.info(f"收到消息: {data}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("连接已关闭,停止监听刷新消息")
        except Exception as e:
            logger.error(f"监听刷新消息时出错: {e}")


async def test_single_inference():
    """测试单个推理请求"""
    client = AsyncPi05Client(host="localhost", port=8765)

    try:
        await client.connect()

        # 准备测试数据
        images = {"base_0_rgb": "faceImg.png", "left_wrist_0_rgb": "leftImg.png", "right_wrist_0_rgb": "rightImg.png"}

        high_level_prompt = "Pick up the flashcard on the table"

        # 执行推理
        logger.info("开始推理...")
        result = await client.infer(
            images=images,
            high_level_prompt=high_level_prompt,
            generate_subtask=True,
            max_decoding_steps=25,
            temperature=0.1,
            subtask_refresh_interval=2.0,  # 每2秒刷新一次子任务
        )

        # 打印结果
        print("推理结果:")
        print(f"状态: {result.get('status')}")
        print(f"动作形状: {np.array(result['actions']).shape}")
        print(f"生成的子任务: {result.get('subtask')}")
        print(f"时序信息: {result.get('timing')}")
        print(f"客户端时序: {result.get('client_timing')}")

    except Exception as e:
        logger.error(f"测试失败: {e}")
    finally:
        await client.disconnect()


async def test_batch_inference():
    """测试批量推理请求"""
    client = AsyncPi05Client(host="localhost", port=8765)

    try:
        await client.connect()

        # 准备批量请求
        requests = [
            {
                "images": {
                    "base_0_rgb": "faceImg.png",
                    "left_wrist_0_rgb": "leftImg.png",
                    "right_wrist_0_rgb": "rightImg.png",
                },
                "high_level_prompt": "Pick up the flashcard on the table",
                "generate_subtask": True,
            },
            {
                "images": {
                    "base_0_rgb": "faceImg.png",
                    "left_wrist_0_rgb": "leftImg.png",
                    "right_wrist_0_rgb": "rightImg.png",
                },
                "high_level_prompt": "Move the pen to the box",
                "generate_subtask": True,
            },
        ]

        # 执行批量推理
        logger.info("开始批量推理...")
        results = await client.batch_infer(requests, delay_between_requests=0.5)

        # 打印结果
        print(f"批量推理完成,处理了 {len(results)} 个请求")
        for i, result in enumerate(results):
            if "error" in result:
                print(f"请求 {i + 1} 失败: {result['error']}")
            else:
                print(f"请求 {i + 1} 成功:")
                print(f"  子任务: {result.get('subtask')}")
                print(f"  动作形状: {np.array(result['actions']).shape}")

    except Exception as e:
        logger.error(f"批量测试失败: {e}")
    finally:
        await client.disconnect()


async def test_periodic_refresh():
    """测试定期刷新功能"""
    client = AsyncPi05Client(host="localhost", port=8765)

    try:
        await client.connect()

        # 准备测试数据
        images = {"base_0_rgb": "faceImg.png", "left_wrist_0_rgb": "leftImg.png", "right_wrist_0_rgb": "rightImg.png"}

        high_level_prompt = "Pick up the flashcard on the table"

        # 定义刷新回调函数
        async def on_refresh(data):
            print(f"\n🔄 子任务刷新 (第{data['refresh_count']}次):")
            print(f"   新子任务: {data['subtask']}")
            print(f"   时间戳: {data['timestamp']}")
            print("-" * 50)

        # 启动监听任务
        listen_task = asyncio.create_task(client.listen_for_refresh_messages(callback=on_refresh))

        # 执行推理并启用定期刷新
        logger.info("开始推理并启用定期刷新...")
        result = await client.infer(
            images=images,
            high_level_prompt=high_level_prompt,
            generate_subtask=True,
            subtask_refresh_interval=2.0,  # 每2秒刷新一次
        )

        print("初始推理结果:")
        print(f"状态: {result.get('status')}")
        print(f"动作形状: {np.array(result['actions']).shape}")
        print(f"初始子任务: {result.get('subtask')}")
        print(f"定期刷新已启用: {result.get('subtask_refresh_enabled')}")
        print(f"刷新间隔: {result.get('subtask_refresh_interval')}秒")
        print("\n等待定期刷新消息... (按 Ctrl+C 停止)")

        # 等待一段时间来观察刷新
        try:
            await asyncio.wait_for(listen_task, timeout=10.0)  # 等待10秒
        except TimeoutError:
            print("测试完成,已观察10秒的刷新过程")

    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        logger.error(f"定期刷新测试失败: {e}")
    finally:
        listen_task.cancel()
        await client.disconnect()


async def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    print("异步 Pi0.5 推理客户端测试")
    print("=" * 50)

    # 测试单个推理
    print("\n1. 测试单个推理请求")
    await test_single_inference()

    # 等待一下
    await asyncio.sleep(2)

    # 测试定期刷新
    print("\n2. 测试定期刷新功能")
    await test_periodic_refresh()

    # 等待一下
    await asyncio.sleep(2)

    # 测试批量推理
    print("\n3. 测试批量推理请求")
    await test_batch_inference()


if __name__ == "__main__":
    asyncio.run(main())

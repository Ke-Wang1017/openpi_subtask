import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional
import websockets
from websockets.server import WebSocketServerProtocol
import numpy as np
import jax
from async_pi05_inference import AsyncPi05Inference

logger = logging.getLogger(__name__)


class AsyncPi05WebSocketServer:
    """基于 WebSocket 的异步 Pi0.5 推理服务器"""
    
    def __init__(self, 
                 host: str = "0.0.0.0", 
                 port: int = 8765,
                 config_name: str = "right_pi05_20",
                 gpu_id: int = 1):
        self.host = host
        self.port = port
        self.inference_engine = AsyncPi05Inference(config_name=config_name, gpu_id=gpu_id)
        self.clients = set()
        self.active_refresh_tasks = {}  # 存储活跃的刷新任务
        
    async def register_client(self, websocket: WebSocketServerProtocol):
        """注册新客户端"""
        self.clients.add(websocket)
        logger.info(f"客户端连接: {websocket.remote_address}")
        
    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """注销客户端"""
        self.clients.discard(websocket)
        
        # 取消该客户端的刷新任务
        if websocket in self.active_refresh_tasks:
            task = self.active_refresh_tasks[websocket]
            task.cancel()
            del self.active_refresh_tasks[websocket]
            logger.info(f"已取消客户端 {websocket.remote_address} 的刷新任务")
        
        logger.info(f"客户端断开: {websocket.remote_address}")
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """处理客户端连接"""
        await self.register_client(websocket)
        
        try:
            # 发送服务器元数据
            metadata = {
                "server_type": "AsyncPi05Inference",
                "version": "1.0.0",
                "capabilities": ["subtask_generation", "action_prediction"],
                "max_decoding_steps": 25,
                "supported_image_types": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
            }
            await websocket.send(json.dumps(metadata))
            
            async for message in websocket:
                try:
                    # 解析请求
                    request = json.loads(message)
                    response = await self.process_request(request)
                    
                    # 发送响应
                    await websocket.send(json.dumps(response))
                    
                except json.JSONDecodeError:
                    error_response = {
                        "error": "Invalid JSON format",
                        "status": "error"
                    }
                    await websocket.send(json.dumps(error_response))
                    
                except Exception as e:
                    logger.error(f"处理请求时出错: {e}")
                    error_response = {
                        "error": str(e),
                        "status": "error"
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"客户端连接关闭: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"处理客户端时出错: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理推理请求"""
        try:
            # 验证请求格式
            if "images" not in request or "high_level_prompt" not in request:
                return {
                    "error": "Missing required fields: images, high_level_prompt",
                    "status": "error"
                }
            
            # 提取请求参数
            images_data = request["images"]
            high_level_prompt = request["high_level_prompt"]
            low_level_prompt = request.get("low_level_prompt", "ABCDEFG")
            state = request.get("state")
            generate_subtask = request.get("generate_subtask", True)
            max_decoding_steps = request.get("max_decoding_steps", 25)
            temperature = request.get("temperature", 0.1)
            noise = request.get("noise")
            subtask_refresh_interval = request.get("subtask_refresh_interval")  # 新增：子任务刷新间隔
            
            # 转换图像数据
            images = {}
            for key, img_data in images_data.items():
                if isinstance(img_data, list):
                    # 假设是 [height, width, channels] 格式
                    img_array = np.array(img_data, dtype=np.uint8)
                else:
                    # 假设已经是 numpy 数组
                    img_array = np.array(img_data, dtype=np.uint8)
                images[key] = img_array
            
            # 转换状态数据
            state_array = None
            if state is not None:
                state_array = np.array(state, dtype=np.float32)
            
            # 转换噪声数据
            noise_array = None
            if noise is not None:
                noise_array = np.array(noise, dtype=np.float32)
            
            # 执行推理
            start_time = time.time()
            results = await self.inference_engine.infer(
                images=images,
                high_level_prompt=high_level_prompt,
                low_level_prompt=low_level_prompt,
                state=state_array,
                generate_subtask=generate_subtask,
                max_decoding_steps=max_decoding_steps,
                temperature=temperature,
                noise=noise_array,
                subtask_refresh_interval=subtask_refresh_interval
            )
            total_time = time.time() - start_time
            
            # 构建响应
            response = {
                "status": "success",
                "actions": results["actions"].tolist(),
                "subtask": results["subtask"],
                "subtask_tokens": results["subtask_tokens"].tolist() if results["subtask_tokens"] is not None else None,
                "state": results["state"].tolist() if results["state"] is not None else None,
                "timing": results["timing"],
                "server_timing": {
                    "total_ms": total_time * 1000
                }
            }
            
            # 如果启用了定期刷新，启动后台任务
            if subtask_refresh_interval is not None and subtask_refresh_interval > 0:
                response["subtask_refresh_interval"] = subtask_refresh_interval
                response["subtask_refresh_enabled"] = True
                
                # 启动定期刷新任务
                refresh_task = asyncio.create_task(
                    self._handle_periodic_refresh(
                        websocket, images, high_level_prompt, low_level_prompt, 
                        state_array, subtask_refresh_interval, max_decoding_steps, temperature
                    )
                )
                self.active_refresh_tasks[websocket] = refresh_task
                logger.info(f"已启动客户端 {websocket.remote_address} 的定期刷新任务，间隔: {subtask_refresh_interval}s")
            else:
                response["subtask_refresh_enabled"] = False
            
            return response
            
        except Exception as e:
            logger.error(f"处理推理请求时出错: {e}")
            return {
                "error": str(e),
                "status": "error"
            }
    
    async def _handle_periodic_refresh(self, 
                                     websocket: WebSocketServerProtocol,
                                     images: Dict[str, np.ndarray],
                                     high_level_prompt: str,
                                     low_level_prompt: str,
                                     state: Optional[np.ndarray],
                                     refresh_interval: float,
                                     max_decoding_steps: int,
                                     temperature: float):
        """处理定期刷新子任务"""
        refresh_count = 0
        
        while True:
            try:
                await asyncio.sleep(refresh_interval)
                refresh_count += 1
                
                logger.info(f"开始第 {refresh_count} 次子任务刷新 (客户端: {websocket.remote_address})")
                
                # 准备新的观察数据
                observation = self.inference_engine.prepare_observation(
                    images, high_level_prompt, low_level_prompt, state
                )
                
                # 生成新的子任务
                rng = jax.random.key(int(time.time() * 1000) % 2**32)
                subtask_tokens, subtask_text = await self.inference_engine.generate_subtask(
                    observation, rng, max_decoding_steps, temperature
                )
                
                # 发送刷新消息给客户端
                refresh_message = {
                    "type": "subtask_refresh",
                    "subtask": subtask_text,
                    "subtask_tokens": subtask_tokens.tolist(),
                    "refresh_count": refresh_count,
                    "timestamp": time.time()
                }
                
                await websocket.send(json.dumps(refresh_message))
                logger.info(f"第 {refresh_count} 次刷新完成，新子任务: {subtask_text}")
                
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"客户端连接已关闭，停止刷新任务: {websocket.remote_address}")
                break
            except asyncio.CancelledError:
                logger.info(f"刷新任务被取消: {websocket.remote_address}")
                break
            except Exception as e:
                logger.error(f"定期刷新出错: {e}")
                await asyncio.sleep(1)  # 出错后等待1秒再重试
    
    async def start_server(self, skip_init: bool = False):
        """启动 WebSocket 服务器
        
        skip_init: 为 True 时跳过模型初始化，仅用于连通性排查。
        """
        logger.info(f"启动异步 Pi0.5 WebSocket 服务器: {self.host}:{self.port}")
        
        if skip_init:
            logger.warning("跳过模型初始化 (--skip-init)。仅用于连通性测试，推理不可用。")
        else:
            # 初始化推理引擎
            logger.info("初始化推理引擎（可能需要较长时间，请耐心等待）...")
            await self.inference_engine.initialize()
            logger.info("推理引擎初始化完成")
        
        # 启动 WebSocket 服务器
        logger.info("正在启动 WebSocket 监听...")
        server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
        
        logger.info(f"服务器已启动，监听 {self.host}:{self.port}")
        
        try:
            await server.wait_closed()
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在关闭服务器...")
            server.close()
            await server.wait_closed()
            logger.info("服务器已关闭")


async def main():
    """启动服务器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Async Pi0.5 WebSocket Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    parser.add_argument("--config", type=str, default="right_pi05_20", help="模型配置名称")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID，CPU 可用 -1")
    parser.add_argument("--skip-init", action="store_true", help="跳过模型初始化，仅用于连通性测试")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别: DEBUG/INFO/WARN/ERROR")
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建并启动服务器
    server = AsyncPi05WebSocketServer(
        host=args.host,
        port=args.port,
        config_name=args.config,
        gpu_id=args.gpu_id
    )
    
    await server.start_server(skip_init=args.skip_init)


if __name__ == "__main__":
    asyncio.run(main())

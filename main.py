import asyncio
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List

# 从models.py导入模拟模型
from models import mock_pose_estimation_model, mock_qwen_vl_model

app = FastAPI()

# --------------------------------------------------------------------------
# 核心状态管理器
# --------------------------------------------------------------------------
class StateManager:
    """为每个WebSocket连接管理状态，确保并发安全。"""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.frame_count = 0              # 总接收帧数计数器
        self.sampled_frames_buffer = []   # 存放采样后的帧，用于动作分类
        self.current_action_frames = []   # 存放当前识别到的动作的所有原始帧
        self.current_action = None        # 当前识别到的动作名称
        self.last_sent_action = None      # 上一个已发送评估的动作，防止重复评估

    async def process_frame(self, frame_bytes: bytes):
        """处理接收到的单帧图像"""
        self.frame_count += 1
        
        # 1. 解码图像
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            print("WARN: 无法解码收到的帧，已跳过。")
            return

        # 2. 如果当前有正在进行的动作，则缓存该帧
        if self.current_action and self.current_action != 'Transitioning':
            self.current_action_frames.append(frame)

        # 3. 按规则采样（每6帧抽1帧）
        if self.frame_count % 6 == 0:
            print(f"DEBUG: 采样第 {self.frame_count} 帧。")
            self.sampled_frames_buffer.append(frame)

            # 4. 如果采样缓冲区满8帧，则进行动作识别
            if len(self.sampled_frames_buffer) == 8:
                await self._run_pose_estimation()
    
    async def _run_pose_estimation(self):
        """运行动作识别并处理状态变化"""
        # 调用模型进行识别
        recognized_action = await mock_pose_estimation_model(self.sampled_frames_buffer)

        # 状态机：处理动作变化
        # 如果识别到了一个新动作，并且这个新动作不是上一个刚评估完的动作
        if recognized_action != self.current_action and recognized_action != self.last_sent_action:
            # 检查上一个动作是否需要评估 (必须有缓存帧且不是'Transitioning')
            if self.current_action and self.current_action != 'Transitioning' and self.current_action_frames:
                await self._run_evaluation(self.current_action, self.current_action_frames)

            # 更新当前动作为新识别的动作，并清空帧缓存
            self.current_action = recognized_action
            self.current_action_frames = []
            
            # 如果新动作不是转换动作，也立即通知前端
            if self.current_action != 'Transitioning':
                 await self.send_json_to_client("info", f"检测到新动作: {self.current_action}")

        # 滑动窗口：移除缓冲区中最老的3帧，为新帧腾出空间
        self.sampled_frames_buffer = self.sampled_frames_buffer[3:]

    async def _run_evaluation(self, action_to_evaluate: str, frames_to_evaluate: List):
        """调用大模型进行评估，并将结果发送给客户端"""
        print(f"INFO: 准备评估已结束的动作 '{action_to_evaluate}'...")
        await self.send_json_to_client("info", f"动作 '{action_to_evaluate}' 已结束，正在生成评估报告...")
        
        # 调用大模型
        evaluation_result = await mock_qwen_vl_model(frames_to_evaluate, action_to_evaluate)
        
        # 发送评估结果
        await self.send_json_to_client("evaluation", evaluation_result)
        self.last_sent_action = action_to_evaluate

    async def handle_disconnect(self):
        """处理连接断开，对未完成的动作进行最后一次评估"""
        print("INFO: 客户端连接断开。")
        if self.current_action and self.current_action != 'Transitioning' and self.current_action_frames:
            print("INFO: 检测到视频流截断，正在对最后一个动作进行评估...")
            await self._run_evaluation(self.current_action, self.current_action_frames)
        print("INFO: 清理完成。")

    async def send_json_to_client(self, msg_type: str, data: any):
        """向客户端发送结构化的JSON消息"""
        await self.websocket.send_json({"type": msg_type, "data": data})

# --------------------------------------------------------------------------
# WebSocket 端点
# --------------------------------------------------------------------------
@app.websocket("/ws/assessment")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state = StateManager(websocket)
    
    await state.send_json_to_client("info", "连接成功！请开始发送视频帧...")
    
    try:
        while True:
            # 接收前端发送的单帧图像数据（bytes）
            frame_bytes = await websocket.receive_bytes()
            await state.process_frame(frame_bytes)
    except WebSocketDisconnect:
        await state.handle_disconnect()
    except Exception as e:
        print(f"ERROR: 发生意外错误: {e}")
        await state.handle_disconnect()

@app.get("/")
def read_root():
    return {"message": "康复评估后端服务正在运行。请使用 /ws/assessment 端点进行连接。"}


import os
from openai import OpenAI
from typing import List, Dict, Optional
from datetime import datetime
import uuid
# class ChatService:
#     def __init__(self):
#         pass
from ..database.models import db, MessageChat, User, RecoveryPlan, Exercise, CalendarSchedule, RecoveryRecord, UserRecoveryPlan
from ..database import database as db_operations

class ChatService:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.default_model = os.getenv("OPENAI_DEFAULT_MODEL", "qwen-plus")
        self.default_system_message = """
        你是一个专业的术后康复训练助手，专门帮助乳腺癌术后患者进行康复训练。
        你能够提供专业的康复建议、解释训练动作、记录训练进度，并回答患者关于康复过程的问题。
        请保持专业、友好和鼓励的态度，使用简单易懂的语言解释医学概念。
        总长度不超过100个字。
        """

    def _get_user_context(self, user_id: int) -> str:
        """获取用户上下文信息，用于增强系统提示"""
        user = db_operations.get_record_by_id(User, user_id)
        if not user:
            return ""

        # 获取用户康复计划
        plans = db.session.query(RecoveryPlan).join(
            UserRecoveryPlan, UserRecoveryPlan.plan_id == RecoveryPlan.plan_id
        ).filter(UserRecoveryPlan.user_id == user_id).all()

        # 获取用户训练进度
        total_exercises = db.session.query(CalendarSchedule).filter(
            CalendarSchedule.user_id == user_id
        ).count()
        completed_exercises = db.session.query(CalendarSchedule).filter(
            CalendarSchedule.user_id == user_id,
            CalendarSchedule.is_completed == True
        ).count()

        # 获取最近一次康复记录
        latest_record = db.session.query(RecoveryRecord).filter(
            RecoveryRecord.user_id == user_id
        ).order_by(RecoveryRecord.record_date.desc()).first()

        context = f"""
        患者信息:
        - 姓名: {user.name or '未知'}
        - 注册日期: {user.registration_date.strftime('%Y-%m-%d') if user.registration_date else '未知'}
        
        康复计划:
        {', '.join([plan.plan_name for plan in plans]) if plans else '暂无康复计划'}
        
        训练进度:
        - 总训练项目: {total_exercises}
        - 已完成项目: {completed_exercises}
        {f"- 最近训练日期: {latest_record.record_date.strftime('%Y-%m-%d')}" if latest_record else ''}
        """
        
        return context

    def _prepare_conversation_messages(self, conversation_id: str, user_message: str) -> List[Dict]:
        """准备对话消息，包括历史消息"""
        # 获取历史消息
        history_messages = db.session.query(MessageChat).filter(
            MessageChat.conversation_id == conversation_id
        ).order_by(MessageChat.timestamp.asc()).all()

        # 转换为OpenAI格式的消息
        messages = [{"role": "system", "content": self.default_system_message}]
        
        for msg in history_messages:
            role = "user" if msg.sender_type == 'user' else "assistant"
            messages.append({"role": role, "content": msg.message_text})
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": user_message})
        
        return messages

    def _save_message_to_db(self, message_data: Dict) -> bool:
        """保存消息到数据库"""
        try:
            # 转换数据格式以匹配数据库操作
            db_data = {
                'conversation_id': message_data.get('conversation_id'),
                'is_follow_up': message_data.get('is_follow_up', False),
                'sender_id': message_data.get('sender_id'),
                'sender_type': message_data.get('sender_type'),
                'receiver_id': message_data.get('receiver_id'),
                'receiver_type': message_data.get('receiver_type'),
                'message_text': message_data.get('message_text'),
                'timestamp': message_data.get('timestamp', datetime.now())
            }
            
            # 使用数据库操作方法添加记录
            result = db_operations.add_record(MessageChat, db_data)
            return result is not None
        except Exception as e:
            print(f"Error saving message to database: {e}")
            return False

    def process_user_message(self, user_id: int, conversation_id: str, message_text: str) -> Dict:
        """处理用户消息并返回助手回复"""
        # 获取用户上下文
        user_context = self._get_user_context(user_id)
        
        # 准备对话消息
        messages = self._prepare_conversation_messages(conversation_id, message_text)
        
        # 如果有用户上下文，增强系统消息
        if user_context:
            messages[0]['content'] += f"\n\n以下是患者的相关信息:\n{user_context}"
        
        try:
            # 调用OpenAI API
            completion = self.client.chat.completions.create(
                model=self.default_model,
                messages=messages
            )
            
            response = completion.model_dump()
            assistant_response = response['choices'][0]['message']['content']
            current_time = datetime.now()
            
            # 保存用户消息到数据库
            user_msg_saved = self._save_message_to_db({
                'conversation_id': conversation_id,
                'sender_id': user_id,
                'sender_type': 'user',
                'receiver_id': 0,  # 0表示系统/助手
                'receiver_type': 'assistant',
                'message_text': message_text,
                'timestamp': current_time
            })
            
            # 保存助手回复到数据库
            assistant_msg_saved = self._save_message_to_db({
                'conversation_id': conversation_id,
                'sender_id': 0,  # 0表示系统/助手
                'sender_type': 'assistant',
                'receiver_id': user_id,
                'receiver_type': 'user',
                'message_text': assistant_response,
                'timestamp': current_time
            })
            
            if not user_msg_saved or not assistant_msg_saved:
                print("Warning: Failed to save one or both messages to database")
            
            return {
                'response': assistant_response,
                'conversation_id': conversation_id,
                'timestamp': current_time.isoformat()
            }
        except Exception as e:
            # 记录错误但不要暴露给用户
            print(f"Error processing chat message: {str(e)}")
            return {
                'response': "抱歉，处理您的请求时出现问题。请稍后再试。",
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }

    def start_new_conversation(self, user_id: int) -> str:
        """开始一个新的对话，返回对话ID"""
        return str(uuid.uuid4())
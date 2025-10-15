import os
import uuid
from openai import OpenAI
from typing import List, Dict, Optional, Any
from datetime import datetime

from .retriever import Retriever 
from ..database.models import db, User, RecoveryPlan, CalendarSchedule, RecoveryRecord, UserRecoveryPlan, ChatHistory
from ..database import database as db_operations

class Consult:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.qwen_model = os.getenv("QWEN_MODEL_NAME", "qwen-plus")
        self.retriever = Retriever()
        self.default_system_message = """
        你是一个专业的术后康复训练助手，专门帮助乳腺癌术后患者进行康复训练。
        你能够提供专业的康复建议、解释训练动作、记录训练进度，并回答患者关于康复过程的问题。
        请保持专业、友好和鼓励的态度，使用简单易懂的语言解释医学概念。
        你的回答应直接切入主题，无需重复问题，并尽量简洁。
        """

    def _get_user_context(self, user_id: int) -> str:
        user = db_operations.get_record_by_id(User, user_id)
        if not user:
            return ""

        plans = db.session.query(RecoveryPlan).join(
            UserRecoveryPlan, UserRecoveryPlan.plan_id == RecoveryPlan.plan_id
        ).filter(UserRecoveryPlan.user_id == user_id).all()

        total_exercises = db.session.query(CalendarSchedule).filter_by(user_id=user_id).count()
        completed_exercises = db.session.query(CalendarSchedule).filter_by(user_id=user_id, is_completed=True).count()

        context = (
            f"患者信息:\n"
            f"- 姓名: {user.name or '未知'}\n"
            f"- 注册日期: {user.registration_date.strftime('%Y-%m-%d') if user.registration_date else '未知'}\n\n"
            f"康复计划:\n"
            f"{', '.join([plan.plan_name for plan in plans]) if plans else '暂无康复计划'}\n\n"
            f"训练进度:\n"
            f"- 总训练项目: {total_exercises}\n"
            f"- 已完成项目: {completed_exercises}\n"
        )
        
        return context

    def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()
        if record and record.chat_history:
            return record.chat_history
        return []

    def _save_conversation_turn(self, user_id: int, conversation_id: str, user_message: str, assistant_message: str):
        record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()
        
        user_msg_obj = {"role": "user", "content": user_message}
        assistant_msg_obj = {"role": "assistant", "content": assistant_message}

        try:
            if record:
                current_history = record.chat_history or []
                current_history.extend([user_msg_obj, assistant_msg_obj])
                update_data = {'chat_history': current_history}
                db_operations.update_record(record, update_data)
            else:
                new_history = [user_msg_obj, assistant_msg_obj]
                data = {
                    'conversation_id': conversation_id,
                    'user_id': user_id,
                    'chat_history': new_history,
                    'is_follow_up': False
                }
                db_operations.add_record(ChatHistory, data)
        except Exception as e:
            print(f"Error saving conversation turn to database: {e}")
            # 根据需要决定是否要回滚事务
            # db.session.rollback()

    def _build_prompt_with_rag(self, query: str, context_docs: List[Dict]) -> str:
        if not context_docs:
            return query
            
        context_str = "\n\n".join([f"参考资料 [{i+1}]:\n{doc['document']['page_content']}" for i, doc in enumerate(context_docs)])
        prompt_template = f"请根据以下提供的参考资料，回答问题。\n\n[参考资料]\n{context_str}\n\n[问题]\n{query}"
        return prompt_template.strip()

    @staticmethod
    def _format_references(retrieved_docs: List[Dict]) -> str:
        if not retrieved_docs:
            return ""
        
        unique_sources = {
            doc.get('document', {}).get('metadata', {}).get('source_file') 
            for doc in retrieved_docs 
            if doc.get('document', {}).get('metadata', {}).get('source_file')
        }
        
        if not unique_sources:
            return ""
            
        references_header = "\n\n---\n**参考文献:**\n"
        references_list = [f"[{i+1}] {source}" for i, source in enumerate(sorted(list(unique_sources)))]
        return references_header + "\n".join(references_list)

    def chat(self, user_id: int, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        retrieved_docs = self.retriever.search(query)
        prompt_with_rag = self._build_prompt_with_rag(query, retrieved_docs)

        messages = [{"role": "system", "content": self.default_system_message}]
        
        user_context = self._get_user_context(user_id)
        if user_context:
            messages.append({"role": "system", "content": f"请结合以下患者历史数据进行回答：\n{user_context}"})
        
        history = self._get_conversation_history(conversation_id)
        messages.extend(history)
        
        messages.append({"role": "user", "content": prompt_with_rag})

        try:
            completion = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=messages
            )
            response = completion.model_dump()
            assistant_response = response['choices'][0]['message']['content']

            self._save_conversation_turn(user_id, conversation_id, query, assistant_response)
            
            references = self._format_references(retrieved_docs)
            final_answer = assistant_response + references

            return {
                'response': final_answer,
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error in Consult.chat: {e}")
            return {
                'response': "抱歉，处理您的请求时出现问题，请稍后再试。",
                'conversation_id': conversation_id
            }

    def summarize_conversation(self, conversation_id: str) -> str:
        history = self._get_conversation_history(conversation_id)
        if not history:
            print(f"ID为 {conversation_id} 的对话为空或不存在，无法总结")
            return ""

        messages = [{"role": "system", "content": "你是一个专业的对话总结助手。"}]
        messages.extend(history)
        messages.append({
            "role": "user",
            "content": "请将上述对话内容精准地总结为一段通顺流畅的摘要，重点概括user提问内容并提炼出对话涉及的症状或不良反应，不超过100字。"
        })
        
        try:
            completion = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=messages
            )
            response = completion.model_dump()
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error summarizing conversation: {e}")
            return "无法生成对话摘要。"
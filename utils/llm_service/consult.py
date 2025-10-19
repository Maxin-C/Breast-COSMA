import os
import uuid
import json
from openai import OpenAI
from typing import List, Dict, Optional, Any
from datetime import datetime

from .retriever import Retriever 
from ..database.models import db, User, RecoveryRecord, ChatHistory, QoL
from ..database import database as db_operations
from sqlalchemy.orm.attributes import flag_modified

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
        不要为患者推荐锻炼动作。
        请保持专业、友好和鼓励的态度，使用简单易懂的语言解释医学概念。
        你的回答应直接切入主题，无需重复问题，并尽量简洁。
        生成的内容必须少于150字，内容必须简洁。
        """

        self.follow_up_forms = []
        form_path = os.getenv("FOLLOW_UP_FORM_PATH")
        if form_path and os.path.exists(form_path):
            for f in os.listdir(form_path):
                if f.endswith('.json'):
                    form_file_path = os.path.join(form_path, f)
                    self.follow_up_forms.append(json.load(open(form_file_path, 'r', encoding='utf-8')))
    
    def process_message(self, user_id: int, query: str, conversation_id: Optional[str] = None, mode: str = 'consult', end_conversation: bool = False) -> Dict[str, Any]:
        is_follow_up_mode = (mode == 'followup')
        original_conversation_id = conversation_id

        if conversation_id:
            record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()
            if record and record.is_follow_up != is_follow_up_mode:
                self.end_and_summarize_conversation(conversation_id)
                conversation_id = None 

        if is_follow_up_mode:
            response = self.chat_followup(user_id=user_id, user_query=query, conversation_id=conversation_id)
        else:
            response = self.chat_consult(user_id=user_id, query=query, conversation_id=conversation_id)
        
        final_conversation_id = response.get('conversation_id')

        if end_conversation and final_conversation_id:
            self.end_and_summarize_conversation(final_conversation_id)
            response['status_message'] = 'Conversation has been ended and summarized.'

        return response

    def _get_user_context(self, user_id: int) -> str:
        user = db_operations.get_record_by_id(User, user_id)
        if not user:
            return ""

        context_parts = []

        user_info = (
            f"患者信息:\n"
            f"- 姓名: {user.name or '未知'}\n"
            f"- 注册日期: {user.registration_date.strftime('%Y-%m-%d') if user.registration_date else '未知'}"
        )
        context_parts.append(user_info)

        chat_summaries = db.session.query(ChatHistory).filter(
            ChatHistory.user_id == user_id,
            ChatHistory.summary.isnot(None)
        ).order_by(ChatHistory.timestamp.asc()).all()
        
        summary_lines = []
        if chat_summaries:
            for record in chat_summaries:
                timestamp = record.timestamp.strftime('%Y-%m-%d %H:%M')
                summary_lines.append(f"- [{timestamp}] {record.summary}")
            context_parts.append("历史对话记录总结:\n" + "\n".join(summary_lines))
        else:
            context_parts.append("历史对话记录总结:\n- 暂无记录")

        qol_records = db.session.query(QoL).filter(
            QoL.user_id == user_id
        ).order_by(QoL.submission_time.asc()).all()
        
        qol_lines = []
        if qol_records:
            for record in qol_records:
                timestamp = record.submission_time.strftime('%Y-%m-%d')
                scores = record.result.get('scoring_result', [])
                score_summary = ", ".join([f"{s.get('module_name', 'N/A')}: {s.get('value', 'N/A')}" for s in scores])
                qol_lines.append(f"- [{timestamp}] {record.form_name}得分: {score_summary}")
            context_parts.append("历史随访记录:\n" + "\n".join(qol_lines))
        else:
            context_parts.append("历史随访记录:\n- 暂无记录")
            
        latest_recovery_record = db.session.query(RecoveryRecord).filter(
            RecoveryRecord.user_id == user_id,
            RecoveryRecord.evaluation_summary.isnot(None)
        ).order_by(RecoveryRecord.record_date.desc()).first()

        if latest_recovery_record:
            timestamp = latest_recovery_record.record_date.strftime('%Y-%m-%d')
            evaluation_summary = f"最新锻炼评估结果 ({timestamp}):\n- {latest_recovery_record.evaluation_summary}"
            context_parts.append(evaluation_summary)
        else:
            context_parts.append("最新锻炼评估结果:\n- 暂无记录")

        return "\n\n".join(context_parts)

    def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()
        if record and record.chat_history:
            return record.chat_history
        return []

    def _save_conversation_turn(self, user_id: int, conversation_id: str, user_message: str, assistant_message: str, is_follow_up: bool):

        record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()
        
        user_msg_obj = {"role": "user", "content": user_message}
        assistant_msg_obj = {"role": "assistant", "content": assistant_message}

        try:
            if record:
                current_history = record.chat_history or []
                current_history.extend([user_msg_obj, assistant_msg_obj])
                record.chat_history = current_history
                flag_modified(record, "chat_history")
                db.session.commit()
            else: 
                new_history = [user_msg_obj, assistant_msg_obj]
                data = {
                    'conversation_id': conversation_id,
                    'user_id': user_id,
                    'chat_history': new_history,
                    'is_follow_up': is_follow_up,
                    'summary': None 
                }
                db_operations.add_record(ChatHistory, data)
        except Exception as e:
            print(f"Error in _save_conversation_turn: {e}")
            db.session.rollback()

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

    def _extract_followup_results(self, history: list) -> Optional[str]:
        if not history:
            return None
            
        messages = [{"role": "system", "content": f'''
请你根据随访人员与患者的对话内容，提取出下面随访表单中对应的随访结果至"form_result"，并根据'scoring_rules'计算随访得分至"scoring_result"。如果随访结果计算方法中条目为可选且没有随访结果的，其"value"设定为"-1"。
随访表单内容如下：
{self.follow_up_forms}
输出格式为json，json结构为：
{{
    "form_result": [
    {{
        "question_id":"", 
        "answer_value":""
    }},
    ...
    ],
    "scoring_result": [
    {{
        "module_name":"",
        "value":""
    }},
    ...
    ]
}}
'''}]
        
        chat_history_text = "对话内容如下：\n" + "\n".join([f"{h['role']}: {h['content']}" for h in history])
        messages.append({"role": "user", "content": chat_history_text})

        try:
            completion = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=messages,
                response_format={"type": "json_object"}
            )
            response = completion.model_dump()
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error extracting follow-up results: {e}")
            return None

    def chat_consult(self, user_id: int, query: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
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

            self._save_conversation_turn(user_id, conversation_id, query, assistant_response, is_follow_up=False)
            
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
    
    def chat_followup(self, user_id: int, user_query: str, conversation_id: Optional[str]) -> Dict[str, Any]:
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            
        if not self.follow_up_forms:
            print("Error: Follow-up forms are not loaded in the Consult service.")
            return {
                'response': "抱歉，随访功能当前不可用，请联系管理员。",
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }
        history = self._get_conversation_history(conversation_id)
        history.append({"role": "user", "content": user_query})

        messages = [{"role": "system", "content": f'''
        你是一名友好和有同情心的医疗助理，对病人进行例行随访，了解他们术后康复情况。你的目标是进行一次自然的对话，并在这个过程中完成随访。

        指令:
        1.  对话式：不要只阅读表单上的问题。结合上下文，将它们转换成自然的日常语言。每次输出内容需要语言简练，不要重复表述类似的意思。
        2.  确认答案：在继续之前，简要确认一下患者的选择以确保准确性。如果需要病人进一步确认，则等待得到准确答案后提出新的问题，否则在确认的同时需要提出新的问题，不能让患者无话可说。
        3.  遵循结构：按照section_id依次问询随访问题，在一个section内的问题可以根据情况自由调整提问顺序，或者将问题合并提问，以减少问答时间。
        4.  处理可选的部分：对于可选的部分（"is_optional": true），首先询问relevance_question。如果病人态度是肯定的，继续回答那个部分的问题。如果是否定的，就跳过这一个section。
        5.  随访完成提示：在完成所有必选随访问题（包括可选section中病人态度肯定时需要提问的问题）后，必须输出“本次随访到此结束，感谢您的支持”。注意，这个判断必须严格，要求在能够保证可以根据对话历史提取出随访结果时才能认为随访已完成。

        Attention: 必须获取所有表单的所有必选随访问题结果。
        Attention: 必须获取所有表单的所有必选随访问题结果。
        Attention: 必须获取所有表单的所有必选随访问题结果。

        随访表单列表内容如下：\n{self.follow_up_forms}
        '''}]
        
        patient_info = self._get_user_context(user_id)
        if patient_info:
            messages.append({"role": "system", "content": f"请结合以下患者历史数据进行自然问询：\n{patient_info}"})
        
        messages.extend(history)

        try:
            completion = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=messages,
                max_tokens=250
            )
            assistant_response = completion.model_dump()['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling LLM in chat_followup: {e}")
            return {
                'response': "抱歉，我在处理您的问题时遇到了一个临时错误，请稍后再试。",
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }

        final_response = {
            'response': assistant_response,
            'conversation_id': conversation_id,
            'timestamp': datetime.now().isoformat(),
            'followup_complete': False
        }

        self._save_conversation_turn(user_id, conversation_id, user_query, assistant_response, is_follow_up=True)

        if "本次随访到此结束，感谢您的支持" in assistant_response:
            print(f"Follow-up for conversation {conversation_id} is complete. Extracting and saving results...")
            final_response['followup_complete'] = True
            
            # Fetch the full, updated history for accurate result extraction
            full_history = self._get_conversation_history(conversation_id)
            results_json_str = self._extract_followup_results(history=full_history)
            
            if results_json_str:
                try:
                    results_data = json.loads(results_json_str)
                    final_response['followup_results'] = results_data
                    
                    # Save the extracted results to the qol_records table
                    qol_record_data = {
                        'user_id': user_id,
                        # Safely get form_name from the first form, or use a default
                        'form_name': self.follow_up_forms[0].get('form_name', 'Comprehensive QoL'),
                        'result': results_data,
                        'submission_time': datetime.now()
                    }
                    db_operations.add_record(QoL, qol_record_data)
                    print(f"Successfully saved QoL results for user {user_id}.")

                except json.JSONDecodeError as e:
                    print(f"ERROR: Failed to parse JSON from follow-up results for conv {conversation_id}: {e}")
                    final_response['followup_results'] = {"error": "Result parsing failed."}
                except Exception as e:
                    print(f"ERROR: Database error while saving QoL record for user {user_id}: {e}")
            else:
                print(f"WARNING: Follow-up complete but failed to extract results for conv {conversation_id}.")

        return final_response

    def summarize_conversation(self, conversation_id: str,  history: Optional[List[Dict]] = None) -> str:
        if history is None:
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
    
    def end_and_summarize_conversation(self, conversation_id: str):
        if not conversation_id:
            return

        record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()

        if not record or not record.chat_history:
            print(f"Warning: Attempted to summarize non-existent or empty conversation: {conversation_id}")
            return
        
        summary = self.summarize_conversation(conversation_id, history=record.chat_history)

        if summary:
            db_operations.update_record(record, {'summary': summary})
            print(f"Successfully saved summary for conversation: {conversation_id}")

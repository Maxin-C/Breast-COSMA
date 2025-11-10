import os
import uuid
import json
from openai import OpenAI
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta # 导入 timedelta
import re 

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

        self.follow_up_forms = json.load(open(os.getenv("FOLLOW_UP_FORM_PATH"), 'r', encoding='utf-8'))
    
    def process_message(self, user_id: int, query: str, conversation_id: Optional[str] = None, mode: str = 'consult', end_conversation: bool = False) -> Dict[str, Any]:
        is_follow_up_mode = (mode == 'followup')
        original_conversation_id = conversation_id
        
        # --- 新增：用于返回给前端的提示信息 ---
        resume_message = None 

        # --- 修改：查找24小时内未完成的随访 ---
        if is_follow_up_mode and not conversation_id:
            twenty_four_hours_ago = datetime.now() - timedelta(hours=24) # 计算24小时前的时间
            
            incomplete_followup = db.session.query(ChatHistory).filter(
                ChatHistory.user_id == user_id,
                ChatHistory.is_follow_up == True,
                ChatHistory.status == 'followup_in_progress', # 查找进行中的随访
                ChatHistory.timestamp >= twenty_four_hours_ago  # 限制在24小时内
            ).order_by(ChatHistory.timestamp.desc()).first()
            
            if incomplete_followup:
                print(f"Resuming incomplete followup conversation (within 24h): {incomplete_followup.conversation_id}")
                conversation_id = incomplete_followup.conversation_id
                
                # --- 新增：设置提示消息 ---
                resume_message = f"检测到您在 {incomplete_followup.timestamp.strftime('%Y-%m-%d %H:%M')} 有一个未完成的随访，我们将继续。"
        # --- 结束修改 ---


        if conversation_id:
            record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()
            
            # --- 修改：处理模式切换 ---
            # 检查模式是否不匹配 (例如，用户在随访中发起了普通咨询)
            if record and record.is_follow_up != is_follow_up_mode:
                # 并且随访仍在进行中
                if record.status == 'followup_in_progress':
                    # 我们只总结，不改变其 'followup_in_progress' 状态，以便下次可以继续
                    print(f"Mode mismatch. Summarizing previous {record.status} conversation {conversation_id}, status remains unchanged.")
                    self.end_and_summarize_conversation(conversation_id, mark_completed=False) # 明确告知不要标记为完成
                
                # 无论如何，都要开始一个新会话
                conversation_id = None
            # --- 结束修改 ---

        if is_follow_up_mode:
            response = self.chat_followup(user_id=user_id, user_query=query, conversation_id=conversation_id, end_conversation=end_conversation)
        else:
            response = self.chat_consult(user_id=user_id, query=query, conversation_id=conversation_id, end_conversation=end_conversation)
        
        final_conversation_id = response.get('conversation_id')

        # --- 新增：添加 resume_message 到最终响应 ---
        if resume_message:
            response['resume_message'] = resume_message
        # --- 结束新增 ---

        if end_conversation and final_conversation_id:
            # --- 修改：手动结束时只总结，不改变状态 ---
            # 状态的改变（如果需要）由 chat_followup 内部的自然完成逻辑处理
            # 这里的 False 确保了随访状态不会被错误地标记为 completed
            self.end_and_summarize_conversation(final_conversation_id, mark_completed=False)
            response['status_message'] = 'Conversation has been ended and summarized.'
            
            # 明确告知前端，随访没有“完成”，只是被“结束”（中断）了
            if response.get('followup_complete') == False:
                 response['status_message'] += " (Follow-up remains in_progress)"
            # --- 结束修改 ---

        return response

    def _get_user_context(self, user_id: int) -> str:
        # ... (此函数内容不变) ...
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
        # ... (此函数内容不变) ...
        record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()
        if record and record.chat_history:
            return record.chat_history
        return []

    def _save_conversation_turn(self, user_id: int, conversation_id: str, user_message: str, assistant_message: str, is_follow_up: bool):

        record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()
        
        user_msg_obj = {"role": "user", "content": user_message}
        assistant_msg_obj = {"role": "assistant", "content": assistant_message}

        # --- 修改：根据模式设置正确的状态 ---
        new_status = 'followup_in_progress' if is_follow_up else 'consult'
        # --- 结束修改 ---

        try:
            if record:
                current_history = record.chat_history or []
                current_history.extend([user_msg_obj, assistant_msg_obj])
                record.chat_history = current_history
                
                # --- 修改：只在会话未*自然完成*时更新状态 ---
                if record.status != 'followup_completed':
                    record.status = new_status
                # --- 结束修改 ---

                flag_modified(record, "chat_history")
                db.session.commit()
            else: 
                new_history = [user_msg_obj, assistant_msg_obj]
                data = {
                    'conversation_id': conversation_id,
                    'user_id': user_id,
                    'chat_history': new_history,
                    'is_follow_up': is_follow_up,
                    'summary': None, 
                    'status': new_status # --- 修改：使用新的状态变量 ---
                }
                db_operations.add_record(ChatHistory, data)
        except Exception as e:
            print(f"Error in _save_conversation_turn: {e}")
            db.session.rollback()

    def _build_prompt_with_rag(self, query: str, context_docs: List[Dict]) -> str:
        # ... (此函数内容不变) ...
        if not context_docs:
            return query
            
        context_str = "\n\n".join([f"参考资料 [{i+1}]:\n{doc['document']['page_content']}" for i, doc in enumerate(context_docs)])
        prompt_template = f"请根据以下提供的参考资料，回答问题。\n\n[参考资料]\n{context_str}\n\n[问题]\n{query}"
        return prompt_template.strip()

    @staticmethod
    def _format_references(retrieved_docs: List[Dict]) -> str:
        # ... (此函数内容不变) ...
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
        # ... (此函数内容不变) ...
        if not history:
            return None
            
        messages = [{"role": "system", "content": f'''
请你根据随访人员与患者的对话内容，提取出下面随访表单中对应的随访结果至"form_result"。
你只需要提取患者的回答，不需要进行任何计算。

随访表单内容如下：
{self.follow_up_forms}

示例：
{{
    "form_result": [
        {
            "question_id": "q1",
            "answer_value": 2
        },
        {
            "question_id": "q2",
            "answer_value": 5
        }
    ]
}}

输出格式为json，json结构为：
{{
    "form_result": [
    {{
        "question_id":"", 
        "answer_value":""
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

    def _calculate_scores(self, form_result_list: List[Dict], scoring_rules: Dict) -> List[Dict]:
        # ... (此函数内容不变) ...
        results = []
        form_data = {
            item['question_id']: item.get('answer_value') 
            for item in form_result_list
        }
        for module_name, rule in scoring_rules.items():
            try:
                formula = rule['formula']
                is_optional = rule.get('is_optional', False)
                item_sum = 0.0
                n = 0 
                item_ids_to_check = []
                if formula['type'] in ["AVERAGE_SCALE", "SUM_SCALE"]:
                    item_ids_to_check = formula['items']
                elif formula['type'] == "COMPLEX_SUM_SCALE":
                    item_ids_to_check = [item['id'] for item in formula['items']]
                has_all_data_for_optional = True
                if is_optional:
                    for item_id in item_ids_to_check:
                        value = form_data.get(item_id)
                        if value is None or not isinstance(value, (int, float)):
                            has_all_data_for_optional = False
                            break
                if is_optional and not has_all_data_for_optional:
                    results.append({"module_name": module_name, "value": -1}) 
                    continue
                if formula['type'] in ["AVERAGE_SCALE", "SUM_SCALE"]:
                    item_operation = formula.get('item_operation', 'val') 
                    for item_id in formula['items']:
                        value = form_data.get(item_id)
                        if value is not None and isinstance(value, (int, float)):
                            if item_operation == "4-val":
                                item_sum += (4.0 - value)
                            elif item_operation == "val":
                                item_sum += float(value)
                            n += 1 
                elif formula['type'] == "COMPLEX_SUM_SCALE":
                    for item in formula['items']:
                        item_id = item['id']
                        item_op = item['op']
                        value = form_data.get(item_id)
                        if value is not None and isinstance(value, (int, float)):
                            if item_op == "4-val":
                                item_sum += (4.0 - value)
                            elif item_op == "val":
                                item_sum += float(value)
                            n += 1 
                if n == 0:
                    if is_optional:
                        results.append({"module_name": module_name, "value": -1}) 
                    else:
                        results.append({"module_name": module_name, "value": None})
                    continue
                n_denominator = 0.0
                if formula['n_source'] == "count":
                    n_denominator = float(n) 
                else:
                    n_denominator = float(formula['n_source'])
                if n_denominator == 0:
                     results.append({"module_name": module_name, "value": None})
                     continue
                final_score = 0.0
                if formula['type'] == "AVERAGE_SCALE":
                    average = item_sum / n_denominator
                    offset = formula.get('offset', 0.0)
                    scale = formula.get('scale', 1.0)
                    final_score = (average + offset) * scale
                elif formula['type'] in ["SUM_SCALE", "COMPLEX_SUM_SCALE"]:
                    scale = formula.get('scale', 1.0)
                    if formula.get('divide_by_n', False):
                        final_score = (item_sum * scale) / n_denominator
                    else:
                        final_score = item_sum * scale
                results.append({"module_name": module_name, "value": round(final_score, 2)})
            except Exception as e:
                print(f"Error calculating score for {module_name}: {e}")
                results.append({"module_name": module_name, "value": None}) 
        return results
        
    def _get_required_question_ids(self, form_json: Dict) -> List[str]:
        # ... (此函数内容不变) ...
        required_q_ids = []
        try:
            for section in form_json.get('sections', []):
                if not section.get('is_optional', False):
                    for q in section.get('questions', []):
                        required_q_ids.append(q['id'])
        except Exception as e:
            print(f"Error parsing form sections: {e}")
        return required_q_ids

    def _get_user_chat_history(self, history):
        # ... (此函数内容不变) ...
        content = ""
        for h in history:
            if h['role'] == 'user':
                content += h['content'] + '\n'
        return content

    def chat_consult(self, user_id: int, query: str, conversation_id: Optional[str] = None, end_conversation: bool = False) -> Dict[str, Any]:
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        if end_conversation:
            return {
                'response': "",
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }

        messages = [{"role": "system", "content": self.default_system_message}]
        
        user_context = self._get_user_context(user_id)
        if user_context:
            messages.append({"role": "system", "content": f"请结合以下患者历史数据进行回答：\n{user_context}"})
        
        history = self._get_conversation_history(conversation_id)
        messages.extend(history)

        prompt_with_rag = query
        if query == '你好':
            prompt_with_rag = query
            retrieved_docs = []
        else:
            retrieved_docs = self.retriever.search(f"问题：{query}；历史对话：{self._get_user_chat_history(history)}")
            prompt_with_rag = self._build_prompt_with_rag(query, retrieved_docs)
        
        messages.append({"role": "user", "content": prompt_with_rag})

        try:
            completion = self.client.chat.completions.create(
                model=self.qwen_model,
                messages=messages
            )
            response = completion.model_dump()
            assistant_response = response['choices'][0]['message']['content']

            # --- 修改：确保 `is_follow_up=False` 被传递 ---
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
    
    def chat_followup(self, user_id: int, user_query: str, conversation_id: Optional[str], end_conversation: bool = False) -> Dict[str, Any]:
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        if end_conversation:
            return {
                'response': "",
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat(),
                'followup_complete': False # 手动结束，未完成
            }
            
        if not self.follow_up_forms:
            print("Error: Follow-up forms are not loaded in the Consult service.")
            return {
                'response': "抱歉，随访功能当前不可用，请联系管理员。",
                'conversation_id': conversation_id,
                'timestamp': datetime.now().isoformat()
            }
        history = self._get_conversation_history(conversation_id)
        
        # 只有当用户有实际输入时才添加到历史记录
        if user_query:
             history.append({"role": "user", "content": user_query})

        required_q_ids = self._get_required_question_ids(self.follow_up_forms)

        messages = [{"role": "system", "content": f'''
        你是一名友好和有同情心的医疗助理，对病人进行例行随访，了解他们术后康复情况。你的目标是进行一次自然的对话，并在这个过程中完成随访。

        指令:
        1.  对话式：不要只阅读表单上的问题。结合上下文，将它们转换成自然的日常语言。每次输出内容需要语言简练，不要重复表述类似的意思。
        2.  确认答案：在继续之前，简要确认一下患者的选择以确保准确性。如果需要病人进一步确认，则等待得到准确答案后提出新的问题，准确答案指的是能够明确对应到具体选项，否则在确认的同时需要提出新的问题，不能让患者无话可说。
        
        3.  遵循结构：按照section_id依次问询随访问题。不要和合并任何问题。

        4.  处理可选的部分：对于可选的部分（"is_optional": true），首先询问relevance_question。如果病人态度是肯定的，继续回答那个部分的问题。如果是否定的，就跳过这一个section。
        
        5.  随访完成提示：在完成所有必选随访问题后，必须输出“本次随访到此结束，感谢您的支持”。
            **必选问题ID列表如下：{required_q_ids}**
            你必须在对话中获得这些ID的明确答案（非'不确定'或'跳过'）后，才能输出结束语。

        Attention: 必须获取所有表单的所有必选随访问题结果。如果用户回答多个问题时答案模糊不清，必须追问到清晰为止。
        Attention: 必须获取所有表单的所有必选随访问题结果。如果用户回答多个问题时答案模糊不清，必须追问到清晰为止。
        Attention: 必须获取所有表单的所有必选随访问题结果。如果用户回答多个问题时答案模糊不清，必须追问到清晰为止。

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
                max_tokens=150
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
        
        # 只有当用户有实际输入时才保存这一轮对话
        if user_query:
            # --- 修改：确保 `is_follow_up=True` 被传递 ---
            self._save_conversation_turn(user_id, conversation_id, user_query, assistant_response, is_follow_up=True)

        if "本次随访到此结束，感谢您的支持" in assistant_response:
            print(f"Follow-up for conversation {conversation_id} is complete. Extracting and saving results...")
            final_response['followup_complete'] = True
            
            full_history = self._get_conversation_history(conversation_id)
            results_json_str = self._extract_followup_results(history=full_history)
            
            if results_json_str:
                try:
                    results_data = json.loads(results_json_str)
                    form_result_list = results_data.get('form_result', [])
                    
                    scoring_result_list = self._calculate_scores(
                        form_result_list, 
                        self.follow_up_forms['scoring_rules']
                    )
                    
                    final_results_obj = {
                        "form_result": form_result_list,
                        "scoring_result": scoring_result_list
                    }
                    
                    final_response['followup_results'] = final_results_obj
                    
                    qol_record_data = {
                        'user_id': user_id,
                        'form_name': self.follow_up_forms.get('form_name', 'Comprehensive QoL'),
                        'result': final_results_obj,
                        'submission_time': datetime.now()
                    }
                    db_operations.add_record(QoL, qol_record_data)
                    print(f"Successfully saved QoL results for user {user_id}.")

                    # --- 修改：将会话标记为 'followup_completed' ---
                    try:
                        record_to_complete = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()
                        if record_to_complete:
                            record_to_complete.status = 'followup_completed' # 使用新状态
                            db.session.commit()
                            print(f"Marked conversation {conversation_id} as followup_completed.")
                    except Exception as e:
                        print(f"ERROR: Failed to mark conversation {conversation_id} as completed: {e}")
                        db.session.rollback()
                    # --- 结束修改 ---

                except json.JSONDecodeError as e:
                    print(f"ERROR: Failed to parse JSON from follow-up results for conv {conversation_id}: {e}")
                    final_response['followup_results'] = {"error": "Result parsing failed."}
                except Exception as e:
                    print(f"ERROR: Database error while saving QoL record for user {user_id}: {e}")
            else:
                print(f"WARNING: Follow-up complete but failed to extract results for conv {conversation_id}.")

        return final_response

    def summarize_conversation(self, conversation_id: str,  history: Optional[List[Dict]] = None) -> str:
        # ... (此函数内容不变) ...
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
    
    # --- 修改：添加 mark_completed 参数 ---
    def end_and_summarize_conversation(self, conversation_id: str, mark_completed: bool = False):
        if not conversation_id:
            return

        record = db.session.query(ChatHistory).filter(ChatHistory.conversation_id == conversation_id).first()

        if not record or not record.chat_history:
            print(f"Warning: Attempted to summarize non-existent or empty conversation: {conversation_id}")
            return
        
        # 仅当没有摘要时才生成
        summary = record.summary
        if not summary:
            summary = self.summarize_conversation(conversation_id, history=record.chat_history)

        update_data = {}
        if summary:
            update_data['summary'] = summary
        
        # --- 修改：根据新逻辑决定是否更新状态 ---
        # 只有在自然完成时（由 chat_followup 调用）才应将 mark_completed 设为 True
        if mark_completed and record.status != 'followup_completed':
            update_data['status'] = 'followup_completed'
            print(f"Marking conversation as completed: {conversation_id}")
        else:
            print(f"Summarizing conversation: {conversation_id}. Status remains '{record.status}'.")
        # --- 结束修改 ---

        if update_data:
            try:
                db_operations.update_record(record, update_data)
            except Exception as e:
                db.session.rollback()
                print(f"Error updating record during summarization: {e}")
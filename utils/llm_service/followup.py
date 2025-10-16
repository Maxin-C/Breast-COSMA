import os
from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()

class FollowUp:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.qwen_model = os.getenv("QWEN_MODEL_NAME", "qwen-plus")

        self.form = []

        for f in os.listdir(os.getenv("FOLLOW_UP_FORM_PATH")):
            self.form.append(json.load(open(os.path.join(os.getenv("FOLLOW_UP_FORM_PATH"), f), 'r')))
    
    def conversation(self, history=[], patient_info=None):
        if self.form == []:
            print("随访表单读取失败")
            return ""
            
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

随访表单列表内容如下：\n{self.form}
        '''}]

        if patient_info is not None:
            messages.append({"role": "system", "content": f"患者在系统中的历史数据如下：\n随访数据：{patient_info['follow_up_result']}\n锻炼评估报告：{patient_info['report_sum']}\n锻炼频率信息：{patient_info['exercise_freq']}\n对话历史总结：{patient_info['history_sum']}\n"})
        
        messages.extend(history)

        completion = self.client.chat.completions.create(
            model=self.qwen_model,
            messages=messages,
            max_tokens=180
        )

        response = completion.model_dump()
        doctor_content = response['choices'][0]['message']['content']
        if "本次随访到此结束，感谢您的支持" in doctor_content:
            print("正在提取随访结果...")
            result = self._get_result(history=history)
            print(result)
        return doctor_content
    
    def _get_result(self, patient_info=None, history=[]):
        if history == []:
            return 
        messages = [{"role": "system", "content": f'''
请你根据随访人员与患者的对话内容，提取出下面随访表单中对应的随访结果至"form_result"，并根据'scoring_rules'计算随访得分至"scoring_result"。如果随访结果计算方法中条目为可选且没有随访结果的，其"value"设定为"-1"。
随访表单内容如下：
{self.form}
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
        if patient_info is not None:
            messages.append({"role": "system", "content": f"患者之前的随访信息和锻炼信息如下：\n{patient_info}"})
        
        chat_history = "对话内容如下：\n"
        for h in history:
            if h['role'] == 'user':
                chat_history += f"患者：{h['content']}\n"
            elif h['role'] == 'assistant':
                chat_history += f"医生：{h['content']}\n"
        
        messages.append({"role": "user", "content": chat_history})

        completion = self.client.chat.completions.create(
            model=self.qwen_model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        response = completion.model_dump()
        return response['choices'][0]['message']['content']

# if __name__ == "__main__":
#     agent = FollowUp()
#     history = []

#     while True:
#         doctor_content = agent.conversation(history=history)
#         print(f"医生：{doctor_content}\n")
#         patient_content = input("患者：")
#         print()
#         history.extend([
#             {"role":"assistant", "content":doctor_content},
#             {"role":"user", "content":patient_content}
#         ])
#         if 'quit' in patient_content:
#             break
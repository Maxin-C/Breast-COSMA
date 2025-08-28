import dashscope
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

class follow_up_model():
    def __init__(self, api_key=None, backbone=None, tokenizer=None):
        if backbone != None and tokenizer != None:
            self.model_state = "local"
            self.model = backbone
            self.tokenizer = tokenizer
        elif api_key != None:
            self.model_state = "cloud"
            dashscope.api_key = api_key
            self.model = dashscope.Generation.Models.qwen_plus
        else:
            print("follow_up_model loading failed!")

    def qwen_model(self, messages):
        if self.model_state == "local":
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        elif self.model_state == "cloud":
            response = dashscope.Generation.call(
                model,
                messages=messages,
                result_format='message',
            )
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                print(f'Request id: {response.request_id}, Status code: {response.status_code}, error code: {response.code}, error message: {response.message}')
                return "error"

    def gen_Conbined_Question(self, questionsList):
        if len(questionsList) > 1:
            prompt = f'''
            你现在是一名乳腺癌随访医生，正在进行乳腺癌术后随访任务，需要根据特定的表单条目，搜集患者的数据。
            你将会拿到问题列表，你需要将问题列表的问题进行合并，一次性获取患者对于这些问题的答案。请一步一步地思考：
            step1：请简短思考这些问题主要是想采集患者哪方面的相关信息？是生理状况方面，还是心理状况方面，或是社会支持方面？
            step2：生成总结问题。你需要将这些问题进行简短地总结，是给予患者一个整体的总结问题。
            请注意，这个总结问题不需要包含所有的问题内容，只需要概括出核心主题，相当于一个问答轮次的开场白，引导患者进一步诉说自己的情况。
            step3：根据总结问题，结合具体的问题列表，生成引导问题。引导问题是对患者的进一步提问，目的是让患者更具体地表达自己的感受和情况。
            你需要构建有效的提问策略，将问题巧妙地融合在聊天中，可以用偏口语化的表述，有效地引导患者说出自己的健康。
            
            注意：请把总结问题和引导问题合并成一句话，只需要输出最终的结果。
            
            举例：
            输入:问题列表：['我感到悲伤', '我感到紧张']；
            输出: 您最近心情怎么样？会偶尔出现悲伤或者紧张的情绪吗？

            输入:问题列表：['脱发使我烦恼', '体重的变化使我烦恼'];
            输出:最近的治疗副作用，比如脱发呀、体重浮动呀，还严重吗？

            输入:问题列表：['我能够享受生活', '我对现在的生活质量感到满意']；
            输出:我想了解一下您对当前生活状态的感觉，您觉得当前的生活质量如何？
            
            现在请处理以下问题：
            输入:问题列表：{questionsList};
            输出:
        '''
        
        else:
            prompt = f'''        
            您是一名正在进行随访调查的专业医生，您需要收集患者报告结果数据，以改善医疗保健质量并提高患者的生活质量。
            请作为医生与患者交互，提出的问题需亲切、专业、简洁，可以使用偏口语化的表述。
            选项仅作为提问策略的参考，无需包含在问题中，问题长度需控制在30字以内，禁止输出表情。

            以下是一些示例：
            题目内容：请问您接受的手术是什么术式？
            选项：["乳房结节麦默通微创手术", "乳房结节开放切除手术"]
            输出：请问您接受的是哪一种手术呢？是乳房结节麦默通手术，还是乳房结节开放切除手术？

            题目内容：我对自己的整体外形很满意
            选项：["非常不同意", "不同意", "既不同意也不反对", "同意", "非常同意"]
            输出：我们认为对自己的整体外形比较满意，您认为呢？是同意这种说法，还是不太同意？

            以下是你需要询问的问题：
            题目内容：{questionsList[0]}
            选项：['一点也不', '有一点', '有些', '相当', '非常']
            输出：
            '''

        messages = [{'role': 'system', 'content': prompt}]
        content = qwen_model(messages)
        return content

    def gen_follow_up_question(self, question,dia):

        prompt = f'''你现在是一名乳腺癌医生，正在进行随访任务，以下是你与患者的对话记录：
        {dia}
        在当前的对话中，患者没有回答或回答得不清晰
        你需要进一步询问患者{question}这些问题。
        
        示例：
        对话：医生：您最近的身体状态怎么样？会不会经常感到精力不足或者需要卧床休息？患者：我最近觉得特别累
        待问问题：我不得不卧床
        输出：术后疲倦是正常的现象，那您需要常常卧床休息来休息吗？

        对话：{dia}
        待问问题：{question}
        输出：
        不能超过60字
        '''
        messages = [{'role': 'assistant', 'content': prompt}]
        content,timer = qwen_model(messages)
        return content

    def extract_combined_options(self, questionList,labels,dia):
        timer = time.time()
        extract_options = []
        total_time = 0
        print(f"问题列表：{questionList}")
        print(len(questionList))

        def extract_single_option(question,labels,dia):
            prompt_degree = f'''
            这是一个信息提取任务。目的是收集患者的健康信息。
            对话中的医生问了很多个问题，但是你只需要关注患者对于问题“{question}”相关的回答。
            根据对话历史，提取患者回复中对应的选项。注意：选项只能限制在{labels}之中，无需解释，只需要输出患者选中选项的内容。

            如果患者没有回答，请输出“不确定”或者“跳过”。
            [不确定]指的是患者的回复模棱两可、回答与题目无关的事情，或者询问问题；
            [跳过]指的是患者明确指出希望跳过、不愿意回答这个问题。
            注意：在对话中，涉及到程度的判断，请进行区分和对应。

            以下是几个参考例子：
            输入: 对话:医生：请问您最近的心情状态如何？包括是否有感到低落、紧张不安或失去战胜疾病的信心，同时，您对于目前应对疾病的方式是否满意，以及是否对病情恶化或生命安全有所担忧？患者：我有一点紧张和担心，蛮怕自己治不好了。
            问题: 请问您最近有没有感觉紧张不安？
            选项: ["不确定","跳过","一点也不","有一点","有些","相当","非常"];
            输出: 有一点
            
            输入: 对话：医生：在手术之后，您对自己的整体外形感到满意吗？/n患者：整形科医生怎么说？/n 
            问题: 您对自己整体外形感到满意吗？
            选项: ["不确定","跳过","非常不同意","不同意","既不同意也不反对","同意","非常同意","非常不同意"] 
            输出: 不确定  

            这是你的提取任务：
            输入: 对话: {dia}
            问题: {question}
            选项: {labels}
            输出:
            '''

            messages = [{'role': 'assistant', 'content': prompt_degree}]

            content = qwen_model(messages)
            # print(f"单个内容的提取选项：{content}")
            return content
        
        for question in questionList:
            pred_option = extract_single_option(question, labels, dia)
            if pred_option in labels:
                extract_options.append(pred_option)
            else:
                extract_options.append(pred_option)

        print(f"最终提取的选项：{extract_options}")
        return extract_options

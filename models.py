import ast
from multiprocessing.pool import INIT
from re import I
from typing import Optional
import os
from pydantic import BaseModel, Field
import requests
import json

os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

INIT_STEP_PROMPT = """
You are an expert problem-solving agent designed to not only solve complex problems but also critically evaluate the quality of your thought process and final answers. 
Your task is to follow a structured approach to generate solutions, assess your thoughts, and provide a rating for each on a scale of 0.1 to 1.0. 
This rating should reflect the accuracy and quality of your reasoning and final answer.

### Instructions:

1. **Understand the Problem:**
   - Carefully analyze the problem provided by the user.
   - Break down the problem into smaller, manageable parts if necessary.
   - Formulate a clear understanding of the problem before proceeding.

2. **Generate Thoughts:**
   - Create multiple thoughts or steps toward solving the problem.
   - For each thought, document your reasoning, ensuring that it is logical and well-founded.

3. **Self-Evaluation:**
   - After generating each thought, evaluate its accuracy and quality.
   - Assign an evaluation score between 0.1 and 1.0. Use the following guidelines:
     - **0.1 to 0.4:** The thought is flawed, inaccurate, or incomplete.
     - **0.5 to 0.7:** The thought is partially correct but may lack detail or full accuracy.
     - **0.8 to 1.0:** The thought is accurate, complete, and well-reasoned.

4. **Generate Final Answer:**
   - Based on your thoughts, synthesize a final answer to the problem.
   - Ensure the final answer is comprehensive and addresses all aspects of the problem.

5. **Final Evaluation:**
   - Evaluate the overall quality and accuracy of your final answer.
   - Provide a final evaluation score based on the same 0.1 to 1.0 scale.

### Output Format:
1. only return the first step and evaluation in JSON format
{
    "thought": your first step of thoughts,
    "evaluation": your first step of evaluation
}
"""

NEXT_STEP_PROMPT = """
You are an expert problem-solving agent designed to not only solve complex problems but also critically evaluate the quality of your thought process and final answers. 
Now, I will give you a list, and you can think about the next step in your reasoning based on the original questions provided in the list and the existing thought processes.
You should provide a rating for your next step on a scale of 0.1 to 1.0. This rating should reflect the accuracy and quality of your reasoning and final answer.

### Instructions:

1. **Understand the Problem:**
   - Carefully analyze the problem provided by the user.
   - Break down the problem into smaller, manageable parts if necessary.
   - Formulate a clear understanding of the problem before proceeding.

2. **Generate Thoughts:**
   - Create multiple thoughts or steps toward solving the problem.
   - For each thought, document your reasoning, ensuring that it is logical and well-founded.

3. **Self-Evaluation:**
   - After generating each thought, evaluate its accuracy and quality.
   - Assign an evaluation score between 0.1 and 1.0. Use the following guidelines:
     - **0.1 to 0.4:** The thought is flawed, inaccurate, or incomplete.
     - **0.5 to 0.7:** The thought is partially correct but may lack detail or full accuracy.
     - **0.8 to 1.0:** The thought is accurate, complete, and well-reasoned.

4. **Generate Final Answer:**
   - Based on your thoughts, synthesize a final answer to the problem.
   - Ensure the final answer is comprehensive and addresses all aspects of the problem.

5. **Final Evaluation:**
   - Evaluate the overall quality and accuracy of your final answer.
   - Provide a final evaluation score based on the same 0.1 to 1.0 scale.

### Output Format:
1. only return the first step and evaluation in JSON format
{
    "thought": your next step of thoughts,
    "evaluation": your next step of evaluation
}
2. the "thought" field should not contain the key word like:'step', 'next', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', just return the content of the next step.

"""

INIT_STEP_PROMPT_CH = """
# 角色
你是一位专门解决问题的智能系统，不仅可以处理复杂问题，还能对自己思维过程和最终答案的质量进行严格的评估。
你会对一个复杂的问题进行拆解，并对你的每一步思考过程在 0.1 到 1.0 的评分范围内提供评分（该评分反映了你的推理准确性以及最终答案的质量）。

# 操作步骤
1. 理解问题：
- 仔细分析用户提供的问题。
- 必要时，将问题分解为更小且易管理的部分。
- 在开始解决前，确保对问题有清晰的理解。
2. 产生思路：
- 生成多个解决问题的思路或步骤。
- 记录每个思路的推理过程，确保逻辑合理，依据充分。
3. 自我评估：
- 在生成每个思路后，评估其准确性和质量。
- 为每个思路打分，范围在 0.1 到 1.0 之间，评分标准如下：
- 0.1 到 0.4： 思路存在明显问题，不准确或不完整。
- 0.5 到 0.7： 思路部分正确，但可能缺乏细节或完全准确性。
- 0.8 到 1.0： 思路准确、全面且推理充足。
4. 生成最终答案：
- 基于思路合成最终答案。
- 确保最终答案是全面的，涵盖了问题的所有方面。
5. 整体评估：
- 对最终答案的整体质量和准确性进行评估。
- 根据相同的 0.1 到 1.0 评分范围提供最终评分。

# 任务
现在我给出一个具体的任务，请返回你的第一步思路和评分。给出的思路需要从用户角度出发，清晰明了，易于理解。

# 输出格式要求
1. 仅返回第一个步骤和评分的 JSON 格式，只需要返回 json，不需要任何解释！
{"thought": "第一步思路", "evaluation": 第一步思路评分}
2. 务必保证这个json需要能被 eval 函数解析，中间不需要任何换行符，中间不需要任何换行符，中间不需要任何换行符。
"""

NEXT_STEP_PROMPT_CH = """
# 角色
你是一位专长于解决问题的智能助手，既能处理复杂问题，又能对自己的思维过程和最终答案质量进行深度评估。
你会对一个复杂的问题进行拆解，并对你的每一步思考过程在 0.1 到 1.0 的评分范围内为每一步提供评分（该评分反映了你的推理准确性以及最终答案的质量）。
注意你的思维具有一定的创造性与发散性。

# 操作步骤
1. 理解问题：
- 仔细分析用户提供的问题。
- 必要时，将问题分解为更小且易管理的部分。
- 在开始解决前，确保对问题有清晰的理解。
2. 产生思路：
- 生成多个解决问题的思路或步骤。
- 记录每个思路的推理过程，确保逻辑合理，依据充分。
3. 自我评估：
- 在生成每个思路后，评估其准确性和质量。
- 为每个思路打分，范围在 0.1 到 1.0 之间，评分标准如下：
- 0.1 到 0.4： 思路存在明显问题，不准确或不完整。
- 0.5 到 0.7： 思路部分正确，但可能缺乏细节或完全准确性。
- 0.8 到 1.0： 思路准确、全面且推理充足。
4. 生成最终答案：
- 基于思路合成最终答案。
- 确保最终答案是全面的，涵盖了问题的所有方面。
5. 整体评估：
- 对最终答案的整体质量和准确性进行评估。
- 根据相同的 0.1 到 1.0 评分范围提供最终评分。

# 任务
接下来，我会提供给你一个原始任务以及目前现有的思考流程。请你结合现有的思考流程，给出一个相对最合理的下一步解决流程，并给出评分然后返回。给出的思路需要从用户角度出发，清晰明了，易于理解。
注意：如果当前现有思考流程错误，你可以直接返回空字符串的思考，然后给出评分0代表任务失败。

# 输出格式要求
1. 仅返回第一个思考步骤和评估结果的 JSON 格式, 只需要返回 json，不需要任何解释！
{"thought": "下一步思路", "evaluation": 下一步思路评分}
代表下一步的内容。

{"thought": "", "evaluation": 0}
代表任务失败。
2. “thought”字段中不应包含类似“步骤”“下一步”“第一个”“第二个”“第三个”“第四个”“第五个”“第六个”“第七个”“第八个”“第九个”“第十个”等关键词，仅返回下一步的内容。
3. 务必保证这个json需要能被 eval 函数解析，中间不需要任何换行符，中间不需要任何换行符，中间不需要任何换行符。
"""

API_URL = "http://xxxx:xxxx/v1/chat/completions"
MODEL_PATH = "/home/pubdata/model_weights/Meta-Llama-3-70B-Instruct"
HEADERS = {
    "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
    "Content-Type": "application/json",
}

class Thought(BaseModel):
    thought: str
    evaluation: Optional[float] = Field(
        description="对思维的评价。它可以是0.1到1.0之间的数值，0.1为最差，1.0为最好。"
    )

def string_to_dict(thought_string):
    """将字符串转换为字典"""
    return ast.literal_eval(thought_string)

def send_request(user_message, system_prompt, temperature=0.7, repetition_penalty=1.1, max_tokens=8192):
    """发送请求到指定的API并返回响应内容"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    payload = json.dumps(
        {
            "model": MODEL_PATH,
            "messages": messages,
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
        }
    )

    response = requests.post(API_URL, headers=HEADERS, data=payload)
    response_content = json.loads(response.text)["choices"][0]["message"]["content"]
    return response_content

def generate_thought(user_message, is_next_step=False):
    """生成思维内容，根据是否为下一步选择不同的系统提示"""
    system_prompt = NEXT_STEP_PROMPT_CH if is_next_step else INIT_STEP_PROMPT_CH
    return send_request(user_message, system_prompt)

def generate(user_message, system_prompt):
    """使用指定的系统提示生成内容"""
    return send_request(user_message, system_prompt)

if __name__ == "__main__":
    task = "怎么学好 python 编程？"
    print(generate_thought(task))

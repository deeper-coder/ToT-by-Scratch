from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import re
import sys
import os
from typing import Optional

from click import confirm
from matplotlib.font_manager import json_dump
import numpy as np
from regex import P

parent_dir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
print(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from models import generate, generate_thought, Thought, string_to_dict


class BFSAgent:
    def __init__(self):
        self.thoughts = []
        self.num_agents = 3
        self.breadth_limit = 5
        self.max_iterations = 4
        self.current_iteration = 0

    def bfs(self, task: str) -> Optional[str]:
        """
        执行广度优先搜索以解决给定的任务。

        :param task: 任务描述字符串。
        :return: 最终答案字符串，如果未找到解决方案则返回 None。
        """
        current_states = [task]
        selected_index = -1

        for iteration in range(1, self.max_iterations + 1):
            self.current_iteration = iteration
            new_states = self._generate_new_states(current_states, iteration > 1)
            
            if not new_states:
                print(f"第 {iteration} 步未生成有效的思维。停止 BFS。")
                break

            evaluations = self._evaluate_states(new_states)
            current_states, (is_done, selected_index) = self._select_best_states(new_states, evaluations)

            if is_done:
                break

        return self._generate_final_answer(current_states, selected_index)

    def _generate_final_answer(self, states, selected_index):
        """
        基于选定的状态生成最终答案。

        :param states: 当前状态列表。
        :param selected_index: 选定状态的索引。
        :return: 最终答案字符串，如果没有可用状态则返回 None。
        """
        if not states:
            return None

        selected_state = states[selected_index] if selected_index != -1 else states[0]
        all_thoughts = self._build_current_thoughts(selected_state)

        print("使用思维树:")
        return all_thoughts
    
    def _refine_final_answer(self, all_thoughts):
        """
        根据所有思维步骤，组织并生成最终的答案。

        :param all_thoughts: 包含任务和思维步骤的字符串。
        :return: 整理后的最终答案字符串。
        """
        sys_prompt = """
        你是一位擅长有条理表达的语言专家。我将提供一个任务及若干具体步骤，请你按要求组织语言，描述解决问题的思路，语句通顺。
        ### 要求：
        1. 分步骤描述解决问题的思路，确保逻辑清晰，表达准确。
        2. 如果有些步骤意思一样，可以合并描述。
        """
        return generate(all_thoughts, sys_prompt)

    def _should_choose(self, x, k=2):
        """
        决定是否基于概率函数选择状态。

        :param x: 状态的索引。
        :param k: 控制概率的参数。
        :return: 如果应选择状态则返回 True，否则返回 False。
        """
        return np.random.random() > 0.75 or np.exp(-k * x) > 0.6

    def _confirm_state(self, state):
        """
        通过模拟专家讨论来确认当前状态是否能导致解决方案。

        :param state: 当前状态。
        :return: 如果状态能导致解决方案则返回 True，否则返回 False。
        """
        current_thoughts = self._build_current_thoughts(state)
        sys_prompt = """
        想象一下三位专家正在讨论给出的现有思维步骤是否能确定解决任务。请你以三位专家的身份，按照以下要求多次进行讨论。
        要求：
        1. 所有专家将提出自己的观点，互相讨论，但不要争吵，以达成共识。
        2. 请把一些常识考虑进去，比如物体受到重力作用，3 * 8 = 24 等。
        3. 如果当前步骤不能解决问题，或者不符合常识，请提出异议。
        4. 返回值：True 表示可以确定解决任务，False 表示不能确定解决任务。
        下面是用户给出的思维步骤，请你讨论是否可以确定解决任务。
        """
        answer = generate(current_thoughts, sys_prompt)
        return answer.rfind("True") != -1

    def _select_best_states(self, new_states, evaluations):
        """
        根据评估结果选择最佳状态。

        :param new_states: 新状态列表。
        :param evaluations: 与状态对应的评估列表。
        :return: 包含最佳状态列表和 (is_done, selected_index) 元组的元组。
        """
        state_evaluation_pairs = sorted(
            zip(new_states, evaluations), key=lambda x: x[1], reverse=True
        )

        best_states = []
        is_done = False
        selected_index = -1
        for i, (state, evaluation) in enumerate(state_evaluation_pairs):
            if evaluation > 0:
                if evaluation == 10:
                    is_done = self._confirm_state(state)
                    if is_done:
                        best_states.append(state)
                        continue
                if i == 0 or self._should_choose(i):
                    best_states.append(state)
            if len(best_states) >= self.breadth_limit:
                break
        return best_states, (is_done, selected_index)

    def _print_best_states(self, best_states):
        """
        打印最佳状态。

        :param best_states: 最佳状态列表。
        """
        for state in best_states:
            print(state)
            print("\n")
        print("----------------------------------------------------------")

    def _evaluate_states(self, new_states):
        """
        评估给定的状态。

        :param new_states: 新状态列表。
        :return: 与状态对应的评估列表。
        """
        return [
            Thought(**string_to_dict(thoughts[-1])).evaluation for thoughts in new_states
        ]

    def _extract_thought(self, thought) -> Optional[str]:
        """
        提取 thought 字段的 JSON 字符串。

        :param thought: JSON 格式的思维字符串。
        :return: 提取的思维 JSON 字符串，如果提取失败则返回 None。
        """
        try:
            thought = thought.replace("\\n", "")
            thought = eval(thought)
        except Exception as e:
            print(e)
            return None
        return json.dumps(thought, ensure_ascii=False)

    def _build_current_thoughts(self, thoughts):
        """
        构建当前思维轨迹的字符串表示。

        :param thoughts: 思维列表。
        :return: 表示思维轨迹的字符串。
        """
        ans = []
        for i, thought in enumerate(thoughts):
            if i == 0:
                ans.append("Task: " + thought)
            else:
                ans.append(f"Step{i}: {Thought(**string_to_dict(thought)).thought}")
        return "\n\n".join(ans)

    def _generate_new_states(self, states, is_next_step=False):
        """
        从当前状态生成新状态。

        :param states: 当前状态列表。
        :param next_step: 是否生成下一步。
        :return: 新状态列表。
        """
        new_states_list = []
        # 创建用于生成思维的部分函数
        generate_partial = partial(generate_thought, is_next_step=is_next_step)
        for state in states:
            with ThreadPoolExecutor() as executor:
                # 构建当前思维
                current_thoughts = self._build_current_thoughts(state) if is_next_step else state
                # 生成新思维列表
                new_thoughts = list(
                    executor.map(generate_partial, [current_thoughts] * self.num_agents)
                )
                for thought in new_thoughts:
                    # 提取思维内容
                    extracted_thought = self._extract_thought(thought)
                    if extracted_thought is not None:
                        # 更新状态
                        new_state = [*state, extracted_thought] if is_next_step else [state, extracted_thought]
                        # 确认新状态是否有效
                        if self._confirm_state(new_state):
                            new_states_list.append(new_state)
        return new_states_list

    def run(self, task):
        """
        运行 BFSAgent 以解决给定的任务。

        :param task: 任务描述字符串。
        :return: 最终答案字符串，如果未找到解决方案则返回 None。
        """
        return self.bfs(task)


if __name__ == "__main__":
    agent = BFSAgent()
    task = """
        Bob 在客厅里。他走到厨房，手里拿着一个杯子。他把一个球放进杯子里，然后把杯子带到卧室。他将杯子倒过来，然后走到花园。他把杯子放在花园里，然后走到车库。
        问：现在球在哪里？
        """
    print(agent.run(task))
    print("----------------------------------------------------------")
    print("不使用思维树:")
    print(generate(task, "你是一个有帮助的助手。"))
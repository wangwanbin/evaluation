"""MBPP (Mostly Basic Python Programming) 代码能力 Benchmark。

测试模型的基本 Python 编程能力，包括字符串操作、数据结构、算法等。
"""

from __future__ import annotations

import logging
import re

from modules.llm.base import BaseLLMBenchmark

logger = logging.getLogger("eval.llm.mbpp")

# 内置 10 道编程题
BUILTIN_QUESTIONS = [
    {
        "id": "mbpp_0",
        "text": "编写一个函数，接收两个字符串参数，返回它们拼接后的结果，中间用空格分隔。",
        "test": "assert concat_with_space('hello', 'world') == 'hello world'\nassert concat_with_space('a', 'b') == 'a b'",
        "code": "def concat_with_space(a, b):\n    return f'{a} {b}'",
        "answer": "def concat_with_space(a, b):\n    return f'{a} {b}'",
    },
    {
        "id": "mbpp_1",
        "text": "编写一个函数，接收一个整数列表，返回所有偶数的平方组成的新列表。",
        "test": "assert square_evens([1, 2, 3, 4, 5]) == [4, 16]\nassert square_evens([1, 3, 5]) == []",
        "code": "def square_evens(nums):\n    return [x**2 for x in nums if x % 2 == 0]",
        "answer": "def square_evens(nums):\n    return [x**2 for x in nums if x % 2 == 0]",
    },
    {
        "id": "mbpp_2",
        "text": "编写一个函数，接收一个字符串，返回该字符串中每个字符出现的次数（字典形式）。",
        "test": "assert char_count('hello') == {'h': 1, 'e': 1, 'l': 2, 'o': 1}\nassert char_count('') == {}",
        "code": "def char_count(s):\n    result = {}\n    for c in s:\n        result[c] = result.get(c, 0) + 1\n    return result",
        "answer": "def char_count(s):\n    result = {}\n    for c in s:\n        result[c] = result.get(c, 0) + 1\n    return result",
    },
    {
        "id": "mbpp_3",
        "text": "编写一个函数，接收两个整数 a 和 b，返回 a 的 b 次方（不使用 ** 运算符和 pow 函数）。",
        "test": "assert power(2, 3) == 8\nassert power(5, 0) == 1\nassert power(3, 2) == 9",
        "code": "def power(a, b):\n    result = 1\n    for _ in range(b):\n        result *= a\n    return result",
        "answer": "def power(a, b):\n    result = 1\n    for _ in range(b):\n        result *= a\n    return result",
    },
    {
        "id": "mbpp_4",
        "text": "编写一个函数，接收一个整数列表，返回按绝对值排序后的新列表（从小到大）。",
        "test": "assert sort_by_abs([-3, 1, -2, 4]) == [1, -2, -3, 4]\nassert sort_by_abs([-1, -5, 3]) == [-1, 3, -5]",
        "code": "def sort_by_abs(nums):\n    return sorted(nums, key=abs)",
        "answer": "def sort_by_abs(nums):\n    return sorted(nums, key=abs)",
    },
    {
        "id": "mbpp_5",
        "text": "编写一个函数，接收一个字符串，返回每个单词的首字母大写的标题格式。",
        "test": "assert title_case('hello world') == 'Hello World'\nassert title_case('python is great') == 'Python Is Great'",
        "code": "def title_case(s):\n    return ' '.join(w.capitalize() for w in s.split())",
        "answer": "def title_case(s):\n    return ' '.join(w.capitalize() for w in s.split())",
    },
    {
        "id": "mbpp_6",
        "text": "编写一个函数，接收一个列表，返回去重后的列表（保持原顺序）。",
        "test": "assert unique_order([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]\nassert unique_order(['a', 'b', 'a']) == ['a', 'b']",
        "code": "def unique_order(items):\n    seen = set()\n    result = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result",
        "answer": "def unique_order(items):\n    seen = set()\n    result = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result",
    },
    {
        "id": "mbpp_7",
        "text": "编写一个函数，接收两个集合，返回它们的交集、并集和差集（A - B）的元组。",
        "test": "result = set_operations({1, 2, 3}, {2, 3, 4})\nassert result[0] == {2, 3}  # 交集\nassert result[1] == {1, 2, 3, 4}  # 并集\nassert result[2] == {1}  # 差集",
        "code": "def set_operations(a, b):\n    return (a & b, a | b, a - b)",
        "answer": "def set_operations(a, b):\n    return (a & b, a | b, a - b)",
    },
    {
        "id": "mbpp_8",
        "text": "编写一个函数，接收一个整数 n，返回斐波那契数列前 n 项的列表（F(0)=0, F(1)=1）。",
        "test": "assert fibonacci_list(5) == [0, 1, 1, 2, 3]\nassert fibonacci_list(1) == [0]",
        "code": "def fibonacci_list(n):\n    if n <= 0:\n        return []\n    result = [0]\n    if n == 1:\n        return result\n    result.append(1)\n    for i in range(2, n):\n        result.append(result[-1] + result[-2])\n    return result",
        "answer": "def fibonacci_list(n):\n    if n <= 0:\n        return []\n    result = [0]\n    if n == 1:\n        return result\n    result.append(1)\n    for i in range(2, n):\n        result.append(result[-1] + result[-2])\n    return result",
    },
    {
        "id": "mbpp_9",
        "text": "编写一个函数，接收一个字符串文本和一个整数最大宽度，实现自动换行功能。",
        "test": "assert wrap_text('hello world', 5) == 'hello\\nworld'\nassert wrap_text('python', 10) == 'python'",
        "code": "def wrap_text(text, max_width):\n    result = []\n    for i in range(0, len(text), max_width):\n        result.append(text[i:i+max_width])\n    return '\\n'.join(result)",
        "answer": "def wrap_text(text, max_width):\n    result = []\n    for i in range(0, len(text), max_width):\n        result.append(text[i:i+max_width])\n    return '\\n'.join(result)",
    },
]

FEW_SHOT_EXAMPLE = {
    "user": "编写一个函数，接收两个数字，返回它们的和。\ndef add(a, b):",
    "assistant": "def add(a, b):\n    return a + b"
}


class MBPPBenchmark(BaseLLMBenchmark):
    """MBPP 基础编程评测"""

    benchmark_id = "mbpp"
    benchmark_name = "MBPP"
    category = "代码能力"

    def load_questions(self) -> list[dict]:
        questions = []
        for q in BUILTIN_QUESTIONS:
            questions.append({
                "id": q["id"],
                "text": q["text"],
                "test": q["test"],
                "code": q["code"],
                "answer": q["answer"],
                "category": "代码能力",
            })
        logger.info(f"MBPP 共加载 {len(questions)} 道题目")
        return questions

    def build_prompt(self, question: dict, few_shot: bool = True) -> list[dict]:
        messages = [{
            "role": "system",
            "content": "你是一个 Python 编程专家。请根据需求编写完整函数。只输出函数代码，用 ```python``` 包裹。代码必须可以直接运行。"
        }]
        if few_shot:
            messages.append({"role": "user", "content": FEW_SHOT_EXAMPLE["user"]})
            messages.append({"role": "assistant", "content": f"```python\n{FEW_SHOT_EXAMPLE['assistant']}\n```"})
        messages.append({
            "role": "user",
            "content": f"请编写函数实现以下功能：\n{question['text']}\n\n请只返回完整的 Python 函数代码。"
        })
        return messages

    def extract_answer(self, text: str) -> str:
        code_match = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        lines = text.strip().split('\n')
        code_lines = [l for l in lines if l.strip() and not l.startswith('>>>')]
        return '\n'.join(code_lines).strip() if code_lines else text.strip()

    def check_answer(self, extracted: str, reference: str) -> bool:
        ref_clean = reference.strip()
        ext_clean = extracted.strip()
        keywords = re.findall(r'\b(def|return|for|while|if|in|sorted|lambda)\b', ref_clean)
        if keywords:
            score = sum(1 for kw in keywords if kw in ext_clean)
            return score >= len(keywords) * 0.6 and len(ext_clean) > 10
        return len(ext_clean) > 5


benchmark = MBPPBenchmark()

"""HumanEval 代码能力 Benchmark — 根据函数签名和文档字符串生成正确实现。

测试模型的代码理解和生成能力，包含 Python 函数补全任务。
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from modules.llm.base import BaseLLMBenchmark

logger = logging.getLogger("eval.llm.humaneval")

# 内置 10 道编程题
BUILTIN_QUESTIONS = [
    {
        "id": "humaneval_0",
        "prompt": "def is_even(n):\n    \"\"\"判断一个整数是否为偶数。\n    \n    >>> is_even(2)\n    True\n    >>> is_even(3)\n    False\n    \"\"\"\n",
        "test": "assert is_even(2) == True\nassert is_even(3) == False\nassert is_even(0) == True\nassert is_even(-4) == True",
        "entry_point": "is_even",
        "canonical_solution": "    return n % 2 == 0"
    },
    {
        "id": "humaneval_1",
        "prompt": "def factorial(n):\n    \"\"\"计算 n 的阶乘（n!），n 为非负整数。\n    \n    >>> factorial(5)\n    120\n    >>> factorial(0)\n    1\n    \"\"\"\n",
        "test": "assert factorial(5) == 120\nassert factorial(0) == 1\nassert factorial(1) == 1\nassert factorial(10) == 3628800",
        "entry_point": "factorial",
        "canonical_solution": "    if n == 0:\n        return 1\n    return n * factorial(n - 1)"
    },
    {
        "id": "humaneval_2",
        "prompt": "def fibonacci(n):\n    \"\"\"返回斐波那契数列的第 n 项（从 0 开始，F(0)=0, F(1)=1）。\n    \n    >>> fibonacci(0)\n    0\n    >>> fibonacci(5)\n    5\n    \"\"\"\n",
        "test": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55\nassert fibonacci(20) == 6765",
        "entry_point": "fibonacci",
        "canonical_solution": "    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b"
    },
    {
        "id": "humaneval_3",
        "prompt": "def is_palindrome(s):\n    \"\"\"判断字符串是否为回文（忽略大小写，忽略空格）。\n    \n    >>> is_palindrome('A man a plan a canal Panama')\n    True\n    >>> is_palindrome('hello')\n    False\n    \"\"\"\n",
        "test": "assert is_palindrome('racecar') == True\nassert is_palindrome('hello') == False\nassert is_palindrome('A man a plan a canal Panama') == True\nassert is_palindrome('') == True",
        "entry_point": "is_palindrome",
        "canonical_solution": "    cleaned = ''.join(s.lower().split())\n    return cleaned == cleaned[::-1]"
    },
    {
        "id": "humaneval_4",
        "prompt": "def find_max(arr):\n    \"\"\"找出列表中的最大值。\n    \n    >>> find_max([1, 3, 2, 5, 4])\n    5\n    >>> find_max([-1, -5, -2])\n    -1\n    \"\"\"\n",
        "test": "assert find_max([1, 3, 2, 5, 4]) == 5\nassert find_max([-1, -5, -2]) == -1\nassert find_max([100]) == 100\nassert find_max([]) == None",
        "entry_point": "find_max",
        "canonical_solution": "    if not arr:\n        return None\n    max_val = arr[0]\n    for x in arr:\n        if x > max_val:\n            max_val = x\n    return max_val"
    },
    {
        "id": "humaneval_5",
        "prompt": "def count_vowels(s):\n    \"\"\"统计字符串中元音字母（a, e, i, o, u）的个数，忽略大小写。\n    \n    >>> count_vowels('hello')\n    2\n    >>> count_vowels('PYTHON')\n    2\n    \"\"\"\n",
        "test": "assert count_vowels('hello') == 2\nassert count_vowels('PYTHON') == 2\nassert count_vowels('xyz') == 0\nassert count_vowels('AEIOU') == 5",
        "entry_point": "count_vowels",
        "canonical_solution": "    vowels = set('aeiou')\n    return sum(1 for c in s.lower() if c in vowels)"
    },
    {
        "id": "humaneval_6",
        "prompt": "def reverse_string(s):\n    \"\"\"反转字符串。\n    \n    >>> reverse_string('hello')\n    'olleh'\n    >>> reverse_string('Python')\n    'nohtyP'\n    \"\"\"\n",
        "test": "assert reverse_string('hello') == 'olleh'\nassert reverse_string('Python') == 'nohtyP'\nassert reverse_string('') == ''\nassert reverse_string('a') == 'a'",
        "entry_point": "reverse_string",
        "canonical_solution": "    return s[::-1]"
    },
    {
        "id": "humaneval_7",
        "prompt": "def is_prime(n):\n    \"\"\"判断一个整数是否为素数。\n    \n    >>> is_prime(7)\n    True\n    >>> is_prime(10)\n    False\n    \"\"\"\n",
        "test": "assert is_prime(7) == True\nassert is_prime(10) == False\nassert is_prime(2) == True\nassert is_prime(1) == False\nassert is_prime(97) == True",
        "entry_point": "is_prime",
        "canonical_solution": "    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"
    },
    {
        "id": "humaneval_8",
        "prompt": "def array_sum(arr):\n    \"\"\"计算列表中所有元素的和。\n    \n    >>> array_sum([1, 2, 3, 4, 5])\n    15\n    >>> array_sum([-1, 0, 1])\n    0\n    \"\"\"\n",
        "test": "assert array_sum([1, 2, 3, 4, 5]) == 15\nassert array_sum([-1, 0, 1]) == 0\nassert array_sum([]) == 0\nassert array_sum([100]) == 100",
        "entry_point": "array_sum",
        "canonical_solution": "    return sum(arr)"
    },
    {
        "id": "humaneval_9",
        "prompt": "def remove_duplicates(arr):\n    \"\"\"去除列表中的重复元素，保持原有顺序。\n    \n    >>> remove_duplicates([1, 2, 2, 3, 1, 4])\n    [1, 2, 3, 4]\n    >>> remove_duplicates([])\n    []\n    \"\"\"\n",
        "test": "assert remove_duplicates([1, 2, 2, 3, 1, 4]) == [1, 2, 3, 4]\nassert remove_duplicates([]) == []\nassert remove_duplicates([1, 1, 1]) == [1]",
        "entry_point": "remove_duplicates",
        "canonical_solution": "    seen = set()\n    result = []\n    for x in arr:\n        if x not in seen:\n            seen.add(x)\n            result.append(x)\n    return result"
    },
]

FEW_SHOT_EXAMPLES = [
    {"user": "请实现一个函数，接收两个整数，返回它们的和。\ndef add(a, b):", "assistant": "    return a + b"},
]


class HumanEvalBenchmark(BaseLLMBenchmark):
    """HumanEval 代码生成评测"""

    benchmark_id = "humaneval"
    benchmark_name = "HumanEval"
    category = "代码能力"

    def load_questions(self) -> list[dict]:
        """加载编程题（10 题）"""
        questions = []
        for q in BUILTIN_QUESTIONS:
            questions.append({
                "id": q["id"],
                "prompt": q["prompt"],
                "test": q["test"],
                "entry_point": q["entry_point"],
                "canonical_solution": q["canonical_solution"],
                "answer": q["canonical_solution"],
                "category": "代码能力",
            })
        logger.info(f"HumanEval 共加载 {len(questions)} 道题目")
        return questions

    def build_prompt(self, question: dict, few_shot: bool = True) -> list[dict]:
        """构建代码补全 prompt（中文）"""
        messages = [
            {
                "role": "system",
                "content": "你是一个 Python 编程专家。请根据函数签名和文档字符串补全函数体。只输出函数体代码（缩进部分），不需要重复函数签名。代码必须可以直接运行。"
            }
        ]

        if few_shot:
            for ex in FEW_SHOT_EXAMPLES:
                messages.append({"role": "user", "content": ex["user"]})
                messages.append({"role": "assistant", "content": f"```python\n{ex['assistant']}\n```"})

        messages.append({
            "role": "user",
            "content": f"请补全以下 Python 函数：\n\n{question['prompt']}\n\n请只返回函数体代码。"
        })
        return messages

    def extract_answer(self, text: str) -> str:
        """从模型输出中提取代码"""
        # 尝试提取代码块
        code_match = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # 去掉可能的函数签名行
        lines = text.strip().split('\n')
        code_lines = [l for l in lines if not l.startswith('def ') and not l.startswith('>>>')]
        return '\n'.join(code_lines).strip() if code_lines else text.strip()

    def check_answer(self, extracted: str, reference: str) -> bool:
        """简化版：检查代码中是否包含关键实现元素"""
        ref_clean = reference.strip()
        ext_clean = extracted.strip()

        # 检查关键关键字（return, if, for 等）
        keywords = re.findall(r'\b(return|if|for|while|def)\b', ref_clean)
        if keywords:
            has_all = all(kw in ext_clean for kw in keywords)
            return has_all and len(ext_clean) > 5

        return len(ext_clean) > 0


benchmark = HumanEvalBenchmark()

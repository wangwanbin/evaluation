"""IFEval 指令遵循 Benchmark — 测试模型是否准确遵循指令中的约束条件。

覆盖约束：
- 格式约束（JSON、列表、邮件等）
- 长度约束（固定字数、范围）
- 内容约束（必须包含/不包含特定词）
- 字数约束
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from modules.llm.base import BaseLLMBenchmark

logger = logging.getLogger("eval.llm.ifeval")

# 内置 15 道指令遵循题
BUILTIN_QUESTIONS = [
    # === 格式约束 ===
    {
        "id": "ifeval_0",
        "instruction": "请用 JSON 格式输出你的回答，包含 name、age、city 三个字段。请直接输出 JSON。",
        "validator": "json",
        "keyword": "name",
        "category": "格式约束"
    },
    {
        "id": "ifeval_1",
        "instruction": "请写一封简短的邮件，以「尊敬的老师：」开头，以「此致敬礼」结尾。不少于 50 字。",
        "validator": "prefix_suffix",
        "prefix": "尊敬的老师：",
        "suffix": "此致敬礼",
        "min_length": 50,
        "category": "格式约束"
    },
    {
        "id": "ifeval_2",
        "instruction": "请用 markdown 列表格式列出你今天的待办事项，至少 3 项，每项以 - 开头。",
        "validator": "bullet_list",
        "min_items": 3,
        "category": "格式约束"
    },
    {
        "id": "ifeval_3",
        "instruction": "请用 Python 列表的形式输出 5 个编程语言名称，格式如 ['Python', 'Java', ...]。",
        "validator": "python_list",
        "min_items": 5,
        "category": "格式约束"
    },

    # === 内容约束 ===
    {
        "id": "ifeval_4",
        "instruction": "请写一段关于人工智能的介绍，必须包含「深度学习」和「自然语言处理」这两个关键词，不少于 80 字。",
        "validator": "keywords",
        "keywords": ["深度学习", "自然语言处理"],
        "min_length": 80,
        "category": "内容约束"
    },
    {
        "id": "ifeval_5",
        "instruction": "请介绍一种你喜欢的水果。要求：1) 不能使用「好吃」这个词 2) 至少写 3 句话 3) 要说明它的颜色和口感。",
        "validator": "forbidden_words",
        "forbidden": ["好吃"],
        "min_sentences": 3,
        "category": "内容约束"
    },
    {
        "id": "ifeval_6",
        "instruction": "请写一段关于环保的文字，必须包含以下所有词语：\n- 可持续发展\n- 碳排放\n- 绿色能源\n不少于 100 字。",
        "validator": "keywords",
        "keywords": ["可持续发展", "碳排放", "绿色能源"],
        "min_length": 100,
        "category": "内容约束"
    },

    # === 长度约束 ===
    {
        "id": "ifeval_7",
        "instruction": "请用恰好 50 个字介绍你自己，不多也不少。请在回答末尾标注实际字数。",
        "validator": "exact_length",
        "target_length": 50,
        "category": "长度约束"
    },
    {
        "id": "ifeval_8",
        "instruction": "请写一段不超过 80 字的短文，主题为「秋天的景色」。",
        "validator": "max_length",
        "max_length": 80,
        "category": "长度约束"
    },
    {
        "id": "ifeval_9",
        "instruction": "请用 3-5 句话简要说明如何学习一门外语。",
        "validator": "sentence_count",
        "min_sentences": 3,
        "max_sentences": 5,
        "category": "长度约束"
    },

    # === 综合性约束 ===
    {
        "id": "ifeval_10",
        "instruction": "请写一段产品介绍文案，要求：\n1) 格式：以「【产品名称】」开头\n2) 必须包含「创新」和「品质」两个词\n3) 不少于 60 字\n4) 结尾以「欢迎选购」结束",
        "validator": "composite",
        "prefix": "【产品名称】",
        "keywords": ["创新", "品质"],
        "min_length": 60,
        "suffix": "欢迎选购",
        "category": "综合性约束"
    },
    {
        "id": "ifeval_11",
        "instruction": "请生成一段对话，格式为：\n甲：...\n乙：...\n甲：...\n要求至少 3 轮对话，话题是关于周末计划。",
        "validator": "dialogue",
        "min_rounds": 3,
        "category": "格式约束"
    },
    {
        "id": "ifeval_12",
        "instruction": "请用分号分隔的格式写出 4 个你的兴趣爱好，例如：阅读；音乐；运动；旅行",
        "validator": "semicolon_list",
        "min_items": 4,
        "category": "格式约束"
    },
    {
        "id": "ifeval_13",
        "instruction": "请不要在回答中使用字母 'a'（不区分大小写），写一段约 30 字的自我介绍。",
        "validator": "no_letter_a",
        "min_length": 20,
        "category": "内容约束"
    },
    {
        "id": "ifeval_14",
        "instruction": "请写一段关于读书的好处的小作文。要求：\n1) 第一句必须为「读书的好处很多。」\n2) 全文至少 100 字\n3) 使用总分总结构\n4) 包含至少一个比喻句",
        "validator": "composite",
        "prefix": "读书的好处很多。",
        "min_length": 100,
        "keywords": ["像", "如", "好比", "仿佛"],
        "category": "综合性约束"
    },
]


class IFEvalBenchmark(BaseLLMBenchmark):
    """IFEval 指令遵循评测"""

    benchmark_id = "ifeval"
    benchmark_name = "IFEval"
    category = "指令遵循"

    def load_questions(self) -> list[dict]:
        """加载指令遵循题（15 题）"""
        questions = []
        for q in BUILTIN_QUESTIONS:
            q_copy = dict(q)
            q_copy["answer"] = "（需校验指令遵循度）"
            questions.append(q_copy)
        logger.info(f"IFEval 共加载 {len(questions)} 道题目")
        return questions

    def build_prompt(self, question: dict, few_shot: bool = True) -> list[dict]:
        """构建 prompt"""
        messages = [
            {
                "role": "system",
                "content": "你是一个指令遵循助手。请严格按照用户提出的要求完成写作任务，注意格式、长度和内容的所有约束。"
            }
        ]

        if few_shot and question.get("validator") == "json":
            messages.append({
                "role": "user",
                "content": "请用 JSON 格式输出，包含 name 和 age 两个字段。"
            })
            messages.append({
                "role": "assistant",
                "content": '{"name": "张三", "age": 25}'
            })

        messages.append({
            "role": "user",
            "content": question["instruction"]
        })
        return messages

    def extract_answer(self, text: str) -> str:
        """返回模型原始输出"""
        return text.strip()

    def check_answer(self, extracted: str, reference: str) -> bool:
        """根据验证器类型判断指令遵循度"""
        # 这个方法在子类中被重写，这里只是占位
        return True

    async def _evaluate_single(self, model, question: dict, idx: int) -> "QuestionResult":
        """重写单题评测以支持指令遵循度评分"""
        from core.schema import QuestionResult

        prompt = self.build_prompt(question)
        validator = question.get("validator", "")
        qid = question.get("id", str(idx))

        try:
            resp = await model.chat_with_usage(prompt)
            output = resp["content"]
            extracted = output.strip()

            # 按规则评分
            score, details = self._validate_response(extracted, question)

            prompt_text = "\n".join(m.get("content", "") for m in prompt if m.get("content"))

            return QuestionResult(
                question_id=qid,
                prompt=prompt_text,
                reference_answer=question.get("instruction", ""),
                model_output=output,
                extracted_answer=extracted,
                score=score,
                is_correct=score >= 0.7,
                latency_ms=resp["latency_ms"],
                category=question.get("category", "指令遵循"),
                metadata=details,
            )

        except Exception as e:
            logger.warning(f"指令 {qid} 评测失败: {e}")
            return QuestionResult(
                question_id=qid,
                prompt=str(question),
                reference_answer="",
                model_output="",
                extracted_answer="",
                score=0.0,
                is_correct=False,
                latency_ms=0.0,
                category=question.get("category", "指令遵循"),
                error=str(e),
            )

    def _validate_response(self, response: str, question: dict) -> tuple[float, dict]:
        """按规则验证指令遵循度，返回 (得分, 详情)"""
        validator = question.get("validator", "")
        passed = []
        failed = []

        if validator == "json":
            try:
                data = json.loads(response)
                keyword = question.get("keyword", "")
                if keyword and keyword in data:
                    passed.append("包含目标字段")
                elif keyword:
                    failed.append(f"缺少字段: {keyword}")
                else:
                    passed.append("有效 JSON")
            except json.JSONDecodeError:
                failed.append("无效 JSON 格式")

        elif validator == "prefix_suffix":
            prefix = question.get("prefix", "")
            suffix = question.get("suffix", "")
            if response.strip().startswith(prefix):
                passed.append("符合开头要求")
            else:
                failed.append(f"期望开头: {prefix}")
            if response.strip().endswith(suffix):
                passed.append("符合结尾要求")
            else:
                failed.append(f"期望结尾: {suffix}")
            if len(response) >= question.get("min_length", 0):
                passed.append("达到最小字数")
            else:
                failed.append(f"字数不足: {len(response)}/{question.get('min_length', 0)}")

        elif validator == "bullet_list":
            items = re.findall(r'^- .+', response, re.MULTILINE)
            min_items = question.get("min_items", 3)
            if len(items) >= min_items:
                passed.append(f"包含 {len(items)} 个列表项")
            else:
                failed.append(f"列表项不足: {len(items)}/{min_items}")

        elif validator == "python_list":
            list_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if list_match:
                items = re.findall(r'"([^"]*)"', list_match.group())
                if len(items) >= question.get("min_items", 5):
                    passed.append(f"包含 {len(items)} 项")
                else:
                    failed.append(f"项数不足: {len(items)}")
            else:
                failed.append("未找到 Python 列表格式")

        elif validator == "keywords":
            keywords = question.get("keywords", [])
            min_len = question.get("min_length", 0)
            missing = [kw for kw in keywords if kw not in response]
            if not missing:
                passed.append("包含所有关键词")
            else:
                failed.append(f"缺少关键词: {missing}")
            if len(response) >= min_len:
                passed.append("达到最小字数")
            else:
                failed.append(f"字数不足: {len(response)}/{min_len}")

        elif validator == "forbidden_words":
            forbidden = question.get("forbidden", [])
            found = [w for w in forbidden if w in response]
            if not found:
                passed.append("未使用禁用词")
            else:
                failed.append(f"使用了禁用词: {found}")
            sentences = [s for s in re.split(r'[。！？!?]', response) if s.strip()]
            if len(sentences) >= question.get("min_sentences", 3):
                passed.append("满足最少句子数")
            else:
                failed.append(f"句子数不足: {len(sentences)}")

        elif validator == "exact_length":
            target = question.get("target_length", 50)
            actual_len = len(response)
            diff = abs(actual_len - target)
            if diff <= 5:
                passed.append(f"字数接近要求: {actual_len}/{target}")
            else:
                failed.append(f"字数偏差大: {actual_len}/{target}")

        elif validator == "max_length":
            max_len = question.get("max_length", 80)
            if len(response) <= max_len:
                passed.append(f"不超字数: {len(response)}/{max_len}")
            else:
                failed.append(f"超字数: {len(response)}/{max_len}")

        elif validator == "sentence_count":
            sentences = [s for s in re.split(r'[。！？!?]', response) if s.strip()]
            min_s = question.get("min_sentences", 3)
            max_s = question.get("max_sentences", 5)
            if min_s <= len(sentences) <= max_s:
                passed.append(f"句子数合规: {len(sentences)}")
            else:
                failed.append(f"句子数: {len(sentences)} (需 {min_s}-{max_s})")

        elif validator == "composite":
            checks = []
            if "prefix" in question:
                if response.strip().startswith(question["prefix"]):
                    passed.append("符合开头要求")
                else:
                    failed.append(f"期望开头: {question['prefix']}")
            if "suffix" in question:
                if response.strip().endswith(question["suffix"]):
                    passed.append("符合结尾要求")
                else:
                    failed.append(f"期望结尾: {question['suffix']}")
            if "keywords" in question:
                missing = [kw for kw in question["keywords"] if kw not in response]
                if not missing:
                    passed.append("包含所有关键词")
                else:
                    failed.append(f"缺少: {missing}")
            if "min_length" in question:
                if len(response) >= question["min_length"]:
                    passed.append("达到最小字数")
                else:
                    failed.append(f"字数不足")

        elif validator == "dialogue":
            lines = response.strip().split('\n')
            dialogue_lines = [l for l in lines if re.match(r'[甲乙]：', l)]
            rounds = len(dialogue_lines)
            if rounds >= question.get("min_rounds", 3) * 2:
                passed.append(f"对话轮次充足: {rounds} 句")
            else:
                failed.append(f"对话轮次不足: {rounds} 句")

        elif validator == "semicolon_list":
            items = [x.strip() for x in response.split('；') if x.strip()]
            if len(items) >= question.get("min_items", 4):
                passed.append(f"包含 {len(items)} 项")
            else:
                failed.append(f"项数不足: {len(items)}")

        elif validator == "no_letter_a":
            if 'a' not in response.lower():
                passed.append("未包含字母 a")
            else:
                failed.append("包含字母 a")
            if len(response) >= question.get("min_length", 20):
                passed.append("达到最小字数")
            else:
                failed.append("字数不足")

        # 计算得分
        total = len(passed) + len(failed)
        score = len(passed) / total if total > 0 else 0.0

        return score, {"passed": passed, "failed": failed, "score": round(score, 2)}


benchmark = IFEvalBenchmark()

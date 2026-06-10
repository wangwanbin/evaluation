"""LLM-as-Judge — 使用独立 Judge 模型评估回答质量。

支持场景：
- 开放式问答的质量评分
- RAG 回答的忠实度/正确性评分
- Agent 任务完成度判定
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from core.config import load_config
from core.model_adapter import ModelAdapter
from core.schema import ModelConfig

logger = logging.getLogger("eval.llm.judge")


class LLMJudge:
    """LLM-as-Judge 评估器"""

    def __init__(self, judge_config: Optional[ModelConfig] = None):
        self.config = load_config()
        self.judge_config = judge_config or self.config.effective_judge_config
        self.model = ModelAdapter(self.judge_config)

    async def score_answer(
        self,
        question: str,
        reference_answer: str,
        model_answer: str,
        rubric: Optional[str] = None,
    ) -> dict:
        """评分单条回答 vs 参考答案

        Returns:
            {"score": float, "reasoning": str, "passed": bool}
        """
        system_prompt = """你是一个公正的答案评分员（Judge）。你的任务是评估模型回答与标准答案的匹配程度。

评分标准：
- 5分：完全正确，与标准答案一致，包含所有关键信息
- 4分：基本正确，包含了核心信息，但有小差异
- 3分：部分正确，包含了部分关键信息
- 2分：大部分错误，仅包含少量正确信息
- 1分：完全错误或无关

请输出JSON格式：
{"score": <1-5的分数>, "reasoning": "<评分理由>"}"""

        user_prompt = f"""问题：{question}

标准答案：{reference_answer}

模型回答：{model_answer}

{rubric or '请根据标准答案评估模型回答的正确性和完整性。'}
请输出JSON格式的评分结果。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            resp = await self.model.chat_with_usage(messages, temperature=0.0, max_tokens=512)
            content = resp["content"].strip()

            # 尝试解析 JSON
            score, reasoning = self._parse_judge_response(content)
            return {
                "score": score,
                "reasoning": reasoning,
                "passed": score >= 4,
                "latency_ms": resp["latency_ms"],
                "tokens": resp["total_tokens"],
            }
        except Exception as e:
            logger.warning(f"Judge 评分失败: {e}")
            return {"score": 0, "reasoning": f"评分失败: {e}", "passed": False, "latency_ms": 0, "tokens": 0}

    async def batch_score(
        self,
        items: list[dict],
        concurrency: int = 5,
    ) -> list[dict]:
        """批量评分"""
        import asyncio
        sem = asyncio.Semaphore(concurrency)

        async def score_one(item: dict) -> dict:
            async with sem:
                result = await self.score_answer(
                    question=item.get("question", ""),
                    reference_answer=item.get("reference", ""),
                    model_answer=item.get("answer", ""),
                    rubric=item.get("rubric"),
                )
                result["id"] = item.get("id", "")
                return result

        tasks = [score_one(item) for item in items]
        return await asyncio.gather(*tasks)

    def _parse_judge_response(self, content: str) -> tuple[float, str]:
        """解析 Judge 的 JSON 输出"""
        # 尝试提取 JSON 块
        json_match = re.search(r'\{(?:[^{}]|"[^"]*")*\}', content)
        if json_match:
            try:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0))
                reasoning = data.get("reasoning", content)[:500]
                return min(max(score, 1), 5), reasoning
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # 回退：提取数字评分
        score_match = re.search(r'(?:分数|得分|score)[：:\s]*(\d+(?:\.?\d+)?)', content, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                return min(max(score, 1), 5), content[:200]
            except ValueError:
                pass

        return 3.0, content[:200]  # 默认中等分

    async def close(self):
        await self.model.close()

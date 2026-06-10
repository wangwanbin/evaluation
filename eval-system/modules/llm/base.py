"""LLM 评测基类 — 提供 Prompt 构建、答案提取、并发推理等公共能力。"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from core.config import load_config
from core.model_adapter import ModelAdapter
from core.schema import BenchmarkResult, ModelConfig, QuestionResult

logger = logging.getLogger("eval.llm")


class BaseLLMBenchmark(ABC):
    """LLM Benchmark 基类"""

    def __init__(self):
        self.config = load_config()

    @abstractmethod
    def load_questions(self) -> list[dict]:
        """加载评测题目列表"""
        ...

    @abstractmethod
    def build_prompt(self, question: dict, few_shot: bool = True) -> list[dict]:
        """构建 messages prompt"""
        ...

    @abstractmethod
    def extract_answer(self, text: str) -> str:
        """从模型输出中提取答案"""
        ...

    @abstractmethod
    def check_answer(self, extracted: str, reference: str) -> bool:
        """判断提取答案是否匹配参考答案"""
        ...

    def get_few_shot_examples(self) -> list[dict]:
        """返回 few-shot 示例（子类可覆盖）"""
        return []

    async def run(
        self,
        model_config: ModelConfig,
        params: Optional[dict] = None,
    ) -> BenchmarkResult:
        """执行完整的 benchmark 评测流程"""
        params = params or {}
        concurrency = params.get("concurrency", self.config.concurrency)

        questions = self.load_questions()
        logger.info(f"加载 {len(questions)} 道题目")

        model = ModelAdapter(model_config, timeout=self.config.timeout)
        benchmark_id = getattr(self, "benchmark_id", "unknown")
        benchmark_name = getattr(self, "benchmark_name", benchmark_id)

        result = BenchmarkResult(
            benchmark_id=benchmark_id,
            benchmark_name=benchmark_name,
            category=getattr(self, "category", ""),
            total_questions=len(questions),
            correct_count=0,
            score=0.0,
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def evaluate_one(q: dict, idx: int) -> QuestionResult:
            async with semaphore:
                return await self._evaluate_single(model, q, idx)

        tasks = [evaluate_one(q, i) for i, q in enumerate(questions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        correct = 0
        errors = 0
        total_latency = 0.0

        for r in results:
            if isinstance(r, Exception):
                logger.error(f"题目评测异常: {r}")
                errors += 1
                continue
            if isinstance(r, QuestionResult):
                result.questions.append(r)
                if r.is_correct:
                    correct += 1
                if r.error:
                    errors += 1
                total_latency += r.latency_ms

        result.correct_count = correct
        result.score = correct / len(questions) if questions else 0.0
        result.error_count = errors
        result.latency_avg_ms = round(total_latency / len(result.questions), 2) if result.questions else 0.0

        await model.close()
        return result

    async def _evaluate_single(
        self,
        model: ModelAdapter,
        question: dict,
        idx: int,
    ) -> QuestionResult:
        """评测单道题目"""
        prompt = self.build_prompt(question)
        reference = question.get("answer", question.get("reference", ""))

        qid = question.get("id", str(idx))
        category = question.get("category", "")

        try:
            resp = await model.chat_with_usage(prompt)

            output = resp["content"]
            extracted = self.extract_answer(output)
            is_correct = self.check_answer(extracted, reference)

            prompt_text = "\n".join(m.get("content", "") for m in prompt if m.get("content"))

            return QuestionResult(
                question_id=qid,
                prompt=prompt_text,
                reference_answer=str(reference),
                model_output=output,
                extracted_answer=extracted,
                score=1.0 if is_correct else 0.0,
                is_correct=is_correct,
                latency_ms=resp["latency_ms"],
                category=category,
            )

        except Exception as e:
            logger.warning(f"题目 {qid} 评测失败: {e}")
            return QuestionResult(
                question_id=qid,
                prompt=str(question),
                reference_answer=str(reference),
                model_output="",
                extracted_answer="",
                score=0.0,
                is_correct=False,
                latency_ms=0.0,
                category=category,
                error=str(e),
            )

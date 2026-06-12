"""评估系统的核心数据模型和接口定义。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class EvalStatus(Enum):
    """评测状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # 部分完成


@dataclass
class ModelConfig:
    """模型配置"""
    api_base: str
    api_key: str = ""
    model_name: str = "default"
    model_type: str = "openai"  # openai | ollama
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.api_key:
            d["api_key"] = self.api_key[:8] + "..."  # 脱敏
        return d


@dataclass
class MetricDef:
    """指标定义"""
    key: str
    name: str
    description: str
    higher_is_better: bool = True
    unit: str = "%"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QuestionResult:
    """单题评测结果"""
    question_id: str
    prompt: str
    reference_answer: str
    model_output: str
    extracted_answer: str
    score: float
    is_correct: bool
    latency_ms: float
    category: str = ""
    metadata: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """单个 Benchmark 的评测结果"""
    benchmark_id: str
    benchmark_name: str
    category: str  # 所属大类
    total_questions: int
    correct_count: int
    score: float  # 0-100
    questions: list[QuestionResult] = field(default_factory=list)
    latency_avg_ms: float = 0.0
    error_count: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def score_pct(self) -> float:
        return round(self.score * 100, 2)

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_name": self.benchmark_name,
            "category": self.category,
            "total_questions": self.total_questions,
            "correct_count": self.correct_count,
            "score": self.score,
            "score_pct": self.score_pct,
            "latency_avg_ms": self.latency_avg_ms,
            "error_count": self.error_count,
            "question_count": len(self.questions),
            "metadata": self.metadata,
        }


@dataclass
class EvalResult:
    """一次完整的评测运行结果"""
    run_id: str
    module: str  # "llm", "rag", "agent"
    model_config: ModelConfig
    judge_config: Optional[ModelConfig] = None
    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    status: EvalStatus = EvalStatus.COMPLETED
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    total_duration_s: float = 0.0
    total_questions: int = 0
    total_correct: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def overall_score(self) -> float:
        if not self.benchmarks:
            return 0.0
        weighted = sum(b.score * b.total_questions for b in self.benchmarks)
        total = sum(b.total_questions for b in self.benchmarks)
        return round(weighted / total, 4) if total > 0 else 0.0

    @property
    def overall_score_pct(self) -> float:
        return round(self.overall_score * 100, 2)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "module": self.module,
            "model_config": self.model_config.to_dict(),
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_duration_s": self.total_duration_s,
            "total_questions": self.total_questions,
            "total_correct": self.total_correct,
            "overall_score": self.overall_score,
            "overall_score_pct": self.overall_score_pct,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class Benchmark:
    """Benchmark 定义"""
    id: str
    name: str
    description: str
    category: str
    num_questions: int = 0
    few_shot_count: int = 0
    metadata: dict = field(default_factory=dict)


class EvalModule(ABC):
    """评估模块基类 — 所有评估模块必须实现此接口"""

    name: str = ""
    version: str = ""
    description: str = ""

    @abstractmethod
    def list_benchmarks(self) -> list[Benchmark]:
        """返回此模块支持的所有 benchmark"""
        ...

    @abstractmethod
    async def run_benchmark(
        self,
        benchmark_id: str,
        model_config: ModelConfig,
        params: dict,
    ) -> BenchmarkResult:
        """运行单个 benchmark 评测"""
        ...

    def get_metrics(self) -> list[MetricDef]:
        """返回此模块使用的评测指标"""
        return []

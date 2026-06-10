"""评测执行器 — 编排评测流程、控制并发、管理超时与重试。"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

from core.config import load_config
from core.model_adapter import ModelAdapter
from core.registry import ModuleRegistry
from core.schema import (
    EvalResult,
    EvalStatus,
    ModelConfig,
)

logger = logging.getLogger("eval.runner")


class EvalRunner:
    """评测执行器"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.config = load_config()
        self.model = ModelAdapter(model_config, timeout=self.config.timeout)

    async def run_benchmark(
        self,
        module_name: str,
        benchmark_id: str,
        params: Optional[dict] = None,
    ) -> EvalResult:
        """执行单个 benchmark 评测"""
        params = params or {}

        module = ModuleRegistry.get_module(module_name)
        if module is None:
            raise ValueError(f"未知模块: {module_name}，可用模块: {ModuleRegistry.list_module_names()}")

        # 验证 benchmark 是否存在
        benchmarks = module.list_benchmarks()
        bench_map = {b.id: b for b in benchmarks}
        if benchmark_id not in bench_map:
            available = [b.id for b in benchmarks]
            raise ValueError(f"模块 '{module_name}' 中未知 benchmark: {benchmark_id}，可用: {available}")

        run_id = f"{module_name}_{benchmark_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = EvalResult(
            run_id=run_id,
            module=module_name,
            model_config=self.model_config,
            status=EvalStatus.RUNNING,
            started_at=datetime.now().isoformat(),
            metadata={"benchmark_id": benchmark_id, **params},
        )

        logger.info(f"开始评测：[{module_name}] {benchmark_id}")
        logger.info(f"模型：{self.model_config.model_name} @ {self.model_config.api_base}")

        start_time = time.time()

        try:
            # 模块执行 benchmark
            bench_result = await module.run_benchmark(
                benchmark_id=benchmark_id,
                model_config=self.model_config,
                params=params,
            )
            result.benchmarks.append(bench_result)
            result.total_questions = bench_result.total_questions
            result.total_correct = bench_result.correct_count
            result.status = EvalStatus.COMPLETED

        except Exception as e:
            logger.error(f"评测失败: {e}", exc_info=True)
            result.status = EvalStatus.FAILED
            result.metadata["error"] = str(e)

        result.completed_at = datetime.now().isoformat()
        result.total_duration_s = round(time.time() - start_time, 2)

        logger.info(f"评测完成: {run_id}")
        logger.info(f"耗时: {result.total_duration_s}s | 得分: {result.overall_score_pct}%")

        return result

    async def run_multiple_benchmarks(
        self,
        module_name: str,
        benchmark_ids: list[str],
        params: Optional[dict] = None,
    ) -> EvalResult:
        """并发执行多个 benchmark 评测"""
        params = params or {}

        module = ModuleRegistry.get_module(module_name)
        if module is None:
            raise ValueError(f"未知模块: {module_name}")

        run_id = f"{module_name}_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = EvalResult(
            run_id=run_id,
            module=module_name,
            model_config=self.model_config,
            status=EvalStatus.RUNNING,
            started_at=datetime.now().isoformat(),
            metadata={"benchmarks": benchmark_ids, **params},
        )

        start_time = time.time()
        semaphore = asyncio.Semaphore(self.config.concurrency)

        async def run_one(bench_id: str) -> None:
            async with semaphore:
                try:
                    bench_result = await module.run_benchmark(
                        benchmark_id=bench_id,
                        model_config=self.model_config,
                        params=params,
                    )
                    result.benchmarks.append(bench_result)
                except Exception as e:
                    logger.error(f"Benchmark {bench_id} 失败: {e}")
                    result.metadata.setdefault("errors", {})[bench_id] = str(e)

        tasks = [run_one(bid) for bid in benchmark_ids]
        await asyncio.gather(*tasks)

        result.total_questions = sum(b.total_questions for b in result.benchmarks)
        result.total_correct = sum(b.correct_count for b in result.benchmarks)
        result.completed_at = datetime.now().isoformat()
        result.total_duration_s = round(time.time() - start_time, 2)

        # 如果有部分失败
        error_count = len(result.metadata.get("errors", {}))
        if error_count > 0 and result.benchmarks:
            result.status = EvalStatus.PARTIAL
        elif error_count > 0:
            result.status = EvalStatus.FAILED
        else:
            result.status = EvalStatus.COMPLETED

        return result

    async def close(self):
        await self.model.close()

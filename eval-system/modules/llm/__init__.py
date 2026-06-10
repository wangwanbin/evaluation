"""大模型基础能力评估模块 (LLM)。

覆盖设计大纲全部 8 大评估维度：
1. 知识储备      → MMLU, C-Eval
2. 数学推理      → GSM8K
3. 逻辑推理      → HellaSwag, PIQA
4. 代码能力      → HumanEval, MBPP
5. 指令遵循      → IFEval
6. 长文本理解    → Needle-in-a-Haystack
7. 多语言能力    → (预留)
8. 安全性/对齐   → (预留)
"""

from core.schema import EvalModule, Benchmark, MetricDef


class LlmEvalModule(EvalModule):
    """LLM 基础能力评估模块"""

    name = "llm"
    version = "1.0.0"
    description = "大模型基础能力评估（知识、推理、代码、指令遵循、长文本等 8 大维度）"

    def __init__(self):
        self._benchmarks: dict[str, Benchmark] = {}
        self._register_benchmarks()

    def _register_benchmarks(self) -> None:
        """注册全部支持的 benchmark"""
        bench_defs = [
            # 1. 知识储备
            Benchmark(
                id="mmlu",
                name="MMLU",
                description="大规模多任务语言理解 — 57 个学科的知识储备评测（内置 15 学科 × 5 题 = 75 题）",
                category="知识储备",
                num_questions=75,
                few_shot_count=5,
                metadata={"subjects": 15},
            ),
            Benchmark(
                id="ceval",
                name="C-Eval",
                description="中文知识评测 — 涵盖人文、社科、理工、医学等学科（内置 30 题）",
                category="知识储备",
                num_questions=30,
                few_shot_count=2,
                metadata={"subjects": 10},
            ),
            # 2. 数学推理
            Benchmark(
                id="gsm8k",
                name="GSM8K",
                description="小学数学应用题 — 多步推理，加减乘除综合运算（内置 40 题）",
                category="数学推理",
                num_questions=40,
                few_shot_count=4,
                metadata={"difficulty": "grade_school"},
            ),
            # 3. 常识/物理推理
            Benchmark(
                id="hellaswag",
                name="HellaSwag",
                description="常识推理 — 从日常场景中选择最合理的后续（内置 15 题）",
                category="常识推理",
                num_questions=15,
                few_shot_count=0,
            ),
            Benchmark(
                id="piqa",
                name="PIQA",
                description="物理常识推理 — 判断完成物理目标的最合理方式（内置 15 题）",
                category="常识推理",
                num_questions=15,
                few_shot_count=0,
            ),
            # 4. 代码能力
            Benchmark(
                id="humaneval",
                name="HumanEval",
                description="Python 函数补全 — 根据签名和文档字符串实现函数（内置 10 题）",
                category="代码能力",
                num_questions=10,
                few_shot_count=1,
            ),
            Benchmark(
                id="mbpp",
                name="MBPP",
                description="基础 Python 编程 — 字符串、数据结构、算法等编程任务（内置 10 题）",
                category="代码能力",
                num_questions=10,
                few_shot_count=1,
            ),
            # 5. 指令遵循
            Benchmark(
                id="ifeval",
                name="IFEval",
                description="指令遵循 — 格式/长度/内容约束的综合评测（内置 15 题）",
                category="指令遵循",
                num_questions=15,
                few_shot_count=1,
                metadata={"validator_types": 10},
            ),
            # 6. 长文本理解
            Benchmark(
                id="needle",
                name="Needle-in-a-Haystack",
                description="大海捞针 — 长文本中检索特定信息（动态生成 15 题）",
                category="长文本理解",
                num_questions=15,
                few_shot_count=0,
                metadata={"avg_length": 3000},
            ),
        ]
        for b in bench_defs:
            self._benchmarks[b.id] = b

    def list_benchmarks(self) -> list[Benchmark]:
        return list(self._benchmarks.values())

    def get_metrics(self) -> list[MetricDef]:
        return [
            MetricDef(key="accuracy", name="准确率", description="正确回答占总数的比例", higher_is_better=True, unit="%"),
            MetricDef(key="latency", name="平均延迟", description="单题平均推理耗时", higher_is_better=False, unit="ms"),
        ]

    async def run_benchmark(self, benchmark_id: str, model_config, params: dict):
        """运行指定 benchmark 评测"""
        if benchmark_id not in self._benchmarks:
            raise ValueError(f"未知 benchmark: {benchmark_id}，可用: {list(self._benchmarks.keys())}")

        # 动态导入对应的 benchmark 实现
        IMPORTS = {
            "mmlu": ("modules.llm.benchmarks.mmlu", "MMLUBenchmark"),
            "ceval": ("modules.llm.benchmarks.ceval", "CEvalBenchmark"),
            "gsm8k": ("modules.llm.benchmarks.gsm8k", "GSM8KBenchmark"),
            "hellaswag": ("modules.llm.benchmarks.hellaswag", "HellaSwagBenchmark"),
            "piqa": ("modules.llm.benchmarks.piqa", "PIQABenchmark"),
            "humaneval": ("modules.llm.benchmarks.humaneval", "HumanEvalBenchmark"),
            "mbpp": ("modules.llm.benchmarks.mbpp", "MBPPBenchmark"),
            "ifeval": ("modules.llm.benchmarks.ifeval", "IFEvalBenchmark"),
            "needle": ("modules.llm.benchmarks.needle", "NeedleInHaystackBenchmark"),
        }
        mod_path, cls_name = IMPORTS.get(benchmark_id, ("", ""))
        if not mod_path:
            raise NotImplementedError(f"Benchmark {benchmark_id} 尚未实现")

        import importlib
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)
        return await cls().run(model_config, params)


# 模块实例 — 被 ModuleRegistry 自动发现
module = LlmEvalModule()

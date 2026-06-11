# AI 能力评估系统（Evaluation Harness）

<p align="center">
  <strong>大模型评测 · RAG 评测 · Agent 评测 · LLM Evaluation Harness</strong>
  <br>
  MMLU · GSM8K · C-Eval · HellaSwag · HumanEval · MBPP · IFEval · PIQA · Needle-in-Haystack
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
  <img src="https://img.shields.io/badge/benchmarks-9-orange" alt="9 Benchmarks">
  <img src="https://img.shields.io/badge/questions-209-brightgreen" alt="209 Questions">
  <img src="https://img.shields.io/badge/LLM-RAG-Agent-purple" alt="LLM RAG Agent">
  <img src="https://img.shields.io/badge/OpenAI_API-compatible-success" alt="OpenAI API Compatible">
  <img src="https://img.shields.io/badge/离线部署-无需外网-blueviolet" alt="离线部署">
  <img src="https://img.shields.io/badge/评测-模型能力评估-red" alt="评测">
</p>

> **大模型评测工具 · LLM Evaluation Harness · 模型能力评估框架**
>
> 这是一个评测夹具（Evaluation Harness）—— 系统本身不运行模型、不提供推理服务。
> 被评测模型通过 **OpenAI 兼容 API** 接入（vLLM / 云端 API / 任意兼容端点）。
> 系统的职责：发 Prompt → 收 Response → 评分 → 出报告。
>
> 适用场景：**大模型选型对比** · **模型微调前后评估** · **RAG 系统检索质量评测** · **Agent 能力验证** ·
> **Benchmark 自动化测试** · **模型能力排行榜** · **AI 安全与对齐评测**

---

## 📦 项目结构

```
eval-system/
├── main.py               # 入口
├── .env                  # 配置（唯一必填：vLLM 地址）
│
├── core/                 # 核心框架
│   ├── config.py         # 配置管理
│   ├── schema.py         # 数据模型 + EvalModule 抽象接口
│   ├── model_adapter.py  # OpenAI 兼容 API 适配器
│   ├── registry.py       # 模块自动发现与注册
│   └── runner.py         # 评测执行器（并发/超时/重试）
│
├── modules/              # 评估模块（插件式）
│   └── llm/              # 大模型基础能力评估
│       ├── base.py       # 基准测试基类
│       ├── judge.py      # LLM-as-Judge 评分器
│       └── benchmarks/   # 各 Benchmark 实现
│           ├── mmlu.py       # 多学科知识（75 题）
│           ├── ceval.py      # 中文知识（30 题）
│           ├── gsm8k.py      # 数学推理（40 题）
│           ├── hellaswag.py  # 常识推理（15 题）
│           ├── piqa.py       # 物理常识（15 题）
│           ├── humaneval.py  # 代码补全（10 题）
│           ├── mbpp.py       # 基础编程（10 题）
│           ├── ifeval.py     # 指令遵循（15 题）
│           └── needle.py     # 长文本检索（15 题）
│
├── storage/              # 数据存储
│   ├── database.py       # SQLite（WAL 模式）
│   └── backup.py         # 自动备份与恢复
│
├── reports/              # 报告生成
│   ├── html.py           # HTML 报告（含雷达图）
│   └── markdown.py       # Markdown 报告
│
├── cli/main.py           # 命令行入口
├── docker/               # Docker 部署
│   ├── Dockerfile.app
│   ├── docker-compose.yml
│   └── images/           # 预构建镜像（.tar）
│
├── datasets/             # 评测数据集（可选，扩展用）
├── results/              # 评测结果输出（自动生成）
├── start.sh              # Linux 一键启动
├── start.bat             # Windows 一键启动
└── setup.py              # Python 包安装
```

---

## 🚀 快速开始

### 方式一：本地 Python 运行（推荐开发调试）

#### 1. 安装依赖

```bash
cd eval-system
pip install -r requirements.txt
# 或直接安装包
pip install -e .
```

#### 2. 配置 .env

```bash
cp .env.example .env
# 编辑 .env，至少修改：
#   EVAL_MODEL_API_BASE=http://你的vLLM地址:8000/v1
```

#### 3. 运行评测

```bash
# 检查环境
python main.py check

# 列出可用评测项
python main.py dataset list

# 测试模型连通性
python main.py test

# 运行单个 Benchmark
python main.py run llm -b mmlu

# 运行多个 Benchmark
python main.py run llm -b mmlu,gsm8k,ceval

# 运行全部评测
python main.py all
```

评测完成后，在 `results/<run_id>/` 下查看报告：

```bash
# 浏览器打开 HTML 报告
start results/<run_id>/report.html

# 终端预览 Markdown 报告
cat results/<run_id>/report.md
```

### 方式二：Docker 部署（推荐生产使用）

#### 1. 构建镜像

```bash
cd eval-system
docker build -f docker/Dockerfile.app -t eval-app:v1.0.0 .
```

#### 2. 配置并启动

```bash
# 编辑 .env 设置 vLLM 地址
cp .env.example .env
# vi .env → 修改 EVAL_MODEL_API_BASE

# 一键启动
./start.sh
```

#### 3. 执行评测

```bash
docker exec -it eval-app python main.py run llm -b mmlu
```

### 方式三：从移动硬盘部署（便携方案）

将 `eval-system/` 整个目录拷贝到目标机器的 Docker 环境中：

```bash
# 1. 拷贝
cp -r /移动硬盘/eval-system ~/

# 2. 配置
cd ~/eval-system
cp .env.example .env
# 编辑: EVAL_MODEL_API_BASE=http://你的vLLM地址:8000/v1

# 3. 启动
./start.sh

# 4. 评测
docker exec -it eval-app python main.py run llm -b mmlu
```

> 目标机器只需安装 Docker，无需 Python、pip 或任何依赖库。

---

## ⚙️ 配置说明

系统只通过 `.env` 配置，位于项目根目录：

```bash
# ===== 必需：被评测模型的 API 地址 =====
EVAL_MODEL_API_BASE=http://localhost:8000/v1    # 必填！vLLM 地址
EVAL_MODEL_API_KEY=                              # API Key（本地留空）
EVAL_MODEL_NAME=qwen2.5-72b                     # 模型名称

# ===== 评测参数（可选） =====
EVAL_CONCURRENCY=10                              # 并发数，默认 10
EVAL_TIMEOUT=120                                 # 单题超时秒数

# ===== Judge 模型（可选，默认复用评测模型） =====
JUDGE_MODEL_API_BASE=
JUDGE_MODEL_API_KEY=
JUDGE_MODEL_NAME=

# ===== 存储路径（可选，默认项目 results/ 目录） =====
# EVAL_DB_PATH=results/eval.db
# EVAL_RESULTS_DIR=results
# EVAL_REPORTS_DIR=results/reports

# ===== 日志 =====
LOG_LEVEL=INFO
```

---

## 📊 评测维度与 Benchmark

| 维度 | Benchmark | 题数 | 说明 |
|------|-----------|:----:|------|
| 知识储备 | MMLU | 75 | 57 学科多任务语言理解 |
| 知识储备 | C-Eval | 30 | 中文全科知识评测 |
| 数学推理 | GSM8K | 40 | 多步数学应用题 |
| 常识推理 | HellaSwag | 15 | 日常场景推理 |
| 物理常识 | PIQA | 15 | 物理交互推理 |
| 代码能力 | HumanEval | 10 | Python 函数补全 |
| 代码能力 | MBPP | 10 | 基础编程任务 |
| 指令遵循 | IFEval | 15 | 格式/长度/内容约束 |
| 长文本理解 | Needle-in-Haystack | 15 | 长文本信息检索 |
| **总计** | **9 个** | **209** | **覆盖 6 大维度** |

> 所有数据内置在 Python 文件中，无需下载，开箱即用。

---

## 💻 CLI 命令参考

```bash
# ===== 系统管理 =====
python main.py init               # 初始化环境
python main.py check              # 环境完整性检查
python main.py logs               # 查看日志

# ===== 模型连通性 =====
python main.py test               # 测试模型可达

# ===== 运行评测 =====
python main.py run llm -b mmlu               # 单个评测项
python main.py run llm -b mmlu,gsm8k         # 多个评测项
python main.py run llm                       # 列出评测项
python main.py all                           # 全量评测

# ===== 结果分析 =====
python main.py report <run_id>               # 查看报告
python main.py compare <id1> <id2>          # 对比两次评测
python main.py leaderboard <benchmark>       # 排行榜

# ===== 数据集管理 =====
python main.py dataset list                  # 列出所有评测项
python main.py dataset import <path>         # 导入自定义数据
python main.py dataset info <name>           # 查看数据统计
```

---

## 🏗️ 架构设计

### 插件化模块

每个评估模块只需继承 `EvalModule` 抽象类即可被自动发现：

```python
from core.schema import EvalModule, Benchmark

class MyModule(EvalModule):
    name = "mymodule"
    description = "我的评测模块"

    def list_benchmarks(self) -> list[Benchmark]: ...
    async def run_benchmark(self, benchmark_id, model_config, params): ...
```

在 `modules/` 下创建目录 + 实现接口即可注册，无需修改核心代码。

### 执行流程

```
用户输入：模型配置 + 评测项选择
    ↓
Step 1: 加载数据集 & 构建 Prompt
    ↓
Step 2: 并发调用模型推理（asyncio + aiohttp）
    ↓
Step 3: 答案提取 & 评分（规则匹配 + LLM-as-Judge）
    ↓
Step 4: 指标汇总 → 生成 EvalResult
    ↓
输出：HTML 报告 + Markdown 报告 + JSON 数据
```

### 存储设计

- **SQLite（WAL 模式）**: 评测元数据、排行榜、对比数据。零配置、自动备份、崩溃恢复
- **JSON 文件**: 逐题完整结果的原始数据，方便二次分析
- **HTML + Markdown**: 双份报告，可视化 + 终端预览

### 模型接入

```
┌──────────────────┐     OpenAI API      ┌──────────────────┐
│   评测系统         │ ◄─────────────────► │   vLLM / 云端 API  │
│   · 发 Prompt      │   /v1/chat/         │                  │
│   · 收 Response    │   completions       │                  │
│   · 评分 · 出报告  │                     │                  │
└──────────────────┘                     └──────────────────┘
```

---

## 🔧 常见问题

| 问题 | 排查 |
|------|------|
| ❌ `name 're' is not defined` | 检查 `import re` |
| ❌ 模型不可达 | `python main.py test` 检查连通性 |
| ❌ 得分全是 0% | 模型输出格式异常，检查 `.env` 模型名称 |
| ❌ OOM | 降低 `.env` 中 `EVAL_CONCURRENCY=3` |
| ❌ 非 Windows 系统启动报错 | 用 `python main.py` 代替 `start.bat` |
| ❌ 数据库损坏 | 系统会自动从备份恢复 |

> **所有评测数据内置，无需下载数据集。** 如需扩展，把 JSON 文件放入 `datasets/llm/` 对应目录即可自动加载。

---

## 📝 扩展指南

### 添加新 Benchmark

在 `modules/llm/benchmarks/` 下新建文件，继承 `BaseLLMBenchmark`：

```python
from modules.llm.base import BaseLLMBenchmark

class MyBenchmark(BaseLLMBenchmark):
    benchmark_id = "mybench"
    benchmark_name = "My Benchmark"
    category = "自定义类别"

    def load_questions(self) -> list[dict]:
        return [{"id": "q1", "question": "...", "answer": "...", ...}]

    def build_prompt(self, question: dict, few_shot=True) -> list[dict]:
        ...

    def extract_answer(self, text: str) -> str:
        ...

    def check_answer(self, extracted: str, reference: str) -> bool:
        ...

benchmark = MyBenchmark()
```

然后在 `modules/llm/__init__.py` 的 `_register_benchmarks()` 中添加注册和导入映射即可。

### 添加全新评估模块

```bash
mkdir modules/my_module
# 创建 modules/my_module/__init__.py，实现 EvalModule 接口
```

系统启动时自动发现。

---

## 📄 报告输出

每次评测自动生成三份文件：

```
results/<run_id>/
├── report.html    # 浏览器打开，含雷达图 + 柱状图
├── report.md      # 终端快速预览
└── data.json      # 结构化原始数据，方便二次分析
```

历史数据可通过 CLI 查询：

```bash
python main.py leaderboard mmlu       # 查看 MMLU 排行榜
python main.py compare <id1> <id2>    # 对比两次评测
```

---

## 📜 技术栈

| 层面 | 方案 |
|------|------|
| 语言 | Python 3.11+ |
| 模型接入 | OpenAI 兼容 API（HTTP） |
| 并发 | asyncio + aiohttp |
| 存储 | SQLite（WAL 模式）+ JSON |
| 报告 | matplotlib + HTML + Markdown |
| 部署 | Docker / 本地 Python 均可 |
| 插件 | ABC 自动发现注册机制 |

---

## 🔍 搜索关键词

本项目可通过以下关键词被搜索到：

**中文关键词：**
`大模型评测` `大模型评估` `LLM评测` `模型能力评估` `模型评测框架` `AI评测系统`
`RAG评估` `RAG评测` `检索增强生成评测` `Agent评测` `Agent评估` `智能体评测`
`Benchmark测试` `基准测试` `模型排行榜` `模型对比` `模型选型` `模型能力测试`
`MMLU评测` `GSM8K评测` `C-Eval评测` `HumanEval评测` `IFEval评测`
`大模型离线评测` `无需外网` `Docker部署` `LLM-as-Judge` `Prompt评测`

**English keywords：**
`LLM Evaluation` `LLM Evaluation Harness` `LLM Benchmark` `Model Evaluation Framework`
`AI Evaluation` `RAG Evaluation` `Agent Evaluation` `LLM-as-Judge`
`MMLU` `GSM8K` `C-Eval` `HellaSwag` `HumanEval` `MBPP` `IFEval` `PIQA`
`OpenAI Compatible API` `vLLM` `Few-shot Evaluation` `Model Capability Testing`
`Machine Learning Benchmark` `NLP Evaluation` `Model Alignment` `Safety Evaluation`

**GitHub Topics 推荐（在仓库右侧设置）：**
```
llm-evaluation, 大模型评测, rag-evaluation, agent-evaluation, evaluation-harness,
benchmark, mmlu, gsm8k, ceval, humaneval, ifeval, hellaswag, mbpp, piqa,
llm-benchmark, model-evaluation, ai-evaluation, llm-as-judge, vllm,
openai-compatible, nlp-benchmark, 人工智能, 模型评估, 基准测试
```

## 相关项目

- [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) — 业界标准的 LLM 评测框架（英文）
- [OpenCompass](https://github.com/open-compass/opencompass) — 上海 AI 实验室的大模型评测平台
- [C-Eval](https://github.com/SJTU-LIT/ceval) — 中文基础模型评估数据集
- [MMLU](https://github.com/hendrycks/test) — 多任务语言理解评测基准
- [vLLM](https://github.com/vllm-project/vllm) — 高性能 LLM 推理服务（本系统推荐配合使用）

---

<div align="center">
  <sub>
    🚀 <strong>AI 能力评估系统</strong> —
    让大模型评测更简单、更便携、更可靠
    <br>
    <em>LLM Evaluation · RAG Evaluation · Agent Evaluation · Benchmark Framework</em>
  </sub>
</div>

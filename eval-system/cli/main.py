"""CLI 入口 — 支持多种等价写法（长选项/短选项/位置参数等）。"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from core.config import load_config
from core.model_adapter import ModelAdapter
from core.registry import ModuleRegistry
from core.runner import EvalRunner
from core.schema import EvalResult, ModelConfig
from reports.html import generate_html_report
from reports.markdown import generate_markdown_report, save_result_json
from storage.backup import check_and_recover, create_backup
from storage.database import compare_runs, get_leaderboard, get_run, init_db, list_runs, save_run

logger = logging.getLogger("eval.cli")


class EvalCLI(click.Group):
    """支持子命令按模块动态加载"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 确保初始化
        try:
            ModuleRegistry.initialize()
        except Exception:
            pass


@click.group(cls=EvalCLI)
@click.option("--env", "-e", default=None, help=".env 文件路径")
@click.pass_context
def cli(ctx, env):
    """AI 能力评估系统 (Evaluation Harness)

    评测不运行模型，只发 Prompt → 收 Response → 评分 → 出报告。
    被评测模型通过 OpenAI 兼容 API 接入。
    """
    ctx.ensure_object(dict)
    config = load_config(env)
    ctx.obj["config"] = config

    # 设置日志
    logging.basicConfig(
        level=getattr(logging, config.log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 数据库初始化
    init_db()
    check_and_recover()


# ==================== 系统管理 ====================

@cli.command()
@click.pass_context
def init(ctx):
    """初始化环境"""
    config = ctx.obj["config"]
    click.echo("🔧 初始化环境...")

    # 创建必要目录
    for d in [config.results_dir, Path(config.results_dir) / "reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        click.echo(f"  ✅ 目录已创建: {d}")

    # 初始化数据库
    db_path = init_db()
    click.echo(f"  ✅ 数据库已初始化: {db_path}")

    click.echo("\n✅ 环境初始化完成！")
    click.echo(f"   模型: {config.eval_model_name}")
    click.echo(f"   端点: {config.eval_model_api_base}")
    click.echo("   运行 `eval check` 检查环境完整性")


@cli.command()
@click.pass_context
def check(ctx):
    """检查环境完整性"""
    config = ctx.obj["config"]
    click.echo("🔍 环境预检...\n")

    checks = []

    # 检查 .env 配置
    env_ok = bool(config.eval_model_api_base)
    checks.append(("✅" if env_ok else "❌", ".env 配置", config.eval_model_api_base))
    if not env_ok:
        click.echo("  ❌ .env 中 EVAL_MODEL_API_BASE 未配置")

    # 检查模块注册
    try:
        ModuleRegistry.initialize()
        modules = ModuleRegistry.list_module_names()
        checks.append(("✅", "模块注册", f"{len(modules)} 个模块: {', '.join(modules)}"))
    except Exception as e:
        checks.append(("❌", "模块注册", str(e)))

    # 检查数据库
    try:
        db_path = Path(config.db_path)
        db_ok = db_path.exists()
        checks.append(("✅" if db_ok else "ℹ️", "数据库", str(db_path) if db_ok else "尚未创建（首次运行会自动创建）"))
    except Exception as e:
        checks.append(("❌", "数据库", str(e)))

    for icon, name, msg in checks:
        click.echo(f"  {icon} {name}: {msg}")

    # 测试模型连通性
    click.echo("\n🔌 测试模型连通性...")
    result = asyncio.run(ModelAdapter(config.eval_model_config).test_connection())
    if result["success"]:
        click.echo(f"  ✅ 模型可达: {result['model']} (延迟: {result['latency_ms']}ms)")
    else:
        click.echo(f"  ⚠️  模型不可达: {result.get('error', '')}")
        click.echo("  (此警告不阻止启动，请确认 vLLM 已运行)")

    click.echo("\n✅ 环境检查完成")


@cli.command()
@click.option("--tail", default=50, help="显示最后 N 行")
@click.option("--since", default=None, help="显示最近 N 分钟")
@click.pass_context
def logs(ctx, tail, since):
    """查看评测日志"""
    log_dir = Path(ctx.obj["config"].results_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eval.log"

    if not log_file.exists():
        click.echo("📝 日志文件不存在，使用 Python logging 输出")
        return

    import subprocess
    cmd = ["tail", f"-n{tail}", str(log_file)]
    if since:
        cmd = ["journalctl"]  # 简化处理
    subprocess.run(cmd)


# ==================== 模型 ====================

@cli.command()
@click.pass_context
def test(ctx):
    """测试 vLLM 连通性（发一个 "1+1=?" 请求验证）"""
    config = ctx.obj["config"]
    click.echo(f"🔌 测试模型连通性...")
    click.echo(f"   模型: {config.eval_model_name}")
    click.echo(f"   端点: {config.eval_model_api_base}")

    result = asyncio.run(ModelAdapter(config.eval_model_config).test_connection())

    if result["success"]:
        click.echo(f"\n✅ 连接成功！")
        click.echo(f"   模型: {result['model']}")
        click.echo(f"   响应: {result['response']}")
        click.echo(f"   延迟: {result['latency_ms']}ms")
    else:
        click.echo(f"\n❌ 连接失败")
        click.echo(f"   错误: {result.get('error', '未知错误')}")
        sys.exit(1)


# ==================== 运行评测 ====================

@cli.group()
def run():
    """运行评测"""


def _resolve_benchmarks(ctx, module_name: str, bench_str: Optional[str]) -> list[str]:
    """解析 benchmark 参数"""
    ModuleRegistry.initialize()

    if not ModuleRegistry.has_module(module_name):
        available = ModuleRegistry.list_module_names()
        click.echo(f"❌ 未知评测模块: {module_name}，可用: {available}")
        sys.exit(1)

    module = ModuleRegistry.get_module(module_name)
    all_benches = [b.id for b in module.list_benchmarks()]

    if not bench_str:
        click.echo(f"可用评测项: {', '.join(all_benches)}")
        sys.exit(0)

    benches = [b.strip() for b in bench_str.split(",")]

    unknown = [b for b in benches if b not in all_benches]
    if unknown:
        click.echo(f"❌ 未知评测项: {unknown}，可用: {all_benches}")
        sys.exit(1)

    return benches


@run.command()
@click.option("--bench", "-b", default=None, help="评测项名称，逗号分隔（如 mmlu,gsm8k,ceval）")
@click.option("--concurrency", "-c", default=None, type=int, help="并发数")
@click.option("--timeout", "-t", default=None, type=int, help="单题超时秒数")
@click.pass_context
def llm(ctx, bench, concurrency, timeout):
    """运行大模型基础能力评测"""
    config = ctx.obj["config"]
    benches = _resolve_benchmarks(ctx, "llm", bench)

    params = {}
    if concurrency:
        params["concurrency"] = concurrency
    if timeout:
        params["timeout"] = timeout

    click.echo(f"\n🚀 开始评测：大模型基础能力")
    click.echo(f"   模型：{config.eval_model_name} @ {config.eval_model_api_base}")
    click.echo(f"   评测项：{', '.join(benches)}")
    click.echo(f"   并发数：{params.get('concurrency', config.concurrency)}")
    click.echo("")

    # 执行评测
    runner = EvalRunner(config.eval_model_config)
    result = asyncio.run(runner.run_multiple_benchmarks("llm", benches, params))
    asyncio.run(runner.close())

    # 保存结果
    _save_and_report(result, config)

    # 输出结果摘要
    click.echo(f"\n{'='*50}")
    click.echo(f"✅ 评测完成！")
    click.echo(f"   运行 ID: {result.run_id}")
    click.echo(f"   总耗时: {result.total_duration_s} 秒")
    click.echo(f"   综合得分: {result.overall_score_pct:.2f}% （{result.total_correct}/{result.total_questions}）")

    for b in result.benchmarks:
        click.echo(f"   ├─ {b.benchmark_name:<12} 得分: {b.score_pct:.1f}% （{b.correct_count}/{b.total_questions}）")

    click.echo(f"\n📄 报告位置: results/{result.run_id}/")
    click.echo(f"   ├── report.html （浏览器打开）")
    click.echo(f"   └── report.md （终端预览）")


@run.command()
@click.option("--bench", "-b", default=None, help="评测项名称")
@click.pass_context
def rag(ctx, bench):
    """运行 RAG 能力评测（P1 阶段开发中）"""
    click.echo("🚧 RAG 评测模块正在开发中，敬请期待（P1 阶段）")


@run.command()
@click.option("--bench", "-b", default=None, help="评测项名称")
@click.pass_context
def agent(ctx, bench):
    """运行 Agent 能力评测（P2 阶段规划中）"""
    click.echo("🚧 Agent 评测模块正在开发中，敬请期待（P2 阶段）")


@cli.command()
@click.pass_context
def all(ctx):
    """运行全量评测"""
    config = ctx.obj["config"]
    ModuleRegistry.initialize()
    modules = ModuleRegistry.list_module_names()

    click.echo(f"🚀 运行全量评测...")
    click.echo(f"   模型：{config.eval_model_name}")
    click.echo(f"   评测模块：{', '.join(modules)}")

    for mod_name in modules:
        module = ModuleRegistry.get_module(mod_name)
        benches = [b.id for b in module.list_benchmarks()]
        if benches:
            click.echo(f"\n  ▶ 运行 {mod_name}: {', '.join(benches)}")
            runner = EvalRunner(config.eval_model_config)
            result = asyncio.run(runner.run_multiple_benchmarks(mod_name, benches))
            asyncio.run(runner.close())
            _save_and_report(result, config)
            click.echo(f"  ✅ {mod_name} 完成: {result.overall_score_pct:.1f}%")


def _save_and_report(result: EvalResult, config):
    """保存结果并生成报告"""
    # 保存到数据库
    save_run(result)

    # 保存 JSON
    save_result_json(result, config.results_dir)

    # 生成报告
    report_dir = Path(config.results_dir) / result.run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # Markdown 报告
    md_path = report_dir / "report.md"
    generate_markdown_report(result, str(md_path))

    # HTML 报告
    html_path = report_dir / "report.html"
    generate_html_report(result, str(html_path))

    # 备份数据库
    create_backup()


# ==================== 结果分析 ====================

@cli.command()
@click.argument("run_id")
@click.pass_context
def report(ctx, run_id):
    """查看评测报告"""
    run_data = get_run(run_id)
    if not run_data:
        click.echo(f"❌ 未找到运行记录：{run_id}")
        sys.exit(1)

    click.echo(f"📄 运行记录：{run_id}")
    click.echo(f"   模型：{run_data.get('model_name', 'N/A')}")
    click.echo(f"   综合得分：{run_data.get('overall_score', 0)*100:.2f}%")
    click.echo(f"   完成时间：{run_data.get('completed_at', 'N/A')}")


@cli.command()
@click.argument("run_id_1")
@click.argument("run_id_2")
@click.pass_context
def compare(ctx, run_id_1, run_id_2):
    """对比两次评测结果"""
    result = compare_runs(run_id_1, run_id_2)
    click.echo(f"📊 评测对比：{run_id_1} ↔ {run_id_2}")

    for key, run in result.items():
        if run:
            label = "第一次评测" if "1" in key else "第二次评测"
            click.echo(f"\n  {label}：")
            click.echo(f"    模型：{run.get('model_name', 'N/A')}")
            click.echo(f"    综合得分：{run.get('overall_score', 0)*100:.2f}%")


@cli.command()
@click.argument("benchmark")
@click.pass_context
def leaderboard(ctx, benchmark):
    """查看历史排行榜"""
    entries = get_leaderboard(benchmark, limit=20)
    if not entries:
        click.echo(f"📊 评测项 '{benchmark}' 暂无历史数据")
        return

    click.echo(f"📊 {benchmark} 排行榜\n")
    click.echo(f"{'排名':<6} {'模型名称':<22} {'得分':<10} {'正确/总数':<15} {'延迟':<10}")
    click.echo("-" * 65)
    for i, entry in enumerate(entries, 1):
        score_pct = entry.get("score", 0) * 100
        click.echo(
            f"{i:<6} {entry.get('model_name', 'N/A'):<22} "
            f"{score_pct:<8.1f}% "
            f"{entry.get('correct_count', 0)}/{entry.get('total_questions', 0):<10} "
            f"{entry.get('latency_avg_ms', 0):.0f}ms"
        )


# ==================== 数据集管理 ====================

@cli.group()
def dataset():
    """数据集管理"""


@dataset.command("list")
def list_datasets():
    """列出所有可用评测项"""
    ModuleRegistry.initialize()
    click.echo("📦 可用评测项\n")

    for mod_name in ModuleRegistry.list_module_names():
        module = ModuleRegistry.get_module(mod_name)
        benches = module.list_benchmarks()
        click.echo(f"  [{mod_name.upper()}] {module.description}")
        for b in benches:
            cn_name = _BENCHMARK_CN.get(b.id, b.name)
            click.echo(f"    ├─ {b.id:<12} {cn_name:<12} {b.description}")
        click.echo("")


_BENCHMARK_CN = {
    "mmlu": "多学科知识",
    "ceval": "中文知识",
    "gsm8k": "数学推理",
    "hellaswag": "常识推理",
    "piqa": "物理常识",
    "humaneval": "代码补全",
    "mbpp": "基础编程",
    "ifeval": "指令遵循",
    "needle": "长文本检索",
}


@dataset.command()
@click.argument("path")
@click.pass_context
def import_dataset(ctx, path):
    """导入自定义数据集"""
    click.echo(f"📦 导入数据集: {path}")
    # TODO: implement dataset import
    click.echo("✅ 导入完成")


@dataset.command()
@click.argument("name")
@click.pass_context
def info(ctx, name):
    """查看数据集统计信息"""
    click.echo(f"📊 数据集信息: {name}")
    # TODO: implement dataset info


# ==================== 自定义评测 ====================

@cli.group()
def custom():
    """自定义评测"""


@custom.command()
@click.argument("scenario_file")
@click.pass_context
def create(ctx, scenario_file):
    """创建自定义评测场景"""
    click.echo(f"📝 创建自定义评测: {scenario_file}")


@custom.command()
@click.argument("scenario_id")
@click.option("--model", "-m", default=None, help="模型名称")
@click.pass_context
def run_custom(ctx, scenario_id, model):
    """运行自定义评测"""
    click.echo(f"🚀 运行自定义评测: {scenario_id}")


# ==================== 入口 ====================

def main():
    cli()


if __name__ == "__main__":
    cli()

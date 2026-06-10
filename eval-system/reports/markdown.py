"""Markdown 报告生成器 — 生成终端可预览的报告。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from core.schema import EvalResult, BenchmarkResult


def generate_markdown_report(result: EvalResult, output_path: Optional[str] = None) -> str:
    """生成 Markdown 格式评测报告"""
    lines = []

    # 标题
    lines.append(f"# 📊 AI 能力评估报告\n")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**运行 ID**: `{result.run_id}`")
    lines.append(f"**状态**: {'✅ 完成' if result.status.value == 'completed' else '⚠️ 部分完成' if result.status.value == 'partial' else '❌ 失败'}")
    lines.append("")

    # 模型信息
    lines.append("---\n")
    lines.append("## 模型信息\n")
    lines.append(f"| 项目 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 模型名称 | {result.model_config.model_name} |")
    lines.append(f"| API 端点 | {result.model_config.api_base} |")
    lines.append(f"| 评测模块 | {result.module} |")
    lines.append(f"| 评测时间 | {result.started_at} ~ {result.completed_at} |")
    lines.append(f"| 总耗时 | {result.total_duration_s}s |")
    lines.append("")

    # 总分概览
    lines.append("---\n")
    lines.append("## 总得分\n")
    lines.append(f"> **{result.overall_score_pct:.2f}%** ({result.total_correct}/{result.total_questions})\n")

    # Benchmark 明细
    lines.append("---\n")
    lines.append("## 各 Benchmark 得分\n")
    lines.append("| Benchmark | 类别 | 正确/总数 | 得分 | 平均延迟 | 错误数 |")
    lines.append("|-----------|------|-----------|------|---------|-------|")

    for b in result.benchmarks:
        score_str = f"**{b.score_pct:.1f}%**" if b.score_pct >= 60 else f"{b.score_pct:.1f}%"
        lines.append(
            f"| {b.benchmark_name} | {b.category} | "
            f"{b.correct_count}/{b.total_questions} | {score_str} | "
            f"{b.latency_avg_ms:.0f}ms | {b.error_count} |"
        )
    lines.append("")

    # 错误样本（仅列出有错误的题目）
    all_errors = [q for b in result.benchmarks for q in b.questions if q.error]
    if all_errors:
        lines.append("---\n")
        lines.append("## ⚠️ 错误样本\n")
        lines.append(f"> 共 {len(all_errors)} 题出现错误\n")
        lines.append("| 题目 ID | 类别 | 错误信息 |")
        lines.append("|---------|------|---------|")
        for q in all_errors[:20]:  # 最多显示 20 条
            lines.append(f"| `{q.question_id}` | {q.category} | {q.error[:100]} |")
        if len(all_errors) > 20:
            lines.append(f"| ... 还有 {len(all_errors) - 20} 条 | | |")

    # 错误题目（答错的题）
    wrong_questions = [q for b in result.benchmarks for q in b.questions if not q.is_correct and not q.error]
    if wrong_questions:
        lines.append("---\n")
        lines.append("## 答错题目展示\n")
        lines.append(f"> 共 {len(wrong_questions)} 题答错，以下展示前 10 题：\n")
        for q in wrong_questions[:10]:
            lines.append(f"### ❌ {q.question_id}\n")
            lines.append(f"- **模型输出**: {q.extracted_answer}")
            lines.append(f"- **正确答案**: {q.reference_answer}")
            lines.append("")

    # 结论
    lines.append("---\n")
    lines.append("## 结论与建议\n")

    if result.overall_score >= 0.8:
        lines.append("✅ **表现优秀** — 模型在评测中表现出色。")
    elif result.overall_score >= 0.6:
        lines.append("👍 **表现良好** — 模型在大部分任务上表现不错，仍有提升空间。")
    elif result.overall_score >= 0.4:
        lines.append("🔧 **表现一般** — 建议检查模型的薄弱领域并进行针对性优化。")
    else:
        lines.append("⚠️ **表现待提升** — 模型在评测任务上表现不佳，建议检查配置或更换模型。")

    if result.benchmarks:
        scores = [(b.benchmark_name, b.score_pct) for b in result.benchmarks]
        scores.sort(key=lambda x: x[1])
        if scores:
            weakest = scores[0]
            strongest = scores[-1]
            lines.append(f"- **最强项**: {strongest[0]} ({strongest[1]:.1f}%)")
            lines.append(f"- **弱项**: {weakest[0]} ({weakest[1]:.1f}%) — 建议加强此方面能力")

    lines.append("")
    lines.append("---\n")
    lines.append(f"*报告由 AI 能力评估系统自动生成 · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    report = "\n".join(lines)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        logger.info(f"Markdown 报告已生成: {out}")

    return report


def save_result_json(result: EvalResult, output_dir: str) -> str:
    """保存 JSON 格式的原始结果数据"""
    out_dir = Path(output_dir) / result.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "data.json"
    json_path.write_text(result.to_json(), encoding="utf-8")

    logger.info(f"JSON 结果已保存: {json_path}")
    return str(json_path)


import logging
logger = logging.getLogger("eval.report")

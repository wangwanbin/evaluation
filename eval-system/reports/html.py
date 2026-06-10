"""HTML 报告生成器 — 单个自包含 HTML 文件（雷达图用 matplotlib 内嵌 base64，离线可用）。"""

from __future__ import annotations

import base64
import io
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 非交互后端
import matplotlib.pyplot as plt
import numpy as np

from core.schema import EvalResult

logger = logging.getLogger("eval.report.html")

# 配置中文字体
_CN_FONTS = ["Noto Sans SC", "Microsoft YaHei", "SimHei", "PingFang SC",
             "WenQuanYi Micro Hei", "Source Han Sans SC", "STXihei"]
for _f in _CN_FONTS:
    try:
        plt.rcParams["font.sans-serif"] = [_f] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue


def _generate_radar_chart_base64(benchmarks: list) -> str:
    """用 matplotlib 生成雷达图，返回 base64 PNG"""
    if not benchmarks:
        return ""

    names = [b.benchmark_name for b in benchmarks]
    scores = [b.score_pct for b in benchmarks]
    num_vars = len(names)

    # 角度（闭合）
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    scores_plot = scores + scores[:1]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("white")

    # 背景网格颜色
    ax.set_facecolor("#fafafa")

    # 绘制刻度网格线（0, 20, 40, 60, 80, 100）
    ax.set_rgrids([20, 40, 60, 80, 100], angle=90, fontsize=9,
                  labels=["20", "40", "60", "80", "100"],
                  color="#cccccc")
    ax.set_ylim(0, 110)

    # 绘制网格线
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.8)
    ax.xaxis.grid(True, color="#e0e0e0", linewidth=0.8)

    # 绘制填充区域
    ax.fill(angles, scores_plot, alpha=0.25, color="#667eea")
    ax.plot(angles, scores_plot, color="#667eea", linewidth=2.2, marker="o",
            markersize=8, markerfacecolor="#667eea", markeredgecolor="white",
            markeredgewidth=2)

    # 在每个数据点上标注分数
    for i, (angle, score) in enumerate(zip(angles[:-1], scores)):
        ax.annotate(
            f"{score:.1f}",
            xy=(angle, score),
            xytext=(0, 12 if score < 95 else -12),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            color="#333",
            ha="center",
            va="center" if score >= 95 else "bottom",
        )

    # 设置标签（轴名称）
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, fontsize=11, fontweight="medium", color="#333")

    # 最外圈 100% 刻度标签
    ax.set_rlabel_position(30)

    # 设置极坐标轴样式
    ax.spines["polar"].set_color("#dddddd")
    ax.spines["polar"].set_linewidth(1)

    # 标题
    ax.set_title("多维能力雷达图", fontsize=16, fontweight="bold",
                 color="#444", pad=20)

    # 输出为 PNG → base64
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def generate_html_report(result: EvalResult, output_path: Optional[str] = None) -> str:
    """生成自包含 HTML 报告（雷达图已内嵌为 base64，离线可打开）"""
    overall_pct = result.overall_score_pct
    status_icon = "✅" if result.status.value == "completed" else "⚠️"

    # 生成雷达图
    radar_img_data = _generate_radar_chart_base64(result.benchmarks)

    # 构建 benchmark 表格行
    bench_rows = ""
    for b in result.benchmarks:
        score_class = "good" if b.score_pct >= 60 else "warn" if b.score_pct >= 40 else "bad"
        bench_rows += f"""
        <tr>
            <td><strong>{b.benchmark_name}</strong></td>
            <td>{b.category}</td>
            <td>{b.correct_count}/{b.total_questions}</td>
            <td class="{score_class}">{b.score_pct:.1f}%</td>
            <td>{b.latency_avg_ms:.0f}ms</td>
            <td>{b.error_count}</td>
        </tr>"""

    # 错误样本
    wrong_items = [
        q for b in result.benchmarks
        for q in b.questions
        if not q.is_correct and not q.error
    ]
    wrong_html = ""
    for q in wrong_items[:5]:
        wrong_html += f"""
        <div class="wrong-item">
            <div class="wrong-header">❌ {q.question_id}</div>
            <div class="wrong-detail"><strong>模型输出:</strong> <code>{q.extracted_answer}</code></div>
            <div class="wrong-detail"><strong>正确答案:</strong> <code>{q.reference_answer}</code></div>
        </div>"""

    # 条形图：得分柱状图（用内联 HTML/CSS 实现，免 JS）
    bars_html = ""
    if result.benchmarks:
        max_score = max(b.score_pct for b in result.benchmarks) or 100
        for b in result.benchmarks:
            pct = b.score_pct
            width = max(8, (pct / max(max_score, 1)) * 100)
            color = "#22c55e" if pct >= 80 else "#f59e0b" if pct >= 50 else "#ef4444"
            bars_html += f"""
            <div class="bar-row">
                <div class="bar-label">{b.benchmark_name}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:{width:.1f}%;background:{color};">
                        <span class="bar-value">{pct:.1f}%</span>
                    </div>
                </div>
            </div>"""

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI 能力评估报告 - {result.run_id}</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'PingFang SC', 'Microsoft YaHei', sans-serif;
           background: #f0f2f5; color: #333; line-height: 1.6; }}
    .container {{ max-width: 1000px; margin: 0 auto; padding: 24px; }}

    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
              color: white; padding: 36px 40px; border-radius: 14px; margin-bottom: 24px; }}
    .header h1 {{ font-size: 28px; margin-bottom: 6px; letter-spacing: 1px; }}
    .header .meta {{ opacity: 0.88; font-size: 14px; line-height: 1.8; }}
    .header .meta code {{ background: rgba(255,255,255,0.15); padding: 1px 8px; border-radius: 4px; font-size: 13px; }}

    .score-card {{ background: white; border-radius: 14px; padding: 32px; margin-bottom: 24px;
                  box-shadow: 0 2px 12px rgba(0,0,0,0.06); text-align: center; }}
    .score-number {{ font-size: 72px; font-weight: 800; color: #667eea; line-height: 1; }}
    .score-label {{ font-size: 15px; color: #999; margin-top: 6px; }}
    .score-sub {{ margin-top: 10px; color: #aaa; font-size: 14px; }}

    .section {{ background: white; border-radius: 14px; padding: 28px; margin-bottom: 24px;
               box-shadow: 0 2px 12px rgba(0,0,0,0.06); }}
    .section h2 {{ font-size: 18px; margin-bottom: 18px; color: #444;
                   border-bottom: 2px solid #f0f2f5; padding-bottom: 10px; }}

    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #f0f2f5; }}
    th {{ background: #f8f9fa; font-weight: 600; color: #555; font-size: 13px; }}
    tr:hover {{ background: #f8f9fa; }}
    .good {{ color: #22c55e; font-weight: 700; }}
    .warn {{ color: #f59e0b; font-weight: 700; }}
    .bad {{ color: #ef4444; font-weight: 700; }}

    .info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .info-item {{ padding: 8px 0; border-bottom: 1px solid #f5f5f5; }}
    .info-label {{ color: #999; font-size: 13px; }}
    .info-value {{ font-weight: 500; font-size: 14px; }}

    .wrong-item {{ background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px;
                  padding: 12px 16px; margin-bottom: 10px; }}
    .wrong-header {{ font-weight: 600; font-size: 14px; margin-bottom: 4px; }}
    .wrong-detail {{ font-size: 13px; color: #666; margin: 3px 0; }}
    .wrong-detail code {{ background: #f1f5f9; padding: 1px 6px; border-radius: 3px; font-size: 13px; }}

    .chart-container {{ text-align: center; padding: 10px 0; }}
    .chart-container img {{ max-width: 100%; height: auto; border-radius: 8px; }}

    .bar-chart {{ margin-top: 8px; }}
    .bar-row {{ display: flex; align-items: center; margin-bottom: 12px; gap: 12px; }}
    .bar-label {{ width: 120px; text-align: right; font-size: 14px; font-weight: 500; color: #555; flex-shrink: 0; }}
    .bar-track {{ flex: 1; height: 28px; background: #f0f2f5; border-radius: 14px; overflow: hidden; position: relative; }}
    .bar-fill {{ height: 100%; border-radius: 14px; display: flex; align-items: center;
                 justify-content: flex-end; padding-right: 10px; transition: width 0.6s ease; min-width: 40px; }}
    .bar-value {{ font-size: 13px; font-weight: 700; color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.2); }}

    .conclusion {{ font-size: 16px; line-height: 1.8; padding: 4px 0; }}
    .conclusion .big {{ font-size: 20px; font-weight: 700; }}

    .footer {{ text-align: center; color: #bbb; font-size: 13px; padding: 24px; }}

    @media (max-width: 640px) {{
        .info-grid {{ grid-template-columns: 1fr; }}
        .bar-label {{ width: 80px; font-size: 12px; }}
        .score-number {{ font-size: 48px; }}
    }}
</style>
</head>
<body>
<div class="container">

    <!-- 头部 -->
    <div class="header">
        <h1>{status_icon} AI 能力评估报告</h1>
        <div class="meta">
            <p>运行 ID: <code>{result.run_id}</code> · {result.module.upper()} 评测</p>
            <p>模型: <strong>{result.model_config.model_name}</strong> · 端点: <code>{result.model_config.api_base}</code></p>
        </div>
    </div>

    <!-- 得分卡片 -->
    <div class="score-card">
        <div class="score-number">{overall_pct:.1f}%</div>
        <div class="score-label">综合得分</div>
        <div class="score-sub">
            {result.total_correct} / {result.total_questions} 题正确 ·
            耗时 {result.total_duration_s}s
            {f' · {result.total_questions // max(1, int(result.total_duration_s))}题/秒' if result.total_duration_s > 0 else ''}
        </div>
    </div>

    <!-- 评测信息 -->
    <div class="section">
        <h2>📋 评测信息</h2>
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">运行 ID</div>
                <div class="info-value"><code>{result.run_id}</code></div>
            </div>
            <div class="info-item">
                <div class="info-label">状态</div>
                <div class="info-value">{'✅ 完成' if result.status.value == 'completed' else '⚠️ 部分完成'}</div>
            </div>
            <div class="info-item">
                <div class="info-label">模型名称</div>
                <div class="info-value">{result.model_config.model_name}</div>
            </div>
            <div class="info-item">
                <div class="info-label">API 端点</div>
                <div class="info-value"><code>{result.model_config.api_base}</code></div>
            </div>
            <div class="info-item">
                <div class="info-label">开始时间</div>
                <div class="info-value">{result.started_at}</div>
            </div>
            <div class="info-item">
                <div class="info-label">完成时间</div>
                <div class="info-value">{result.completed_at}</div>
            </div>
        </div>
    </div>

    <!-- 雷达图 -->
    {f"""
    <div class="section">
        <h2>📈 多维能力雷达图</h2>
        <div class="chart-container">
            <img src="{radar_img_data}" alt="能力雷达图" />
        </div>
    </div>
    """ if radar_img_data else ""}

    <!-- 得分条形图 -->
    {f"""
    <div class="section">
        <h2>📊 各 Benchmark 得分</h2>
        <div class="bar-chart">
            {bars_html}
        </div>
    </div>
    """ if bars_html else ""}

    <!-- 得分明细表 -->
    <div class="section">
        <h2>📋 得分明细</h2>
        <table>
            <thead>
                <tr>
                    <th>Benchmark</th>
                    <th>类别</th>
                    <th>正确/总数</th>
                    <th>得分</th>
                    <th>平均延迟</th>
                    <th>错误数</th>
                </tr>
            </thead>
            <tbody>
                {bench_rows}
            </tbody>
        </table>
    </div>

    <!-- 答错题目 -->
    {f"""
    <div class="section">
        <h2>❌ 答错题目（前 5 题）</h2>
        {wrong_html}
    </div>
    """ if wrong_html else ""}

    <!-- 结论 -->
    <div class="section">
        <h2>💡 结论</h2>
        <div class="conclusion">
            <span class="big">
            {"✅ 表现优秀" if result.overall_score >= 0.8
             else "👍 表现良好" if result.overall_score >= 0.6
             else "🔧 表现一般" if result.overall_score >= 0.4
             else "⚠️ 表现待提升"}
            </span>
            — 综合得分 {overall_pct:.1f}%
        </div>
    </div>

    <!-- 底部 -->
    <div class="footer">
        <p>由 AI 能力评估系统自动生成 · {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p style="font-size:11px;">此报告为独立 HTML 文件，所有内容已内嵌，离线可打开</p>
    </div>

</div>
</body>
</html>"""

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info(f"HTML 报告已生成: {out}")

    return html

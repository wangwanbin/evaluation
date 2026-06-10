# AI 能力评估系统 — 应用镜像构建文件
# 构建时完成所有依赖安装，宿主机零依赖

FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

# 配置国内镜像源（构建加速）
COPY requirements.txt .
RUN pip install --no-cache-dir \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    -r requirements.txt

# === 精简运行镜像 ===
FROM python:3.11-slim-bookworm

LABEL maintainer="eval-system"
LABEL description="AI 能力评估系统 - Evaluation Harness"

WORKDIR /app

# 从 builder 复制已安装的包
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 配置国内镜像源环境变量（供 Python 包下载使用）
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装 tini 作为 init 进程（正确处理信号）
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    && rm -rf /var/lib/apt/lists/*

# 复制应用代码
COPY core/ /app/core/
COPY modules/ /app/modules/
COPY cli/ /app/cli/
COPY reports/ /app/reports/
COPY storage/ /app/storage/
COPY main.py /app/
COPY .env.example /app/.env.example

# 创建数据目录（挂载点）
RUN mkdir -p /app/datasets /app/results/runs /app/results/reports /app/results/backups

# 默认环境变量
ENV EVAL_DB_PATH=/app/results/eval.db
ENV EVAL_RESULTS_DIR=/app/results/runs
ENV EVAL_REPORTS_DIR=/app/results/reports

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "from core.config import load_config; cfg = load_config(); exit(0)" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "cli.main", "--help"]

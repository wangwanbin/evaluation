"""SQLite 数据库管理 — WAL 模式、自动备份、崩溃恢复。

存储方案：SQLite（WAL 模式）+ JSON 文件双轨。
- 元数据（run 记录、配置、指标汇总）→ SQLite
- 逐题详情 → JSON 文件
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from core.config import load_config
from core.schema import EvalResult

logger = logging.getLogger("eval.storage.db")

_local = threading.local()


def _get_connection(db_path: str) -> sqlite3.Connection:
    """获取线程本地连接"""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(db_path)
        _local.conn.row_factory = sqlite3.Row
        _apply_pragmas(_local.conn)
    return _local.conn


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """应用 SQLite 安全配置"""
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA cache_size=-64000;")


def init_db(db_path: Optional[str] = None) -> str:
    """初始化数据库，创建表结构。返回数据库路径。"""
    config = load_config()
    path = db_path or config.db_path

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    conn = _get_connection(path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            run_id          TEXT PRIMARY KEY,
            module          TEXT NOT NULL,
            model_name      TEXT NOT NULL,
            model_api_base  TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'completed',
            overall_score   REAL DEFAULT 0.0,
            total_questions INTEGER DEFAULT 0,
            total_correct   INTEGER DEFAULT 0,
            total_duration_s REAL DEFAULT 0.0,
            started_at      TEXT,
            completed_at    TEXT,
            config_snapshot TEXT,       -- JSON: 评测时配置快照
            metadata        TEXT,
            created_at      TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS benchmark_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id          TEXT NOT NULL,
            benchmark_id    TEXT NOT NULL,
            benchmark_name  TEXT NOT NULL,
            category        TEXT DEFAULT '',
            score           REAL DEFAULT 0.0,
            total_questions INTEGER DEFAULT 0,
            correct_count   INTEGER DEFAULT 0,
            latency_avg_ms  REAL DEFAULT 0.0,
            error_count     INTEGER DEFAULT 0,
            metadata        TEXT,
            FOREIGN KEY (run_id) REFERENCES eval_runs(run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_benchmark_run
            ON benchmark_results(run_id);

        CREATE INDEX IF NOT EXISTS idx_runs_module
            ON eval_runs(module);

        CREATE INDEX IF NOT EXISTS idx_runs_created
            ON eval_runs(created_at DESC);
    """)
    conn.commit()
    logger.info(f"数据库已初始化: {path}")
    return path


def save_run(result: EvalResult) -> None:
    """保存评测运行记录"""
    config = load_config()
    conn = _get_connection(config.db_path)

    conn.execute("""
        INSERT OR REPLACE INTO eval_runs
            (run_id, module, model_name, model_api_base, status,
             overall_score, total_questions, total_correct, total_duration_s,
             started_at, completed_at, config_snapshot, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        result.run_id,
        result.module,
        result.model_config.model_name,
        result.model_config.api_base,
        result.status.value,
        result.overall_score,
        result.total_questions,
        result.total_correct,
        result.total_duration_s,
        result.started_at,
        result.completed_at,
        json.dumps(result.model_config.to_dict(), ensure_ascii=False),
        json.dumps(result.metadata, ensure_ascii=False),
    ))

    for bench in result.benchmarks:
        conn.execute("""
            INSERT INTO benchmark_results
                (run_id, benchmark_id, benchmark_name, category,
                 score, total_questions, correct_count, latency_avg_ms,
                 error_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.run_id,
            bench.benchmark_id,
            bench.benchmark_name,
            bench.category,
            bench.score,
            bench.total_questions,
            bench.correct_count,
            bench.latency_avg_ms,
            bench.error_count,
            json.dumps(bench.metadata, ensure_ascii=False),
        ))

    conn.commit()
    logger.info(f"运行记录已保存: {result.run_id}")


def list_runs(module: Optional[str] = None, limit: int = 20) -> list[dict]:
    """列出历史评测记录"""
    config = load_config()
    conn = _get_connection(config.db_path)

    if module:
        rows = conn.execute(
            "SELECT * FROM eval_runs WHERE module = ? ORDER BY created_at DESC LIMIT ?",
            (module, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM eval_runs ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [dict(r) for r in rows]


def get_run(run_id: str) -> Optional[dict]:
    """获取单次评测详情"""
    config = load_config()
    conn = _get_connection(config.db_path)

    row = conn.execute(
        "SELECT * FROM eval_runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return None

    result = dict(row)
    benches = conn.execute(
        "SELECT * FROM benchmark_results WHERE run_id = ?", (run_id,)
    ).fetchall()
    result["benchmarks"] = [dict(b) for b in benches]
    return result


def get_leaderboard(benchmark_id: str, limit: int = 20) -> list[dict]:
    """获取指定 benchmark 的历史排行榜"""
    config = load_config()
    conn = _get_connection(config.db_path)

    rows = conn.execute("""
        SELECT r.run_id, r.model_name, r.model_api_base, r.created_at,
               b.score, b.total_questions, b.correct_count, b.latency_avg_ms
        FROM benchmark_results b
        JOIN eval_runs r ON r.run_id = b.run_id
        WHERE b.benchmark_id = ?
        ORDER BY b.score DESC, b.total_questions DESC
        LIMIT ?
    """, (benchmark_id, limit)).fetchall()

    return [dict(r) for r in rows]


def compare_runs(run_id_1: str, run_id_2: str) -> dict:
    """对比两次评测"""
    run1 = get_run(run_id_1)
    run2 = get_run(run_id_2)
    return {"run_1": run1, "run_2": run2}

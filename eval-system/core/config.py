"""配置管理 — 从 .env 加载配置，提供统一访问接口。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from core.schema import ModelConfig


@dataclass
class Config:
    """系统配置，从 .env 加载"""

    # 被评测模型
    eval_model_api_base: str = "http://localhost:8000/v1"
    eval_model_api_key: str = ""
    eval_model_name: str = "qwen2.5-72b"

    # Judge 模型
    judge_model_api_base: str = ""
    judge_model_api_key: str = ""
    judge_model_name: str = ""

    # RAG 端点
    rag_api_base: str = ""
    rag_api_key: str = ""

    # 评测参数
    concurrency: int = 10
    timeout: int = 120

    # 存储路径
    db_path: str = ""
    results_dir: str = ""
    reports_dir: str = ""

    # 日志
    log_level: str = "INFO"

    _loaded: bool = False

    @property
    def eval_model_config(self) -> ModelConfig:
        return ModelConfig(
            api_base=self.eval_model_api_base,
            api_key=self.eval_model_api_key,
            model_name=self.eval_model_name,
        )

    @property
    def judge_model_config(self) -> Optional[ModelConfig]:
        if not self.judge_model_api_base:
            return None
        return ModelConfig(
            api_base=self.judge_model_api_base,
            api_key=self.judge_model_api_key,
            model_name=self.judge_model_name or self.eval_model_name,
        )

    @property
    def effective_judge_config(self) -> ModelConfig:
        """返回 judge 配置，如果没有独立配置则用 eval 配置"""
        return self.judge_model_config or self.eval_model_config

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if k.startswith("_") or v is None:
                continue
            if "key" in k.lower() and v:
                d[k] = v[:8] + "..." if len(v) > 8 else "***"
            else:
                d[k] = v
        return d


_config: Optional[Config] = None


def load_config(env_file: Optional[str] = None) -> Config:
    """加载配置（带缓存）"""
    global _config
    if _config is not None:
        return _config

    if env_file:
        load_dotenv(env_file)
    else:
        # 从 CWD 往上找 .env
        cwd = Path.cwd()
        for parent in [cwd] + list(cwd.parents):
            ef = parent / ".env"
            if ef.exists():
                load_dotenv(ef)
                break

    cfg = Config(
        eval_model_api_base=_env_str("EVAL_MODEL_API_BASE", "http://localhost:8000/v1"),
        eval_model_api_key=_env_str("EVAL_MODEL_API_KEY", ""),
        eval_model_name=_env_str("EVAL_MODEL_NAME", "qwen2.5-72b"),
        judge_model_api_base=_env_str("JUDGE_MODEL_API_BASE", ""),
        judge_model_api_key=_env_str("JUDGE_MODEL_API_KEY", ""),
        judge_model_name=_env_str("JUDGE_MODEL_NAME", ""),
        rag_api_base=_env_str("RAG_API_BASE", ""),
        rag_api_key=_env_str("RAG_API_KEY", ""),
        concurrency=_env_int("EVAL_CONCURRENCY", 10),
        timeout=_env_int("EVAL_TIMEOUT", 120),
        db_path=_env_str("EVAL_DB_PATH", ""),
        results_dir=_env_str("EVAL_RESULTS_DIR", ""),
        reports_dir=_env_str("EVAL_REPORTS_DIR", ""),
        log_level=_env_str("LOG_LEVEL", "INFO"),
        _loaded=True,
    )

    # 设置默认路径
    base = _find_project_root()
    if not cfg.db_path:
        cfg.db_path = str(base / "results" / "eval.db")
    if not cfg.results_dir:
        cfg.results_dir = str(base / "results")
    if not cfg.reports_dir:
        cfg.reports_dir = str(base / "results" / "reports")

    _config = cfg
    return cfg


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _find_project_root() -> Path:
    """从 .env 所在目录或 CWD 推断项目根目录"""
    cwd = Path.cwd()
    for p in [cwd] + list(cwd.parents):
        if (p / ".env").exists() or (p / "core" / "schema.py").exists():
            return p
    return cwd

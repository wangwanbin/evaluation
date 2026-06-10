"""自动备份与崩溃恢复机制。

保留最近 5 次备份，超过自动清理。
启动时检测 WAL 文件 → 崩溃恢复 → 完整性检查。
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.config import load_config

logger = logging.getLogger("eval.storage.backup")

MAX_BACKUPS = 5


def ensure_backup_dir() -> Path:
    """确保备份目录存在"""
    config = load_config()
    backup_dir = Path(config.db_path).parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def create_backup() -> Optional[Path]:
    """创建数据库备份（非阻塞复制）"""
    config = load_config()
    db_path = Path(config.db_path)
    if not db_path.exists():
        return None

    backup_dir = ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"eval_{timestamp}.db"

    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"数据库已备份: {backup_path}")
        _cleanup_old_backups(backup_dir)
        return backup_path
    except Exception as e:
        logger.warning(f"备份失败: {e}")
        return None


def _cleanup_old_backups(backup_dir: Path) -> None:
    """清理超出保留数量的旧备份"""
    backups = sorted(backup_dir.glob("eval_*.db"))
    while len(backups) > MAX_BACKUPS:
        oldest = backups.pop(0)
        oldest.unlink(missing_ok=True)
        logger.info(f"已清理旧备份: {oldest}")


def check_and_recover() -> dict:
    """启动时检查并执行崩溃恢复

    Returns:
        {"status": "ok"|"recovered"|"failed", "message": str}
    """
    config = load_config()
    db_path = Path(config.db_path)
    if not db_path.exists():
        return {"status": "ok", "message": "数据库不存在，将新建"}

    wal_path = db_path.with_suffix(".db-wal")
    result = {"status": "ok", "message": ""}

    # 检查 WAL 文件（异常退出标志）
    if wal_path.exists() and wal_path.stat().st_size > 0:
        logger.info("检测到 WAL 文件（上次可能异常退出），正在回放...")
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.close()
            logger.info("WAL 回放完成")
            result["message"] = "WAL 回放完成，数据已恢复"
        except Exception as e:
            logger.warning(f"WAL 回放失败: {e}")
            result["status"] = "failed"
            result["message"] = f"WAL 回放失败: {e}"

    # 完整性检查
    try:
        conn = sqlite3.connect(str(db_path))
        integrity = conn.execute("PRAGMA integrity_check;").fetchone()
        if integrity and integrity[0] != "ok":
            logger.warning(f"数据库完整性检查失败: {integrity[0]}")
            result = _recover_from_backup(db_path)
        else:
            if not result["message"]:
                result["message"] = "数据库完整性检查通过"
        conn.close()
    except Exception as e:
        logger.error(f"完整性检查异常: {e}")
        result = _recover_from_backup(db_path)

    return result


def _recover_from_backup(db_path: Path) -> dict:
    """从最新备份恢复数据库"""
    backup_dir = db_path.parent / "backups"
    if not backup_dir.exists():
        return {"status": "failed", "message": "无可用备份，无法恢复"}

    backups = sorted(backup_dir.glob("eval_*.db"))
    if not backups:
        return {"status": "failed", "message": "无可用备份，无法恢复"}

    latest = backups[-1]
    try:
        # 重命名损坏文件
        corrupted = db_path.with_suffix(".db.corrupted")
        db_path.rename(corrupted)
        # 从备份恢复
        shutil.copy2(latest, db_path)
        logger.info(f"已从备份恢复: {latest}")
        return {"status": "recovered", "message": f"已从 {latest.name} 恢复"}
    except Exception as e:
        return {"status": "failed", "message": f"恢复失败: {e}"}

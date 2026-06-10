"""模块注册与发现 — 自动扫描 modules/ 目录，注册所有 EvalModule 实现。"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from pathlib import Path
from typing import Optional

from core.schema import EvalModule


class ModuleRegistry:
    """模块注册表 — 单例，管理所有评估模块"""

    _modules: dict[str, EvalModule] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls, modules_path: Optional[str] = None) -> None:
        """扫描并注册所有模块（只执行一次）"""
        if cls._initialized:
            return
        cls._modules = {}

        # 确定模块搜索路径
        search_path = Path(modules_path) if modules_path else (Path(__file__).parent.parent / "modules")

        if not search_path.exists():
            cls._initialized = True
            return

        # 递归扫描 modules/ 下的所有包
        cls._scan_package(search_path, "modules")

        cls._initialized = True

    @classmethod
    def _scan_package(cls, pkg_path: Path, import_path: str) -> None:
        """扫描一个 Python 包，寻找 EvalModule 子类"""
        if not pkg_path.is_dir():
            return
        if not (pkg_path / "__init__.py").exists():
            return

        try:
            pkg = importlib.import_module(import_path)
        except Exception as e:
            return

        # 找出包内所有 EvalModule 子类
        for name, obj in inspect.getmembers(pkg):
            if (inspect.isclass(obj) and issubclass(obj, EvalModule)
                    and obj is not EvalModule and not inspect.isabstract(obj)):
                try:
                    instance = obj()
                    if instance.name:
                        cls._modules[instance.name] = instance
                except Exception:
                    pass

        # 递归扫描子包
        if hasattr(pkg, "__path__"):
            for importer, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
                sub_import = f"{import_path}.{modname}"
                if ispkg:
                    sub_path = pkg_path / modname
                    cls._scan_package(sub_path, sub_import)

    @classmethod
    def get_module(cls, name: str) -> Optional[EvalModule]:
        """按名称获取评估模块"""
        return cls._modules.get(name)

    @classmethod
    def list_modules(cls) -> list[EvalModule]:
        """列出所有已注册的模块"""
        return list(cls._modules.values())

    @classmethod
    def list_module_names(cls) -> list[str]:
        return list(cls._modules.keys())

    @classmethod
    def has_module(cls, name: str) -> bool:
        return name in cls._modules

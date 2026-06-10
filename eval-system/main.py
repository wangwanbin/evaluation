#!/usr/bin/env python3
"""AI 能力评估系统 — 入口点"""

import sys
import os

# 确保项目根在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.main import cli

if __name__ == "__main__":
    cli()

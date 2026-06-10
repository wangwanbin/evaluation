#!/usr/bin/env python3
"""AI 能力评估系统 — 安装脚本"""

from setuptools import find_packages, setup

setup(
    name="eval-system",
    version="1.0.0",
    description="AI 能力评估系统 - Evaluation Harness",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "aiohttp>=3.9.0",
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "rich>=13.7.0",
        "jinja2>=3.1.0",
        "matplotlib>=3.8.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "eval=cli.main:cli",
        ],
    },
)

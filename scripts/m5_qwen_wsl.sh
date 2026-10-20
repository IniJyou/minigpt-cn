#!/usr/bin/env bash
# 只使用已准备的M5环境；不安装依赖或下载模型。
set -euo pipefail
project_root=/mnt/d/aiworkspace/transformer
export PYTHONPATH="$project_root/src"
export PYTHONDONTWRITEBYTECODE=1
exec /home/luoyu/.venvs/minigpt-cn-m5/bin/python -B -m minigpt_cn.m5_qwen_run "$@"

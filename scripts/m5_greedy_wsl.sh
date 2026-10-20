#!/usr/bin/env bash
# 使用已有独立M5环境，默认inspect；不安装、不下载、不训练。
set -euo pipefail
export PYTHONPATH=/mnt/d/aiworkspace/transformer/src
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
exec /home/luoyu/.venvs/minigpt-cn-m5/bin/python -B -m minigpt_cn.m5_greedy_run "$@"

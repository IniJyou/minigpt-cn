#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/mnt/d/aiworkspace/transformer/src
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
exec /home/luoyu/.venvs/minigpt-cn-m5/bin/python -B -m minigpt_cn.m5_demo "$@"

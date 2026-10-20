#!/usr/bin/env bash
# 使用已验收的独立M5环境；默认inspect，不安装、不下载。
set -euo pipefail
project_root=/mnt/d/aiworkspace/transformer
export PYTHONPATH="$project_root/src"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export UV_OFFLINE=1
exec /home/luoyu/.venvs/minigpt-cn-m5/bin/python -B -m minigpt_cn.m5_lora_run "$@"

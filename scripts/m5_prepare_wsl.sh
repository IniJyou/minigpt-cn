#!/usr/bin/env bash
# 用户在 WSL 执行：创建独立环境、安装推理依赖、下载固定版本模型。
# 不激活或修改已有 33M/作业环境；不运行模型推理或训练。
set -euo pipefail

phase="${1:-all}"
case "$phase" in
    all|create|deps|model) ;;
    *) printf '用法：bash %s [all|create|deps|model]\n' "$0" >&2; exit 2 ;;
esac

project_root=/mnt/d/aiworkspace/transformer
uv_bin=/home/luoyu/.local/bin/uv
export M5_ENV=/home/luoyu/.venvs/minigpt-cn-m5
export M5_MODEL_DIR="$project_root/data/raw/m5_models/qwen3_0_6b_c1899de"
export HF_HOME="$project_root/data/raw/m5_hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export UV_CACHE_DIR="$project_root/data/raw/m5_uv_cache"
# 环境在 Linux 文件系统、下载缓存在 D 盘，跨文件系统使用复制。
export UV_LINK_MODE=copy
export PYTHONNOUSERSITE=1

if [[ ! -f "$project_root/pyproject.toml" || ! -x "$uv_bin" ]]; then
    printf '未找到预期项目或 uv，请先核对 WSL 路径。\n' >&2
    exit 1
fi

require_m5_python() {
    "$M5_ENV/bin/python" - <<'PY'
import os
import sys
from pathlib import Path

expected = Path(os.environ["M5_ENV"])
if Path(sys.prefix) != expected or sys.prefix == sys.base_prefix:
    raise SystemExit("解释器不是预期的独立 M5 虚拟环境，停止。")
PY
}

if [[ "$phase" == all || "$phase" == create ]]; then
    if [[ -e "$M5_ENV" || -L "$M5_ENV" ]]; then
        printf 'M5 环境已存在，停止创建。先核对；本脚本创建后的安装重试用 deps。\n' >&2
        exit 1
    fi
    "$uv_bin" venv --python /usr/bin/python3 "$M5_ENV"
fi

if [[ "$phase" == all || "$phase" == deps ]]; then
    require_m5_python
    "$uv_bin" pip install --python "$M5_ENV/bin/python" \
        'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu126
    "$uv_bin" pip install --python "$M5_ENV/bin/python" \
        'transformers==4.57.6' 'accelerate==1.10.1'
    "$uv_bin" pip check --python "$M5_ENV/bin/python"
    mkdir -p "$project_root/logs"
    deps_file=$(mktemp "$project_root/logs/m5_dependencies_XXXXXXXX.txt")
    "$uv_bin" pip freeze --python "$M5_ENV/bin/python" > "$deps_file"
    printf '完整依赖版本清单：%s\n' "$deps_file"
fi

if [[ "$phase" == all || "$phase" == model ]]; then
    require_m5_python
    "$M5_ENV/bin/python" - <<'PY'
import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from importlib.metadata import distributions, version
from pathlib import Path

from huggingface_hub import snapshot_download

expected_versions = {
    "torch": "2.8.0+cu126", "transformers": "4.57.6", "accelerate": "1.10.1",
}
for name, expected in expected_versions.items():
    if version(name) != expected:
        raise SystemExit(f"{name} 版本不符；先核对独立环境，停止下载。")

repo_id = "Qwen/Qwen3-0.6B"
revision = "c1899de289a04d12100db370d81485cdf75e47ca"
model_dir = Path(os.environ["M5_MODEL_DIR"])
files = [
    "model.safetensors", "config.json", "generation_config.json",
    "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
    "README.md", "LICENSE",
]
snapshot_download(
    repo_id=repo_id, revision=revision, local_dir=str(model_dir),
    allow_patterns=files, token=False,
)

# 预期摘要来自上轮固定 revision 的官方元数据；不靠文件名判断版本。
expected_hashes = {
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
}
file_info = {}
for name in files:
    path = model_dir / name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if name in expected_hashes and actual != expected_hashes[name]:
        raise SystemExit(f"{name} SHA256 不符，保留现场并停止。")
    file_info[name] = {"bytes": path.stat().st_size, "sha256": actual}

template = json.loads(
    (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
)["chat_template"]
template_sha = hashlib.sha256(template.encode("utf-8")).hexdigest()
if template_sha != "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8":
    raise SystemExit("聊天模板摘要不符，保留现场并停止。")

log_root = Path("/mnt/d/aiworkspace/transformer/logs")
log_root.mkdir(parents=True, exist_ok=True)
run_dir = Path(tempfile.mkdtemp(prefix="m5_model_prepared_", dir=log_root))
receipt = {
    "completed_utc": datetime.now(timezone.utc).isoformat(),
    "environment": os.environ["M5_ENV"], "python": platform.python_version(),
    "packages": {d.metadata["Name"]: d.version for d in distributions()},
    "repo_id": repo_id, "revision": revision, "model_directory": str(model_dir),
    "cache_paths": {key: os.environ[key] for key in (
        "HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HF_ASSETS_CACHE",
        "HF_DATASETS_CACHE", "UV_CACHE_DIR",
    )},
    "files": file_info, "chat_template_sha256": template_sha,
    "weight_loading_run": False, "generation_run": False, "training_run": False,
}
(run_dir / "receipt.json").write_text(
    json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
)
print(f"文件下载与摘要核对完成：{run_dir / 'receipt.json'}")
print("尚未加载权重、运行推理或训练；下一步由助手只读验收。")
PY
fi

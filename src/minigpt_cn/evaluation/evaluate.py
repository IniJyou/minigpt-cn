"""从项目 checkpoint 配置重建模型，对完整 uint16 文件计算 loss / perplexity。"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from minigpt_cn.data.tokens import read_tokens
from minigpt_cn.training.checkpoint import file_sha256, load_checkpoint
from minigpt_cn.training.optim import check_finite


def check_runtime(device, precision):
    if device not in {"cpu", "cuda"} or precision not in {"float32", "bfloat16"}:
        raise ValueError("设备仅支持 cpu/cuda；精度仅支持 float32/bfloat16")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；CPU 调试请显式设置 cpu/float32")
    if device == "cuda" and precision == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16，请使用 float32")


@torch.no_grad()
def evaluate(model, tokens, precision="float32"):
    """与原 eval_pilot.py 相同的逐块上下文；包括最后不足一个上下文的尾块。"""
    if len(tokens) < 2:
        raise ValueError("完整验证至少需要两个 token")
    device = next(model.parameters()).device
    check_runtime(device.type, precision)
    was_training = model.training
    model.eval()
    total_nll, total_targets = 0.0, 0
    started = time.perf_counter()
    try:
        for start in range(0, len(tokens) - 1, model.config.context_length):
            end = min(start + model.config.context_length, len(tokens) - 1)
            # x/y: [1,sequence]，尾块的 sequence 可小于 context_length。
            x = torch.from_numpy(np.asarray(tokens[start:end], dtype=np.int64))[None].to(device)
            y = torch.from_numpy(np.asarray(tokens[start + 1 : end + 1], dtype=np.int64))[None].to(
                device
            )
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
            ):
                output = model(x, targets=y)
            check_finite([("validation_loss", output.loss)], f"验证 start={start}")
            # 平均 NLL = sum(batch_mean_loss * batch_targets)/sum(batch_targets)。
            # 不能对长短不同的块简单平均；item() 等待 CUDA 结果后再累计耗时。
            total_nll += output.loss.item() * y.numel()
            total_targets += y.numel()
    finally:
        model.train(was_training)
    if total_targets != len(tokens) - 1:
        raise AssertionError("完整验证目标覆盖不正确")
    mean_loss = total_nll / total_targets
    return {
        "targets": total_targets,
        "loss": mean_loss,
        "ppl": math.exp(mean_loss) if mean_loss < math.log(sys.float_info.max) else None,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        check_runtime(args.device, args.precision)
        if args.output is not None and args.output.exists():
            raise FileExistsError("报告已存在，请选择新的 --output 文件")
        loaded = load_checkpoint(args.checkpoint, device=args.device, tokenizer_path=args.tokenizer)
        tokens = read_tokens(args.data, loaded.model.config.vocab_size)
        report = {
            "iteration": loaded.iteration,
            "device": args.device,
            "precision": args.precision,
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "data_sha256": file_sha256(args.data),
            **evaluate(loaded.model, tokens, args.precision),
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return report
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

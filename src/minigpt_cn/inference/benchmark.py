"""同一 checkpoint 的 cached/uncached greedy 对照与小型推理基准。

复用生成模块的窗口策略；预热后多次测量，报告中位数，不把 pilot 数字当作正式33M预算。
"""

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from minigpt_cn.evaluation.evaluate import check_runtime
from minigpt_cn.inference.generate import _check_logits, generate, generation_step
from minigpt_cn.training.checkpoint import file_sha256, load_checkpoint


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _measure(model, prompt_ids, max_new_tokens, precision, use_cache):
    device = prompt_ids.device
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    generated, cache = prompt_ids.clone(), None
    incremental_calls, rebuilds, cache_bytes, first_seconds = 0, 0, 0, None
    started = time.perf_counter()
    for index in range(max_new_tokens):
        if use_cache and cache is not None:
            if cache[0][0].shape[-2] >= model.config.context_length:
                rebuilds += 1
            else:
                incremental_calls += 1
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
        ):
            output = generation_step(model, generated, cache, use_cache=use_cache)
        cache = output.cache
        _check_logits(output.logits[:, -1, :])
        if cache is not None:
            # 持久K/V张量大小，不等于CUDA峰值（峰值还包含权重、临时激活和拼接副本）。
            cache_bytes = max(
                cache_bytes, sum(t.numel() * t.element_size() for pair in cache for t in pair)
            )
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_id), dim=1)
        if index == 0:
            _sync(device)
            # 包含 prefill、argmax 和拼接，不包含加载模型/tokenizer/编码 prompt。
            first_seconds = time.perf_counter() - started
    _sync(device)
    total_seconds = time.perf_counter() - started
    return generated, {
        "generation_seconds": total_seconds,
        "first_token_seconds": first_seconds,
        "decode_tokens_per_second": (max_new_tokens - 1) / (total_seconds - first_seconds),
        "overall_tokens_per_second": max_new_tokens / total_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "kv_cache_bytes": cache_bytes,
        "incremental_calls": incremental_calls,
        "window_rebuilds": rebuilds,
    }


def benchmark(model, prompt_ids, *, max_new_tokens=64, precision="float32", repeats=3, warmup=1):
    for name, value, minimum in (
        ("max_new_tokens", max_new_tokens, 2),
        ("repeats", repeats, 1),
        ("warmup", warmup, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} 必须为 >= {minimum} 的整数")
    # 复用生成输入校验。固定生成长度、greedy、无EOS，避免两组工作量不同。
    generate(model, prompt_ids, 0, greedy=True, precision=precision)
    was_training = model.training
    model.eval()
    results, expected = {}, None
    try:
        for use_cache in (False, True):
            for _ in range(warmup):
                _measure(model, prompt_ids, max_new_tokens, precision, use_cache)
            measurements = []
            for _ in range(repeats):
                ids, metrics = _measure(model, prompt_ids, max_new_tokens, precision, use_cache)
                if expected is None:
                    expected = ids.clone()
                if not torch.equal(ids, expected):
                    index = (ids != expected).nonzero()[0, 1].item()
                    raise AssertionError(
                        f"cached/uncached greedy 不一致，首个差异在完整序列位置 {index}"
                    )
                measurements.append(metrics)
            # 速度/延迟取中位数；显存峰值取多次测量的最大值，不做选择性报告。
            summary = {
                name: statistics.median(row[name] for row in measurements)
                for name in (
                    "generation_seconds",
                    "first_token_seconds",
                    "decode_tokens_per_second",
                    "overall_tokens_per_second",
                )
            }
            summary.update(
                {
                    "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in measurements)
                    if prompt_ids.device.type == "cuda"
                    else None,
                    "kv_cache_bytes": max(row["kv_cache_bytes"] for row in measurements),
                    "incremental_calls": measurements[0]["incremental_calls"],
                    "window_rebuilds": measurements[0]["window_rebuilds"],
                    "trials": measurements,
                }
            )
            results["cached" if use_cache else "uncached"] = summary
    finally:
        model.train(was_training)
    return {
        "status": "passed",
        "greedy_ids_equal": True,
        "full_ids": expected[0].tolist(),
        "prompt_tokens": prompt_ids.shape[1],
        "generated_tokens": max_new_tokens,
        "device": str(prompt_ids.device),
        "precision": precision,
        "repeats": repeats,
        "warmup": warmup,
        "window_policy": "rebuild_on_slide",
        "total_speedup": results["uncached"]["generation_seconds"]
        / results["cached"]["generation_seconds"],
        **results,
        "timing_scope": "greedy generation only; excludes checkpoint load, tokenization, detokenization; EOS disabled",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        check_runtime(args.device, args.precision)
        if not args.prompt:
            raise ValueError("prompt 不能为空")
        if args.output is not None and args.output.exists():
            raise FileExistsError("报告已存在，请换新 --output 文件名")
        loaded = load_checkpoint(args.checkpoint, device=args.device, tokenizer_path=args.tokenizer)
        prompt_ids = torch.tensor(
            [loaded.tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device
        )
        report = benchmark(
            loaded.model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            precision=args.precision,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        report.update(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": file_sha256(args.checkpoint),
                "iteration": loaded.iteration,
                "model_config": asdict(loaded.model.config),
                "torch_version": str(torch.__version__),
                "hardware": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                "prompt": args.prompt,
            }
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                json.dump(report, file, ensure_ascii=False, indent=2)
                file.write("\n")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        AssertionError,
        FloatingPointError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

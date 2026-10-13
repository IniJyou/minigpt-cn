"""配置驱动的项目短训练入口；复用作业训练链路，不进行数据下载或云端操作。"""

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from minigpt_cn.config import RunConfig, TrainingConfig, load_run_config
from minigpt_cn.data.tokens import get_batch, read_tokens
from minigpt_cn.evaluation.evaluate import check_runtime, evaluate
from minigpt_cn.model import MiniGPT
from minigpt_cn.tokenizer import BPETokenizer
from minigpt_cn.training.checkpoint import file_sha256, load_checkpoint, save_checkpoint
from minigpt_cn.training.optim import AdamW, check_finite, clip_gradient_norm, lr_cosine_schedule


def train(run: RunConfig, out_dir, *, resume=None, stop_after=None):
    """新运行从随机权重开始；resume 只接受项目训练 checkpoint，不把旧 pilot 当续训。"""
    config = run.training
    check_runtime(config.device, config.precision)
    if stop_after is not None and (
        isinstance(stop_after, bool) or not isinstance(stop_after, int) or stop_after <= 0
    ):
        raise ValueError("stop_after 必须为正整数（本次停止时的总更新数）")
    end_step = min(config.max_steps, stop_after) if stop_after is not None else config.max_steps
    # 数据/tokenizer 预检查在模型构建前执行；不复制、不修改原 pilot 文件。
    tokenizer = BPETokenizer.load(run.data.tokenizer_path)
    if tokenizer.vocab_size != run.model.vocab_size or set(tokenizer.vocab) != set(
        range(run.model.vocab_size)
    ):
        raise ValueError("实际 tokenizer 的词表大小/连续 ID 必须与模型一致")
    train_tokens = read_tokens(run.data.train_path, run.model.vocab_size)
    valid_tokens = read_tokens(run.data.valid_path, run.model.vocab_size)
    if len(train_tokens) <= run.model.context_length:
        raise ValueError("训练集长度必须大于 context_length")
    hashes = {
        name: file_sha256(getattr(run.data, name))
        for name in ("train_path", "valid_path", "tokenizer_path")
    }
    start_step, previous_seconds, parent = 0, 0.0, None
    if resume is not None:
        # 先核对续训契约再恢复 RNG；这里只读取可信项目文件，不动态导入优化器。
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1 or payload.get("resume_supported") is not True:
            raise ValueError("--resume 需要项目训练 checkpoint；旧 pilot 导入文件仅用于推理")
        previous = TrainingConfig(**payload["training_config"])
        # 仅允许更改总训练终点，不隐式更改调度曲线、batch、精度或优化器超参数。
        if replace(previous, max_steps=config.max_steps) != config:
            raise ValueError("续训仅允许调整 max_steps；其他训练配置必须与 checkpoint 相同")
        if payload["model_config"] != asdict(run.model) or payload["data_config"] != asdict(
            run.data
        ):
            raise ValueError("续训的模型/数据配置与 checkpoint 不匹配")
        state = payload.get("run_state")
        if not isinstance(state, dict) or state.get("data_sha256") != hashes:
            raise ValueError("续训需要D阶段的语料校验和；文件内容不一致或缺少记录")
        if state.get("torch_version") != str(torch.__version__):
            raise ValueError("精确续训要求相同 PyTorch 版本")
        previous_seconds = state["elapsed_seconds"]
        if (
            not isinstance(previous_seconds, (int, float))
            or not np.isfinite(previous_seconds)
            or previous_seconds < 0
        ):
            raise ValueError("checkpoint 累计耗时非法")
        start_step = payload["iteration"]
        if isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0:
            raise ValueError("checkpoint 已完成步数非法")
        parent = {"path": str(Path(resume).resolve()), "sha256": file_sha256(resume)}
        del payload  # 不让预检查的 CPU 权重/矩状态与实际加载的第二份长期并存。
        if start_step >= end_step:
            print(f"已完成 {start_step} 步，本次目标 {end_step} 步：无需训练，不保存新文件。")
            return {"status": "already_complete", "iteration": start_step}
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise FileExistsError("输出目录已存在；每个新实验/续训片段使用新目录，不覆盖已有产物")
    # 原作业：固定随机种子；项目适配：补齐 Python，并用独立发生器采样 batch。
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    sampler = torch.Generator(device="cpu").manual_seed(config.seed)
    model = MiniGPT(run.model, device=config.device, dtype=torch.float32)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    if resume is not None:
        loaded = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            tokenizer_path=run.data.tokenizer_path,
            expected_model_config=run.model,
            restore_rng=True,
            sampler=sampler,
        )
        start_step = loaded.iteration
        # load_state_dict 会恢复旧参数组，记录真实生效值，不用命令行冒充实际值。
        print(f"从已完成 {start_step} 步恢复；实际优化器设置：{loaded.optimizer_settings}")
    out_dir.mkdir(parents=True)
    metadata = {
        "model": asdict(run.model),
        "training": asdict(config),
        "data": asdict(run.data),
        "data_sha256": hashes,
        "torch_version": str(torch.__version__),
        "parent_checkpoint": parent,
        "optimizer_settings": [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer.param_groups
        ],
        "parameters": sum(p.numel() for p in model.parameters()),
        "time_scope": "training/validation/checkpoint-loop; includes final checkpoint; excludes startup",
    }
    (out_dir / "run.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"参数量：{metadata['parameters']}；训练/验证 token：{len(train_tokens)}/{len(valid_tokens)}"
    )
    started = time.perf_counter()

    def elapsed():
        if config.device == "cuda":
            torch.cuda.synchronize()  # 仅在记录耗时/保存时等待 GPU，不在每个算子后强制同步。
        return time.perf_counter() - started

    def save(path, completed):
        return save_checkpoint(
            path,
            model,
            tokenizer_path=run.data.tokenizer_path,
            iteration=completed,
            optimizer=optimizer,
            training_config=config,
            data_config=run.data,
            sampler=sampler,
            run_state={
                "data_sha256": hashes,
                "elapsed_seconds": previous_seconds + elapsed(),
                "torch_version": str(torch.__version__),
                "parent_checkpoint": parent,
            },
        )

    # 本地 JSONL 是当前阶段的日志，不接入 W&B，避免上传或新增外部依赖。
    with (out_dir / "metrics.jsonl").open("x", encoding="utf-8") as log:

        def record(values):
            seconds = elapsed()
            row = {
                "process_seconds": seconds,
                "cumulative_seconds": previous_seconds + seconds,
                **values,
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(row, ensure_ascii=False))

        validation = evaluate(model, valid_tokens, config.precision)
        record({"step": start_step, "event": "validation", **validation})
        for it in range(start_step, end_step):
            model.train()
            # 原作业：先设置学习率，再清梯度、前向、反向、裁剪、更新。
            lr = lr_cosine_schedule(
                it,
                config.learning_rate,
                config.min_learning_rate,
                config.warmup_steps,
                config.lr_decay_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = get_batch(
                train_tokens,
                config.batch_size,
                run.model.context_length,
                config.device,
                generator=sampler,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=config.device,
                dtype=torch.bfloat16,
                enabled=config.precision == "bfloat16",
            ):
                # x/y: [batch,sequence] -> logits: [batch,sequence,vocabulary] -> 标量 loss。
                # 参数与梯度保持 FP32，cross_entropy 内部把 logits 转成 FP32。
                loss = model(x, targets=y).loss
            check_finite([("train_loss", loss)], f"第 {it + 1} 次更新前")
            loss.backward()
            check_finite(((name, p.grad) for name, p in model.named_parameters()), "反向梯度")
            grad_norm = clip_gradient_norm(model.parameters(), config.max_grad_norm)
            optimizer.step()
            check_finite(model.named_parameters(), "更新后参数")
            check_finite(
                (
                    (key, value)
                    for state in optimizer.state.values()
                    for key, value in state.items()
                    if torch.is_tensor(value)
                ),
                "优化器矩状态",
            )
            completed = it + 1  # checkpoint/log 的 step 始终是已完成更新数。
            if completed % config.log_interval == 0 or completed == end_step:
                record(
                    {
                        "step": completed,
                        "event": "train",
                        "loss": loss.item(),
                        "lr": lr,
                        "grad_norm_before_clip": grad_norm,
                    }
                )
            if completed % config.eval_interval == 0 or completed == end_step:
                validation = evaluate(model, valid_tokens, config.precision)
                record({"step": completed, "event": "validation", **validation})
            if completed % config.save_interval == 0:
                save(out_dir / f"step_{completed:06d}.pt", completed)
        # 无 finally 保存：NaN/Inf 或其他异常时不能产生“成功的最终 checkpoint”。
        final_path = save(out_dir / "ckpt_final.pt", completed)
        seconds = elapsed()
        result = {
            "status": "completed" if completed == config.max_steps else "stopped_at_boundary",
            "iteration": completed,
            "validation": validation,
            "checkpoint": str(final_path),
            "process_seconds": seconds,
            "cumulative_seconds": previous_seconds + seconds,
        }
        record({"step": completed, "event": "finished", **result})
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int, help="覆盖训练终点，不改变 lr_decay_steps")
    parser.add_argument("--stop-after", type=int, help="到指定总更新数后保存退出，保留原训练配置")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        run = load_run_config(args.config)
        if args.max_steps is not None:
            run = replace(run, training=replace(run.training, max_steps=args.max_steps))
        return train(run, args.out_dir, resume=args.resume, stop_after=args.stop_after)
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

"""中文生成入口：项目 checkpoint + tokenizer + prompt -> token IDs 和文本。

来源为作业 generate.py；保留 temperature、top-p、裁剪和 EOS 的核心逻辑。
项目适配：读取 ModelOutput.logits，增加 greedy、精度控制和可复现的独立采样器。
F阶段可选 KV Cache；窗口滑动后重建，保持原无缓存裁剪语义。
"""

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from minigpt_cn.evaluation.evaluate import check_runtime
from minigpt_cn.model.components import softmax
from minigpt_cn.training.checkpoint import file_sha256, load_checkpoint


def _validate_top_p(p):
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        raise TypeError("top_p 必须为数值")
    if not math.isfinite(p) or not 0 < p <= 1:
        raise ValueError("top_p 必须满足 0 < top_p <= 1")


def _check_logits(logits):
    # 允许 -inf 屏蔽候选，但每行必须至少有一个有限候选；不能让 softmax 得到 NaN。
    if (
        torch.isnan(logits).any()
        or torch.isposinf(logits).any()
        or not torch.isfinite(logits).any(-1).all()
    ):
        raise FloatingPointError("生成 logits 含 NaN/+Inf 或所有候选均被屏蔽")


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """保留累计概率首次达到/超过 p 的最小候选集合；不修改输入 logits。"""
    _validate_top_p(p)
    if not isinstance(logits, torch.Tensor) or not logits.is_floating_point():
        raise TypeError("logits 必须为浮点张量")
    if logits.ndim < 1 or logits.numel() == 0:
        raise ValueError("logits 必须包含非空 vocabulary 维")
    _check_logits(logits)
    # 原作业：对词表分数从大到小排序，形状 [...,vocabulary] 不变。
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    # softmax 得到概率，再沿词表维累加，例如 [0.5,0.25,0.25] -> [0.5,0.75,1]。
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    # 原作业的 >= 边界：p=0.5 只保留第一项，p=0.6 保留前两项。
    remove = cumulative_probs >= p
    # 标记向右移一格：跨过阈值的那个候选也必须保留，且最高概率项始终保留。
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    # 散回原词表 ID 顺序，不能直接把排序后的 mask 用在原 logits 上。
    original_order = torch.zeros_like(logits, dtype=torch.bool)
    original_order.scatter_(dim=-1, index=sorted_indices, src=remove)
    return logits.masked_fill(original_order, -torch.inf)


def generation_step(model, generated, cache=None, *, use_cache=False):
    """一次前向，供生成/基准共用；调用者负责 eval/no_grad 与 autocast。"""
    context = generated[:, -model.config.context_length :]
    if not use_cache:
        return model(context)
    if cache is None or cache[0][0].shape[-2] >= model.config.context_length:
        # 第一次 prefill 或窗口滑动：用最新窗口重建各层 K/V，RoPE 位置从0开始。
        # 不能只切掉旧 K/V：深层历史表示可能仍携带已移除 token 的信息。
        return model(context, use_cache=True)
    # 上一轮刚采样出的 token 尚未进入缓存；这次只计算它，[1,1]。
    return model(generated[:, -1:], use_cache=True, kv_cache=cache)


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    *,
    greedy: bool = False,
    precision: str = "float32",
    generator: torch.Generator | None = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """单条生成，返回包含原 prompt 的 [1,prompt_length+generated_length] 整数张量。"""
    # 原作业：prompt_ids 是 token ID，不随模型前向精度转换为 BF16。
    if not isinstance(prompt_ids, torch.Tensor):
        raise TypeError("prompt_ids 必须是 torch.Tensor")
    if prompt_ids.ndim != 2 or min(prompt_ids.shape) == 0:
        raise ValueError("prompt_ids 必须为非空 [batch,sequence] 张量")
    if prompt_ids.shape[0] != 1:
        raise ValueError("generate 目前只支持 batch_size=1")
    if prompt_ids.dtype != torch.long:
        raise TypeError("prompt_ids 必须为 torch.long")
    device = next(model.parameters()).device
    if prompt_ids.device != device:
        raise ValueError("prompt_ids 与模型必须位于同一设备")
    vocab_size = model.config.vocab_size
    if (prompt_ids < 0).any() or (prompt_ids >= vocab_size).any():
        raise ValueError(f"prompt_ids 超出词表范围 [0,{vocab_size - 1}]")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens 必须为 int")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 必须为非负整数")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise TypeError("temperature 必须为数值")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature 必须为有限正数")
    _validate_top_p(top_p)
    if not isinstance(greedy, bool):
        raise TypeError("greedy 必须为 bool")
    if not isinstance(use_cache, bool):
        raise TypeError("use_cache 必须为 bool")
    if eos_token_id is not None:
        if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
            raise TypeError("eos_token_id 必须为 int 或 None")
        if not 0 <= eos_token_id < vocab_size:
            raise ValueError("eos_token_id 超出词表范围")
    check_runtime(device.type, precision)
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator 必须为 torch.Generator 或 None")
        # PyTorch 的默认 GPU 发生器可能报告 cuda，而参数报告 cuda:0；它们可以是同一设备。
        generator_device = generator.device
        generator_index = generator_device.index
        if generator_device.type == "cuda" and generator_index is None:
            generator_index = torch.cuda.current_device()
        if generator_device.type != device.type or (
            device.type == "cuda" and generator_index != device.index
        ):
            raise ValueError("采样 Generator 与模型必须位于同一设备")
    # 原作业：提前拷贝一份，避免修改原 prompt，零新增 token 也返回独立副本。
    generated = prompt_ids.clone()
    cache = None  # 每次调用独立建立，不能把其他 prompt 的缓存带进来。
    was_training = model.training
    model.eval()  # 关闭 dropout；@no_grad 确保整个前向不记录梯度。
    try:
        for _ in range(max_new_tokens):
            # 原作业：只把最后 context_length 个 ID 送进模型，完整输出仍保留全部 prompt。
            # 形状 [1,total_length] -> [1,min(total_length,context_length)]。
            # 裁剪窗口的 RoPE 位置重新从0开始，与原无缓存生成语义相同。
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bfloat16"
            ):
                output = generation_step(model, generated, cache, use_cache=use_cache)
                logits = output.logits  # [1,sequence,V]；增量 decode 的 sequence=1。
            cache = output.cache
            # 只有最后一个位置用于预测下一个 token：[1,sequence,V] -> [1,V]。
            # 项目适配：采样概率在 FP32 中计算；BF16 只影响模型前向。
            logits = logits[:, -1, :].float()
            _check_logits(logits)
            if greedy:
                # 项目新增：直接选最高分 ID，不抽样、不消耗采样 RNG，也不应用温度/top-p。
                next_token = logits.argmax(dim=-1, keepdim=True)  # [1,1]。
            else:
                # 原作业温度公式：P(i)=exp(z_i/T)/sum_j exp(z_j/T)。分母不加额外 epsilon。
                # T 小使分布更集中，T 大使分布更平；极小 T 若溢出则明确报错。
                if temperature != 1.0:
                    logits = logits / temperature
                _check_logits(logits)
                if top_p < 1.0:
                    logits = _top_p_filter(logits, top_p)
                probs = softmax(logits, dim=-1)  # [1,V]，被屏蔽候选的概率为0。
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)  # [1,1]。
            # 拼接后长度增加1，遇到 EOS 也先保留这个 ID 再停止。
            generated = torch.cat((generated, next_token), dim=1)
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    finally:
        # 项目适配：库函数结束时恢复调用者原来的模式，包括异常路径。
        model.train(was_training)
    return generated


def _decode_report(tokenizer, ids):
    # token 可只包含半个 UTF-8 字符，必须先拼接全部字节再解码，不能逐 token decode。
    raw = b"".join(tokenizer.vocab[index] for index in ids)
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        # 保留原 tokenizer 的 replace 行为，同时明确报告“确实存在非法字节”。
        # 不能只数文本里的 � 来判断非法 UTF-8，因为用户也可能输入真正的 U+FFFD 字符。
        return raw.decode("utf-8", errors="replace"), True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="移动文件后可显式提供同一版本 tokenizer")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--use-cache", action="store_true", help="窗口未满时增量推理，窗口滑动时重建缓存"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    eos = parser.add_mutually_exclusive_group()
    eos.add_argument("--eos-token", default="<|endoftext|>")
    eos.add_argument("--no-eos", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="另存原始 IDs、文本和解码设置，不覆盖已有文件")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        check_runtime(args.device, args.precision)
        if not args.prompt:
            raise ValueError("prompt 不能为空")
        if not 0 <= args.seed < 2**64:
            raise ValueError("seed 必须为 [0,2**64) 的整数")
        if args.output is not None and args.output.exists():
            raise FileExistsError("输出已存在，请选择新文件名；不覆盖已有样例")
        loaded = load_checkpoint(args.checkpoint, device=args.device, tokenizer_path=args.tokenizer)
        tokenizer = loaded.tokenizer
        eos_id = None
        if not args.no_eos:
            if args.eos_token not in tokenizer.special_tokens:
                raise ValueError("EOS 必须是 tokenizer 中的特殊 token；可用 --no-eos 禁用")
            eos_ids = tokenizer.encode(args.eos_token)
            if len(eos_ids) != 1:
                raise ValueError("EOS 必须编码为一个 token")
            eos_id = eos_ids[0]
        prompt_ids = torch.tensor(
            [tokenizer.encode(args.prompt)], dtype=torch.long, device=args.device
        )
        # 在模型初始化/加载之后设置独立采样器，不依赖这些操作消耗了多少全局随机数。
        sampler = torch.Generator(device=prompt_ids.device).manual_seed(args.seed)
        result = generate(
            loaded.model,
            prompt_ids,
            args.max_new_tokens,
            eos_id,
            args.temperature,
            args.top_p,
            greedy=args.greedy,
            precision=args.precision,
            generator=sampler,
            use_cache=args.use_cache,
        )
        full_ids = result[0].tolist()
        new_ids = full_ids[prompt_ids.shape[1] :]
        full_text, full_invalid = _decode_report(tokenizer, full_ids)
        continuation, continuation_invalid = _decode_report(tokenizer, new_ids)
        report = {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "iteration": loaded.iteration,
            "model_config": asdict(loaded.model.config),
            "device": args.device,
            "precision": args.precision,
            "seed": args.seed,
            "decoding": "greedy" if args.greedy else "sampling",
            "use_cache": args.use_cache,
            "window_policy": "rebuild_on_slide",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "eos_token_id": eos_id,
            "max_new_tokens": args.max_new_tokens,
            "generated_tokens": len(new_ids),
            "stop_reason": "eos" if new_ids and new_ids[-1] == eos_id else "max_new_tokens",
            "prompt": args.prompt,
            "prompt_ids": prompt_ids[0].tolist(),
            "full_ids": full_ids,
            "generated_ids": new_ids,
            "full_text": full_text,
            "continuation": continuation,
            "full_utf8_invalid": full_invalid,
            "continuation_utf8_invalid": continuation_invalid,
            "replacement_character_count": continuation.count("\ufffd"),
        }
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                json.dump(report, file, ensure_ascii=False, indent=2)
                file.write("\n")
        cache_label = "开启（滑动时重建）" if args.use_cache else "关闭"
        print(
            f"checkpoint iteration={loaded.iteration}；模式={report['decoding']}；缓存={cache_label}"
        )
        print(f"Prompt：{args.prompt}\n续写：\n{continuation}")
        print(f"新增 {len(new_ids)} 个 token；停止原因={report['stop_reason']}")
        if full_invalid or continuation_invalid:
            print("警告：生成字节含非法或不完整 UTF-8，已保留替换字符 �。", file=sys.stderr)
        return report
    except (OSError, ValueError, TypeError, RuntimeError, FloatingPointError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

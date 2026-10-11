"""用同一份旧 pilot 权重、tokenizer 和完整验证集，对照作业与项目实现。

这是一次性迁移验收入口；只有这里按显式路径导入作业参考实现。
项目模型、tokenizer 和常规 CPU 测试都不依赖作业仓库。
"""

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from minigpt_cn.config import load_model_config
from minigpt_cn.model import MiniGPT
from minigpt_cn.tokenizer import BPETokenizer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tokens(path, vocab_size):
    path = Path(path)
    if path.stat().st_size < 4 or path.stat().st_size % 2:
        raise ValueError("验证文件必须包含至少两个原始 uint16 token，且字节数为偶数")
    with path.open("rb") as file:
        if file.read(6) == b"\x93NUMPY":
            raise ValueError("需要原始 uint16 二进制，不能传入带文件头的 .npy")
    tokens = np.memmap(path, dtype="<u2", mode="r")
    if int(tokens.max()) >= vocab_size:
        raise ValueError("验证数据的 token ID 超出模型词表")
    return tokens


@torch.no_grad()
def compare_validation(reference, project, tokens, reference_loss, device, precision):
    """逐块比对完整 logits 与 NLL，包括最后不足 context_length 的尾块。"""
    if len(tokens) < 2:
        raise ValueError("完整验证至少需要两个 token")
    reference.eval()
    project.eval()
    total_reference, total_project, total_targets, max_error = 0.0, 0.0, 0, 0.0
    started = time.perf_counter()
    context = project.config.context_length
    for start in range(0, len(tokens) - 1, context):
        end = min(start + context, len(tokens) - 1)
        x = torch.from_numpy(np.asarray(tokens[start:end], dtype=np.int64)).unsqueeze(0).to(device)
        y = (
            torch.from_numpy(np.asarray(tokens[start + 1 : end + 1], dtype=np.int64))
            .unsqueeze(0)
            .to(device)
        )
        with torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=precision == "bfloat16"
        ):
            expected = reference(x)
            actual = project(x, targets=y)
            expected_loss = reference_loss(expected.float(), y)
        if not torch.isfinite(actual.loss) or not torch.isfinite(expected_loss):
            raise FloatingPointError(f"验证块 start={start} 出现非有限 loss")
        # 同一进程/设备/精度、同一权重：本轮要求逐元素完全一致，不只比较 argmax。
        torch.testing.assert_close(actual.logits, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.loss, expected_loss, rtol=0, atol=0)
        max_error = max(max_error, (actual.logits.float() - expected.float()).abs().max().item())
        count = y.numel()
        total_reference += expected_loss.item() * count
        total_project += actual.loss.item() * count
        total_targets += count
    if total_targets != len(tokens) - 1:
        raise AssertionError("完整验证目标覆盖不正确")
    # 按目标数加权：不能对长短不同的 batch loss 简单取平均。
    reference_mean, project_mean = total_reference / total_targets, total_project / total_targets
    return {
        "targets": total_targets,
        "reference_loss": reference_mean,
        "project_loss": project_mean,
        "reference_ppl": math.exp(reference_mean),
        "project_ppl": math.exp(project_mean),
        "max_logits_abs_error": max_error,
        "loss_abs_error": abs(reference_mean - project_mean),
        "elapsed_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def compare_greedy(reference, project, tokenizer, device, precision):
    results = []
    eos = tokenizer.encode("<|endoftext|>")
    if len(eos) != 1:
        raise AssertionError("EOS 必须编码为单个 token")
    for prompt in ("数学是一门", "在计算机科学中，", "中国的历史"):
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated = ids.clone()
        for _ in range(32):
            cropped = generated[:, -project.config.context_length :]
            with torch.autocast(
                device_type=device, dtype=torch.bfloat16, enabled=precision == "bfloat16"
            ):
                expected = reference(cropped)[:, -1].argmax(-1, keepdim=True)
                actual = project(cropped).logits[:, -1].argmax(-1, keepdim=True)
            if not torch.equal(actual, expected):
                raise AssertionError(f"prompt={prompt!r} 的 greedy token 不一致")
            generated = torch.cat((generated, actual), dim=1)
            if actual.item() == eos[0]:
                break
        results.append(
            {
                "prompt": prompt,
                "ids": generated[0].tolist(),
                "text": tokenizer.decode(generated[0].tolist()),
            }
        )
    return results


def verify(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请选择 --device cpu --precision float32")
    config = load_model_config(args.config)
    root = args.assignment_root.resolve()
    checkpoint_path = root / "out/pilot_3m_bpe2048/ckpt_final.pt"
    tokenizer_path = root / "data/processed/tokenizer_2048.json"
    valid_path = root / "data/processed/valid.bin"
    documents_path = root / "data/processed/valid.jsonl"

    # 只在显式验收入口加载原作业，生产代码不作路径注入或跨仓库 import。
    sys.path.insert(0, str(root))
    from cs336_basics.train_pilot_tokenizer import load_tokenizer
    from cs336_basics.transformer_lm import TransformerLM
    from cs336_basics.utils import cross_entropy as reference_loss

    tokenizer = BPETokenizer.load(tokenizer_path)
    original_tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("模型词表必须与实际 tokenizer 大小相同")
    if tokenizer.vocab != original_tokenizer.vocab or tokenizer.merges != original_tokenizer.merges:
        raise AssertionError("tokenizer 词表或 merge 顺序发生变化")
    texts = ["", "中文百科😀\n第二行", "<|endoftext|>"]
    documents = [
        json.loads(line)
        for line in documents_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts.extend(document["text"] for document in documents)
    for text in texts:
        encoded = tokenizer.encode(text)
        if encoded != original_tokenizer.encode(text) or tokenizer.decode(encoded) != text:
            raise AssertionError("tokenizer 编码或 UTF-8 round-trip 不一致")
    tokens = read_tokens(valid_path, config.vocab_size)
    reconstructed = []
    for document in documents:
        reconstructed.extend(tokenizer.encode(document["text"]))
        reconstructed.extend(tokenizer.encode("<|endoftext|>"))
    if not np.array_equal(tokens, reconstructed):
        raise AssertionError("按验证文档重新编码的 ID 与原 valid.bin 不一致")

    reference = TransformerLM(
        vocab_size=config.vocab_size,
        max_seq_len=config.context_length,
        d_model=config.d_model,
        num_layers=config.n_layers,
        num_heads=config.n_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=args.device,
        dtype=torch.float32,
        tie_embeddings=config.tie_embeddings,
        dropout=config.dropout,
        embedding_init_std=config.embedding_init_std,
    )
    project = MiniGPT(config, device=args.device, dtype=torch.float32)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    if config.tie_embeddings and not torch.equal(state["token_embeddings.W"], state["lm_head.W"]):
        raise ValueError("旧 checkpoint 的两份共享权重键数值不一致")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise FloatingPointError("checkpoint 含 NaN/Inf")
    reference.load_state_dict(state, strict=True)
    project.load_state_dict(state, strict=True)
    if set(reference.state_dict()) != set(project.state_dict()):
        raise AssertionError("迁移前后 state_dict 键不一致")
    for model in (reference, project):
        if model.token_embeddings.W is not model.lm_head.W:
            raise AssertionError("embedding 和 LM head 必须共享同一个 Parameter")
        if sum(parameter is model.token_embeddings.W for parameter in model.parameters()) != 1:
            raise AssertionError("共享参数没有去重")
    parameter_count = sum(parameter.numel() for parameter in project.parameters())
    if parameter_count != sum(parameter.numel() for parameter in reference.parameters()):
        raise AssertionError("唯一参数量不一致")
    validation = compare_validation(
        reference, project, tokens, reference_loss, args.device, args.precision
    )
    greedy = compare_greedy(reference, project, tokenizer, args.device, args.precision)
    report = {
        "status": "passed",
        "checkpoint_iteration": checkpoint["iteration"],
        "unique_parameters": parameter_count,
        "model_config": asdict(config),
        "torch_version": torch.__version__,
        "device": args.device,
        "precision": args.precision,
        "hardware": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in (
                ("checkpoint", checkpoint_path),
                ("tokenizer", tokenizer_path),
                ("valid_tokens", valid_path),
                ("valid_documents", documents_path),
                ("config", args.config),
            )
        },
        "tokenizer_documents_checked": len(documents),
        "validation": validation,
        "reference_source_sha256": {
            name: sha256(root / "cs336_basics" / name)
            for name in (
                "transformer_lm.py",
                "transformer_block.py",
                "attention.py",
                "rope.py",
                "linear.py",
                "embedding.py",
                "rmsnorm.py",
                "swiglu.py",
                "dropout.py",
                "utils.py",
                "tokenizer.py",
                "train_pilot_tokenizer.py",
            )
        },
        "shared_parameter_identity": True,
        "shared_parameter_unique": True,
        "greedy_samples": greedy,
        "scope": "无缓存迁移一致性；缓存路径另行验收；不重训、不证明正式 33M 资源或中文生成质量",
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[3] / "configs/pilot_3m.yaml"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    verify(args)


if __name__ == "__main__":
    main()

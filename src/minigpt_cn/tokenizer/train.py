"""小语料 BPE 训练/保存 CLI；只读UTF-8训练文本，不下载、不清洗、不训练模型。"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from minigpt_cn.tokenizer import BPETokenizer, train_bpe


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_round_trip(tokenizer, text):
    # IDs是变长一维整数列表，不是[B,S]模型输入；未知字符仍可拆成初始UTF-8字节。
    if tokenizer.decode(tokenizer.encode(text)) != text:
        raise RuntimeError("tokenizer 编解码 round-trip 检查失败")
    for index, token in enumerate(tokenizer.special_tokens, start=256):
        if tokenizer.encode(token) != [index]:
            raise RuntimeError("特殊token必须保持声明顺序，并编码为单个ID")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="仅训练集的UTF-8纯文本")
    parser.add_argument(
        "--vocab-size", type=int, required=True, help="包含初始字节/特殊token的目标上限"
    )
    parser.add_argument("--output", type=Path, required=True, help="新的tokenizer JSON路径，不覆盖")
    parser.add_argument(
        "--special-token",
        action="append",
        help="可重复；不传则默认<|endoftext|>，传入则替换默认列表",
    )
    parser.add_argument("--check-text", default="今天学习 Transformer，第3讲介绍注意力机制。😀")
    args = parser.parse_args(argv)
    try:
        source, output = args.input.resolve(), args.output.resolve()
        if not source.is_file():
            raise ValueError("--input 必须是已存在的UTF-8纯文本文件")
        if source.suffix.lower() in {".jsonl", ".json", ".bin", ".npy", ".pt"}:
            raise ValueError("--input 需要纯文本，不直接读取JSONL/JSON、token二进制或权重")
        if source.stat().st_size == 0:
            raise ValueError("--input 训练文本不能为空")
        if args.output.exists() or args.output.is_symlink() or output.exists():
            raise FileExistsError("--output 已存在，请换新文件名；不覆盖已有tokenizer")
        if output.suffix.lower() != ".json":
            raise ValueError("--output 必须使用.json后缀")
        specials = ["<|endoftext|>"] if args.special_token is None else args.special_token
        source_hash = _sha256(source)
        started = time.perf_counter()
        vocab, merges = train_bpe(source, args.vocab_size, specials)
        training_seconds = time.perf_counter() - started  # 仅训练函数，不含hash/保存/重载。
        if _sha256(source) != source_hash:
            raise RuntimeError("训练期间输入文本发生变化，请固定输入版本后重试")
        tokenizer = BPETokenizer(vocab, merges, specials)
        _check_round_trip(tokenizer, args.check_text)
        output.parent.mkdir(parents=True, exist_ok=True)
        # x模式再次防守：即使预检查之后有人创建同名文件，也不能覆盖它。
        tokenizer.save(output, overwrite=False)
        restored = BPETokenizer.load(output)
        if (restored.vocab, restored.merges, restored.special_tokens) != (vocab, merges, specials):
            raise RuntimeError("tokenizer 保存重载后词表/merge顺序/特殊token不一致")
        _check_round_trip(restored, args.check_text)
        report = {
            "input": str(source),
            "input_sha256": source_hash,
            "input_bytes": source.stat().st_size,
            "output": str(output),
            "tokenizer_sha256": _sha256(output),
            "requested_vocab_size": args.vocab_size,
            "actual_vocab_size": restored.vocab_size,
            "merges": len(merges),
            "special_token_ids": {t: restored.encode(t)[0] for t in specials},
            "training_seconds": training_seconds,
            "reload_verified": True,
            "round_trip_verified": True,
        }
        # 词表不足不补伪token，明确提示；后续模型必须匹配actual_vocab_size。
        if restored.vocab_size < args.vocab_size:
            print(
                "警告：语料的可合并pair已耗尽，实际词表小于目标；没有填充token。", file=sys.stderr
            )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

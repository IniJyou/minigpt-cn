"""固定 pilot 三集合 → 同版本 tokenizer 的无文件头小端 uint16；不训练 BPE。"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from minigpt_cn.data.export_text import EOS, SPLIT_RULE, SUFFIX, _load_manifest
from minigpt_cn.data.prepare import _reject_nonfinite, _sha256
from minigpt_cn.data.splitting import choose_split
from minigpt_cn.tokenizer import BPETokenizer


def _load_tokenizer(path, expected_hash, expected_size):
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(c not in "0123456789abcdef" for c in expected_hash)
    ):
        raise ValueError("必须提供已验收 tokenizer 的64位小写 SHA-256")
    if type(expected_size) is not int or not 257 <= expected_size <= 65536:
        raise ValueError("词表必须包含字节和EOS，且可由 uint16 表示（257..65536）")
    if _sha256(path) != expected_hash:
        raise ValueError("tokenizer SHA-256 与指定的已验收版本不一致")
    try:
        tokenizer = BPETokenizer.load(path)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("tokenizer JSON 格式不完整或损坏") from error
    if tokenizer.vocab_size != expected_size:
        raise ValueError("实际 tokenizer 词表与指定词表大小不一致，不填充token")
    if set(tokenizer.vocab) != set(range(expected_size)) or any(
        tokenizer.vocab[i] != bytes([i]) for i in range(256)
    ):
        raise ValueError("tokenizer ID必须连续，且保留256个初始字节")
    if (
        tokenizer.special_tokens != [EOS]
        or tokenizer.encode(EOS) != [256]
        or len(tokenizer.merges) != expected_size - 257
    ):
        raise ValueError("必须使用仅含EOS（ID256）的已验收BPE词表与有序merges")
    if _sha256(path) != expected_hash:
        raise RuntimeError("加载期间 tokenizer 发生变化")
    return tokenizer


def _check_source(path, info):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"集合必须是普通 JSONL 文件，不能是符号链接：{path}")
    if any(
        type(info[key]) is not int or info[key] <= 0 for key in ("bytes", "documents", "characters")
    ):
        raise ValueError(f"集合清单的字节/文档/字符数必须为正整数：{path}")
    if path.stat().st_size != info["bytes"] or _sha256(path) != info["sha256"]:
        raise ValueError(f"集合字节数或 SHA-256 与来源清单不一致：{path}")


def _encode_split(source, info, target, tokenizer, seed):
    documents = characters = tokens = 0
    source_digest, encoded_digest = hashlib.sha256(), hashlib.sha256()
    split = source.stem
    with source.open("rb") as fin, target.open("xb") as fout:
        for line_number, raw in enumerate(fin, start=1):
            source_digest.update(raw)
            try:
                record = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError("每行必须是含 text 字符串的 JSON 对象")
                text = record["text"]
                if choose_split(text, seed=seed) != split:
                    raise ValueError("正文不属于当前固定集合，不能重新划分或挪文档")
                if EOS in text:
                    raise ValueError("正文包含保留的EOS字面量，不能冒充文档边界")
                piece = text + SUFFIX
                ids = tokenizer.encode(piece)
                # ids为一维变长整数序列；先查范围再转uint16，防止溢出后悄悄回绕。
                if not ids or min(ids) < 0 or max(ids) >= tokenizer.vocab_size:
                    raise ValueError("编码ID超出 tokenizer 词表，不能转换为uint16")
                if ids.count(256) != 1 or ids[-2:] != [256, 10]:
                    raise ValueError("每篇必须有且仅有一个EOS，结尾为EOS加LF")
                if tokenizer.decode(ids) != piece:
                    raise RuntimeError("正文/EOS/LF 编解码往返不一致")
                # 按篇写入，不缓存整份token语料；<u2明确固定小端、无.npy文件头。
                encoded = np.asarray(ids, dtype="<u2").tobytes()
                fout.write(encoded)
                encoded_digest.update(encoded)
                documents += 1
                characters += len(text)
                tokens += len(ids)
            except (ValueError, TypeError, RuntimeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
    if source_digest.hexdigest() != info["sha256"]:
        raise RuntimeError(f"编码期间输入发生变化：{source}")
    if documents != info["documents"] or characters != info["characters"]:
        raise ValueError(f"集合文档数或字符数与来源清单不一致：{source}")
    digest = _sha256(target)
    if target.stat().st_size != tokens * 2 or digest != encoded_digest.hexdigest():
        raise RuntimeError(f"保存读回的uint16字节与实际编码不一致：{target}")
    # 读回磁盘ID，核对范围和每篇边界；正文往返已覆盖全部文档。
    stored = np.memmap(target, dtype="<u2", mode="r")
    minimum, maximum = int(stored.min()), int(stored.max())
    ends = np.flatnonzero(stored == 256)
    valid = (
        maximum < tokenizer.vocab_size
        and len(ends) == documents
        and int(ends[-1]) == tokens - 2
        and np.all(stored[ends + 1] == 10)
    )
    del stored  # Windows下及时关闭映射，避免保留文件句柄。
    if not valid:
        raise RuntimeError(f"磁盘ID范围或EOS/LF边界不一致：{target}")
    return {
        "path": target.name,
        "documents": documents,
        "characters": characters,
        "tokens": tokens,
        "bytes": target.stat().st_size,
        "sha256": digest,
        "min_id": minimum,
        "max_id": maximum,
        "eos_count": len(ends),
        "round_trip_verified": True,
        "readback_verified": True,
    }


def encode_splits(
    manifest_path: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    expected_tokenizer_sha256: str,
    expected_vocab_size: int = 16000,
) -> dict:
    """只编码已有pilot三集合；成功清单最后创建，失败的部分产物保留供诊断。"""
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("--output-dir 已存在，先检查已有产物；不覆盖、不盲目重编码")
    manifest_path, tokenizer_path = Path(manifest_path).resolve(), Path(tokenizer_path).resolve()
    manifest, manifest_hash = _load_manifest(manifest_path)
    if manifest["processing"]["split_rule"] != SPLIT_RULE:
        raise ValueError("本入口仅编码固定pilot正文哈希划分，不接作品候选或重新划分")
    tokenizer = _load_tokenizer(tokenizer_path, expected_tokenizer_sha256, expected_vocab_size)
    sources = {
        split: manifest_path.parent / f"{split}.jsonl" for split in ("train", "valid", "test")
    }
    for split, source in sources.items():
        _check_source(source, manifest["files"][split])
    # 全部输入通过检查后才创建新目录；并发创建同名目录也不能覆盖。
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    files = {
        split: _encode_split(
            source,
            manifest["files"][split],
            output / f"{split}.bin",
            tokenizer,
            manifest["processing"]["seed"],
        )
        for split, source in sources.items()
    }
    for split, source in sources.items():
        _check_source(source, manifest["files"][split])
    if (
        _sha256(manifest_path) != manifest_hash
        or _sha256(tokenizer_path) != expected_tokenizer_sha256
    ):
        raise RuntimeError("编码期间来源清单或 tokenizer 发生变化，不写成功清单")
    report = {
        "format_version": 1,
        "kind": "encoded_pilot_tokens",
        "status": "completed",
        "purpose": "pilot_loop_verification",
        "source_manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "source_splits": {
            split: {**manifest["files"][split], "path": str(source)}
            for split, source in sources.items()
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "sha256": expected_tokenizer_sha256,
            "vocab_size": tokenizer.vocab_size,
            "merges": len(tokenizer.merges),
            "special_token_ids": {EOS: 256},
        },
        "storage": {"dtype": "<u2", "byte_order": "little", "header": False},
        "boundary": {"special_token": EOS, "suffix": SUFFIX, "encoding": "utf-8"},
        "processing": {"split_rule": SPLIT_RULE, "seed": manifest["processing"]["seed"]},
        "encoding_seconds": time.perf_counter() - started,
        "files": files,
    }
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="固定pilot三集合来源清单")
    parser.add_argument("--tokenizer", type=Path, required=True, help="已保存验收的tokenizer JSON")
    parser.add_argument("--tokenizer-sha256", required=True, help="已验收版本的SHA-256，不允许混用")
    parser.add_argument("--vocab-size", type=int, default=16000, help="必须等于实际词表，默认16000")
    parser.add_argument("--output-dir", type=Path, required=True, help="新的输出目录，不覆盖")
    args = parser.parse_args(argv)
    try:
        report = encode_splits(
            args.manifest,
            args.tokenizer,
            args.output_dir,
            expected_tokenizer_sha256=args.tokenizer_sha256,
            expected_vocab_size=args.vocab_size,
        )
        print(
            json.dumps(
                {"output_dir": str(args.output_dir.resolve()), **report},
                ensure_ascii=False,
                indent=2,
            )
        )
        return report
    except FileExistsError as error:
        parser.error(
            f"{error}\n请先执行 docs/token-encoding-guide.md 中的独立只读验收，判断已有产物是否完整。"
        )
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as error:
        parser.error(
            f"{error}\n本次未成功完成编码；若留下缺少完整 manifest.json 的新产物，不可用于训练。"
        )


if __name__ == "__main__":
    main()

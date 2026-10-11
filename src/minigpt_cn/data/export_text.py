"""只从已验收清单的 train.jsonl 导出 BPE 训练纯文本；不训练或编码 token。"""

import argparse
import hashlib
import json
from pathlib import Path

from minigpt_cn.data.prepare import _reject_nonfinite, _sha256
from minigpt_cn.data.sampling import HASH_RULE, _HashPrefixSampler
from minigpt_cn.data.splitting import choose_split

EOS = "<|endoftext|>"
SUFFIX = EOS + "\n"
SPLIT_RULE = "SHA256(UTF8(seed + NUL + text)) % 100; 0..97=train, 98=valid, 99=test"


def _load_manifest(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
    if not isinstance(manifest, dict):
        raise ValueError("数据处理清单必须是 JSON 对象")
    try:
        if type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
            raise ValueError("仅支持版本1的数据处理清单")
        files = manifest["files"]
        if not isinstance(files, dict) or set(files) != {"train", "valid", "test"}:
            raise ValueError("清单必须包含三个 split")
        for name, info in files.items():
            if info["path"] != f"{name}.jsonl":
                raise ValueError("清单中的 split 路径必须是固定文件名，不能改指其他文件")
            if type(info["documents"]) is not int or info["documents"] <= 0:
                raise ValueError("清单中的三个 split 必须非空")
        if manifest["counts"]["saved"] != sum(info["documents"] for info in files.values()):
            raise ValueError("清单的文档总数不一致")
        train = files["train"]
        if any(type(train[key]) is not int or train[key] <= 0 for key in ("bytes", "characters")):
            raise ValueError("train 的字节数/字符数必须是正整数")
        digest = train["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("train 的 SHA-256 格式不正确")
        if manifest["processing"]["split_rule"] != SPLIT_RULE:
            raise ValueError("清单的划分规则不受支持")
        choose_split("参数校验", seed=manifest["processing"]["seed"])
    except (KeyError, TypeError) as error:
        raise ValueError("不完整或结构错误的数据处理清单") from error
    return manifest, hashlib.sha256(raw).hexdigest()


def export_train_text(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    max_bytes: int | None = None,
    selection: str = "prefix",
    sample_seed: int | None = None,
) -> dict:
    """验证完整 train，导出完整文档前缀；可选源顺序或独立哈希顺序。

    默认 prefix 保留原行为。hash 必须显式给预算，sample_seed 默认42，与划分种子独立。
    不读取 valid/test 正文，不再次清洗。输出目录必须新建；失败可能留下部分 train.txt，
    只有最终 export_manifest.json 写出后才是成功导出。
    """
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("max_bytes 必须是正整数或 None")
    if selection not in ("prefix", "hash"):
        raise ValueError("selection 必须是 prefix 或 hash")
    if sample_seed is not None and type(sample_seed) is not int:
        raise TypeError("sample_seed 必须是整数或 None，不能是 bool")
    if selection == "prefix" and sample_seed is not None:
        raise ValueError("sample_seed 仅用于 selection=hash，不能用于源文件前缀")
    if selection == "hash" and max_bytes is None:
        raise ValueError("selection=hash 必须显式指定 max_bytes，不能无预算缓存文档")
    sampler = (
        _HashPrefixSampler(
            max_bytes, 42 if sample_seed is None else sample_seed, len(SUFFIX.encode())
        )
        if selection == "hash"
        else None
    )
    manifest_path = Path(manifest_path).resolve()
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("--output-dir 已存在，请换新目录；不覆盖旧文本")
    manifest, manifest_hash = _load_manifest(manifest_path)
    info = manifest["files"]["train"]
    # 来源固定为清单同目录的 train.jsonl；不开放 --split，也不跟随 train 文件的符号链接。
    source = manifest_path.parent / "train.jsonl"
    if source.is_symlink() or not source.is_file():
        raise ValueError("train.jsonl 必须是清单同目录下的普通文件，不能是符号链接")
    if source.stat().st_size != info["bytes"] or _sha256(source) != info["sha256"]:
        raise ValueError("train.jsonl 的字节数或 SHA-256 与清单不一致")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    documents = characters = exported = exported_characters = written_bytes = 0
    accepting = True
    digest = hashlib.sha256()
    source_stats: dict[str | None, dict] = {}
    text_path = output / "train.txt"
    with source.open("rb") as fin, text_path.open("xb") as fout:
        for line_number, line in enumerate(fin, start=1):
            digest.update(line)
            try:
                record = json.loads(line.decode("utf-8"), parse_constant=_reject_nonfinite)
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError("每行必须是含 text 字符串的 JSON 对象")
                text = record["text"]
                if (
                    not text.strip()
                    or choose_split(text, seed=manifest["processing"]["seed"]) != "train"
                ):
                    raise ValueError("发现空正文或不属于 train 的正文")
                if EOS in text:
                    raise ValueError("正文包含保留的 EOS 字面量，不能冒充文档边界")
                documents += 1
                characters += len(text)
                # 与原 pilot 相同：正文不再清洗，仅追加一个 EOS 和 LF。
                # 二进制写 UTF-8，避免 Windows 自动把 LF 改成 CRLF。
                piece = (text + SUFFIX).encode("utf-8")
                if sampler is not None:
                    # 来源只用于统计，不影响排序。支持 source 字符串或 source.name；
                    # 其他结构/缺失统一记为 null，不从标题、URL 猜测站点或许可。
                    label = record.get("source")
                    if isinstance(label, dict):
                        label = label.get("name")
                    if not isinstance(label, str) or not label.strip():
                        label = None
                    sampler.add(text, label)
                elif accepting and (max_bytes is None or written_bytes + len(piece) <= max_bytes):
                    fout.write(piece)
                    exported += 1
                    exported_characters += len(text)
                    written_bytes += len(piece)
                else:
                    # 到达预算即锁定前缀，不跳过大文档再选后面的小文档，也不切半个字符。
                    # 仍扫描余下 train，核对整份输入的哈希、文档数与字符数。
                    accepting = False
            except (ValueError, TypeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
        if sampler is not None:
            # 先验证完整输入再写样本，不能只校验入选文档而漏掉文件后面的坏数据。
            if digest.hexdigest() != info["sha256"]:
                raise RuntimeError("导出期间 train.jsonl 发生变化")
            if documents != info["documents"] or characters != info["characters"]:
                raise ValueError("train 的文档数或正文字符数与清单不一致")
            for candidate in sampler.ordered():
                if written_bytes + candidate.bytes > max_bytes:
                    break  # 不写越界文档，不截断 UTF-8，也不跳过它去填更短的文档。
                fout.write(candidate.key[1] + SUFFIX.encode("utf-8"))
                exported += 1
                exported_characters += candidate.characters
                written_bytes += candidate.bytes
                stats = source_stats.setdefault(
                    candidate.source, {"documents": 0, "characters": 0, "bytes": 0}
                )
                stats["documents"] += 1
                stats["characters"] += candidate.characters
                stats["bytes"] += candidate.bytes  # 包含该来源入选文档的 EOS/LF。
    if digest.hexdigest() != info["sha256"] or _sha256(source) != info["sha256"]:
        raise RuntimeError("导出期间 train.jsonl 发生变化")
    if _sha256(manifest_path) != manifest_hash:
        raise RuntimeError("导出期间来源清单发生变化")
    if documents != info["documents"] or characters != info["characters"]:
        raise ValueError("train 的文档数或正文字符数与清单不一致")
    if exported == 0:
        raise ValueError("字节预算容不下第一篇完整文档；请增大 max_bytes")
    if text_path.stat().st_size != written_bytes:
        raise RuntimeError("导出文件字节数不一致")
    report = {
        "format_version": 1,
        "kind": "bpe_training_text",
        "source_manifest": {"path": str(manifest_path), "sha256": manifest_hash},
        "source_split": {**info, "split": "train", "path": str(source)},
        "selection": {"strategy": "whole_document_prefix", "max_bytes": max_bytes},
        "boundary": {"special_token": EOS, "suffix": SUFFIX, "encoding": "utf-8"},
        "output": {
            "path": "train.txt",
            "documents": exported,
            "characters": exported_characters,
            "bytes": written_bytes,
            "sha256": _sha256(text_path),
        },
    }
    if sampler is not None:
        report["selection"].update(
            strategy="whole_document_hash_prefix",
            algorithm_version=1,
            sample_seed=sampler.seed,
            hash_rule=HASH_RULE,
        )
        report["output"]["sources"] = [
            {"source": label, **source_stats[label]}
            for label in sorted(source_stats, key=lambda label: (label is not None, label or ""))
        ]
    # 新的导出清单独立保存，不修改数据处理清单或旧 BPE 样本。
    with (output / "export_manifest.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, required=True, help="已验收的数据处理 manifest.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="新的文本输出目录，不覆盖")
    parser.add_argument("--max-bytes", type=int, help="含 EOS/LF 的字节上限；不传则导出全部 train")
    parser.add_argument(
        "--selection",
        choices=("prefix", "hash"),
        default="prefix",
        help="prefix 按源文件顺序（默认）；hash 按固定哈希顺序，必须给 --max-bytes",
    )
    parser.add_argument("--sample-seed", type=int, help="仅用于 hash 的采样种子，默认42")
    args = parser.parse_args(argv)
    try:
        report = export_train_text(
            args.manifest,
            args.output_dir,
            max_bytes=args.max_bytes,
            selection=args.selection,
            sample_seed=args.sample_seed,
        )
        print(
            json.dumps(
                {"output_dir": str(args.output_dir.resolve()), **report},
                ensure_ascii=False,
                indent=2,
            )
        )
        return report
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(
            f"{error}\n未成功导出；缺少完整 export_manifest.json 的文本不可用于 BPE 训练。"
        )


if __name__ == "__main__":
    main()

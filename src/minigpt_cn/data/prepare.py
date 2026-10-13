"""已提取正文的 JSONL 处理入口；不下载 dump，不训练 tokenizer，不编码 token。"""

import argparse
import hashlib
import json
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

from minigpt_cn import __version__
from minigpt_cn.data.cleaning import clean_document, normalize_text
from minigpt_cn.data.splitting import choose_split


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_nonfinite(value: str) -> None:
    # Python 的 JSON 解析默认接受 NaN/Infinity，但它们不是合法的 JSON 数据。
    raise ValueError(f"JSON 不能包含 {value}")


def prepare_jsonl(
    input_paths: list[str | Path],
    output_dir: str | Path,
    *,
    seed: int = 42,
    min_chars: int = 1,
    max_chars: int | None = None,
    to_simplified: bool = True,
) -> dict:
    """清洗、精确去重、长度过滤、hash 划分；保留首次记录的非正文元数据。

    input_paths 是路径列表，顺序决定重复正文保留哪份来源记录。
    output_dir 必须不存在。失败可能留下本次新建的部分文件，成功才写 manifest.json；
    不删除这些诊断文件，也不覆盖旧输出。seen 保存全部唯一正文，尚非外存去重方案。
    """
    if isinstance(input_paths, (str, bytes, Path)) or not input_paths:
        raise ValueError("input_paths 必须是非空的路径列表")
    sources = [Path(path).resolve() for path in input_paths]
    if len(set(sources)) != len(sources):
        raise ValueError("--input 不应重复指定同一个文件")
    if any(not path.is_file() for path in sources):
        raise ValueError("--input 必须是已存在的 UTF-8 JSONL 文件")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("--output-dir 已存在，请换新目录；不覆盖旧数据")
    output = output.resolve()
    # 复用已有函数校验选项，让空输入文件也不能绕过非法阈值/种子。
    clean_document("", min_chars=min_chars, max_chars=max_chars, to_simplified=to_simplified)
    choose_split("参数校验", seed=seed)
    inputs = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sources
    ]
    # mkdir 的 exist_ok=False 是第二道防线：预检查后别人创建同名目录也不能覆盖。
    output.mkdir(parents=True, exist_ok=False)
    counts = {"read": 0, "blank_lines": 0, "empty": 0, "duplicates": 0, "length_filtered": 0}
    stats = {name: {"documents": 0, "characters": 0} for name in ("train", "valid", "test")}
    seen: set[str] = set()
    with ExitStack() as stack:
        handles = {
            name: stack.enter_context(
                (output / f"{name}.jsonl").open("x", encoding="utf-8", newline="\n")
            )
            for name in stats
        }
        for source, info in zip(sources, inputs):
            digest = hashlib.sha256()
            # 按原始字节行读取，既能精确核对输入哈希，也能报告坏 UTF-8 所在的行。
            with source.open("rb") as file:
                for line_number, raw_line in enumerate(file, start=1):
                    digest.update(raw_line)
                    try:
                        line = raw_line.decode("utf-8")
                        if not line.strip():
                            counts["blank_lines"] += 1
                            continue
                        counts["read"] += 1
                        record = json.loads(line, parse_constant=_reject_nonfinite)
                        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                            raise ValueError("每行必须是 JSON 对象，且 text 字段必须是字符串")
                        text = normalize_text(record["text"], to_simplified=to_simplified)
                        if not text:
                            counts["empty"] += 1
                            continue
                        # 与 deduplicate_texts 相同的正文 set 规则；这里还要保留原记录的元数据。
                        # 去重先于长度过滤：同一不合长度的正文重现时计作重复，而非再计一次长度过滤。
                        if text in seen:
                            counts["duplicates"] += 1
                            continue
                        seen.add(text)
                        if len(text) < min_chars or (
                            max_chars is not None and len(text) > max_chars
                        ):
                            counts["length_filtered"] += 1
                            continue
                        # 仅替换正文；标题、URL、来源、许可等其他字段原样保留。
                        record["text"] = text
                        split = choose_split(text, seed=seed)
                        handles[split].write(
                            json.dumps(record, ensure_ascii=False, allow_nan=False)
                        )
                        handles[split].write("\n")
                        stats[split]["documents"] += 1
                        stats[split]["characters"] += len(text)
                    except (ValueError, TypeError) as error:
                        raise ValueError(f"{source}:{line_number}: {error}") from error
            if digest.hexdigest() != info["sha256"]:
                raise RuntimeError(f"处理时输入发生变化：{source}")

    # 文件已关闭，再核对输入版本与输出哈希；不能把混合版本的数据记为成功实验。
    for source, info in zip(sources, inputs):
        if _sha256(source) != info["sha256"] or source.stat().st_size != info["bytes"]:
            raise RuntimeError(f"处理时输入发生变化：{source}")
    counts["saved"] = sum(item["documents"] for item in stats.values())
    if counts["read"] != sum(
        counts[key] for key in ("empty", "duplicates", "length_filtered", "saved")
    ):
        raise RuntimeError("处理计数不平衡")
    empty_splits = [name for name, item in stats.items() if item["documents"] == 0]
    if empty_splits:
        raise ValueError(f"划分为空：{', '.join(empty_splits)}；请增加语料，不为凑比例挪文档")
    for name, item in stats.items():
        path = output / f"{name}.jsonl"
        item.update(path=path.name, bytes=path.stat().st_size, sha256=_sha256(path))
    manifest = {
        "format_version": 1,
        "package_version": __version__,
        "inputs": inputs,
        "processing": {
            "normalization": "NFC",
            "to_simplified": to_simplified,
            "opencc_version": version("OpenCC"),
            "opencc_config": "t2s.json" if to_simplified else None,
            "deduplication": "exact_normalized_text_first_occurrence",
            "min_chars": min_chars,
            "max_chars": max_chars,
            "length_unit": "unicode_code_points",
            "seed": seed,
            "split_rule": "SHA256(UTF8(seed + NUL + text)) % 100; 0..97=train, 98=valid, 99=test",
        },
        "counts": counts,
        "files": stats,
    }
    # 成功清单最后写。文件均以 x 模式创建，绝不覆盖；失败目录需换名字重跑。
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, action="append", required=True, help="可重复，顺序保留"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="新的输出目录，不覆盖")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--keep-traditional", action="store_true", help="关闭默认的繁转简")
    args = parser.parse_args(argv)
    try:
        manifest = prepare_jsonl(
            args.input,
            args.output_dir,
            seed=args.seed,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            to_simplified=not args.keep_traditional,
        )
        print(
            json.dumps(
                {"output_dir": str(args.output_dir.resolve()), **manifest},
                ensure_ascii=False,
                indent=2,
            )
        )
        return manifest
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(
            f"{error}\n本次未完成；若留下新目录，缺少完整 manifest.json 的输出不可用于训练。"
        )


if __name__ == "__main__":
    main()

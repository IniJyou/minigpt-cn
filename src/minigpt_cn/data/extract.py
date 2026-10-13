"""已验证dump → 纯正文候选/隔离/审计清单；本阶段不去重、划分或训练。"""

import argparse
import bz2
import hashlib
import json
import sys
from collections import Counter
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from urllib.parse import quote, urlparse

from minigpt_cn.data.parse_xml import (
    DEFAULT_XML_BYTES,
    HOSTS,
    _iter_pages,
    _small_file,
    _verified_input,
)
from minigpt_cn.data.prepare import _reject_nonfinite, _sha256
from minigpt_cn.data.wikitext import ALGORITHM, extract_wikitext


def _write(file, record):
    file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def _review_key(record):
    return record["source"]["name"], record["page_id"], record["revision_id"]


def _load_reviews(path):
    """小样本人工审阅文件，最多2MiB；不是全站版权的自动判定器。"""
    if path is None:
        return {}, None
    raw = _small_file(Path(path))
    reviews = {}
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            record = json.loads(line, parse_constant=_reject_nonfinite)
            key = _review_key(record)
            if key[0] not in HOSTS or any(type(i) is not int or i < 1 for i in key[1:]):
                raise ValueError("来源/ID不合法")
            if key in reviews:
                raise ValueError("重复审批记录")
            if record["decision"] not in {"approved", "rejected"}:
                raise ValueError("decision必须为approved/rejected；不能提交未填写的模板")
            for field in ("wikitext_sha256", "text_sha256"):
                value = record[field]
                if not isinstance(value, str) or len(value) != 64:
                    raise ValueError("需要两份完整SHA-256")
                int(value, 16)
            for field in ("reviewer", "reviewed_at", "notes"):
                if not isinstance(record[field], str) or not record[field].strip():
                    raise ValueError(f"{field}不能为空")
            if record["decision"] == "approved":
                if record["quality_checked"] is not True:
                    raise ValueError("approved必须显式quality_checked=true")
                for field in ("license_identifier", "work_id"):
                    if not isinstance(record[field], str) or not record[field].strip():
                        raise ValueError(f"{field}不能为空")
                for field in ("license_evidence", "work_evidence"):
                    urls = record[field]
                    if (
                        not isinstance(urls, list)
                        or not urls
                        or not all(
                            isinstance(url, str)
                            and urlparse(url).scheme == "https"
                            and urlparse(url).netloc
                            for url in urls
                        )
                    ):
                        raise ValueError(f"{field}需要非空HTTPS证据URL列表")
            reviews[key] = record
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"审阅文件第{number}行：{error}") from error
    return reviews, {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _stats():
    return {
        "pages_scanned": 0,
        "xml_bytes_read": 0,
        "xml_document_complete": False,
        "peak_root_children": 0,
        "stop_reason": None,
        "siteinfo": None,
    }


def extract_bodies(
    download_manifests,
    output_dir,
    *,
    max_pages=100,
    max_xml_bytes=DEFAULT_XML_BYTES,
    min_chars=20,
    max_raw_chars=1000000,
    reviews_path=None,
):
    """每个来源有独立扫描/解压预算；max_pages=None显式扫描到XML EOF。

    原始压缩包在处理前后完整重算摘要。内存保留一页AST和固定计数，不收集整个语料。
    approved表示人工声明已审阅，不是程序提供法律保证；阶段2仍须去重/作品级划分。
    """
    if max_pages is not None and (type(max_pages) is not int or not 1 <= max_pages <= 10000):
        raise ValueError("样本max_pages必须为1～10000的整数；全量使用None/--full")
    for name, value in (
        ("max_xml_bytes", max_xml_bytes),
        ("min_chars", min_chars),
        ("max_raw_chars", max_raw_chars),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name}必须为正整数")
    paths = [Path(path) for path in download_manifests]
    if not paths or any(path.is_symlink() for path in paths):
        raise ValueError("需要至少一份普通下载清单，不能是符号链接")
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("不能重复传入同一下载清单")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("--output-dir已存在，请换新目录；不覆盖旧正文")
    started = perf_counter()
    reviews, review_input = _load_reviews(reviews_path)
    code_paths = {
        name: Path(__file__).with_name(name)
        for name in ("extract.py", "wikitext.py", "parse_xml.py", "cleaning.py")
    }
    code_hashes = {name: _sha256(path) for name, path in code_paths.items()}
    inputs = [_verified_input(path.resolve()) for path in paths]
    identities = {(source["name"], source["dump_date"]) for source, _, _ in inputs}
    if len(identities) != len(inputs):
        raise ValueError("同一来源/版本不能重复提取")
    output.mkdir(parents=True, exist_ok=False)
    counts = Counter(approved=0, quarantine=0, rejected=0, candidates=0)
    reasons, issues, used_reviews, sources = Counter(), Counter(), set(), []
    examples = []
    names = ("approved.jsonl", "quarantine.jsonl", "rejected.jsonl", "review_queue.jsonl")
    with ExitStack() as stack:
        files = {
            name: stack.enter_context((output / name).open("x", encoding="utf-8", newline="\n"))
            for name in names
        }
        for source, dump, input_info in inputs:
            source_name = source["name"]
            stats = _stats()
            source_counts = Counter(approved=0, quarantine=0, rejected=0, candidates=0)
            with bz2.open(dump, "rb") as stream:
                for skip, raw in _iter_pages(stream, source_name, max_pages, max_xml_bytes, stats):
                    record = {key: value for key, value in raw.items() if key != "wikitext"}
                    record.update(
                        source={"name": source_name, "dump_date": source["dump_date"]},
                        license=source["license"],
                    )
                    host = HOSTS[source_name]
                    record["url"] = (
                        host + "/wiki/" + quote(raw["title"].replace(" ", "_"), safe="/")
                    )
                    record["history_url"] = (
                        f"{host}/w/index.php?curid={raw['page_id']}&action=history"
                    )
                    if "revision_id" in raw:
                        record["revision_url"] = f"{host}/w/index.php?oldid={raw['revision_id']}"
                    rejected = ["xml_" + skip] if skip else []
                    if not skip:
                        body = extract_wikitext(
                            raw["wikitext"],
                            source=source_name,
                            title=raw["title"],
                            min_chars=min_chars,
                            max_raw_chars=max_raw_chars,
                        )
                        record.update(body)
                        if source_name == "zhwiki" and "work_group" in record:
                            # 标题可能改名；跨版本核对时页面ID比标题哈希更稳定。
                            record["work_group"]["suggested_id"] = f"zhwiki:page:{raw['page_id']}"
                        record["text_sha256"] = hashlib.sha256(body["text"].encode()).hexdigest()
                        rejected = body["reject_reasons"]
                        issues.update(body["issues"])
                    review = None if skip else reviews.get(_review_key(record))
                    if review is not None:
                        # 权重式审批不可复用到新revision/不同清理结果；只改标题也要重新核对身份。
                        used_reviews.add(_review_key(record))
                        if any(
                            review[field] != record[field]
                            for field in ("wikitext_sha256", "text_sha256")
                        ):
                            raise ValueError(f"审批摘要过期：{source_name}/{raw['title']}")
                        if review["decision"] == "approved" and rejected:
                            raise ValueError(
                                "结构性拒绝的页面不能人工批准；需补齐正文/修正规则再提取"
                            )
                        if review["decision"] == "rejected":
                            rejected = [*rejected, "human_rejected"]
                    if rejected:
                        category = "rejected"
                        record.pop("text", None)  # 不把不完整正文输出为可直接训练的text。
                        record["reject_reasons"] = rejected
                        reasons.update(rejected)
                    else:
                        counts["candidates"] += 1
                        source_counts["candidates"] += 1
                        if review is not None:
                            category = "approved"
                            record["review"] = review
                            record["work_group"].update(status="reviewed", id=review["work_id"])
                        else:
                            category = "quarantine"
                            record["quarantine_reasons"] = [
                                "license_not_reviewed",
                                "quality_not_reviewed",
                            ]
                            if source_name == "zhwikisource":
                                record["quarantine_reasons"].append("work_group_not_reviewed")
                            template = {
                                key: record[key]
                                for key in (
                                    "source",
                                    "page_id",
                                    "revision_id",
                                    "wikitext_sha256",
                                    "text_sha256",
                                )
                            }
                            template.update(
                                title=raw["title"],
                                decision="pending",
                                quality_checked=False,
                                reviewer="",
                                reviewed_at="",
                                notes="",
                                license_identifier="",
                                license_evidence=[],
                                work_id="",
                                work_evidence=[],
                            )
                            _write(files["review_queue.jsonl"], template)
                    record["status"] = category
                    _write(files[category + ".jsonl"], record)
                    counts[category] += 1
                    source_counts[category] += 1
                    if len(examples) < 12 and category != "approved":
                        examples.append(
                            {
                                "source": source_name,
                                "title": raw["title"],
                                "status": category,
                                "reasons": rejected,
                                "issues": record.get("issues", []),
                            }
                        )
                    if stats["pages_scanned"] % 1000 == 0:
                        print(f"{source_name}: 已扫描{stats['pages_scanned']}页", file=sys.stderr)
            if stats["pages_scanned"] != sum(
                source_counts[k] for k in ("approved", "quarantine", "rejected")
            ):
                raise RuntimeError("来源页数与输出记录数不平衡")
            sources.append(
                {"source": source, "input": input_info, "xml": stats, "counts": dict(source_counts)}
            )
    if set(reviews) != used_reviews:
        raise ValueError("审阅文件有未遇到的revision；检查来源、扫描范围或过期ID")
    for source, dump, input_info in inputs:
        final_source, final_dump, final_info = _verified_input(
            Path(input_info["download_manifest"])
        )
        if (final_source, final_dump, final_info) != (source, dump, input_info):
            raise RuntimeError("正文提取期间输入发生变化")
    if _load_reviews(reviews_path)[1] != review_input:
        raise RuntimeError("正文提取期间审批文件发生变化")
    if {name: _sha256(path) for name, path in code_paths.items()} != code_hashes:
        raise RuntimeError("正文提取期间处理代码发生变化；请换新目录重跑")
    quality = {
        "counts": dict(counts),
        "reject_reasons": dict(reasons),
        "issues": dict(issues),
        "examples": examples,
        "all_licenses_automatically_approved": False,
        "unknown_templates_require_human_quality_check": True,
    }
    with (output / "quality_report.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(quality, ensure_ascii=False, indent=2) + "\n")
    report = {
        "format_version": 1,
        "kind": "wikimedia_body_extraction",
        "status": "completed",
        "scope": "full_dump" if max_pages is None else "prefix_sample",
        "ready_for_training": False,
        "counts": dict(counts),
        "sources": sources,
        "processing": {
            "algorithm": ALGORITHM,
            "mwparserfromhell": version("mwparserfromhell"),
            "opencc": version("OpenCC"),
            "max_pages_per_source": max_pages,
            "max_xml_bytes_per_source": max_xml_bytes,
            "min_chars": min_chars,
            "max_raw_chars": max_raw_chars,
            "to_simplified_body": False,
            "work_key_to_simplified": True,
            "deduplication": None,
            "split": None,
            "transclusion_expanded": False,
            "code_sha256": code_hashes,
        },
        "review_input": review_input,
        "elapsed_seconds": perf_counter() - started,
        "files": {
            name: {
                "path": name,
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in (*names, "quality_report.json")
        },
    }
    # 成功标记最后才写；断电/异常的部分输出不能被下游视为已验收语料。
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--max-pages", type=int, help="每来源扫描页数，默认100，最大10000")
    scope.add_argument("--full", action="store_true", help="全量，必须显式指定解压字节预算")
    parser.add_argument("--max-xml-bytes", type=int)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-raw-chars", type=int, default=1000000)
    parser.add_argument("--reviews", type=Path, help="已人工填写的审批JSONL，不接受pending模板")
    args = parser.parse_args(argv)
    if args.full and args.max_xml_bytes is None:
        parser.error("--full必须显式提供 --max-xml-bytes；预算是解压读取量，不是RAM分配")
    try:
        print("核对本地压缩包和快照摘要（开始/结束各一次），不联网。", file=sys.stderr)
        report = extract_bodies(
            args.download_manifest,
            args.output_dir,
            max_pages=None
            if args.full
            else (args.max_pages if args.max_pages is not None else 100),
            max_xml_bytes=args.max_xml_bytes
            if args.max_xml_bytes is not None
            else DEFAULT_XML_BYTES,
            min_chars=args.min_chars,
            max_raw_chars=args.max_raw_chars,
            reviews_path=args.reviews,
        )
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "scope": report["scope"],
                    "counts": report["counts"],
                    "elapsed_seconds": report["elapsed_seconds"],
                    "ready_for_training": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return report
    except (OSError, EOFError, ValueError, TypeError, RuntimeError) as error:
        parser.error(f"{error}\n提取未完成；无完整manifest.json的输出不能用于后续处理。")


if __name__ == "__main__":
    main()

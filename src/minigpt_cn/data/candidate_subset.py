"""中断候选 → 有预算的完整文档子集；不解压XML、不批准、不去重/划分/训练。"""

import argparse
import hashlib
import heapq
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

from minigpt_cn.data.download import _load_source, _validate_status
from minigpt_cn.data.parse_xml import HOSTS, _small_file
from minigpt_cn.data.prepare import _reject_nonfinite, _sha256

DEFAULT_BODY_BYTES = 32 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024 * 1024
HASH_RULE = "SHA256(UTF8(candidate-subset-v1 + NUL + seed + NUL) + UTF8(text)); tie=UTF8(text),canonical_JSON"
ISSUES = {
    "invalid_entity",
    "unknown_template",
    "unknown_tag",
    "unexpanded_parser_function",
    "unexpanded_argument",
    "unknown_node",
}
SCOPE = "interrupted_scan_prefix_subset"


def _json(record):
    return json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _signature(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"需要普通文件，不能是符号链接：{path}")
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _provenance(path):
    """仅核对已保存的下载清单/小快照，不打开压缩包，不冒充重新验证dump内容。"""
    raw = _small_file(path)
    m = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
    try:
        if (
            type(m["format_version"]) is not int
            or m["format_version"] != 1
            or m["kind"] != "wikimedia_dump_download"
            or m["status"] != "verified"
        ):
            raise ValueError("需要原始verified下载清单")
        source, snapshots = m["source"]["name"], {}
        for name, key in (
            ("source_config.yaml", "source_config"),
            ("dumpstatus.json", "official_status"),
        ):
            data = _small_file(path.parent / name)
            digest = hashlib.sha256(data).hexdigest()
            if m[key]["path"] != name or m[key]["sha256"] != digest:
                raise ValueError("下载快照路径/摘要不匹配")
            snapshots[name] = {"sha256": digest, "data": data}
        info, config = _load_source(path.parent / "source_config.yaml", source)
        if config != snapshots["source_config.yaml"]["data"] or m["source"] != {
            "name": source,
            **info,
        }:
            raise ValueError("下载来源与固定快照不一致")
        status = _validate_status(info, snapshots["dumpstatus.json"]["data"])
        if any(m["official_status"][k] != v for k, v in status.items()):
            raise ValueError("官方状态快照与下载清单不一致")
        dump = m["file"]
        if (
            dump["path"] != info["filename"]
            or dump["bytes"] != info["bytes"]
            or dump["sha1"] != info["sha1"]
        ):
            raise ValueError("声明的压缩包信息与固定配置不一致")
        _digest(dump["sha256"])
    except (KeyError, TypeError) as error:
        raise ValueError("下载清单缺少字段或结构错误") from error
    return {
        "source": m["source"],
        "download_manifest": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
        "snapshots": {k: v["sha256"] for k, v in snapshots.items()},
        "declared_dump": dump,
        "dump_content_rechecked": False,
        "raw_wikitext_revalidated": False,
    }


def _digest(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("需要小写十六进制SHA-256")


def _candidate(record, source):
    """核对记录内部一致性与来源声明；不证明其正文与原dump修订逐字对应。"""
    try:
        name, date = source["name"], source["dump_date"]
        if (
            record["source"] != {"name": name, "dump_date": date}
            or record["license"] != source["license"]
        ):
            raise ValueError("候选来源/许可声明与下载快照不一致")
        if record["status"] != "quarantine" or "review" in record or record["reject_reasons"] != []:
            raise ValueError("只接受未审批、无结构性排除的quarantine候选")
        if type(record["namespace"]) is not int or record["namespace"] != 0:
            raise ValueError("候选必须属于主命名空间")
        for key in ("page_id", "revision_id"):
            if type(record[key]) is not int or record[key] < 1:
                raise ValueError("页面/修订ID必须是正整数")
        for key in ("title", "timestamp", "text"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"{key}必须是非空字符串")
        text = record["text"]
        if "<|endoftext|>" in text or any(
            (ord(c) < 32 or ord(c) == 127) and c not in "\r\n\t" for c in text
        ):
            raise ValueError("候选不能包含EOS字面量/非法控制字符")
        _digest(record["wikitext_sha256"])
        _digest(record["text_sha256"])
        body = text.encode("utf-8")
        if hashlib.sha256(body).hexdigest() != record["text_sha256"]:
            raise ValueError("正文SHA-256不匹配")
        if type(record["metrics"]["text_chars"]) is not int or record["metrics"][
            "text_chars"
        ] != len(text):
            raise ValueError("正文字符计数不匹配")
        issues = record["issues"]
        if not isinstance(issues, list) or not all(
            isinstance(i, str) and i in ISSUES for i in issues
        ):
            raise ValueError("候选质量警告结构错误或含结构性问题")
        reasons = record["quarantine_reasons"]
        if not isinstance(reasons, list) or not {
            "license_not_reviewed",
            "quality_not_reviewed",
        } <= set(reasons):
            raise ValueError("候选必须保留许可/质量待审状态")
        expected = "page_identity" if name == "zhwiki" else "needs_review"
        if record["work_group"]["status"] != expected:
            raise ValueError("作品分组状态不应被自动批准")
        host = HOSTS[name]
        urls = {
            "url": host + "/wiki/" + quote(record["title"].replace(" ", "_"), safe="/"),
            "revision_url": f"{host}/w/index.php?oldid={record['revision_id']}",
            "history_url": f"{host}/w/index.php?curid={record['page_id']}&action=history",
        }
        if any(record[k] != v for k, v in urls.items()):
            raise ValueError("候选页面/修订/历史URL与身份不匹配")
        return body
    except (KeyError, TypeError) as error:
        raise ValueError("候选记录缺少字段或结构错误") from error


def _records(path):
    with path.open("rb") as file:
        number = 0
        while raw := file.readline(MAX_LINE_BYTES + 1):
            number += 1
            try:
                # 原写入器每条记录以LF结束；缺LF按中断尾行处理，不悄悄舍弃。
                if len(raw) > MAX_LINE_BYTES or not raw.endswith(b"\n") or not raw.strip():
                    raise ValueError("空行、超过16MiB或截断尾行（缺LF）")
                record = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
                if not isinstance(record, dict):
                    raise ValueError("每行必须是JSON对象")
                yield number, raw, record
            except (ValueError, RecursionError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error


@dataclass
class _Pick:
    key: tuple[bytes, bytes, bytes]

    def __lt__(self, other):
        return self.key > other.key  # 最大堆：先丢弃哈希顺序最靠后的候选。

    @property
    def body_bytes(self):
        return len(self.key[1])


def _key(record, body, seed):
    digest = hashlib.sha256(f"candidate-subset-v1\0{seed}\0".encode() + body).digest()
    # 相同正文保留不同来源；规范JSON为最终平局键，保证重排输入不改变输出。
    return digest, body, _json(record).encode("utf-8")


def sample_candidates(
    candidates, download_manifest, output_dir, *, max_body_bytes=DEFAULT_BODY_BYTES, sample_seed=42
):
    if type(max_body_bytes) is not int or max_body_bytes < 1:
        raise ValueError("max_body_bytes必须是正整数，不能是bool")
    if type(sample_seed) is not int:
        raise TypeError("sample_seed必须是整数，不能是bool")
    path, download, output = Path(candidates), Path(download_manifest), Path(output_dir)
    initial = _signature(path)
    if (path.parent / "manifest.json").exists():
        raise ValueError("原目录已有提取清单；本入口仅处理明确声明的中断候选")
    if output.exists() or output.is_symlink():
        raise FileExistsError("--output-dir已存在，请换新目录；不覆盖候选")
    path, download, output = path.resolve(), download.absolute(), output.resolve()
    started = perf_counter()
    provenance = _provenance(download)
    code_hash = _sha256(Path(__file__))
    heap, retained, lines, file_bytes, body_bytes, chars = [], 0, 0, 0, 0, 0
    digest, warnings = hashlib.sha256(), Counter()
    for number, raw, record in _records(path):
        try:
            body = _candidate(record, provenance["source"])
            # 同一次规范JSON编码用于有限值检查、稳定平局键和最终写出，避免重复编码全语料。
            pick = _Pick(_key(record, body, sample_seed))
        except (ValueError, RecursionError) as error:
            raise ValueError(f"{path}:{number}: {error}") from error
        digest.update(raw)
        lines += 1
        file_bytes += len(raw)
        body_bytes += len(body)
        chars += len(record["text"])
        warnings.update(record["issues"])
        heapq.heappush(heap, pick)
        retained += pick.body_bytes
        # 保留“累计字节首次覆盖预算”的最短哈希前缀。最终越界那篇不写。
        # 正文字节保留量 < 预算 + 一篇；元数据/对象额外占RAM，非硬RSS上限。
        while retained - heap[0].body_bytes >= max_body_bytes:
            retained -= heapq.heappop(heap).body_bytes
        if lines % 10000 == 0:
            print(f"已核对{lines}条落盘候选（不是XML页面）", file=sys.stderr)
    if not lines:
        raise ValueError("候选文件为空")
    ordered = sorted(heap, key=lambda pick: pick.key)
    selected, used = [], 0
    for pick in ordered:
        if used + pick.body_bytes > max_body_bytes:
            break  # 不截断，也不跳过大文档后继续填小文档。
        selected.append(pick)
        used += pick.body_bytes
    if not selected:
        raise ValueError("预算容不下哈希排序后的第一篇完整正文；请调整预算")
    if (
        _signature(path) != initial
        or file_bytes != initial[2]
        or _sha256(path) != digest.hexdigest()
        or (path.parent / "manifest.json").exists()
    ):
        raise RuntimeError("读取期间候选文件发生变化")
    if _provenance(download) != provenance or _sha256(Path(__file__)) != code_hash:
        raise RuntimeError("采样期间来源元数据或处理代码发生变化")
    output.mkdir(parents=True, exist_ok=False)
    chosen_warnings, examples, chosen_chars = Counter(), [], 0
    with (output / "candidates.jsonl").open("xb") as file:
        for pick in selected:
            file.write(pick.key[2] + b"\n")
            record = json.loads(pick.key[2])
            chosen_chars += len(record["text"])
            chosen_warnings.update(record["issues"])
            if len(examples) < 8:
                examples.append(
                    {
                        "title": record["title"],
                        "issues": record["issues"],
                        "preview": record["text"][:160],
                    }
                )
    quality = {
        "input_issues": dict(sorted(warnings.items())),
        "selected_issues": dict(sorted(chosen_warnings.items())),
        "examples": examples,
        "quality_reviewed": False,
        "license_reviewed": False,
    }
    with (output / "quality_report.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(quality, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    report = {
        "format_version": 1,
        "kind": "interrupted_candidate_subset",
        "status": "completed",
        "scope": SCOPE,
        "ready_for_training": False,
        "input": {
            "path": str(path),
            "bytes": file_bytes,
            "sha256": digest.hexdigest(),
            "original_extraction_manifest_present": False,
            "original_extraction_completed": False,
            "original_xml_pages_scanned": None,
        },
        "provenance": provenance,
        "processing": {
            "algorithm": "candidate_subset_v1",
            "max_body_bytes": max_body_bytes,
            "sample_seed": sample_seed,
            "hash_rule": HASH_RULE,
            "budget_unit": "UTF8_body_bytes",
            "deduplication": None,
            "split": None,
            "to_simplified": False,
            "code_sha256": code_hash,
            "internal_consistency_checked": True,
        },
        "counts": {
            "input_documents": lines,
            "input_body_bytes": body_bytes,
            "input_characters": chars,
            "selected_documents": len(selected),
            "selected_body_bytes": used,
            "selected_characters": chosen_chars,
            "approved": 0,
        },
        "files": {
            name: {
                "path": name,
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in ("candidates.jsonl", "quality_report.json")
        },
        "elapsed_seconds": perf_counter() - started,
    }
    # 写成功标记前重新读子集复核；既不改旧目录，也不补原来的全量manifest。
    _verify(output, report)
    if (
        _signature(path) != initial
        or _provenance(download) != provenance
        or _sha256(Path(__file__)) != code_hash
        or (path.parent / "manifest.json").exists()
    ):
        raise RuntimeError("写出期间候选/来源元数据/代码发生变化")
    report["elapsed_seconds"] = perf_counter() - started
    with (output / "subset_manifest.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def _verify(root, m):
    if (
        not isinstance(m, dict)
        or type(m.get("format_version")) is not int
        or m["format_version"] != 1
        or m.get("kind") != "interrupted_candidate_subset"
        or m.get("status") != "completed"
        or m.get("scope") != SCOPE
        or m.get("ready_for_training") is not False
        or type(m["counts"]["approved"]) is not int
        or m["counts"]["approved"] != 0
        or m["processing"]["hash_rule"] != HASH_RULE
        or m["input"]["original_extraction_manifest_present"] is not False
        or m["input"]["original_extraction_completed"] is not False
        or m["input"]["original_xml_pages_scanned"] is not None
        or m["provenance"]["dump_content_rechecked"] is not False
        or m["provenance"]["raw_wikitext_revalidated"] is not False
        or m["processing"]["deduplication"] is not None
        or m["processing"]["split"] is not None
        or m["processing"]["to_simplified"] is not False
    ):
        raise ValueError("需要完整的中断候选子集清单，不能修改为训练就绪/全量成功")
    budget, seed = m["processing"]["max_body_bytes"], m["processing"]["sample_seed"]
    if type(budget) is not int or budget < 1 or type(seed) is not int:
        raise ValueError("子集预算/seed不合法")
    n, body_bytes, chars, last = 0, 0, 0, None
    warnings = Counter()
    for name in ("candidates.jsonl", "quality_report.json"):
        info = m["files"][name]
        path = root / name
        if (
            info["path"] != name
            or _signature(path)[2] != info["bytes"]
            or _sha256(path) != info["sha256"]
        ):
            raise ValueError("子集文件路径/字节数/摘要不匹配")
    for _, _, record in _records(root / "candidates.jsonl"):
        body = _candidate(record, m["provenance"]["source"])
        key = _key(record, body, m["processing"]["sample_seed"])
        if last is not None and key < last:
            raise ValueError("候选未按固定哈希顺序排列")
        last = key
        n += 1
        body_bytes += len(body)
        chars += len(record["text"])
        warnings.update(record["issues"])
    c = m["counts"]
    if (
        n < 1
        or n != c["selected_documents"]
        or body_bytes != c["selected_body_bytes"]
        or chars != c["selected_characters"]
        or body_bytes > m["processing"]["max_body_bytes"]
        or c["input_documents"] < n
        or c["input_body_bytes"] < body_bytes
        or c["input_characters"] < chars
    ):
        raise ValueError("子集计数或正文预算不一致")
    quality = json.loads(
        _small_file(root / "quality_report.json").decode("utf-8"), parse_constant=_reject_nonfinite
    )
    if (
        quality["quality_reviewed"] is not False
        or quality["license_reviewed"] is not False
        or quality["selected_issues"] != dict(warnings)
    ):
        raise ValueError("质量报告不能自动标记审阅完成，警告计数必须一致")
    return {
        "passed": True,
        "documents": n,
        "body_bytes": body_bytes,
        "scope": SCOPE,
        "ready_for_training": False,
    }


def verify_subset(manifest_path):
    path = Path(manifest_path)
    return _verify(
        path.absolute().parent,
        json.loads(_small_file(path).decode("utf-8"), parse_constant=_reject_nonfinite),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--candidates", type=Path)
    mode.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--download-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_BODY_BYTES)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.verify_manifest:
            if args.download_manifest or args.output_dir:
                raise ValueError("只读复核模式不使用下载清单/输出目录参数")
            report = verify_subset(args.verify_manifest)
        else:
            if args.download_manifest is None or args.output_dir is None:
                raise ValueError("采样需要--download-manifest和--output-dir")
            print(
                "只核对已有JSONL和小型来源快照；不打开dump、不解压XML、不批准候选。",
                file=sys.stderr,
            )
            full = sample_candidates(
                args.candidates,
                args.download_manifest,
                args.output_dir,
                max_body_bytes=args.max_body_bytes,
                sample_seed=args.sample_seed,
            )
            report = {
                "output_dir": str(args.output_dir),
                "scope": full["scope"],
                "counts": full["counts"],
                "elapsed_seconds": full["elapsed_seconds"],
                "ready_for_training": False,
            }
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return report
    except KeyboardInterrupt:
        parser.exit(130, "用户中断；无完整且通过复核的subset_manifest.json不能视为成功。\n")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(f"{error}\n子集未完成；不修改原候选，不能补写原全量成功清单。")


if __name__ == "__main__":
    main()

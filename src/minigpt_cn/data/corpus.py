"""阶段2：仅整理已审批正文，磁盘精确去重、作品级固定划分与泄漏复核。"""

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import ExitStack, closing
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlparse

from minigpt_cn.data.cleaning import clean_document, normalize_text
from minigpt_cn.data.parse_xml import HOSTS, _small_file
from minigpt_cn.data.prepare import _reject_nonfinite, _sha256

SPLITS = ("train", "valid", "test")
WORK_SPLIT_RULE = "SHA256(UTF8(corpus-work-v1 + NUL + seed + NUL + work_id)) % 100; 0..97=train, 98=valid, 99=test"
MAX_RECORD_BYTES = 16 * 1024 * 1024


def choose_work_split(work_id, seed=42):
    """桶比例98/1/1不是精确篇数比例；同一作品的所有章节使用同一键。

    work_id必须来自人工确认，不能用每章不同的正文哈希，也不自动采纳标题分组建议。
    """
    if not isinstance(work_id, str) or not work_id.strip() or "\0" in work_id:
        raise ValueError("work_id必须是非空字符串，不能含NUL")
    if type(seed) is not int:
        raise TypeError("seed必须是整数，不能是bool")
    digest = hashlib.sha256(f"corpus-work-v1\0{seed}\0{work_id}".encode()).hexdigest()
    bucket = int(digest, 16) % 100
    return "train" if bucket < 98 else ("valid" if bucket == 98 else "test")


def _json(record):
    return json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _lines(path, expected_sha256=None):
    # 限制单行字节，避免readline一次分配巨型输入；不是进程RSS硬上限。
    with path.open("rb") as file:
        number = 0
        digest = hashlib.sha256() if expected_sha256 is not None else None
        while raw := file.readline(MAX_RECORD_BYTES + 1):
            number += 1
            if digest is not None:
                digest.update(raw)
            try:
                if len(raw) > MAX_RECORD_BYTES or not raw.strip():
                    raise ValueError("JSONL行为空或超过16MiB")
                record = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
                if not isinstance(record, dict):
                    raise ValueError("每行必须是JSON对象")
                yield record
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error
        if digest is not None and digest.hexdigest() != expected_sha256:
            raise RuntimeError("读取期间approved正文文件发生变化")


def _file_info(path, **extra):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"需要普通文件，不能是符号链接：{path}")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path), **extra}


def _check_file(root, name, info):
    if not isinstance(info, dict) or not {"path", "bytes", "sha256"} <= info.keys():
        raise ValueError(f"文件摘要结构不完整：{name}")
    if info.get("path") != name or _file_info(root / name) != {
        key: info[key] for key in ("path", "bytes", "sha256")
    }:
        raise ValueError(f"文件路径/大小/SHA-256与清单不一致：{name}")


def _extraction(path):
    raw = _small_file(path)
    m = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
    try:
        if (
            type(m["format_version"]) is not int
            or m["format_version"] != 1
            or m["kind"] != "wikimedia_body_extraction"
            or m["status"] != "completed"
            or m["scope"] not in {"prefix_sample", "full_dump"}
        ):
            raise ValueError("需要阶段1完整正文提取清单，不能传普通JSONL或隔离文件")
        n = m["counts"]["approved"]
        if type(n) is not int or n < 1:
            raise ValueError("approved正文为空：先核对许可、质量和作品分组；不能使用quarantine代替")
        _check_file(path.parent, "approved.jsonl", m["files"]["approved.jsonl"])
        source_ids = {(s["source"]["name"], s["source"]["dump_date"]) for s in m["sources"]}
        if not source_ids or any(name not in HOSTS for name, _ in source_ids):
            raise ValueError("正文提取清单来源不合法")
    except (KeyError, TypeError) as error:
        raise ValueError("正文提取清单缺少字段或结构错误") from error
    return m, source_ids, hashlib.sha256(raw).hexdigest()


def _approved(record, source_ids):
    """审批文件被更换/手工编辑时仍校验，不能只看status字符串就放行。"""
    try:
        source = record["source"]
        if (source["name"], source["dump_date"]) not in source_ids:
            raise ValueError("记录来源与提取清单不一致")
        if (
            record["status"] != "approved"
            or type(record["namespace"]) is not int
            or record["namespace"] != 0
        ):
            raise ValueError("仅接受已审批的主命名空间正文")
        for field in ("page_id", "revision_id"):
            if type(record[field]) is not int or record[field] < 1:
                raise ValueError("页面/修订ID必须是正整数")
        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("审批正文必须是非空字符串")
        if hashlib.sha256(text.encode()).hexdigest() != record["text_sha256"]:
            raise ValueError("审批正文的SHA-256不匹配")
        if (
            not isinstance(record["wikitext_sha256"], str)
            or len(record["wikitext_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in record["wikitext_sha256"])
        ):
            raise ValueError("原文SHA-256格式不正确")
        if "<|endoftext|>" in text:
            raise ValueError("正文不能包含保留的EOS字面量")
        if any((ord(c) < 32 or ord(c) == 127) and c not in "\r\n\t" for c in text):
            raise ValueError("正文不能包含非法控制字符")
        review = record["review"]
        if review["decision"] != "approved" or review["quality_checked"] is not True:
            raise ValueError("缺少明确的人工质量/许可审批")
        for field in ("source", "page_id", "revision_id", "wikitext_sha256", "text_sha256"):
            if review[field] != record[field]:
                raise ValueError("审批与正文来源/ID/摘要不一致")
        for field in ("reviewer", "reviewed_at", "notes", "license_identifier", "work_id"):
            if not isinstance(review[field], str) or not review[field].strip():
                raise ValueError(f"审批{field}不能为空")
        for field in ("license_evidence", "work_evidence"):
            values = review[field]
            if (
                not isinstance(values, list)
                or not values
                or not all(
                    isinstance(v, str) and urlparse(v).scheme == "https" and urlparse(v).netloc
                    for v in values
                )
            ):
                raise ValueError(f"审批{field}需要非空HTTPS证据列表")
        group = record["work_group"]
        if group["status"] != "reviewed" or group["id"] != review["work_id"]:
            raise ValueError("必须使用已确认的work_id，不能直接采用suggested_id")
        if record.get("reject_reasons"):
            raise ValueError("结构性排除的正文不能进入语料")
        choose_work_split(group["id"])
        return group["id"]
    except (KeyError, TypeError) as error:
        raise ValueError("已审批记录缺少字段或结构错误") from error


def _prepared(metadata, text, digest, split):
    record = json.loads(metadata)
    record.update(
        text=text,
        status="prepared",
        extracted_text_sha256=record["text_sha256"],
        text_sha256=digest,
        corpus={"split": split, "normalized_text_sha256": digest},
    )
    return record


def _verify_outputs(root, m, db):
    """重新读取输出而非只信计数器；严格哈希顺序同时检查重复/覆盖，额外内存O(一篇)。"""
    seed, counts = m["processing"]["seed"], {}
    for name in SPLITS:
        last, n, characters = "", 0, 0
        for record in _lines(root / f"{name}.jsonl"):
            text = record["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("集合存在非法正文")
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest <= last:
                raise ValueError("输出未按正文SHA-256严格递增，存在重复或顺序错误")
            last = digest
            row = db.execute(
                "SELECT text,metadata,split,conflict FROM documents WHERE digest=?", (digest,)
            ).fetchone()
            if (
                row is None
                or row[3]
                or row[2] != name
                or row[0] != text
                or record != _prepared(row[1], text, digest, name)
                or choose_work_split(record["work_group"]["id"], seed) != name
            ):
                raise ValueError("正文/分组/集合与磁盘去重索引不一致，可能存在泄漏")
            n += 1
            characters += len(text)
        expected = db.execute(
            "SELECT COUNT(*) FROM documents WHERE conflict=0 AND split=?", (name,)
        ).fetchone()[0]
        if (
            n != expected
            or n != m["files"][name]["documents"]
            or characters != m["files"][name]["characters"]
        ):
            raise ValueError("集合的文档数/字符数与索引或清单不一致")
        if n == 0:
            raise ValueError(f"划分为空：{name}；增加已审批语料，不为凑比例挪动作品")
        counts[name] = n
    for work_id, split in db.execute("SELECT work_id,split FROM works"):
        if choose_work_split(work_id, seed) != split:
            raise ValueError("作品的固定划分与索引不一致")
    conflicts = db.execute(
        "SELECT COUNT(*) FROM origins o JOIN documents d ON o.digest=d.digest "
        "WHERE d.conflict=0 AND o.split != d.split"
    ).fetchone()[0]
    if conflicts:
        raise ValueError("重复正文的来源作品跨集合，存在作品泄漏")
    # 审计文件也逐项核对：不能只留下代表记录、悄悄丢弃重复文档的其他来源。
    origin_count, last_key = 0, ("", "")
    for record in _lines(root / "origins.jsonl"):
        key = record["text_sha256"], record["origin"]
        row = db.execute(
            "SELECT o.digest,o.work_id,o.split,o.metadata,d.conflict FROM origins o "
            "JOIN documents d ON o.digest=d.digest WHERE o.origin=?",
            (record["origin"],),
        ).fetchone()
        if (
            key <= last_key
            or row is None
            or key[0] != row[0]
            or record["work_id"] != row[1]
            or record["assigned_split"] != row[2]
            or record["document"] != json.loads(row[3])
            or record["excluded"] is not bool(row[4])
        ):
            raise ValueError("来源审计文件顺序/记录与索引不一致")
        last_key = key
        origin_count += 1
    c = m["counts"]
    unique = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    excluded = db.execute("SELECT COUNT(*) FROM documents WHERE conflict=1").fetchone()[0]
    if (
        origin_count != db.execute("SELECT COUNT(*) FROM origins").fetchone()[0]
        or origin_count != c["eligible"]
        or unique != c["unique"]
        or excluded != c["cross_split_duplicates_excluded"]
        or c["duplicates"] != c["eligible"] - unique
        or c["read"] != c["eligible"] + c["empty"] + c["length_filtered"]
        or c["saved"] != unique - excluded
    ):
        raise ValueError("输入/去重/排除/来源审计计数不平衡")
    if sum(counts.values()) != m["counts"]["saved"]:
        raise ValueError("语料总数不一致")
    return {
        "passed": True,
        "duplicate_text_across_splits": 0,
        "work_groups_across_splits": 0,
        "documents": counts,
    }


def prepare_corpus(
    extraction_manifests,
    output_dir,
    *,
    seed=42,
    min_chars=20,
    max_chars=1000000,
    to_simplified=True,
):
    """全局精确去重保留所有来源；旧prepare/pilot接口不变，不训练BPE或编码。

    去重键h=SHA256(UTF8(规范化正文))，命中后再比较完整正文，避免把碰撞当重复。
    分组归属冲突的相同正文整篇排除，不移动整部作品，也不连接成巨大的作品连通分量。
    """
    choose_work_split("参数校验", seed)
    clean_document(
        "参数校验", min_chars=min_chars, max_chars=max_chars, to_simplified=to_simplified
    )
    if isinstance(extraction_manifests, (str, Path)) or not extraction_manifests:
        raise ValueError("extraction_manifests必须是非空路径列表")
    paths = [Path(p) for p in extraction_manifests]
    if any(p.is_symlink() for p in paths) or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("提取清单不能是符号链接或重复路径")
    paths = sorted(p.resolve() for p in paths)
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("--output-dir已存在，请换新目录；不覆盖旧语料")
    inputs = [(p, *_extraction(p)) for p in paths]
    code_hashes = {
        name: _sha256(Path(__file__).with_name(name)) for name in ("corpus.py", "cleaning.py")
    }
    output.mkdir(parents=True, exist_ok=False)
    counts = {"read": 0, "empty": 0, "length_filtered": 0, "eligible": 0}
    stats = {s: {"documents": 0, "characters": 0} for s in SPLITS}
    with closing(sqlite3.connect(output / "dedup.sqlite3")) as db, ExitStack() as stack:
        # SQLite自带，无新增依赖。缓存目标8MiB，临时结构可落盘；不是总RSS硬限制。
        db.executescript("""
            PRAGMA cache_size=-8192; PRAGMA temp_store=FILE;
            CREATE TABLE documents(digest TEXT PRIMARY KEY,text TEXT,canonical TEXT,
                                   metadata TEXT,split TEXT,conflict INTEGER);
            CREATE TABLE works(work_id TEXT PRIMARY KEY,split TEXT);
            CREATE TABLE origins(origin TEXT PRIMARY KEY,digest TEXT,work_id TEXT,split TEXT,metadata TEXT);
            CREATE INDEX origin_body ON origins(digest,origin);
            CREATE INDEX document_split ON documents(split,conflict,digest);
        """)
        filtered = stack.enter_context(
            (output / "filtered.jsonl").open("x", encoding="utf-8", newline="\n")
        )
        for path, manifest, source_ids, _ in inputs:
            n = 0
            for record in _lines(
                path.parent / "approved.jsonl", manifest["files"]["approved.jsonl"]["sha256"]
            ):
                work_id = _approved(record, source_ids)
                n += 1
                counts["read"] += 1
                text = normalize_text(record["text"], to_simplified=to_simplified)
                metadata = {k: v for k, v in record.items() if k != "text"}
                if (
                    not text
                    or len(text) < min_chars
                    or (max_chars is not None and len(text) > max_chars)
                ):
                    reason = "empty" if not text else "length_filtered"
                    counts[reason] += 1
                    filtered.write(_json({"reason": reason, "document": metadata}) + "\n")
                    continue
                counts["eligible"] += 1
                digest = hashlib.sha256(text.encode()).hexdigest()
                split = choose_work_split(work_id, seed)
                # 固定身份排序决定代表记录，而非文件顺序；重复来源全部留在origins。
                origin = json.dumps(
                    [
                        record["source"]["name"],
                        record["source"]["dump_date"],
                        record["page_id"],
                        record["revision_id"],
                    ],
                    ensure_ascii=True,
                )
                data = _json(metadata)
                try:
                    db.execute(
                        "INSERT INTO origins VALUES(?,?,?,?,?)",
                        (origin, digest, work_id, split, data),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError("同一页面修订被重复输入；不要合并重叠的审批产物") from error
                db.execute("INSERT OR IGNORE INTO works VALUES(?,?)", (work_id, split))
                old = db.execute(
                    "SELECT text,canonical,split FROM documents WHERE digest=?", (digest,)
                ).fetchone()
                if old is None:
                    db.execute(
                        "INSERT INTO documents VALUES(?,?,?,?,?,0)",
                        (digest, text, origin, data, split),
                    )
                else:
                    if old[0] != text:
                        raise ValueError("正文SHA-256碰撞，停止处理而非错误去重")
                    if old[2] != split:
                        db.execute("UPDATE documents SET conflict=1 WHERE digest=?", (digest,))
                    if origin < old[1]:
                        db.execute(
                            "UPDATE documents SET canonical=?,metadata=? WHERE digest=?",
                            (origin, data, digest),
                        )
                if counts["read"] % 1000 == 0:
                    db.commit()
                    print(f"已整理{counts['read']}条已审批正文", file=sys.stderr)
            if n != manifest["counts"]["approved"]:
                raise ValueError("approved文档数与提取清单不一致")
        db.commit()
        counts["unique"] = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        counts["duplicates"] = counts["eligible"] - counts["unique"]
        counts["cross_split_duplicates_excluded"] = db.execute(
            "SELECT COUNT(*) FROM documents WHERE conflict=1"
        ).fetchone()[0]
        counts["saved"] = counts["unique"] - counts["cross_split_duplicates_excluded"]
        handles = {
            s: stack.enter_context(
                (output / f"{s}.jsonl").open("x", encoding="utf-8", newline="\n")
            )
            for s in SPLITS
        }
        conflicts_file = stack.enter_context(
            (output / "conflicts.jsonl").open("x", encoding="utf-8", newline="\n")
        )
        for digest, text, data, split, conflict in db.execute(
            "SELECT digest,text,metadata,split,conflict FROM documents ORDER BY digest"
        ):
            if conflict:
                conflicts_file.write(
                    _json({"text_sha256": digest, "reason": "duplicate_work_split_conflict"}) + "\n"
                )
            else:
                handles[split].write(_json(_prepared(data, text, digest, split)) + "\n")
                stats[split]["documents"] += 1
                stats[split]["characters"] += len(text)
        origins_file = stack.enter_context(
            (output / "origins.jsonl").open("x", encoding="utf-8", newline="\n")
        )
        for origin, digest, work_id, split, data, conflict in db.execute(
            "SELECT o.origin,o.digest,o.work_id,o.split,o.metadata,d.conflict FROM origins o "
            "JOIN documents d ON o.digest=d.digest ORDER BY o.digest,o.origin"
        ):
            origins_file.write(
                _json(
                    {
                        "origin": origin,
                        "text_sha256": digest,
                        "work_id": work_id,
                        "assigned_split": split,
                        "excluded": bool(conflict),
                        "document": json.loads(data),
                    }
                )
                + "\n"
            )
        for file in (*handles.values(), filtered, conflicts_file, origins_file):
            file.flush()
        report = {
            "format_version": 1,
            "kind": "prepared_work_corpus",
            "status": "completed",
            "scope": "full_dump"
            if all(m["scope"] == "full_dump" for _, m, _, _ in inputs)
            else "prefix_sample",
            "ready_for_training": False,
            "counts": counts,
            "inputs": [
                {"path": str(p), "sha256": h, "approved": m["files"]["approved.jsonl"]}
                for p, m, _, h in inputs
            ],
            "processing": {
                "seed": seed,
                "split_rule": WORK_SPLIT_RULE,
                "normalization": "NFC",
                "to_simplified": to_simplified,
                "opencc_version": version("OpenCC"),
                "opencc_config": "t2s.json" if to_simplified else None,
                "min_chars": min_chars,
                "max_chars": max_chars,
                "deduplication": "sha256_then_exact_normalized_text",
                "duplicate_conflict_policy": "exclude_body_without_moving_works",
                "canonical_record": "minimum_source_identity",
                "output_order": "text_sha256",
                "code_sha256": code_hashes,
            },
            "files": {s: _file_info(output / f"{s}.jsonl", **stats[s]) for s in SPLITS},
        }
        report["leakage_check"] = _verify_outputs(output, report, db)
    report["audit_files"] = {
        name: _file_info(output / name)
        for name in ("origins.jsonl", "filtered.jsonl", "conflicts.jsonl", "dedup.sqlite3")
    }
    for path, manifest, _, h in inputs:
        if _sha256(path) != h:
            raise RuntimeError("整理期间提取清单发生变化")
        _check_file(path.parent, "approved.jsonl", manifest["files"]["approved.jsonl"])
    if {name: _sha256(Path(__file__).with_name(name)) for name in code_hashes} != code_hashes:
        raise RuntimeError("整理期间处理代码发生变化")
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def verify_corpus(manifest_path):
    """只读复核输出摘要/磁盘索引/作品归属，复核不依赖原始dump仍存在。"""
    path = Path(manifest_path)
    m = json.loads(_small_file(path).decode("utf-8"), parse_constant=_reject_nonfinite)
    if not isinstance(m, dict):
        raise ValueError("阶段2清单必须是JSON对象")
    if (
        m.get("kind") != "prepared_work_corpus"
        or m.get("status") != "completed"
        or type(m.get("format_version")) is not int
        or m.get("format_version") != 1
    ):
        raise ValueError("需要完整阶段2语料清单")
    if m["processing"]["split_rule"] != WORK_SPLIT_RULE:
        raise ValueError("不支持的作品划分规则")
    root = path.resolve().parent
    for name in SPLITS:
        _check_file(root, f"{name}.jsonl", m["files"][name])
    for name in ("origins.jsonl", "filtered.jsonl", "conflicts.jsonl", "dedup.sqlite3"):
        _check_file(root, name, m["audit_files"][name])
    with closing(sqlite3.connect((root / "dedup.sqlite3").as_uri() + "?mode=ro", uri=True)) as db:
        result = _verify_outputs(root, m, db)
    if result != m["leakage_check"]:
        raise ValueError("泄漏复核结果与清单不一致")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--extraction-manifest", type=Path, action="append")
    mode.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=1000000)
    parser.add_argument("--keep-traditional", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.verify_manifest:
            if args.output_dir:
                raise ValueError("复核模式不使用--output-dir")
            report = verify_corpus(args.verify_manifest)
        else:
            if args.output_dir is None:
                raise ValueError("整理模式需要--output-dir")
            report = prepare_corpus(
                args.extraction_manifest,
                args.output_dir,
                seed=args.seed,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                to_simplified=not args.keep_traditional,
            )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return report
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, sqlite3.Error) as error:
        parser.error(f"{error}\n阶段2未完成；无完整manifest.json不能用于后续BPE。")


if __name__ == "__main__":
    main()

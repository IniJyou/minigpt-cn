"""阶段2离线虚构审批数据：TEST-LICENSE不代表任何真实页面获授权。"""

import hashlib
import json
import sqlite3

import pytest
from test_parse_xml import document, make_download, page
from test_wikitext import approve

from minigpt_cn.data import corpus as cli
from minigpt_cn.data.export_text import export_train_text
from minigpt_cn.data.extract import extract_bodies


def work_for(split, prefix="作品"):
    for i in range(10000):
        key = f"test:{prefix}:{i}"
        if cli.choose_work_split(key) == split:
            return key
    raise AssertionError("没有找到测试用桶")


def record(text, work_id, page_id=1, source="zhwikisource"):
    digest = hashlib.sha256(text.encode()).hexdigest()
    identity = {
        "source": {"name": source, "dump_date": "20261001"},
        "page_id": page_id,
        "revision_id": 10000 + page_id,
        "wikitext_sha256": digest,
        "text_sha256": digest,
    }
    review = {
        **identity,
        "decision": "approved",
        "quality_checked": True,
        "reviewer": "offline-test-only",
        "reviewed_at": "2026-10-12",
        "notes": "纯虚构单元测试记录，不代表真实许可",
        "license_identifier": "TEST-LICENSE",
        "license_evidence": ["https://example.org/test-license"],
        "work_id": work_id,
        "work_evidence": ["https://example.org/test-work"],
    }
    return {
        **identity,
        "status": "approved",
        "namespace": 0,
        "text": text,
        "title": f"离线测试/第{page_id}章",
        "timestamp": "2026-10-01T00:00:00Z",
        "review": review,
        "reject_reasons": [],
        "work_group": {"id": work_id, "status": "reviewed"},
        "license": {"summary": "TEST-LICENSE", "review_status": "per_document_review_required"},
    }


def fixture_input(directory, records, *, scope="prefix_sample"):
    directory.mkdir()
    path = directory / "approved.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    m = {
        "format_version": 1,
        "kind": "wikimedia_body_extraction",
        "status": "completed",
        "scope": scope,
        "counts": {"approved": len(records)},
        "sources": [
            {"source": {"name": name, "dump_date": "20261001"}}
            for name in sorted({r["source"]["name"] for r in records})
        ],
        "files": {"approved.jsonl": cli._file_info(path)},
    }
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps(m), encoding="utf-8")
    return manifest


def base_records():
    return [
        record(f"测试正文第{i}篇。这是独立的虚构中文文本，不用于正式模型训练。", work_for(s), i)
        for i, s in enumerate(cli.SPLITS, 1)
    ]


def load_rows(root, name):
    return [json.loads(s) for s in (root / name).read_text(encoding="utf-8").splitlines()]


def test_work_split_formula_shape_independent_of_chapter_text():
    work = work_for("valid")
    expected = int(hashlib.sha256(f"corpus-work-v1\0{42}\0{work}".encode()).hexdigest(), 16) % 100
    assert expected == 98 and cli.choose_work_split(work) == "valid"
    assert {cli.choose_work_split(work) for _ in range(5)} == {"valid"}


@pytest.mark.parametrize(
    "work,seed,error",
    [
        ("", 42, ValueError),
        ("\0作品", 42, ValueError),
        (None, 42, ValueError),
        ("作品", True, TypeError),
        ("作品", 1.5, TypeError),
    ],
)
def test_invalid_work_split_options(work, seed, error):
    with pytest.raises(error):
        cli.choose_work_split(work, seed)


def test_full_pipeline_chapters_hashes_provenance_and_readonly_verify(tmp_path):
    rows = base_records()
    rows.append(record("同一作品的另一章，正文不同也应分在同一个集合。", work_for("valid"), 4))
    path = fixture_input(tmp_path / "input", rows)
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    root = tmp_path / "out"
    report = cli.prepare_corpus([path], root)
    assert report["counts"]["saved"] == 4
    assert report["scope"] == "prefix_sample" and not report["ready_for_training"]
    assert report["files"]["valid"]["documents"] == 2
    assert report["leakage_check"]["passed"]
    assert len(load_rows(root, "origins.jsonl")) == 4
    for s in cli.SPLITS:
        docs = load_rows(root, f"{s}.jsonl")
        for r in docs:
            assert r["status"] == "prepared"
            assert r["text_sha256"] == hashlib.sha256(r["text"].encode()).hexdigest()
            assert cli.choose_work_split(r["work_group"]["id"]) == s
    out_before = {p.name: p.read_bytes() for p in root.iterdir()}
    assert cli.verify_corpus(root / "manifest.json") == report["leakage_check"]
    assert {p.name: p.read_bytes() for p in root.iterdir()} == out_before
    assert {p.name: p.read_bytes() for p in path.parent.iterdir()} == before
    with pytest.raises(FileExistsError):
        cli.prepare_corpus([path], root)


def test_simplification_duplicate_keeps_all_sources_and_original_approval(tmp_path):
    rows = base_records()
    traditional = "繁體中文測試，保留文章來源與審閱記錄，不會混用新舊摘要。"
    simplified = "繁体中文测试，保留文章来源与审阅记录，不会混用新旧摘要。"
    rows += [
        record(traditional, work_for("train"), 4, "zhwiki"),
        record(simplified, work_for("train", "另一作品"), 5),
    ]
    path = fixture_input(tmp_path / "in", rows)
    root = tmp_path / "out"
    report = cli.prepare_corpus([path], root)
    assert report["counts"]["duplicates"] == 1 and report["counts"]["saved"] == 4
    train = load_rows(root, "train.jsonl")
    item = next(r for r in train if r["text"] == simplified)
    assert item["source"]["name"] == "zhwiki"
    assert item["review"]["text_sha256"] == hashlib.sha256(traditional.encode()).hexdigest()
    assert item["extracted_text_sha256"] == item["review"]["text_sha256"]
    origins = [
        r for r in load_rows(root, "origins.jsonl") if r["text_sha256"] == item["text_sha256"]
    ]
    assert {r["document"]["source"]["name"] for r in origins} == {"zhwiki", "zhwikisource"}
    assert all(not r["excluded"] for r in origins)
    kept = cli.prepare_corpus([path], tmp_path / "traditional", to_simplified=False)
    assert kept["counts"]["duplicates"] == 0


def test_duplicate_across_work_splits_excluded_without_moving_works(tmp_path):
    text = "两个不同作品中出现的相同正文，不应导致作品内容跨集合。"
    rows = base_records() + [record(text, work_for("train"), 4), record(text, work_for("valid"), 5)]
    path = fixture_input(tmp_path / "in", rows)
    root = tmp_path / "out"
    report = cli.prepare_corpus([path], root)
    assert report["counts"] == {
        "read": 5,
        "empty": 0,
        "length_filtered": 0,
        "eligible": 5,
        "unique": 4,
        "duplicates": 1,
        "cross_split_duplicates_excluded": 1,
        "saved": 3,
    }
    assert all(r["text"] != text for s in cli.SPLITS for r in load_rows(root, f"{s}.jsonl"))
    assert len(load_rows(root, "conflicts.jsonl")) == 1
    excluded = [r for r in load_rows(root, "origins.jsonl") if r["excluded"]]
    assert len(excluded) == 2 and {r["assigned_split"] for r in excluded} == {"train", "valid"}
    assert cli.verify_corpus(root / "manifest.json")["passed"]


def test_order_independence_and_multiple_sources(tmp_path):
    rows = base_records() + [record(base_records()[0]["text"], work_for("train"), 4, "zhwiki")]
    a = fixture_input(tmp_path / "a", rows)
    b = fixture_input(tmp_path / "b", list(reversed(rows)))
    cli.prepare_corpus([a], tmp_path / "outa")
    cli.prepare_corpus([b], tmp_path / "outb")
    for name in [*(f"{s}.jsonl" for s in cli.SPLITS), "origins.jsonl", "conflicts.jsonl"]:
        assert (tmp_path / "outa" / name).read_bytes() == (tmp_path / "outb" / name).read_bytes()


def test_multiple_manifests_global_dedup(tmp_path):
    rows = base_records()
    a = fixture_input(tmp_path / "a", rows[:2])
    b = fixture_input(
        tmp_path / "b", rows[2:] + [record(rows[0]["text"], work_for("train"), 4, "zhwiki")]
    )
    report = cli.prepare_corpus([b, a], tmp_path / "out")
    assert report["counts"]["duplicates"] == 1 and report["counts"]["saved"] == 3
    assert len(report["inputs"]) == 2


def test_actual_stage1_format_to_stage2_and_bpe_export(tmp_path):
    pages = [
        page(
            title=f"虚构作品{i}",
            text=f"第{i}部虚构作品的正文，仅用于离线接口测试，不声明任何真实版权结论。",
        ).replace("<id>1</id>", f"<id>{i}</id>", 1)
        for i in range(1, 4)
    ]
    path, _ = make_download(tmp_path, document(pages, "zhwikisource"), "zhwikisource")
    extract_bodies([path], tmp_path / "unreviewed")
    queue = load_rows(tmp_path / "unreviewed", "review_queue.jsonl")
    assert len(queue) == 3
    reviews = tmp_path / "reviews.jsonl"
    approved = [{**approve(r), "work_id": work_for(s)} for r, s in zip(queue, cli.SPLITS)]
    reviews.write_text("".join(json.dumps(r) + "\n" for r in approved), encoding="utf-8")
    extract_bodies([path], tmp_path / "approved", reviews_path=reviews)
    cli.prepare_corpus([tmp_path / "approved" / "manifest.json"], tmp_path / "corpus")
    assert cli.verify_corpus(tmp_path / "corpus" / "manifest.json")["passed"]
    report = export_train_text(tmp_path / "corpus" / "manifest.json", tmp_path / "text")
    assert report["output"]["documents"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"status": "quarantine"},
        {"text_sha256": "0" * 64},
        {"namespace": 10},
        {"work_group": {"status": "needs_review", "suggested_id": "猜测"}},
        {"page_id": True},
        {"reject_reasons": ["residual_markup"]},
    ],
)
def test_unapproved_or_inconsistent_record_fails_closed(tmp_path, change):
    rows = base_records()
    rows[0] = {**rows[0], **change}
    path = fixture_input(tmp_path / "in", rows)
    with pytest.raises(ValueError):
        cli.prepare_corpus([path], tmp_path / "out")
    assert not (tmp_path / "out" / "manifest.json").exists()


@pytest.mark.parametrize(
    "case", ["empty_approved", "bad_hash", "path_escape", "wrong_kind", "bad_file_structure"]
)
def test_invalid_manifest_fails_before_output(tmp_path, case):
    path = fixture_input(tmp_path / "in", base_records())
    m = json.loads(path.read_text())
    if case == "empty_approved":
        m["counts"]["approved"] = 0
    if case == "bad_hash":
        m["files"]["approved.jsonl"]["sha256"] = "0" * 64
    if case == "path_escape":
        m["files"]["approved.jsonl"]["path"] = "../quarantine.jsonl"
    if case == "wrong_kind":
        m["kind"] = "pilot_jsonl"
    if case == "bad_file_structure":
        m["files"]["approved.jsonl"] = None
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        cli.prepare_corpus([path], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_empty_split_and_filtering_are_not_silent_success(tmp_path):
    rows = base_records() + [record("短", work_for("train"), 4)]
    path = fixture_input(tmp_path / "in", rows)
    m = cli.prepare_corpus([path], tmp_path / "out")
    assert m["counts"]["length_filtered"] == 1
    assert len(load_rows(tmp_path / "out", "filtered.jsonl")) == 1
    lone = fixture_input(tmp_path / "lone", rows[:1])
    with pytest.raises(ValueError, match="划分为空"):
        cli.prepare_corpus([lone], tmp_path / "empty")
    assert not (tmp_path / "empty" / "manifest.json").exists()


def test_tampered_split_even_with_rewritten_hash_detected(tmp_path):
    path = fixture_input(tmp_path / "in", base_records())
    root = tmp_path / "out"
    cli.prepare_corpus([path], root)
    manifest = root / "manifest.json"
    m = json.loads(manifest.read_text())
    train = load_rows(root, "train.jsonl")
    train[0]["work_group"]["id"] = work_for("valid")
    (root / "train.jsonl").write_text(
        json.dumps(train[0], ensure_ascii=False) + "\n", encoding="utf-8"
    )
    m["files"]["train"].update(cli._file_info(root / "train.jsonl"))
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="泄漏"):
        cli.verify_corpus(manifest)


def test_duplicate_origin_rejected_and_disk_index_used(tmp_path):
    rows = base_records()
    path = fixture_input(tmp_path / "in", rows + [rows[0]])
    with pytest.raises(ValueError, match="重复输入"):
        cli.prepare_corpus([path], tmp_path / "out")
    assert (tmp_path / "out" / "dedup.sqlite3").exists()


def test_group_corpus_export_uses_train_only_without_old_text_split_rule(tmp_path):
    rows = base_records()
    path = fixture_input(tmp_path / "in", rows)
    root = tmp_path / "out"
    cli.prepare_corpus([path], root)
    exported = export_train_text(
        root / "manifest.json", tmp_path / "bpe", selection="hash", max_bytes=16777216
    )
    assert exported["source_split"]["split"] == "train"
    text = load_rows(root, "train.jsonl")[0]["text"]
    assert (tmp_path / "bpe" / "train.txt").read_text(encoding="utf-8") == text + "<|endoftext|>\n"


def test_cli_prepare_and_verify(tmp_path, capsys):
    path = fixture_input(tmp_path / "in", base_records())
    root = tmp_path / "out"
    cli.main(["--extraction-manifest", str(path), "--output-dir", str(root)])
    cli.main(["--verify-manifest", str(root / "manifest.json")])
    assert '"passed": true' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["--extraction-manifest", str(path)])


def test_sql_index_detects_hidden_origin_work_leak(tmp_path):
    path = fixture_input(tmp_path / "in", base_records())
    root = tmp_path / "out"
    cli.prepare_corpus([path], root)
    with sqlite3.connect(root / "dedup.sqlite3") as db:
        db.execute("UPDATE origins SET split='valid' WHERE split='train'")
    manifest = root / "manifest.json"
    m = json.loads(manifest.read_text())
    m["audit_files"]["dedup.sqlite3"] = cli._file_info(root / "dedup.sqlite3")
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="作品泄漏"):
        cli.verify_corpus(manifest)


def test_missing_origin_even_with_updated_file_hash_is_detected(tmp_path):
    path = fixture_input(tmp_path / "in", base_records())
    root = tmp_path / "out"
    cli.prepare_corpus([path], root)
    rows = load_rows(root, "origins.jsonl")
    (root / "origins.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows[1:]), encoding="utf-8"
    )
    manifest = root / "manifest.json"
    m = json.loads(manifest.read_text())
    m["audit_files"]["origins.jsonl"] = cli._file_info(root / "origins.jsonl")
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="计数不平衡"):
        cli.verify_corpus(manifest)


def test_input_changed_after_read_blocks_manifest(tmp_path, monkeypatch):
    path = fixture_input(tmp_path / "in", base_records())
    original = cli._verify_outputs

    def changed(root, m, db):
        result = original(root, m, db)
        with path.open("a", encoding="utf-8") as file:
            file.write("\n")
        return result

    monkeypatch.setattr(cli, "_verify_outputs", changed)
    with pytest.raises(RuntimeError, match="清单发生变化"):
        cli.prepare_corpus([path], tmp_path / "out")
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_streamed_input_hash_is_checked_at_eof(tmp_path):
    path = fixture_input(tmp_path / "in", base_records())
    with pytest.raises(RuntimeError, match="读取期间"):
        list(cli._lines(path.parent / "approved.jsonl", "0" * 64))

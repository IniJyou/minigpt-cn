"""质量筛选不冒充许可审批，原正文保留，分层抽检可复现。"""

import hashlib
import json

import pytest
from test_candidate_subset import candidate, fixture_input, write_records

from minigpt_cn.data import candidate_subset, quality
from minigpt_cn.data.corpus import prepare_corpus
from minigpt_cn.data.export_text import export_train_text

BODY = "這是一篇介紹數學與歷史的中文正文，保留完整的敘述與標點。" * 6


def record(text=BODY, **extra):
    return {
        "text": text,
        "source": {"name": "zhwiki"},
        "issues": [],
        "unknown_templates": [],
        **extra,
    }


@pytest.mark.parametrize(
    "text,extra,category,reason",
    [
        (BODY, {}, "eligible", None),
        (BODY, {"issues": ["unknown_template"], "unknown_templates": ["langx"]}, "eligible", None),
        ("漢" * 99, {}, "excluded", "short_body"),
        ("漢" * 100, {}, "eligible", None),
        ("漢" * 50000, {}, "eligible", None),
        ("漢" * 50001, {}, "excluded", "long_body"),
        ("漢" * 25 + "x" * 75, {}, "eligible", None),
        ("漢" * 24 + "x" * 76, {}, "excluded", "low_han_ratio"),
        (BODY + "\ufffd", {}, "excluded", "replacement_character"),
        (BODY + "{{遺漏}}", {}, "excluded", "residual_markup"),
        (BODY + "<span>內容</span>", {}, "excluded", "residual_markup"),
        (BODY + "（\n ）", {}, "held", "empty_parentheses"),
        (BODY + "( )", {}, "held", "empty_parentheses"),
        (BODY + "（ ， ）", {}, "held", "empty_parentheses"),
        (BODY + "( , )", {}, "held", "empty_parentheses"),
        (BODY + "（正常內容）", {}, "eligible", None),
        (BODY + "\n2004年\n", {}, "held", "standalone_year"),
        (BODY + "2004年發生的事件", {}, "eligible", None),
        (BODY + "-{繁體}-", {}, "excluded", "residual_markup"),
        (BODY + "\n\n" + "小節標題\n\n" * 5, {}, "held", "consecutive_short_lines"),
        (BODY + "\n" + "小節標題\n" * 4, {}, "eligible", None),
        (BODY + "\n" + "完整的句子。\n" * 5, {}, "eligible", None),
        (BODY, {"issues": ["unknown_tag"]}, "held", "unexpanded_syntax"),
        (BODY, {"issues": ["unexpanded_parser_function"]}, "held", "unexpanded_syntax"),
        (
            BODY,
            {"unknown_templates": [str(i) for i in range(16)]},
            "held",
            "template_names_possibly_truncated",
        ),
        (BODY, {"unknown_templates": ["x" * 100]}, "held", "template_names_possibly_truncated"),
        (BODY, {"issues": ["unknown_template"]}, "held", "missing_template_names"),
        (BODY, {"unknown_templates": ["copyvio"]}, "held", "license_hint_requires_review"),
        (BODY, {"license_hints": ["pd-old"]}, "held", "license_hint_requires_review"),
        (BODY, {"source": {"name": "zhwikisource"}}, "held", "source_requires_separate_policy"),
    ],
    ids=[f"rule-{i}" for i in range(30)],
)
def test_quality_rules_and_boundaries(text, extra, category, reason):
    r = record(text, **extra)
    before = json.dumps(r, ensure_ascii=False)
    q = quality.classify(r)
    assert q["category"] == category
    assert q["body_bytes"] == len(text.encode())
    if reason:
        assert reason in q["reasons"]
    assert json.dumps(r, ensure_ascii=False) == before


@pytest.mark.parametrize(
    "extra", [{"unknown_templates": "x"}, {"unknown_templates": [3]}, {"license_hints": [""]}]
)
def test_invalid_metadata(extra):
    with pytest.raises(ValueError, match="字符串列表"):
        quality.classify(record(**extra))


def make_subset(tmp_path):
    path, download, dump, src, _ = fixture_input(tmp_path)
    rows = [candidate(src, BODY + str(i), i) for i in range(1, 25)]
    rows += [
        candidate(src, BODY + "（ ）", 90),
        candidate(src, BODY + "\n2004年", 91),
        candidate(src, "短文", 92),
        candidate(src, BODY + "\ufffd", 93),
    ]
    write_records(path, rows)
    root = tmp_path / "subset"
    candidate_subset.sample_candidates(path, download, root, max_body_bytes=100000)
    # 只删除tmp_path里的虚构dump，证明质量入口不读取/重解压压缩包。
    dump.unlink()
    return root / "subset_manifest.json", rows


def test_artifacts_preserve_input_and_pending_status(tmp_path):
    manifest, rows = make_subset(tmp_path)
    before = {p: p.read_bytes() for p in manifest.parent.iterdir()}
    out = tmp_path / "quality"
    m = quality.audit_quality(manifest, out)
    assert m["counts"]["documents"] == {"eligible": 24, "held": 2, "excluded": 2}
    assert sum(m["counts"]["body_bytes"].values()) == sum(len(r["text"].encode()) for r in rows)
    assert not m["ready_for_training"] and not m["license_reviewed"] and not m["quality_reviewed"]
    assert m["counts"]["approved"] == 0
    packets = [r for c in quality.CATEGORIES for _, _, r in quality._records(out / f"{c}.jsonl")]
    assert {r["document"]["page_id"] for r in packets} == {r["page_id"] for r in rows}
    original = {r["page_id"]: r for r in rows}
    assert all(p["document"] == original[p["document"]["page_id"]] for p in packets)
    assert all(p["document"]["status"] == "quarantine" for p in packets)
    artifacts_before = {p: p.read_bytes() for p in out.iterdir()}
    assert quality.verify_quality(out / "quality_manifest.json")["passed"]
    assert {p: p.read_bytes() for p in out.iterdir()} == artifacts_before
    assert {p: p.read_bytes() for p in before} == before
    # 同一输入再次执行：三个分类文件、抽检JSONL和报告逐字节一致。
    quality.audit_quality(manifest, tmp_path / "again")
    for name in m["files"]:
        assert (out / name).read_bytes() == (tmp_path / "again" / name).read_bytes()
    samples = [r for _, _, r in quality._records(out / "review_samples.jsonl")]
    assert len(samples) == 20  # 两个eligible层各8条；held/excluded各2条。
    # 独立实现抽检域、身份与排序，不能只比较重复调用生产函数。
    for stratum in quality.STRATA:
        group = [p for p in packets if quality._stratum(p) == stratum]

        def key(p):
            r = p["document"]
            identity = [r["source"], r["page_id"], r["revision_id"], r["text_sha256"]]
            raw = json.dumps(
                identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
            return hashlib.sha256(b"quality-review-v1\0" + raw).hexdigest()

        expected = sorted(group, key=key)[:8]
        assert [s["document"] for s in samples if s["stratum"] == stratum] == [
            p["document"] for p in expected
        ]
    for downstream in (prepare_corpus, export_train_text):
        with pytest.raises(ValueError):
            args = (
                [out / "quality_manifest.json"]
                if downstream is prepare_corpus
                else out / "quality_manifest.json"
            )
            downstream(args, tmp_path / "blocked")


@pytest.mark.parametrize(
    "mutation", ["ready", "counts", "policy", "file", "classification", "sample"]
)
def test_tampering_rejected(tmp_path, mutation):
    manifest, _ = make_subset(tmp_path)
    out = tmp_path / "quality"
    m = quality.audit_quality(manifest, out)
    if mutation == "ready":
        m["ready_for_training"] = True
    elif mutation == "counts":
        m["counts"]["documents"]["eligible"] += 1
    elif mutation == "policy":
        m["policy"]["min_nonspace_chars"] = 1
    else:
        name = "review_samples.jsonl" if mutation == "sample" else "eligible.jsonl"
        file = out / name
        packets = [r for _, _, r in quality._records(file)]
        if mutation == "classification":
            packets[0]["quality"]["han_ratio"] = 0
        else:
            packets.reverse()
        write_records(file, packets)
        if mutation != "file":
            m["files"][name]["sha256"] = quality._sha256(file)
            m["files"][name]["bytes"] = file.stat().st_size
    (out / "quality_manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError):
        quality.verify_quality(out / "quality_manifest.json")


def test_existing_output_invalid_subset_and_cli(tmp_path, capsys):
    manifest, _ = make_subset(tmp_path)
    out = tmp_path / "quality"
    out.mkdir()
    with pytest.raises(FileExistsError):
        quality.audit_quality(manifest, out)
    with pytest.raises(SystemExit) as error:
        quality.main(["--subset-manifest", str(manifest)])
    assert error.value.code == 2
    assert "筛选需要" in capsys.readouterr().err
    (manifest.parent / "candidates.jsonl").write_bytes(b"truncated")
    with pytest.raises(ValueError):
        quality.audit_quality(manifest, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_changed_input_during_audit_has_no_success_manifest(tmp_path, monkeypatch):
    manifest, _ = make_subset(tmp_path)
    classify = quality.classify
    changed = False

    def mutate(r):
        nonlocal changed
        if not changed:
            with (manifest.parent / "candidates.jsonl").open("ab") as file:
                file.write(b"\n")
            changed = True
        return classify(r)

    monkeypatch.setattr(quality, "classify", mutate)
    out = tmp_path / "changed"
    with pytest.raises((ValueError, RuntimeError)):
        quality.audit_quality(manifest, out)
    assert not (out / "quality_manifest.json").exists()


def test_cli_read_only_verify_and_interrupt(tmp_path, capsys, monkeypatch):
    manifest, _ = make_subset(tmp_path)
    out = tmp_path / "quality"
    quality.main(["--subset-manifest", str(manifest), "--output-dir", str(out)])
    result = quality.main(["--verify-manifest", str(out / "quality_manifest.json")])
    assert result["passed"] and not result["ready_for_training"]

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(quality, "audit_quality", interrupted)
    with pytest.raises(SystemExit) as error:
        quality.main(
            ["--subset-manifest", str(manifest), "--output-dir", str(tmp_path / "interrupt")]
        )
    assert error.value.code == 130
    assert "已中断" in capsys.readouterr().err

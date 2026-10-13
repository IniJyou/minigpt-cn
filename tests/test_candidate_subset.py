"""中断来源子集：独立全排序对照、完整输入核验、待审状态和不访问dump。"""

import hashlib
import json
import random
from urllib.parse import quote

import pytest
from test_parse_xml import document, make_download, page

from minigpt_cn.data import candidate_subset as cli
from minigpt_cn.data.corpus import prepare_corpus
from minigpt_cn.data.export_text import export_train_text


def candidate(source, text, number):
    name = source["name"]
    title = f"測試/第{number}章"
    host = cli.HOSTS[name]
    issues = ["unknown_template"] if number % 3 == 0 else []
    return {
        "source": {"name": name, "dump_date": source["dump_date"]},
        "license": source["license"],
        "title": title,
        "namespace": 0,
        "page_id": number,
        "revision_id": number + 100,
        "timestamp": "2026-10-01T00:00:00Z",
        "status": "quarantine",
        "text": text,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "wikitext_sha256": hashlib.sha256(("'''" + text + "'''").encode()).hexdigest(),
        "metrics": {"text_chars": len(text)},
        "issues": issues,
        "unknown_templates": ["待查模板"] if issues else [],
        "reject_reasons": [],
        "quarantine_reasons": ["license_not_reviewed", "quality_not_reviewed"],
        "work_group": {
            "status": "page_identity" if name == "zhwiki" else "needs_review",
            "suggested_id": f"{name}:page:{number}",
            "chapter": str(number),
        },
        "url": host + "/wiki/" + quote(title.replace(" ", "_"), safe="/"),
        "revision_url": f"{host}/w/index.php?oldid={number + 100}",
        "history_url": f"{host}/w/index.php?curid={number}&action=history",
    }


def write_records(path, rows):
    path.write_bytes(b"".join(json.dumps(r, ensure_ascii=False).encode() + b"\n" for r in rows))


def fixture_input(tmp_path, source="zhwiki"):
    manifest, dump = make_download(tmp_path, document([page()], source), source)
    src = json.loads(manifest.read_text(encoding="utf-8"))["source"]
    partial = tmp_path / "partial"
    partial.mkdir()
    path = partial / "quarantine.jsonl"
    rows = [
        candidate(src, f"文档{i}：繁體中文😀\n第二段" + "甲" * (i % 7), i) for i in range(1, 41)
    ]
    write_records(path, rows)
    return path, manifest, dump, src, rows


def reference(rows, budget, seed):
    # 不使用生产堆/生产key函数；直接全排序求预算内完整前缀。
    def key(r):
        body = r["text"].encode()
        digest = hashlib.sha256(f"candidate-subset-v1\0{seed}\0".encode() + body).digest()
        canonical = json.dumps(
            r, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return digest, body, canonical

    chosen, used = [], 0
    for r in sorted(rows, key=key):
        size = len(r["text"].encode())
        if used + size > budget:
            break
        chosen.append(r)
        used += size
    return chosen, used


def records(root):
    return [json.loads(line) for line in (root / "candidates.jsonl").read_bytes().splitlines()]


@pytest.mark.parametrize("seed", [0, 42, -7])
@pytest.mark.parametrize("budget_kind", ["first", "before_second", "all"])
def test_matches_independent_full_sort_and_utf8_budget(tmp_path, seed, budget_kind):
    path, manifest, _, _, rows = fixture_input(tmp_path)
    full, total = reference(rows, 100000, seed)
    budget = {
        "first": len(full[0]["text"].encode()),
        "before_second": sum(len(r["text"].encode()) for r in full[:2]) - 1,
        "all": total + 100,
    }[budget_kind]
    expected, used = reference(rows, budget, seed)
    root = tmp_path / "sample"
    m = cli.sample_candidates(path, manifest, root, max_body_bytes=budget, sample_seed=seed)
    assert records(root) == expected
    assert m["counts"]["selected_body_bytes"] == used <= budget
    assert m["counts"]["selected_documents"] == len(expected)
    assert m["counts"]["input_documents"] == len(rows)
    assert m["counts"]["approved"] == 0 and not m["ready_for_training"]
    assert m["scope"] == "interrupted_scan_prefix_subset"
    assert not m["provenance"]["dump_content_rechecked"]
    assert not m["provenance"]["raw_wikitext_revalidated"]
    assert cli.verify_subset(root / "subset_manifest.json")["passed"]


@pytest.mark.parametrize("source", ["zhwiki", "zhwikisource"])
def test_preserves_source_warnings_traditional_and_inputs_without_dump(tmp_path, source):
    path, manifest, dump, _, rows = fixture_input(tmp_path, source)
    # 删除的仅是tmp_path内虚构压缩包：证明入口不打开/重算原dump，也不进行XML解压。
    dump.unlink()
    before = {
        p: p.read_bytes()
        for p in (
            path,
            manifest,
            manifest.parent / "source_config.yaml",
            manifest.parent / "dumpstatus.json",
        )
    }
    root = tmp_path / "sample"
    cli.sample_candidates(path, manifest, root, max_body_bytes=100000)
    assert records(root) == reference(rows, 100000, 42)[0]
    assert all(r["status"] == "quarantine" and "繁體" in r["text"] for r in records(root))
    assert any(r["issues"] for r in records(root))
    quality = json.loads((root / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["quality_reviewed"] is False and quality["license_reviewed"] is False
    assert {p: p.read_bytes() for p in before} == before
    out_before = {p.name: p.read_bytes() for p in root.iterdir()}
    cli.verify_subset(root / "subset_manifest.json")
    assert {p.name: p.read_bytes() for p in root.iterdir()} == out_before
    assert {p.name for p in root.iterdir()} == {
        "candidates.jsonl",
        "quality_report.json",
        "subset_manifest.json",
    }


def test_reordering_and_duplicate_body_metadata_ties(tmp_path):
    path, manifest, _, src, rows = fixture_input(tmp_path)
    rows += [candidate(src, rows[0]["text"], 999)]
    write_records(path, rows)
    cli.sample_candidates(path, manifest, tmp_path / "a", max_body_bytes=900)
    random.Random(123).shuffle(rows)
    write_records(path, rows)
    cli.sample_candidates(path, manifest, tmp_path / "b", max_body_bytes=900)
    assert (tmp_path / "a" / "candidates.jsonl").read_bytes() == (
        tmp_path / "b" / "candidates.jsonl"
    ).read_bytes()
    assert records(tmp_path / "b") == reference(rows, 900, 42)[0]


def test_metadata_is_not_charged_to_body_budget(tmp_path):
    path, manifest, _, _, rows = fixture_input(tmp_path)
    rows = [rows[0]]
    rows[0]["extra_metadata"] = "保留来源" * 1000
    write_records(path, rows)
    size = len(rows[0]["text"].encode())
    m = cli.sample_candidates(path, manifest, tmp_path / "sample", max_body_bytes=size)
    assert m["counts"]["selected_body_bytes"] == size
    assert m["files"]["candidates.jsonl"]["bytes"] > size
    assert records(tmp_path / "sample") == rows


@pytest.mark.parametrize("tail", [b'{"broken":', b"", b"\xff\n", b"\n", b"NaN\n"])
def test_bad_last_line_not_ignored_after_budget_filled(tmp_path, tail):
    path, manifest, _, _, _ = fixture_input(tmp_path)
    if tail == b"":
        path.write_bytes(path.read_bytes()[:-1])  # 有效JSON但缺LF也按中断尾行拒绝。
    else:
        with path.open("ab") as file:
            file.write(tail)
    with pytest.raises(ValueError):
        cli.sample_candidates(path, manifest, tmp_path / "out", max_body_bytes=100)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "approved"),
        ("text_sha256", "0" * 64),
        ("namespace", 10),
        ("page_id", True),
        ("issues", ["requires_transclusion"]),
        ("quarantine_reasons", []),
        ("work_group", {"status": "reviewed"}),
        ("revision_url", "https://example.org/incorrect"),
        ("license", {}),
    ],
)
def test_bad_candidate_fails_closed_before_output(tmp_path, field, value):
    path, manifest, _, _, rows = fixture_input(tmp_path)
    rows[-1][field] = value
    write_records(path, rows)
    with pytest.raises(ValueError):
        cli.sample_candidates(path, manifest, tmp_path / "out", max_body_bytes=100)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"max_body_bytes": 0}, ValueError),
        ({"max_body_bytes": True}, ValueError),
        ({"sample_seed": True}, TypeError),
        ({"sample_seed": 1.5}, TypeError),
    ],
)
def test_invalid_options_fail_before_file_read(tmp_path, kwargs, error):
    with pytest.raises(error):
        cli.sample_candidates(
            tmp_path / "missing", tmp_path / "missing_manifest", tmp_path / "out", **kwargs
        )
    assert not (tmp_path / "out").exists()


def test_empty_and_too_small_and_existing_output(tmp_path):
    path, manifest, _, _, _ = fixture_input(tmp_path)
    with pytest.raises(ValueError, match="第一篇"):
        cli.sample_candidates(path, manifest, tmp_path / "small", max_body_bytes=1)
    assert not (tmp_path / "small").exists()
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        cli.sample_candidates(path, manifest, output)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="为空"):
        cli.sample_candidates(path, manifest, tmp_path / "empty")


def test_overflowing_nested_json_number_not_silently_written(tmp_path):
    path, manifest, _, _, rows = fixture_input(tmp_path)
    # 1e999不是NaN字面量，但Python解析为Inf；稳定编码时仍必须拒绝嵌套的非有限值。
    with path.open("ab") as file:
        raw = json.dumps(rows[-1], ensure_ascii=False).encode()
        file.write(raw[:-1] + b',"extra_metadata":1e999}\n')
    with pytest.raises(ValueError):
        cli.sample_candidates(path, manifest, tmp_path / "out", max_body_bytes=100)
    assert not (tmp_path / "out").exists()


def test_completed_input_directory_and_changed_snapshot_rejected(tmp_path):
    path, manifest, _, _, _ = fixture_input(tmp_path)
    (path.parent / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="已有提取清单"):
        cli.sample_candidates(path, manifest, tmp_path / "out")
    (path.parent / "manifest.json").unlink()
    with (manifest.parent / "source_config.yaml").open("ab") as file:
        file.write(b"\n")
    with pytest.raises(ValueError, match="快照"):
        cli.sample_candidates(path, manifest, tmp_path / "bad_snapshot")


def test_source_changed_during_read_prevents_subset(tmp_path, monkeypatch):
    path, manifest, _, src, _ = fixture_input(tmp_path)
    original, done = cli._candidate, []

    def changing(record, source):
        body = original(record, source)
        if not done:
            done.append(True)
            with path.open("ab") as file:
                file.write(
                    json.dumps(candidate(src, "后来追加的中文", 900), ensure_ascii=False).encode()
                    + b"\n"
                )
        return body

    monkeypatch.setattr(cli, "_candidate", changing)
    with pytest.raises(RuntimeError, match="候选文件发生变化"):
        cli.sample_candidates(path, manifest, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field,value", [("ready_for_training", True), ("scope", "full_dump")])
def test_manifest_cannot_claim_training_ready_or_full_dump(tmp_path, field, value):
    path, manifest, _, _, _ = fixture_input(tmp_path)
    root = tmp_path / "sample"
    cli.sample_candidates(path, manifest, root, max_body_bytes=900)
    report = root / "subset_manifest.json"
    m = json.loads(report.read_text(encoding="utf-8"))
    m[field] = value
    report.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="不能修改"):
        cli.verify_subset(report)


def test_hash_and_count_tampering_and_downstream_gates(tmp_path):
    path, manifest, _, _, _ = fixture_input(tmp_path)
    root = tmp_path / "sample"
    cli.sample_candidates(path, manifest, root, max_body_bytes=900)
    report = root / "subset_manifest.json"
    with pytest.raises(ValueError):
        prepare_corpus([report], tmp_path / "corpus")
    with pytest.raises(ValueError):
        export_train_text(report, tmp_path / "bpe")
    assert not (tmp_path / "corpus").exists() and not (tmp_path / "bpe").exists()
    with (root / "candidates.jsonl").open("ab") as file:
        file.write(b"\n")
    with pytest.raises(ValueError, match="摘要"):
        cli.verify_subset(report)


def test_cli_modes_and_friendly_interrupt(tmp_path, capsys, monkeypatch):
    path, manifest, _, _, _ = fixture_input(tmp_path)
    root = tmp_path / "sample"
    cli.main(
        [
            "--candidates",
            str(path),
            "--download-manifest",
            str(manifest),
            "--output-dir",
            str(root),
            "--max-body-bytes",
            "900",
        ]
    )
    cli.main(["--verify-manifest", str(root / "subset_manifest.json")])
    assert '"ready_for_training": false' in capsys.readouterr().out

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "sample_candidates", interrupted)
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--candidates",
                str(path),
                "--download-manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "stopped"),
            ]
        )
    assert error.value.code == 130
    assert "用户中断" in capsys.readouterr().err


def test_symlink_input_not_followed(tmp_path):
    path, manifest, _, _, _ = fixture_input(tmp_path)
    link = tmp_path / "alias.jsonl"
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("此Windows环境不允许创建符号链接")
    with pytest.raises(ValueError, match="符号链接"):
        cli.sample_candidates(link, manifest, tmp_path / "out")

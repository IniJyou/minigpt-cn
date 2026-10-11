"""哈希采样：独立排序对照、堆前缀不变量、完整 train 校验和旧模式兼容。"""

import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from minigpt_cn.data import export_text as cli
from minigpt_cn.data import sampling
from minigpt_cn.data.prepare import prepare_jsonl
from minigpt_cn.data.splitting import choose_split


def ordered(records, seed=42):
    # 测试 oracle 直接对全部正文排序，不调用生产堆或其 key 函数。
    return sorted(
        records,
        key=lambda r: (
            hashlib.sha256(f"bpe-sample-v1\0{seed}\0{r['text']}".encode()).digest(),
            r["text"].encode(),
        ),
    )


def prefix_bytes(records, budget):
    raw = b""
    for record in records:
        piece = (record["text"] + cli.SUFFIX).encode()
        if len(raw) + len(piece) > budget:
            break
        raw += piece
    return raw


def prepare(tmp_path, name, records):
    path = tmp_path / f"{name}.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        + "\n".join(json.dumps({"text": t}) for t in ("边界文档5", "边界文档28"))
        + "\n",
        encoding="utf-8",
    )
    directory = tmp_path / name
    prepare_jsonl([path], directory, to_simplified=False)
    return directory


@pytest.fixture
def dataset(tmp_path):
    texts = [f"采样文档{i}😀\n第二段" for i in range(30)]
    texts = [t for t in texts if choose_split(t) == "train"][:12]
    labels = ["zhwiki", {"name": "zhwikisource"}, None, {"name": 7}]
    records = [
        {"text": text, "source": labels[i % 4], "title": "标题不应进入纯文本"}
        for i, text in enumerate(texts)
    ]
    return prepare(tmp_path, "prepared", records), records


def export(dataset, output, **kwargs):
    directory, _ = dataset
    return cli.export_train_text(directory / "manifest.json", output, selection="hash", **kwargs)


def rewrite_train(directory, records):
    source = directory / "train.jsonl"
    raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode()
    source.write_bytes(raw)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    info = manifest["files"]["train"]
    info.update(
        documents=len(records),
        characters=sum(len(r["text"]) for r in records),
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    manifest["counts"]["saved"] = sum(v["documents"] for v in manifest["files"].values())
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.parametrize("seed", [0, 42, -7])
@pytest.mark.parametrize("budget_kind", ["exact", "before_second", "all"])
def test_export_matches_independent_hash_sort_and_boundary_budget(
    dataset, tmp_path, seed, budget_kind
):
    records = ordered(dataset[1], seed)
    sizes = [len((r["text"] + cli.SUFFIX).encode()) for r in records]
    budget = {"exact": sizes[0], "before_second": sum(sizes[:2]) - 1, "all": sum(sizes)}[
        budget_kind
    ]
    output = tmp_path / "export"
    report = export(dataset, output, max_bytes=budget, sample_seed=seed)
    raw = (output / "train.txt").read_bytes()
    assert raw == prefix_bytes(records, budget)
    assert report["selection"] == {
        "strategy": "whole_document_hash_prefix",
        "max_bytes": budget,
        "algorithm_version": 1,
        "sample_seed": seed,
        "hash_rule": sampling.HASH_RULE,
    }
    assert report["output"]["bytes"] == len(raw) <= budget
    assert report["output"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["output"]["documents"] == raw.count(cli.EOS.encode())
    assert report["output"]["characters"] == sum(
        len(t) for t in raw.decode().split(cli.SUFFIX)[:-1]
    )
    assert b"\r\n" not in raw and "标题" not in raw.decode()
    assert {p.name for p in output.iterdir()} == {"train.txt", "export_manifest.json"}
    assert json.loads((output / "export_manifest.json").read_text(encoding="utf-8")) == report


def test_reordering_is_invariant_but_sample_seed_changes_order(dataset, tmp_path):
    directory, records = dataset
    reversed_dir = prepare(tmp_path, "reversed", list(reversed(records)))
    a = export(dataset, tmp_path / "a", max_bytes=160)
    b = export((reversed_dir, records), tmp_path / "b", max_bytes=160)
    assert (tmp_path / "a/train.txt").read_bytes() == (tmp_path / "b/train.txt").read_bytes()
    assert a["output"] == b["output"] and a["selection"] == b["selection"]
    assert a["source_split"]["sha256"] != b["source_split"]["sha256"]
    c = export(dataset, tmp_path / "c", max_bytes=160, sample_seed=43)
    assert c["output"]["sha256"] != a["output"]["sha256"]
    assert (tmp_path / "c/train.txt").read_bytes() == prefix_bytes(ordered(records, 43), 160)
    assert directory.is_dir()


def test_hash_export_does_not_read_valid_test_or_modify_inputs(dataset, tmp_path, monkeypatch):
    directory, _ = dataset
    snapshots = {p: p.read_bytes() for p in directory.iterdir()}
    original_open = Path.open

    def guard(path, *args, **kwargs):
        if path.name in {"valid.jsonl", "test.jsonl"}:
            raise AssertionError("采样不应读取 valid/test 正文")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guard)
        export(dataset, tmp_path / "export", max_bytes=160)
    assert all(p.read_bytes() == before for p, before in snapshots.items())


def test_source_statistics_include_unknown_and_document_boundaries(dataset, tmp_path):
    report = export(dataset, tmp_path / "export", max_bytes=10000)
    expected = {}
    for record in dataset[1]:
        label = record["source"]
        if isinstance(label, dict):
            label = label["name"]
        if not isinstance(label, str):
            label = None
        stats = expected.setdefault(label, {"documents": 0, "characters": 0, "bytes": 0})
        stats["documents"] += 1
        stats["characters"] += len(record["text"])
        stats["bytes"] += len((record["text"] + cli.SUFFIX).encode())
    assert {
        s["source"]: {k: v for k, v in s.items() if k != "source"}
        for s in report["output"]["sources"]
    } == expected
    for field in ("documents", "characters", "bytes"):
        assert sum(s[field] for s in report["output"]["sources"]) == report["output"][field]


@pytest.mark.parametrize("seed", range(10))
def test_heap_matches_full_sorted_prefix_after_every_addition(seed):
    rng = random.Random(seed)
    texts = [f"文档{i}:" + "😀x" * rng.randrange(1, 30) for i in range(60)]
    rng.shuffle(texts)
    for budget in (1, 80, 500, 20000):
        sampler = sampling._HashPrefixSampler(budget, seed, len(cli.SUFFIX.encode()))
        seen = []
        for text in texts:
            sampler.add(text, "toy")
            seen.append({"text": text})
            expected, total = [], 0
            for record in ordered(seen, seed):
                expected.append(record["text"].encode())
                total += len((record["text"] + cli.SUFFIX).encode())
                if total >= budget:
                    break
            actual = sampler.ordered()
            assert [c.key[1] for c in actual] == expected
            assert sampler._bytes == sum(c.bytes for c in actual) == total
            assert total < budget or total - actual[-1].bytes < budget


def test_hash_tie_uses_utf8_body_not_source_order(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(sampling, "_sampling_key", lambda text, seed: (b"same", text.encode()))
    export(dataset, tmp_path / "export", max_bytes=10000)
    expected = sorted(dataset[1], key=lambda r: r["text"].encode())
    assert (tmp_path / "export/train.txt").read_bytes() == prefix_bytes(expected, 10000)


def test_boundary_document_is_not_skipped_to_fill_smaller_documents(dataset, tmp_path, monkeypatch):
    directory, _ = dataset
    texts = ["小文档1", "非常长的文档" * 20, "甲"]
    assert all(choose_split(t) == "train" for t in texts)
    rewrite_train(directory, [{"text": t} for t in texts])
    ranks = {t: i.to_bytes(1, "big") for i, t in enumerate(texts)}
    monkeypatch.setattr(sampling, "_sampling_key", lambda text, seed: (ranks[text], text.encode()))
    first = (texts[0] + cli.SUFFIX).encode()
    report = export(dataset, tmp_path / "export", max_bytes=len(first) + 20)
    assert report["output"]["documents"] == 1
    assert (tmp_path / "export/train.txt").read_bytes() == first


@pytest.mark.parametrize(
    "options, error",
    [
        ({}, ValueError),
        ({"max_bytes": 0}, ValueError),
        ({"max_bytes": True}, ValueError),
        ({"max_bytes": 1.5}, ValueError),
        ({"max_bytes": 100, "sample_seed": True}, TypeError),
        ({"max_bytes": 100, "sample_seed": 1.5}, TypeError),
        ({"max_bytes": 100, "sample_seed": "42"}, TypeError),
    ],
)
def test_invalid_hash_options_fail_before_creating_output(dataset, tmp_path, options, error):
    output = tmp_path / "export"
    with pytest.raises(error):
        export(dataset, output, **options)
    assert not output.exists()


@pytest.mark.parametrize("options", [{"selection": "unknown"}, {"sample_seed": 42}])
def test_prefix_rejects_unknown_mode_or_ignored_sample_seed(dataset, tmp_path, options):
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        cli.export_train_text(dataset[0] / "manifest.json", output, **options)
    assert not output.exists()


def test_first_hash_document_too_large_is_failure_not_empty_success(dataset, tmp_path):
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="第一篇完整文档"):
        export(dataset, output, max_bytes=1)
    assert not (output / "export_manifest.json").exists()


@pytest.mark.parametrize("bad_text", ["", "边界文档5", "边界文档91<|endoftext|>"])
def test_entire_train_is_validated_even_if_bad_last_document_cannot_be_selected(
    dataset, tmp_path, bad_text
):
    directory, records = dataset
    rewrite_train(directory, records + [{"text": bad_text}])
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="空正文|不属于 train|保留的 EOS"):
        export(dataset, output, max_bytes=80)
    assert not (output / "export_manifest.json").exists()


@pytest.mark.parametrize("field", ["documents", "characters"])
def test_full_counts_are_checked_before_writing_hash_sample(dataset, tmp_path, field):
    path = dataset[0] / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["files"]["train"][field] += 1
    if field == "documents":
        manifest["counts"]["saved"] += 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="文档数或正文字符数"):
        export(dataset, output, max_bytes=80)
    assert (output / "train.txt").read_bytes() == b""
    assert not (output / "export_manifest.json").exists()


@pytest.mark.parametrize("filename", ["train.jsonl", "manifest.json"])
def test_source_changes_after_selection_never_produce_success_manifest(
    dataset, tmp_path, monkeypatch, filename
):
    original = sampling._HashPrefixSampler.ordered

    def mutate(sampler):
        path = dataset[0] / filename
        path.write_bytes(path.read_bytes() + b"\n")
        return original(sampler)

    monkeypatch.setattr(sampling._HashPrefixSampler, "ordered", mutate)
    output = tmp_path / "export"
    with pytest.raises(RuntimeError, match="发生变化"):
        export(dataset, output, max_bytes=80)
    assert not (output / "export_manifest.json").exists()


def test_existing_directory_cannot_be_overwritten(dataset, tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    marker = output / "train.txt"
    marker.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        export(dataset, output, max_bytes=80)
    assert marker.read_bytes() == b"original"


def test_hash_module_entry_and_default_prefix_compatibility(dataset, tmp_path):
    directory, records = dataset
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "minigpt_cn.data.export_text",
            "--manifest",
            str(directory / "manifest.json"),
            "--output-dir",
            str(tmp_path / "hash"),
            "--selection",
            "hash",
            "--sample-seed",
            "42",
            "--max-bytes",
            "160",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert json.loads(result.stdout)["selection"]["sample_seed"] == 42
    assert "RuntimeWarning" not in result.stderr
    report = cli.export_train_text(directory / "manifest.json", tmp_path / "prefix", max_bytes=160)
    assert (tmp_path / "prefix/train.txt").read_bytes() == prefix_bytes(records, 160)
    assert report["selection"] == {"strategy": "whole_document_prefix", "max_bytes": 160}
    assert "sources" not in report["output"]

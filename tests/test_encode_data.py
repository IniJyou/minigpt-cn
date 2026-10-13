"""固定三集合编码验收：版本绑定、原始uint16、完整正文边界、失败不报成功。"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from minigpt_cn.data import encode as cli
from minigpt_cn.data.prepare import _sha256, prepare_jsonl
from minigpt_cn.data.splitting import choose_split
from minigpt_cn.tokenizer import BPETokenizer


@pytest.fixture
def corpus(tmp_path):
    groups = {split: [] for split in ("train", "valid", "test")}
    for i in range(10000):
        text = f"数学與繁體😀第{i}篇\n第二行 é"
        split = choose_split(text)
        if len(groups[split]) < 2:
            groups[split].append({"text": text})
        if all(len(records) == 2 for records in groups.values()):
            break
    raw = tmp_path / "raw.jsonl"
    raw.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n"
            for records in groups.values()
            for record in records
        ),
        encoding="utf-8",
    )
    source = tmp_path / "prepared"
    prepare_jsonl([raw], source, to_simplified=False)
    # 测试使用真实的257词条字节基线，便于独立计算ID；不填充伪词条。
    tokenizer = BPETokenizer(
        {**{i: bytes([i]) for i in range(256)}, 256: b"<|endoftext|>"}, [], [cli.EOS]
    )
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path, overwrite=False)
    return source / "manifest.json", tokenizer_path


def encode(corpus, output, **kwargs):
    manifest, tokenizer = corpus
    options = {"expected_tokenizer_sha256": _sha256(tokenizer), "expected_vocab_size": 257}
    options.update(kwargs)
    return cli.encode_splits(manifest, tokenizer, output, **options)


def rewrite_source(corpus, split, records=None, **metadata):
    manifest_path, _ = corpus
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = manifest_path.parent / f"{split}.jsonl"
    if records is not None:
        source.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        )
    manifest["files"][split].update(bytes=source.stat().st_size, sha256=_sha256(source), **metadata)
    manifest["counts"]["saved"] = sum(info["documents"] for info in manifest["files"].values())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")


def test_all_splits_match_independent_byte_ids_and_preserve_inputs(corpus, tmp_path):
    manifest_path, tokenizer = corpus
    before = {p: p.read_bytes() for p in (tokenizer, *manifest_path.parent.iterdir())}
    output = tmp_path / "encoded"
    report = encode(corpus, output)
    assert report == json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert report["storage"] == {"dtype": "<u2", "byte_order": "little", "header": False}
    assert report["purpose"] == "pilot_loop_verification" and report["status"] == "completed"
    assert report["tokenizer"]["sha256"] == _sha256(tokenizer)
    assert report["source_manifest"]["sha256"] == _sha256(manifest_path)
    assert report["encoding_seconds"] >= 0
    for split, info in report["files"].items():
        source = manifest_path.parent / f"{split}.jsonl"
        texts = [
            json.loads(line)["text"] for line in source.read_text(encoding="utf-8").splitlines()
        ]
        # 字节词表的独立预期：UTF-8字节ID后追加256和LF的10。
        expected = [i for text in texts for i in (*text.encode("utf-8"), 256, 10)]
        target = output / info["path"]
        assert target.read_bytes() == np.asarray(expected, dtype="<u2").tobytes()
        assert target.read_bytes().endswith(b"\x00\x01\x0a\x00")
        assert info["tokens"] == len(expected) and info["bytes"] == 2 * len(expected)
        assert info["sha256"] == _sha256(target)
        assert info["documents"] == info["eos_count"] == 2
        assert info["characters"] == sum(map(len, texts))
        assert info["min_id"] == min(expected) and info["max_id"] == 256
        assert info["round_trip_verified"] and info["readback_verified"]
        restored = BPETokenizer.load(tokenizer)
        assert restored.decode(np.fromfile(target, dtype="<u2")) == "".join(
            t + cli.SUFFIX for t in texts
        )
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert {p.name for p in output.iterdir()} == {
        "train.bin",
        "valid.bin",
        "test.bin",
        "manifest.json",
    }


def test_repeated_encoding_to_new_directories_is_byte_identical(corpus, tmp_path):
    first = encode(corpus, tmp_path / "first")
    second = encode(corpus, tmp_path / "second")
    assert first["files"] == second["files"]


def test_existing_output_is_rejected_before_reading_inputs(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "train.bin"
    marker.write_bytes(b"original")
    with pytest.raises(FileExistsError, match="先检查已有产物"):
        cli.encode_splits("missing", "missing", output, expected_tokenizer_sha256="0" * 64)
    assert marker.read_bytes() == b"original"


@pytest.mark.parametrize(
    "options",
    [
        {"expected_tokenizer_sha256": "0" * 64},
        {"expected_tokenizer_sha256": "bad"},
        {"expected_vocab_size": 16000},
        {"expected_vocab_size": 65537},
        {"expected_vocab_size": True},
    ],
)
def test_reject_wrong_version_or_uint16_vocabulary_before_output(corpus, tmp_path, options):
    output = tmp_path / "encoded"
    with pytest.raises(ValueError):
        encode(corpus, output, **options)
    assert not output.exists()


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_all_source_hashes_are_checked_before_output(corpus, tmp_path, split):
    source = corpus[0].parent / f"{split}.jsonl"
    with source.open("ab") as f:
        f.write(b" ")
    output = tmp_path / "encoded"
    with pytest.raises(ValueError, match="来源清单不一致"):
        encode(corpus, output)
    assert not output.exists()


@pytest.mark.parametrize("fault", ["wrong_split", "eos", "malformed", "documents", "characters"])
def test_bad_records_or_counts_leave_no_success_manifest(corpus, tmp_path, fault):
    split = "train"
    if fault == "wrong_split":
        records = [
            json.loads(line)
            for line in (corpus[0].parent / "valid.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        rewrite_source(corpus, split, records=records)
    elif fault == "eos":
        text = next(
            f"文档{i}{cli.EOS}" for i in range(1000) if choose_split(f"文档{i}{cli.EOS}") == split
        )
        rewrite_source(corpus, split, records=[{"text": text}, {"text": text}])
    elif fault == "malformed":
        rewrite_source(corpus, split, records=[{"text": 42}])
    elif fault == "documents":
        rewrite_source(corpus, split, documents=3)
    else:
        rewrite_source(corpus, split, characters=1)
    output = tmp_path / "encoded"
    with pytest.raises(ValueError):
        encode(corpus, output)
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("bad_id", [-1, 65536])
def test_id_overflow_is_rejected_before_uint16_conversion(corpus, tmp_path, monkeypatch, bad_id):
    original = BPETokenizer.encode

    def poisoned(tokenizer, text):
        return [bad_id] if text.endswith(cli.SUFFIX) else original(tokenizer, text)

    monkeypatch.setattr(BPETokenizer, "encode", poisoned)
    output = tmp_path / "encoded"
    with pytest.raises(ValueError, match="编码ID超出"):
        encode(corpus, output)
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("which", [0, 1])
def test_changed_manifest_or_tokenizer_never_gets_success(corpus, tmp_path, monkeypatch, which):
    original = BPETokenizer.encode
    changed = False

    def mutate(tokenizer, text):
        nonlocal changed
        if text.endswith(cli.SUFFIX) and not changed:
            with corpus[which].open("ab") as f:
                f.write(b" ")
            changed = True
        return original(tokenizer, text)

    monkeypatch.setattr(BPETokenizer, "encode", mutate)
    output = tmp_path / "encoded"
    with pytest.raises(RuntimeError, match="来源清单或 tokenizer 发生变化"):
        encode(corpus, output)
    assert not (output / "manifest.json").exists()


def test_disk_corruption_does_not_get_success_manifest(corpus, tmp_path, monkeypatch):
    original = cli._sha256

    def corrupt(path):
        if path.suffix == ".bin":
            with path.open("r+b") as f:
                f.write(b"\xff")
        return original(path)

    monkeypatch.setattr(cli, "_sha256", corrupt)
    output = tmp_path / "encoded"
    with pytest.raises(RuntimeError, match="保存读回"):
        encode(corpus, output)
    assert not (output / "manifest.json").exists()


def test_module_cli_reports_success_and_clear_failure(corpus, tmp_path):
    output = tmp_path / "encoded"
    args = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "minigpt_cn.data.encode",
        "--manifest",
        str(corpus[0]),
        "--tokenizer",
        str(corpus[1]),
        "--tokenizer-sha256",
        _sha256(corpus[1]),
        "--vocab-size",
        "257",
        "--output-dir",
        str(output),
    ]
    result = subprocess.run(args, capture_output=True, encoding="utf-8", check=True, timeout=30)
    assert json.loads(result.stdout)["status"] == "completed"
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    failed = subprocess.run(args, capture_output=True, encoding="utf-8", timeout=30)
    assert failed.returncode == 2 and "不覆盖" in failed.stderr
    assert "独立只读验收" in failed.stderr and not failed.stdout
    assert "不可用于训练" not in failed.stderr and "编码未完成" not in failed.stderr
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
    # 输入版本错误仍必须失败，并明确说明只有失败留下的部分新产物不可使用。
    wrong_version = args.copy()
    wrong_version[wrong_version.index("--tokenizer-sha256") + 1] = "0" * 64
    wrong_version[-1] = str(tmp_path / "failed_new_output")
    invalid = subprocess.run(wrong_version, capture_output=True, encoding="utf-8", timeout=30)
    assert invalid.returncode == 2 and "本次未成功完成编码" in invalid.stderr
    assert "若留下" in invalid.stderr and "不可用于训练" in invalid.stderr
    assert not Path(wrong_version[-1]).exists() and not invalid.stdout

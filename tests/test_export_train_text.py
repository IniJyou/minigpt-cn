"""BPE 训练文本导出：只读 train、整篇前缀、EOS边界、来源校验与防覆盖。"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from minigpt_cn.data import export_text as cli
from minigpt_cn.data.prepare import prepare_jsonl

TEXTS = ["边界文档91", "边界文档150", "甲"]  # 三份固定 train 样本，按此顺序写入。


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "raw.jsonl"
    records = [{"text": text, "title": "标题不会导出"} for text in TEXTS]
    records += [{"text": text} for text in ("边界文档5", "边界文档28")]
    source.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    directory = tmp_path / "prepared"
    prepare_jsonl([source], directory)
    return directory


def arguments(prepared, output):
    return ["--manifest", str(prepared / "manifest.json"), "--output-dir", str(output)]


def change_manifest(prepared, modify):
    path = prepared / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    modify(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def replace_train(prepared, text):
    path = prepared / "train.jsonl"
    raw = (json.dumps({"text": text}, ensure_ascii=False) + "\n").encode("utf-8")
    path.write_bytes(raw)

    def update(manifest):
        info = manifest["files"]["train"]
        info.update(
            documents=1,
            characters=len(text),
            bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        manifest["counts"]["saved"] = 3

    change_manifest(prepared, update)


def test_full_export_keeps_text_and_boundaries_without_reading_valid_or_test(
    prepared, tmp_path, monkeypatch, capsys
):
    snapshots = {path: path.read_bytes() for path in prepared.iterdir()}
    original_open = Path.open

    def guard_open(path, *args, **kwargs):
        if path.name in {"valid.jsonl", "test.jsonl"}:
            raise AssertionError("BPE 导出不能读取验证/测试正文")
        return original_open(path, *args, **kwargs)

    output = tmp_path / "nested" / "export"
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guard_open)
        report = cli.main(arguments(prepared, output))
    printed = json.loads(capsys.readouterr().out)
    assert printed.pop("output_dir") == str(output.resolve())
    assert (
        printed
        == report
        == json.loads((output / "export_manifest.json").read_text(encoding="utf-8"))
    )
    raw = (output / "train.txt").read_bytes()
    assert raw == "".join(text + cli.SUFFIX for text in TEXTS).encode("utf-8")
    assert raw.count(cli.EOS.encode()) == len(TEXTS) and b"\r\n" not in raw
    assert b"title" not in raw and "标题" not in raw.decode("utf-8")
    assert report["source_split"]["split"] == "train"
    assert report["source_split"]["path"] == str(prepared / "train.jsonl")
    assert (
        report["source_manifest"]["sha256"]
        == hashlib.sha256(snapshots[prepared / "manifest.json"]).hexdigest()
    )
    info = report["output"]
    assert info["documents"] == 3 and info["characters"] == sum(map(len, TEXTS))
    assert info["bytes"] == len(raw) and info["sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["selection"]["max_bytes"] is None
    assert {path.name for path in output.iterdir()} == {"train.txt", "export_manifest.json"}
    assert all(path.read_bytes() == before for path, before in snapshots.items())


@pytest.mark.parametrize("extra_budget", [0, 17])
def test_budget_is_inclusive_whole_document_prefix_not_skip_and_fill(
    prepared, tmp_path, extra_budget
):
    first = (TEXTS[0] + cli.SUFFIX).encode()
    budget = len(first) + extra_budget
    output = tmp_path / "export"
    report = cli.main(arguments(prepared, output) + ["--max-bytes", str(budget)])
    # 第二篇放不下时锁定前缀；第三篇即使能填进余量，也不能跳过第二篇去挑它。
    assert (output / "train.txt").read_bytes() == first
    assert report["source_split"]["documents"] == 3
    assert report["output"]["documents"] == 1 and report["output"]["bytes"] <= budget


def test_too_small_budget_fails_without_success_manifest(prepared, tmp_path):
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="第一篇完整文档"):
        cli.export_train_text(prepared / "manifest.json", output, max_bytes=1)
    assert not (output / "export_manifest.json").exists()


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_fails_before_creating_output(prepared, tmp_path, budget):
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="max_bytes"):
        cli.export_train_text(prepared / "manifest.json", output, max_bytes=budget)
    assert not output.exists()


def test_tampered_train_fails_before_creating_output(prepared, tmp_path, capsys):
    source = prepared / "train.jsonl"
    source.write_bytes(source.read_bytes() + b"\n")
    output = tmp_path / "export"
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(prepared, output))
    captured = capsys.readouterr()
    assert error.value.code == 2 and captured.out == "" and "清单不一致" in captured.err
    assert not output.exists()


@pytest.mark.parametrize(
    "field, value",
    [
        ("format_version", 2),
        ("format_version", True),
        ("files", []),
        ("files", ["train", "valid", "test"]),
    ],
)
def test_invalid_manifest_is_rejected(prepared, tmp_path, field, value):
    change_manifest(prepared, lambda manifest: manifest.update({field: value}))
    output = tmp_path / "export"
    with pytest.raises(ValueError):
        cli.export_train_text(prepared / "manifest.json", output)
    assert not output.exists()


@pytest.mark.parametrize("path", ["valid.jsonl", "../train.jsonl", "/tmp/other.jsonl"])
def test_train_path_cannot_be_redirected_by_manifest(prepared, tmp_path, path):
    change_manifest(prepared, lambda manifest: manifest["files"]["train"].update(path=path))
    with pytest.raises(ValueError, match="路径"):
        cli.export_train_text(prepared / "manifest.json", tmp_path / "export")


@pytest.mark.parametrize("field", ["documents", "characters"])
def test_full_train_counters_are_verified_even_when_export_is_limited(prepared, tmp_path, field):
    def modify(manifest):
        manifest["files"]["train"][field] += 1
        if field == "documents":
            manifest["counts"]["saved"] += 1

    change_manifest(prepared, modify)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="文档数或正文字符数"):
        cli.export_train_text(prepared / "manifest.json", output, max_bytes=32)
    assert not (output / "export_manifest.json").exists()


def test_wrong_split_text_is_rejected_even_if_manifest_hash_is_updated(prepared, tmp_path):
    replace_train(prepared, "边界文档5")  # valid 的固定向量，不能伪装成 train。
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="不属于 train"):
        cli.export_train_text(prepared / "manifest.json", output)
    assert not (output / "export_manifest.json").exists()


def test_reserved_eos_in_body_is_rejected(prepared, tmp_path):
    text = "边界文档91" + cli.EOS
    assert cli.choose_split(text) == "train"
    replace_train(prepared, text)
    with pytest.raises(ValueError, match="保留的 EOS"):
        cli.export_train_text(prepared / "manifest.json", tmp_path / "export")


def test_source_changes_during_export_do_not_produce_success_manifest(
    prepared, tmp_path, monkeypatch
):
    source = prepared / "train.jsonl"
    before = source.read_bytes()
    original_choose = cli.choose_split

    def change_source(text, **kwargs):
        if text == TEXTS[0]:
            source.write_bytes(before.replace(b"1", b"2", 1))
        return original_choose(text, **kwargs)

    monkeypatch.setattr(cli, "choose_split", change_source)
    output = tmp_path / "export"
    with pytest.raises(RuntimeError, match="train.jsonl 发生变化"):
        cli.export_train_text(prepared / "manifest.json", output)
    assert not (output / "export_manifest.json").exists()


def test_manifest_changes_during_export_do_not_produce_success_manifest(
    prepared, tmp_path, monkeypatch
):
    path = prepared / "manifest.json"
    before = path.read_bytes()
    original_choose = cli.choose_split

    def change_source(text, **kwargs):
        if text == TEXTS[0]:
            path.write_bytes(before + b"\n")
        return original_choose(text, **kwargs)

    monkeypatch.setattr(cli, "choose_split", change_source)
    output = tmp_path / "export"
    with pytest.raises(RuntimeError, match="来源清单发生变化"):
        cli.export_train_text(path, output)
    assert not (output / "export_manifest.json").exists()


def test_existing_output_and_precheck_race_cannot_overwrite_data(prepared, tmp_path, monkeypatch):
    output = tmp_path / "export"
    output.mkdir()
    marker = output / "train.txt"
    marker.write_bytes(b"original text")
    with pytest.raises(FileExistsError):
        cli.export_train_text(prepared / "manifest.json", output)
    assert marker.read_bytes() == b"original text"

    concurrent = tmp_path / "concurrent"
    original_hash = cli._sha256

    def create_output(path):
        concurrent.mkdir()
        (concurrent / "marker").write_bytes(b"someone else's text")
        return original_hash(path)

    monkeypatch.setattr(cli, "_sha256", create_output)
    with pytest.raises(FileExistsError):
        cli.export_train_text(prepared / "manifest.json", concurrent)
    assert {path.name for path in concurrent.iterdir()} == {"marker"}


@pytest.mark.parametrize("kind", ["output", "train"])
def test_symbolic_links_are_rejected(prepared, tmp_path, kind):
    output = tmp_path / "export"
    if kind == "output":
        link, target = output, tmp_path / "missing-target"
    else:
        link, target = prepared / "train.jsonl", tmp_path / "real-train.jsonl"
        target.write_bytes(link.read_bytes())
        link.unlink()
    try:
        link.symlink_to(target, target_is_directory=kind == "output")
    except OSError:
        pytest.skip("此 Windows 环境未授予创建符号链接权限")
    with pytest.raises((FileExistsError, ValueError)):
        cli.export_train_text(prepared / "manifest.json", output)


def test_module_entry_exports_without_import_warning(prepared, tmp_path):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "minigpt_cn.data.export_text"]
        + arguments(prepared, tmp_path / "export"),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert json.loads(result.stdout)["output"]["documents"] == 3
    assert "RuntimeWarning" not in result.stderr

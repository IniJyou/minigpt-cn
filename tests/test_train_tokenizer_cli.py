"""小语料CLI回归：不覆盖、重载、实际词表大小和清楚的失败；不依赖真实数据。"""

import hashlib
import json
import subprocess
import sys

import pytest

from minigpt_cn.tokenizer import BPETokenizer, train_bpe
from minigpt_cn.tokenizer import train as cli


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "train.txt"
    path.write_text("数学与计算机。😀\n数学与语言。<|endoftext|>\n" * 3, encoding="utf-8")
    return path


def arguments(source, output, target=300):
    return ["--input", str(source), "--vocab-size", str(target), "--output", str(output)]


def test_cli_trains_saves_reloads_and_reports_actual_ids_and_hashes(source, tmp_path, capsys):
    output = tmp_path / "nested" / "tokenizer.json"
    before = source.read_bytes()
    report = cli.main(arguments(source, output))
    assert json.loads(capsys.readouterr().out) == report
    assert report["input_sha256"] == hashlib.sha256(before).hexdigest()
    assert report["input_bytes"] == len(before) and source.read_bytes() == before
    assert report["tokenizer_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    vocab, merges = train_bpe(source, 300, ["<|endoftext|>"])
    restored = BPETokenizer.load(output)
    assert (restored.vocab, restored.merges) == (vocab, merges)
    assert report["actual_vocab_size"] == len(vocab) <= report["requested_vocab_size"] == 300
    assert report["merges"] == len(merges) and report["training_seconds"] >= 0
    assert report["reload_verified"] and report["round_trip_verified"]
    assert report["special_token_ids"] == {"<|endoftext|>": 256}
    text = "未见过的文字 🦉\n"
    assert restored.decode(restored.encode(text)) == text


def test_explicit_special_tokens_replace_default_and_keep_order(source, tmp_path):
    report = cli.main(
        arguments(source, tmp_path / "tokenizer.json")
        + ["--special-token", "<doc>", "--special-token", "<end>", "--check-text", "中文😀<end>"]
    )
    assert report["special_token_ids"] == {"<doc>": 256, "<end>": 257}


def test_small_corpus_reports_underfilled_vocabulary_without_padding(source, tmp_path, capsys):
    source.write_text("a1b2", encoding="utf-8")
    report = cli.main(arguments(source, tmp_path / "tokenizer.json"))
    assert report["actual_vocab_size"] == 257 and report["merges"] == 0
    assert "实际词表小于目标" in capsys.readouterr().err


def test_existing_output_is_rejected_before_training(source, tmp_path, monkeypatch, capsys):
    output = tmp_path / "tokenizer.json"
    output.write_bytes(b"original tokenizer")

    def must_not_train(*args):
        raise AssertionError("已有输出时不应开始训练")

    monkeypatch.setattr(cli, "train_bpe", must_not_train)
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(source, output))
    assert error.value.code == 2 and output.read_bytes() == b"original tokenizer"
    assert "已存在" in capsys.readouterr().err


def test_cli_module_entry_runs_in_a_fresh_process(source, tmp_path):
    output = tmp_path / "tokenizer.json"
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "minigpt_cn.tokenizer.train"]
        + arguments(source, output, 257),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    report = json.loads(result.stdout)
    assert report["actual_vocab_size"] == 257 and report["reload_verified"]
    assert "RuntimeWarning" not in result.stderr


def test_existing_broken_symlink_is_not_filled_or_overwritten(source, tmp_path):
    target = tmp_path / "missing-target.json"
    link = tmp_path / "tokenizer.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("此Windows环境未授予创建符号链接权限")
    with pytest.raises(SystemExit):
        cli.main(arguments(source, link))
    assert link.is_symlink() and not target.exists()


def test_save_exclusive_mode_protects_against_output_created_during_training(
    source, tmp_path, monkeypatch
):
    output = tmp_path / "tokenizer.json"

    def concurrent_output(*args):
        result = train_bpe(*args)
        output.write_bytes(b"someone else's tokenizer")
        return result

    monkeypatch.setattr(cli, "train_bpe", concurrent_output)
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(source, output))
    assert error.value.code == 2 and output.read_bytes() == b"someone else's tokenizer"


def test_changed_input_is_rejected_before_saving(source, tmp_path, monkeypatch, capsys):
    output = tmp_path / "tokenizer.json"

    def changed_input(*args):
        result = train_bpe(*args)
        source.write_text("输入已改变", encoding="utf-8")
        return result

    monkeypatch.setattr(cli, "train_bpe", changed_input)
    with pytest.raises(SystemExit):
        cli.main(arguments(source, output))
    assert not output.exists() and "输入文本发生变化" in capsys.readouterr().err


@pytest.mark.parametrize(
    "fault", ["missing", "directory", "empty", "binary", "jsonl", "budget", "suffix"]
)
def test_invalid_cli_input_reports_error_without_output(source, tmp_path, fault, capsys):
    output = tmp_path / "tokenizer.json"
    target = 300
    if fault == "missing":
        source = tmp_path / "missing.txt"
    elif fault == "directory":
        source = tmp_path
    elif fault == "empty":
        source.write_bytes(b"")
    elif fault == "binary":
        source.write_bytes(b"\xff")
    elif fault == "jsonl":
        source = source.rename(tmp_path / "train.jsonl")
    elif fault == "budget":
        target = 256  # 默认EOS也占一个位置，至少257。
    elif fault == "suffix":
        output = tmp_path / "tokenizer.txt"
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(source, output, target))
    assert error.value.code == 2 and not output.exists()
    assert "error:" in capsys.readouterr().err


def test_failed_reload_is_not_reported_as_success(source, tmp_path, monkeypatch, capsys):
    fake = BPETokenizer({i: bytes([i]) for i in range(256)}, [])
    monkeypatch.setattr(cli.BPETokenizer, "load", lambda path: fake)
    with pytest.raises(SystemExit):
        cli.main(arguments(source, tmp_path / "tokenizer.json"))
    captured = capsys.readouterr()
    assert "保存重载" in captured.err and not captured.out


def test_tokenizer_save_preserves_default_but_supports_explicit_no_overwrite(tmp_path):
    tokenizer = BPETokenizer({i: bytes([i]) for i in range(256)}, [])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path, overwrite=False)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        tokenizer.save(path, overwrite=False)
    assert path.read_bytes() == before
    tokenizer.save(path)  # 原默认API仍可覆盖，未改变已有调用语义。
    assert path.read_bytes() == before

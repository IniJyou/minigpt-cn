"""JSONL 处理入口测试：内存规则接文件、来源字段保留、不覆盖与失败不报成功。"""

import hashlib
import json
import subprocess
import sys

import pytest

from minigpt_cn.data import choose_split
from minigpt_cn.data import prepare as cli

RECORDS = [
    {"text": "  边界文档91\r\n", "id": "train", "title": "繁體原標題", "url": "example:1"},
    {"text": "边界文档5", "id": "valid", "source": {"name": "toy", "license": "test-only"}},
    {"text": "边界文档28", "id": "test", "extra": [1, "😀", None]},
]


def write_records(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "raw.jsonl"
    write_records(path, RECORDS)
    return path


def arguments(source, output):
    return ["--input", str(source), "--output-dir", str(output)]


def read_outputs(output):
    return {
        name: [
            json.loads(line)
            for line in (output / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for name in ("train", "valid", "test")
    }


def test_cli_preserves_metadata_and_reports_counts_hashes_and_rules(source, tmp_path, capsys):
    extra = [
        {"text": "臺灣"},
        {"text": "台湾"},  # 规范化后重复：去重先于长度过滤。
        {"text": "a" * 7},
        {"text": " \t\r\n"},
        {"text": "边界文档5", "id": "later", "source": "later source"},
    ]
    write_records(source, RECORDS + extra)
    source.write_bytes(b"\n" + source.read_bytes() + b" \t\n")
    before = source.read_bytes()
    output = tmp_path / "nested" / "prepared"
    manifest = cli.main(arguments(source, output) + ["--min-chars", "5", "--max-chars", "6"])
    printed = json.loads(capsys.readouterr().out)
    assert printed.pop("output_dir") == str(output.resolve())
    assert printed == manifest == json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert {path.name for path in output.iterdir()} == {
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
        "manifest.json",
    }
    assert source.read_bytes() == before
    assert manifest["inputs"] == [
        {
            "path": str(source.resolve()),
            "bytes": len(before),
            "sha256": hashlib.sha256(before).hexdigest(),
        }
    ]
    assert manifest["counts"] == {
        "read": 8,
        "blank_lines": 2,
        "empty": 1,
        "duplicates": 2,
        "length_filtered": 2,
        "saved": 3,
    }
    groups = read_outputs(output)
    for name, record in zip(("train", "valid", "test"), RECORDS):
        expected = dict(record, text=record["text"].strip())
        assert groups[name] == [expected]
        info = manifest["files"][name]
        contents = (output / info["path"]).read_bytes()
        assert info["documents"] == 1 and info["characters"] == len(expected["text"])
        assert (
            info["bytes"] == len(contents)
            and info["sha256"] == hashlib.sha256(contents).hexdigest()
        )
        assert b"\r\n" not in contents  # 各系统都写 LF；JSON 正文里的换行已转义。
        assert choose_split(expected["text"]) == name
    rules = manifest["processing"]
    assert rules["seed"] == 42 and rules["normalization"] == "NFC"
    assert rules["opencc_version"] == "1.4.2" and rules["opencc_config"] == "t2s.json"
    assert rules["min_chars"] == 5 and rules["max_chars"] == 6


def test_multiple_inputs_deduplicate_globally_and_keep_first_metadata(source, tmp_path):
    second = tmp_path / "second.jsonl"
    write_records(second, [{"text": "边界文档5", "id": "later"}, {"text": "新文档", "id": "new"}])
    output = tmp_path / "prepared"
    manifest = cli.prepare_jsonl([source, second], output)
    groups = read_outputs(output)
    assert manifest["counts"]["read"] == 5 and manifest["counts"]["duplicates"] == 1
    assert manifest["counts"]["saved"] == 4
    valid = next(record for record in groups["valid"] if record["text"] == "边界文档5")
    assert valid == RECORDS[1]
    assert len(manifest["inputs"]) == 2
    assert groups[choose_split("新文档")][-1] == {"text": "新文档", "id": "new"}


def test_repeated_processing_is_byte_identical_and_manifest_is_deterministic(source, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    assert cli.prepare_jsonl([source], first) == cli.prepare_jsonl([source], second)
    assert all(
        (first / path.name).read_bytes() == (second / path.name).read_bytes()
        for path in first.iterdir()
    )


def test_keep_traditional_flag_is_recorded_and_only_body_is_processed(source, tmp_path):
    write_records(source, RECORDS + [{"text": " 臺灣 ", "title": "繁體原標題"}])
    output = tmp_path / "prepared"
    manifest = cli.main(arguments(source, output) + ["--keep-traditional"])
    assert manifest["processing"]["to_simplified"] is False
    assert manifest["processing"]["opencc_config"] is None
    assert any(
        record == {"text": "臺灣", "title": "繁體原標題"}
        for group in read_outputs(output).values()
        for record in group
    )


def test_existing_output_even_empty_is_never_reused(source, tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    with pytest.raises(FileExistsError, match="已存在"):
        cli.prepare_jsonl([source], output)
    assert list(output.iterdir()) == []
    marker = output / "manifest.json"
    marker.write_bytes(b"someone else's manifest")
    with pytest.raises(FileExistsError):
        cli.prepare_jsonl([source], output)
    assert marker.read_bytes() == b"someone else's manifest"


def test_output_created_during_precheck_is_not_overwritten(source, tmp_path, monkeypatch):
    output = tmp_path / "prepared"
    original_hash = cli._sha256

    def create_output(path):
        output.mkdir()
        (output / "marker").write_bytes(b"someone else's data")
        return original_hash(path)

    monkeypatch.setattr(cli, "_sha256", create_output)
    with pytest.raises(FileExistsError):
        cli.prepare_jsonl([source], output)
    assert {path.name for path in output.iterdir()} == {"marker"}
    assert (output / "marker").read_bytes() == b"someone else's data"


def test_broken_output_symlink_is_rejected(source, tmp_path):
    target, link = tmp_path / "missing-target", tmp_path / "prepared"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("此 Windows 环境未授予创建符号链接权限")
    with pytest.raises(FileExistsError):
        cli.prepare_jsonl([source], link)
    assert link.is_symlink() and not target.exists()


@pytest.mark.parametrize(
    "bad_line",
    [
        b"not json\n",
        b"[]\n",
        b"{}\n",
        b'{"text":null}\n',
        b'{"text":42}\n',
        b'{"text":"body","score":NaN}\n',
        b"\xff\n",
    ],
)
def test_bad_record_reports_file_and_line_without_success_manifest(
    source, tmp_path, capsys, bad_line
):
    source.write_bytes(source.read_bytes() + bad_line)
    output = tmp_path / "prepared"
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(source, output))
    captured = capsys.readouterr()
    assert error.value.code == 2 and captured.out == ""
    assert f"{source.resolve()}:4:" in captured.err
    assert not (output / "manifest.json").exists()


@pytest.mark.parametrize("contents", ["", '{"text":"只落在一个集合"}\n'])
def test_empty_splits_fail_without_moving_documents_or_writing_manifest(source, tmp_path, contents):
    source.write_text(contents, encoding="utf-8")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="划分为空"):
        cli.prepare_jsonl([source], output)
    assert not (output / "manifest.json").exists()
    assert sum(len(group) for group in read_outputs(output).values()) == (1 if contents else 0)


@pytest.mark.parametrize(
    "options",
    [
        {"min_chars": 0},
        {"max_chars": 0},
        {"min_chars": True},
        {"seed": True},
        {"to_simplified": "true"},
    ],
)
def test_bad_options_fail_before_creating_output(source, tmp_path, options):
    output = tmp_path / "prepared"
    with pytest.raises((ValueError, TypeError)):
        cli.prepare_jsonl([source], output, **options)
    assert not output.exists()


def test_invalid_input_paths_are_rejected_before_creating_output(source, tmp_path):
    for index, paths in enumerate(
        ([], str(source), [source, source], [tmp_path / "missing"], [tmp_path])
    ):
        output = tmp_path / f"prepared-{index}"
        with pytest.raises(ValueError):
            cli.prepare_jsonl(paths, output)
        assert not output.exists()


def test_changed_source_does_not_produce_success_manifest(source, tmp_path, monkeypatch):
    original_normalize = cli.normalize_text
    before = source.read_bytes()

    def change_source(text, **options):
        source.write_bytes(before + b"\n")
        return original_normalize(text, **options)

    monkeypatch.setattr(cli, "normalize_text", change_source)
    output = tmp_path / "prepared"
    with pytest.raises(RuntimeError, match="输入发生变化"):
        cli.prepare_jsonl([source], output)
    assert not (output / "manifest.json").exists()


def test_module_entry_runs_in_fresh_process_and_does_not_import_torch(source, tmp_path):
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "minigpt_cn.data.prepare"]
        + arguments(source, tmp_path / "prepared"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    assert json.loads(result.stdout)["counts"]["saved"] == 3
    assert "RuntimeWarning" not in result.stderr
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import minigpt_cn.data.prepare; assert 'torch' not in sys.modules",
        ],
        check=True,
        timeout=30,
    )

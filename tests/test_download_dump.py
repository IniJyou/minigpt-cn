"""小型离线响应验收下载；测试不会访问公网或下载真实大型 dump。"""

import bz2
import hashlib
import io
import json
import subprocess
import sys

import pytest
import yaml

from minigpt_cn.data import download as cli


class Response(io.BytesIO):
    def __init__(self, body, *, status=200, headers=None, fail=False):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}
        self.fail = fail

    def read(self, size=-1):
        if self.fail:
            if self.tell():
                raise TimeoutError("模拟读取超时")
            size = 1
        return super().read(size)


@pytest.fixture
def remote(tmp_path, monkeypatch):
    config = yaml.safe_load(cli.DEFAULT_CONFIG.read_bytes())
    body = bz2.compress("<mediawiki>仅测试下载，不是正式语料</mediawiki>".encode())
    for info in config["sources"].values():
        info.update(bytes=len(body), sha1=hashlib.sha1(body).hexdigest())
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    state = {
        "config": config,
        "path": path,
        "body": body,
        "requests": [],
        "headers": {},
        "status": 200,
        "fail": False,
        "override_metadata": None,
        "mutate_metadata": None,
    }

    def open_response(url, timeout, *, offset=0):
        state["requests"].append(url)
        assert timeout == 30
        assert offset == 0  # 原入口测试不接受意外的续传请求；续传另用受控响应测试。
        for info in config["sources"].values():
            if url == info["status_url"]:
                status = {
                    "jobs": {
                        info["job"]: {
                            "status": "done",
                            "updated": "test-only",
                            "files": {
                                info["filename"]: {
                                    "size": info["bytes"],
                                    "sha1": info["sha1"],
                                    "url": info["url"].removeprefix("https://dumps.wikimedia.org"),
                                }
                            },
                        }
                    }
                }
                if state["mutate_metadata"]:
                    state["mutate_metadata"](status)
                raw = state["override_metadata"]
                if raw is None:
                    raw = json.dumps(status).encode()
                return Response(raw)
            if url == info["url"]:
                return Response(
                    state["body"],
                    status=state["status"],
                    headers=state["headers"],
                    fail=state["fail"],
                )
        raise AssertionError(f"意外请求：{url}")

    monkeypatch.setattr(cli, "_open", open_response)
    return state


def run(remote, output, source="zhwiki", **options):
    return cli.download_dump(remote["path"], source, output, **options)


def test_default_preview_is_offline_and_creates_nothing(remote, tmp_path):
    output = tmp_path / "missing" / "preview"
    report = run(remote, output)
    assert report["status"] == "planned" and report["file"] is None
    assert report["official_status"] is None and report["ready_for_training"] is False
    assert remote["requests"] == [] and not output.parent.exists()


@pytest.mark.parametrize("source", ["zhwiki", "zhwikisource"])
def test_remote_preview_reads_only_status_and_does_not_create_output(remote, tmp_path, source):
    output = tmp_path / "preview"
    report = run(remote, output, source, check_remote=True)
    assert report["status"] == "planned" and report["official_status"]["status"] == "done"
    assert remote["requests"] == [remote["config"]["sources"][source]["status_url"]]
    assert not output.exists() and report["file"] is None


@pytest.mark.parametrize("source", ["zhwiki", "zhwikisource"])
def test_download_verifies_local_bytes_and_writes_complete_provenance(remote, tmp_path, source):
    output = tmp_path / "download"
    before = remote["path"].read_bytes()
    report = run(remote, output, source, download=True)
    info = remote["config"]["sources"][source]
    assert remote["requests"] == [info["status_url"], info["url"]]
    assert report["status"] == "verified" and not report["ready_for_training"]
    assert report["file"] == {
        "path": info["filename"],
        "bytes": len(remote["body"]),
        "sha1": hashlib.sha1(remote["body"]).hexdigest(),
        "sha256": hashlib.sha256(remote["body"]).hexdigest(),
    }
    assert (output / info["filename"]).read_bytes() == remote["body"]
    assert (output / "source_config.yaml").read_bytes() == before == remote["path"].read_bytes()
    assert cli._sha256(output / "source_config.yaml") == report["source_config"]["sha256"]
    assert cli._sha256(output / "dumpstatus.json") == report["official_status"]["sha256"]
    assert json.loads((output / "download_manifest.json").read_text(encoding="utf-8")) == report
    assert {p.name for p in output.iterdir()} == {
        info["filename"],
        "source_config.yaml",
        "dumpstatus.json",
        "download_manifest.json",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("dump_date", "latest"),
        ("dump_date", "20260230"),
        ("dump_date", 20261001),
        ("filename", "../escape.xml.bz2"),
        ("url", "http://dumps.wikimedia.org/x"),
        ("status_url", "https://example.invalid/status"),
        ("bytes", 0),
        ("bytes", True),
        ("sha1", "F" * 40),
        ("sha1", "0" * 39),
        ("job", None),
        ("license", None),
    ],
)
def test_invalid_source_config_fails_before_network_or_writes(remote, tmp_path, field, value):
    remote["config"]["sources"]["zhwiki"][field] = value
    remote["path"].write_text(yaml.safe_dump(remote["config"]), encoding="utf-8")
    output = tmp_path / "download"
    with pytest.raises(ValueError):
        run(remote, output, download=True)
    assert not output.exists() and remote["requests"] == []


@pytest.mark.parametrize("value", [None, [], {"format_version": True}, {"format_version": 2}])
def test_bad_catalog_structure_is_rejected(remote, tmp_path, value):
    remote["path"].write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(ValueError):
        run(remote, tmp_path / "download")
    assert remote["requests"] == []


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_timeout_is_rejected(remote, tmp_path, value):
    with pytest.raises(ValueError, match="timeout"):
        run(remote, tmp_path / "download", timeout=value)


@pytest.mark.parametrize("field,value", [("size", 1), ("sha1", "0" * 40), ("url", "/latest/file")])
def test_changed_official_metadata_never_silently_changes_config(remote, tmp_path, field, value):
    info = remote["config"]["sources"]["zhwiki"]
    remote["mutate_metadata"] = lambda status: status["jobs"][info["job"]]["files"][
        info["filename"]
    ].update({field: value})
    output = tmp_path / "download"
    with pytest.raises(ValueError, match="元数据与固定配置不一致"):
        run(remote, output, download=True)
    assert not output.exists() and remote["requests"] == [info["status_url"]]


def test_unfinished_job_cannot_be_downloaded(remote, tmp_path):
    info = remote["config"]["sources"]["zhwiki"]
    remote["mutate_metadata"] = lambda status: status["jobs"][info["job"]].update(status="running")
    with pytest.raises(ValueError, match="尚未 done"):
        run(remote, tmp_path / "download", download=True)


@pytest.mark.parametrize(
    "raw",
    [b"[]", b"{broken", b"\xff", b'{"jobs": NaN}', b"x" * (cli.STATUS_LIMIT + 1)],
    ids=["wrong_shape", "bad_json", "bad_utf8", "nonfinite", "oversized"],
)
def test_bad_or_oversized_status_is_rejected(remote, tmp_path, raw):
    remote["override_metadata"] = raw
    with pytest.raises(ValueError):
        run(remote, tmp_path / "download", download=True)
    assert not (tmp_path / "download").exists()


def test_disk_preflight_checks_reserve_before_network(remote, tmp_path, monkeypatch):
    free = remote["config"]["sources"]["zhwiki"]["bytes"] + cli.RESERVE_BYTES - 1
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda path: type("Disk", (), {"free": free})())
    with pytest.raises(ValueError, match="磁盘空间不足"):
        run(remote, tmp_path / "download", download=True)
    assert remote["requests"] == []


def test_preview_rejects_file_as_parent_without_network(remote, tmp_path):
    parent = tmp_path / "file"
    parent.write_bytes(b"original")
    with pytest.raises(ValueError, match="父路径必须是目录"):
        run(remote, parent / "download")
    assert parent.read_bytes() == b"original" and remote["requests"] == []


@pytest.mark.parametrize(
    "problem", ["short", "long", "checksum", "timeout", "length", "encoding", "206"]
)
def test_transfer_failure_retains_diagnostics_without_success_manifest(remote, tmp_path, problem):
    if problem == "short":
        remote["body"] = remote["body"][:-1]
    elif problem == "long":
        remote["body"] += b"x"
    elif problem == "checksum":
        remote["body"] = b"X" + remote["body"][1:]
    elif problem == "timeout":
        remote["fail"] = True
    elif problem == "length":
        remote["headers"] = {"Content-Length": "1"}
    elif problem == "encoding":
        remote["headers"] = {"Content-Encoding": "gzip"}
    else:
        remote["status"] = 206
    output = tmp_path / "download"
    with pytest.raises((ValueError, ConnectionError)):
        run(remote, output, download=True, retries=0)
    assert (output / "source_config.yaml").exists() and (output / "dumpstatus.json").exists()
    assert not (output / "download_manifest.json").exists()
    assert not (output / remote["config"]["sources"]["zhwiki"]["filename"]).exists()


def test_existing_directory_and_creation_race_cannot_overwrite_files(remote, tmp_path):
    output = tmp_path / "download"
    output.mkdir()
    marker = output / "marker"
    marker.write_bytes(b"someone else's data")
    with pytest.raises(FileExistsError):
        run(remote, output, download=True)
    assert marker.read_bytes() == b"someone else's data" and remote["requests"] == []

    raced = tmp_path / "raced"

    def create_after_precheck(status):
        raced.mkdir()
        (raced / "marker").write_bytes(b"concurrent data")

    remote["mutate_metadata"] = create_after_precheck
    with pytest.raises(FileExistsError):
        run(remote, raced, download=True)
    assert {p.name for p in raced.iterdir()} == {"marker"}


def test_config_change_during_download_blocks_publication(remote, tmp_path, monkeypatch):
    original = cli._download_file

    def mutate(info, partial, timeout, **options):
        result = original(info, partial, timeout, **options)
        remote["path"].write_bytes(remote["path"].read_bytes() + b"\n")
        return result

    monkeypatch.setattr(cli, "_download_file", mutate)
    output = tmp_path / "download"
    with pytest.raises(RuntimeError, match="配置发生变化"):
        run(remote, output, download=True)
    assert not (output / "download_manifest.json").exists()
    assert (output / (remote["config"]["sources"]["zhwiki"]["filename"] + ".part")).exists()


def test_publish_does_not_overwrite_an_existing_final_file(tmp_path):
    partial, final = tmp_path / "new.part", tmp_path / "old"
    partial.write_bytes(b"new")
    final.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        cli._publish(partial, final)
    assert partial.read_bytes() == b"new" and final.read_bytes() == b"old"


def test_publication_failure_has_no_success_manifest(remote, tmp_path, monkeypatch):
    original = cli._publish

    def fail_manifest(partial, final):
        if final.name == "download_manifest.json":
            raise OSError("模拟清单定名失败")
        original(partial, final)

    monkeypatch.setattr(cli, "_publish", fail_manifest)
    output = tmp_path / "download"
    with pytest.raises(OSError):
        run(remote, output, download=True)
    assert not (output / "download_manifest.json").exists()
    assert (output / "download_manifest.json.part").exists()


def test_redirects_are_not_followed():
    assert (
        cli._NoRedirect().redirect_request(None, None, 302, "redirect", {}, "http://other") is None
    )


def test_module_preview_is_safe_and_prints_valid_json(remote, tmp_path):
    output = tmp_path / "preview"
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "minigpt_cn.data.download",
            "--source",
            "zhwiki",
            "--sources-config",
            str(remote["path"]),
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert json.loads(result.stdout)["status"] == "planned" and not output.exists()
    assert "RuntimeWarning" not in result.stderr

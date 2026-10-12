"""用百字节响应测试断线续传，不连接公网，也不改用户的真实.part。"""

import bz2
import hashlib
import io
import json
from pathlib import Path

import pytest
import yaml

from minigpt_cn.data import download as cli


class Response(io.BytesIO):
    def __init__(self, body, *, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


@pytest.fixture
def server(tmp_path, monkeypatch):
    body = bz2.compress("<mediawiki>断线后不丢前缀，最终核对完整文件</mediawiki>".encode())
    config = yaml.safe_load(cli.DEFAULT_CONFIG.read_bytes())
    for info in config["sources"].values():
        info.update(bytes=len(body), sha1=hashlib.sha1(body).hexdigest())
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    info = config["sources"]["zhwiki"]
    status = json.dumps(
        {
            "jobs": {
                info["job"]: {
                    "status": "done",
                    "files": {
                        info["filename"]: {
                            "size": len(body),
                            "sha1": info["sha1"],
                            "url": info["url"].removeprefix("https://dumps.wikimedia.org"),
                        }
                    },
                }
            }
        }
    ).encode()
    state = {
        "body": body,
        "path": path,
        "info": info,
        "status": status,
        "offsets": [],
        "respond": None,
        "sleeps": [],
    }

    def open_response(url, timeout, *, offset=0):
        if url == info["status_url"]:
            return Response(status)
        assert url == info["url"]
        state["offsets"].append(offset)
        if state["respond"]:
            return state["respond"](offset)
        headers = {"Content-Length": str(len(body) - offset)}
        if offset:
            headers["Content-Range"] = f"bytes {offset}-{len(body) - 1}/{len(body)}"
        return Response(body[offset:], status=206 if offset else 200, headers=headers)

    monkeypatch.setattr(cli, "_open", open_response)
    monkeypatch.setattr(cli, "sleep", state["sleeps"].append)
    return state


def unfinished(server, tmp_path, size=17):
    output = tmp_path / "download"
    output.mkdir()
    (output / "source_config.yaml").write_bytes(server["path"].read_bytes())
    (output / "dumpstatus.json").write_bytes(server["status"])
    partial = output / (server["info"]["filename"] + ".part")
    partial.write_bytes(server["body"][:size])
    return output, partial


def resume(server, output, **options):
    return cli.download_dump(
        server["path"], "zhwiki", output, download=True, resume=True, **options
    )


@pytest.mark.parametrize("size", [0, 17, "complete"])
def test_resume_preserves_prefix_snapshots_and_verifies_whole_file(server, tmp_path, size):
    size = len(server["body"]) if size == "complete" else size
    output, partial = unfinished(server, tmp_path, size)
    snapshots = {
        name: (output / name).read_bytes() for name in ("source_config.yaml", "dumpstatus.json")
    }
    report = resume(server, output)
    assert server["offsets"] == ([] if size == len(server["body"]) else [size])
    assert report["transfer"]["resume_bytes"] == size
    assert report["status"] == "verified" and not report["ready_for_training"]
    assert (output / report["file"]["path"]).read_bytes() == server["body"]
    assert report["file"]["sha256"] == hashlib.sha256(server["body"]).hexdigest()
    assert not partial.exists() and not (output / ".download.lock").exists()
    assert json.loads((output / "download_manifest.json").read_text(encoding="utf-8")) == report
    for name, raw in snapshots.items():
        assert (output / name).read_bytes() == raw


@pytest.mark.parametrize(
    "problem", ["ignored", "missing", "start", "end", "total", "length", "encoding", "overlong"]
)
def test_invalid_range_never_appends_or_retries(server, tmp_path, problem):
    output, partial = unfinished(server, tmp_path)
    before = partial.read_bytes()
    total, offset = len(server["body"]), len(before)
    headers = {"Content-Range": f"bytes {offset}-{total - 1}/{total}"}
    status, body = 206, server["body"][offset:]
    if problem == "ignored":
        status, body = 200, server["body"]
    elif problem == "missing":
        headers = {}
    elif problem == "start":
        headers["Content-Range"] = f"bytes 0-{total - 1}/{total}"
    elif problem == "end":
        headers["Content-Range"] = f"bytes {offset}-{total - 2}/{total}"
    elif problem == "total":
        headers["Content-Range"] = f"bytes {offset}-{total - 1}/{total + 1}"
    elif problem == "length":
        headers["Content-Length"] = str(total)
    elif problem == "encoding":
        headers["Content-Encoding"] = "gzip"
    else:
        body += b"extra"
    server["respond"] = lambda offset: Response(body, status=status, headers=headers)
    with pytest.raises(ValueError):
        resume(server, output)
    assert partial.read_bytes() == before and server["sleeps"] == []
    assert not (output / "download_manifest.json").exists()


@pytest.mark.parametrize("failure", ["eof", "timeout", "incomplete_read", "connect"])
def test_retry_uses_actual_written_offset_then_completes(server, tmp_path, failure):
    output, partial = unfinished(server, tmp_path)
    offset, total = partial.stat().st_size, len(server["body"])

    class InterruptedResponse(Response):
        def read(self, size=-1):
            if self.tell() == 0:
                return super().read(5)
            if failure == "timeout":
                raise TimeoutError("模拟读取超时")
            if failure == "incomplete_read":
                raise cli.IncompleteRead(b"not-written", 100)
            return b""

    def respond(start):
        if len(server["offsets"]) == 1:
            if failure == "connect":
                raise cli._InterruptedDownload("模拟连接失败")
            return InterruptedResponse(
                server["body"][start:],
                status=206,
                headers={"Content-Range": f"bytes {start}-{total - 1}/{total}"},
            )
        return Response(
            server["body"][start:],
            status=206,
            headers={"Content-Range": f"bytes {start}-{total - 1}/{total}"},
        )

    server["respond"] = respond
    report = resume(server, output, retries=1)
    assert server["offsets"] == [offset, offset + (0 if failure == "connect" else 5)]
    assert server["sleeps"] == [1]
    assert (output / report["file"]["path"]).read_bytes() == server["body"]


def test_new_download_also_retries_without_repeating_the_prefix(server, tmp_path):
    total = len(server["body"])

    def respond(offset):
        if not offset:
            return Response(server["body"][:17])
        return Response(
            server["body"][offset:],
            status=206,
            headers={"Content-Range": f"bytes {offset}-{total - 1}/{total}"},
        )

    server["respond"] = respond
    report = cli.download_dump(server["path"], "zhwiki", tmp_path / "new", download=True)
    assert server["offsets"] == [0, 17]
    assert (Path(report["output_dir"]) / report["file"]["path"]).read_bytes() == server["body"]


def test_retries_are_bounded_and_leave_resumable_partial(server, tmp_path):
    output, partial = unfinished(server, tmp_path)
    total = len(server["body"])
    server["respond"] = lambda start: Response(
        server["body"][start : start + 1],
        status=206,
        headers={"Content-Range": f"bytes {start}-{total - 1}/{total}"},
    )
    with pytest.raises(ConnectionError, match="已用完2次重试"):
        resume(server, output, retries=2)
    assert server["offsets"] == [17, 18, 19] and server["sleeps"] == [1, 2]
    assert partial.read_bytes() == server["body"][:20]
    assert not (output / ".download.lock").exists()
    assert not (output / "download_manifest.json").exists()


@pytest.mark.parametrize(
    "problem", ["config", "status", "source", "oversized", "final", "manifest", "lock", "missing"]
)
def test_wrong_or_completed_directory_is_not_modified(server, tmp_path, problem):
    output, partial = unfinished(server, tmp_path)
    if problem == "config":
        (output / "source_config.yaml").write_bytes(b"wrong config")
    elif problem == "status":
        (output / "dumpstatus.json").write_bytes(b"{}")
    elif problem == "source":
        partial.rename(output / "other-source.part")
    elif problem == "oversized":
        partial.write_bytes(server["body"] + b"x")
    elif problem in ("final", "manifest", "lock"):
        name = {
            "final": server["info"]["filename"],
            "manifest": "download_manifest.json",
            "lock": ".download.lock",
        }[problem]
        (output / name).write_bytes(b"do not overwrite")
    else:
        (output / "dumpstatus.json").unlink()
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError):
        resume(server, output)
    assert server["offsets"] == []
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_bad_existing_prefix_fails_final_hash_without_publication(server, tmp_path):
    output, partial = unfinished(server, tmp_path)
    partial.write_bytes(b"X" + partial.read_bytes()[1:])
    with pytest.raises(ValueError, match="SHA-1"):
        resume(server, output)
    assert partial.read_bytes()[0:1] == b"X"
    assert not (output / "download_manifest.json").exists()


def test_resume_disk_precheck_only_requires_remaining_bytes(server, tmp_path, monkeypatch):
    output, partial = unfinished(server, tmp_path)
    needed = len(server["body"]) - partial.stat().st_size + cli.RESERVE_BYTES
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda path: type("Disk", (), {"free": needed})())
    assert resume(server, output)["status"] == "verified"


@pytest.mark.parametrize("retries", [-1, 21, True, 1.5])
def test_invalid_retry_count_creates_nothing(server, tmp_path, retries):
    with pytest.raises(ValueError, match="retries"):
        cli.download_dump(server["path"], "zhwiki", tmp_path / "new", retries=retries)
    assert not (tmp_path / "new").exists()


def test_resume_requires_explicit_download(server, tmp_path):
    with pytest.raises(ValueError, match="一起使用"):
        cli.download_dump(server["path"], "zhwiki", tmp_path / "new", resume=True)


def test_range_header_is_sent_without_weakening_tls_or_redirect_rules(monkeypatch):
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return Response(b"x")

    monkeypatch.setattr(cli, "build_opener", lambda handler: Opener())
    cli._open("https://example.invalid/file", 30, offset=17)
    assert requests[0].get_header("Range") == "bytes=17-"
    assert requests[0].get_header("Accept-encoding") == "identity"


@pytest.mark.parametrize("code", [403, 404, 416, 429, 503])
def test_only_transient_http_errors_are_retryable(monkeypatch, code):
    class Opener:
        def open(self, request, timeout):
            raise cli.HTTPError(request.full_url, code, "test", {}, None)

    monkeypatch.setattr(cli, "build_opener", lambda handler: Opener())
    expected = cli._InterruptedDownload if code in (429, 503) else cli.HTTPError
    with pytest.raises(expected):
        cli._open("https://example.invalid/file", 30)


def test_certificate_error_is_not_retried(monkeypatch):
    class Opener:
        def open(self, request, timeout):
            raise cli.URLError(cli.ssl.SSLCertVerificationError("test certificate"))

    monkeypatch.setattr(cli, "build_opener", lambda handler: Opener())
    with pytest.raises(cli.URLError):
        cli._open("https://example.invalid/file", 30)


def test_snapshot_change_during_resume_blocks_publication(server, tmp_path, monkeypatch):
    output, partial = unfinished(server, tmp_path)
    original = cli._download_file

    def mutate(info, partial, timeout, **options):
        result = original(info, partial, timeout, **options)
        (output / "dumpstatus.json").write_bytes(b"changed")
        return result

    monkeypatch.setattr(cli, "_download_file", mutate)
    with pytest.raises(RuntimeError, match="快照发生变化"):
        resume(server, output)
    assert partial.exists() and not (output / "download_manifest.json").exists()


def test_keyboard_interrupt_retains_prefix_and_releases_lock(server, tmp_path):
    output, partial = unfinished(server, tmp_path)

    def interrupt(offset):
        raise KeyboardInterrupt

    server["respond"] = interrupt
    with pytest.raises(KeyboardInterrupt):
        resume(server, output)
    assert partial.read_bytes() == server["body"][:17]
    assert not (output / ".download.lock").exists()

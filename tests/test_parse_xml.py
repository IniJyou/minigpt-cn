"""本地百字节/小型bz2验证XML事件解析；不下载或解压整份正式dump。"""

import bz2
import hashlib
import json
import subprocess
import sys
from html import escape

import pytest
import yaml

from minigpt_cn.data import parse_xml as cli
from minigpt_cn.data.download import DEFAULT_CONFIG

URI = "http://www.mediawiki.org/xml/export-0.11/"


def page(
    title="測試/第一章",
    text="原始{{模板}} [[链接]] & 😀",
    *,
    namespace=0,
    redirect=False,
    model="wikitext",
    text_flags="",
    revision=True,
):
    revision_xml = ""
    if revision:
        revision_xml = (
            "<revision><id>11</id><timestamp>2026-10-01T00:00:00Z</timestamp>"
            "<contributor><id>999</id></contributor>"
            f"<model>{model}</model><format>text/x-wiki</format>"
            f"<text {text_flags}>{escape(text)}</text>"
            "<content><role>other</role><text>不能误取辅助槽位</text></content></revision>"
        )
    return (
        f"<page><title>{escape(title)}</title><ns>{namespace}</ns><id>1</id>"
        + ('<redirect title="目标"/>' if redirect else "")
        + revision_xml
        + "</page>"
    )


def document(pages, source="zhwiki", *, uri=URI):
    return (
        f'<mediawiki xmlns="{uri}" version="0.11" xml:lang="zh">'
        f"<siteinfo><dbname>{source}</dbname><sitename>测试站点</sitename></siteinfo>"
        + "".join(pages)
        + "</mediawiki>"
    ).encode()


def make_download(tmp_path, xml, source="zhwiki", *, compressed=None):
    directory = tmp_path / "download"
    directory.mkdir()
    body = bz2.compress(xml) if compressed is None else compressed
    config = yaml.safe_load(DEFAULT_CONFIG.read_bytes())
    info = config["sources"][source]
    info.update(bytes=len(body), sha1=hashlib.sha1(body).hexdigest())
    raw_config = yaml.safe_dump(config, allow_unicode=True).encode()
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
                        },
                    },
                }
            }
        }
    ).encode()
    (directory / "source_config.yaml").write_bytes(raw_config)
    (directory / "dumpstatus.json").write_bytes(status)
    dump = directory / info["filename"]
    dump.write_bytes(body)
    official = cli._validate_status(info, status)
    manifest = {
        "format_version": 1,
        "kind": "wikimedia_dump_download",
        "status": "verified",
        "source": {"name": source, **info},
        "ready_for_training": False,
        "source_config": {
            "path": "source_config.yaml",
            "sha256": hashlib.sha256(raw_config).hexdigest(),
        },
        "official_status": {**official, "path": "dumpstatus.json"},
        "file": {
            "path": info["filename"],
            "bytes": len(body),
            "sha1": info["sha1"],
            "sha256": hashlib.sha256(body).hexdigest(),
        },
    }
    path = directory / "download_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return path, dump


def records(output):
    return [
        json.loads(line)
        for line in (output / "wikitext.jsonl").read_text(encoding="utf-8").splitlines()
    ]


@pytest.mark.parametrize("source", ["zhwiki", "zhwikisource"])
def test_preserves_wikitext_direct_ids_urls_licenses_and_input_files(tmp_path, source):
    text = "  繁體 {{模板}}\n[[数学|數學]] & <ref>參考</ref> 😀 e\u0301  "
    path, dump = make_download(tmp_path, document([page(text=text)], source), source)
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    output = tmp_path / "sample"
    report = cli.sample_xml(path, output)
    record = records(output)[0]
    assert record["wikitext"] == text and "text" not in record
    assert record["page_id"] == 1 and record["revision_id"] == 11
    assert record["timestamp"] == "2026-10-01T00:00:00Z"
    assert record["source"] == {"name": source, "dump_date": "20261001"}
    assert record["license"]["review_status"] == "per_document_review_required"
    assert (
        record["url"] == cli.HOSTS[source] + "/wiki/%E6%B8%AC%E8%A9%A6/%E7%AC%AC%E4%B8%80%E7%AB%A0"
    )
    assert record["revision_url"].endswith("?oldid=11")
    assert record["history_url"].endswith("?curid=1&action=history")
    assert record["wikitext_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert report["scope"] == "prefix_sample" and not report["ready_for_training"]
    assert report["counts"]["xml_document_complete"]
    assert report["file"]["sha256"] == cli._sha256(output / "wikitext.jsonl")
    assert report["file"]["bytes"] == (output / "wikitext.jsonl").stat().st_size
    assert json.loads((output / "sample_manifest.json").read_text(encoding="utf-8")) == report
    assert {p.name: p.read_bytes() for p in path.parent.iterdir()} == before


def test_counts_all_scan_pages_and_each_filter_reason(tmp_path):
    missing = page().replace("<text >原始{{模板}} [[链接]] &amp; 😀</text>", "")
    xml = document(
        [
            page(),
            page(namespace=10),
            page(redirect=True),
            page(revision=False),
            page(model="json"),
            missing,
            page(text_flags='deleted="deleted"'),
            page(text_flags='location="https://example.invalid/no-download"'),
            page(text=" \n"),
        ]
    )
    path, _ = make_download(tmp_path, xml)
    report = cli.sample_xml(path, tmp_path / "sample")
    counts = report["counts"]
    assert counts["pages_scanned"] == 9 and counts["saved"] == 1
    assert set(counts["skipped"].values()) == {1}


def test_page_limit_is_scanned_pages_not_saved_documents(tmp_path):
    path, _ = make_download(tmp_path, document([page(namespace=4), page(), page(title="后续页")]))
    output = tmp_path / "sample"
    report = cli.sample_xml(path, output, max_pages=2)
    assert report["counts"]["pages_scanned"] == 2 and report["counts"]["saved"] == 1
    assert report["counts"]["stop_reason"] == "page_limit"
    assert not report["counts"]["xml_document_complete"]
    assert len(records(output)) == 1


def test_many_pages_release_root_children_and_keep_byte_budget(tmp_path):
    path, _ = make_download(tmp_path, document([page(title=f"页{n}") for n in range(2000)]))
    report = cli.sample_xml(path, tmp_path / "sample", max_pages=1500)
    counts = report["counts"]
    assert counts["saved"] == counts["pages_scanned"] == 1500
    assert counts["peak_root_children"] < 100  # 只残留固定读取块预读的少量页，不累积1500个空节点。
    assert counts["xml_bytes_read"] < cli.DEFAULT_XML_BYTES


@pytest.mark.parametrize(
    "problem",
    [
        "version",
        "namespace",
        "site",
        "missing_site",
        "id",
        "revision_id",
        "timestamp",
        "history",
        "nested_text",
        "truncated",
    ],
)
def test_bad_structure_has_no_success_manifest(tmp_path, problem):
    xml = document([page()])
    if problem == "version":
        xml = xml.replace(b'version="0.11"', b'version="0.10"')
    elif problem == "namespace":
        xml = xml.replace(URI.encode(), b"https://example.invalid/xml/")
    elif problem == "site":
        xml = xml.replace(b"<dbname>zhwiki</dbname>", b"<dbname>zhwikisource</dbname>")
    elif problem == "missing_site":
        start, end = xml.index(b"<siteinfo>"), xml.index(b"</siteinfo>") + len(b"</siteinfo>")
        xml = xml[:start] + xml[end:]
    elif problem == "id":
        xml = xml.replace(b"<id>1</id>", b"<id>0</id>")
    elif problem == "revision_id":
        xml = xml.replace(b"<id>11</id>", b"<id>0</id>")
    elif problem == "timestamp":
        xml = xml.replace(b"<timestamp>2026-10-01T00:00:00Z</timestamp>", b"")
    elif problem == "history":
        revision = xml[xml.index(b"<revision>") : xml.index(b"</revision>") + len(b"</revision>")]
        xml = xml.replace(b"</page>", revision + b"</page>")
    elif problem == "nested_text":
        xml = xml.replace(b"<text >", b"<text ><nested/>")
    else:
        xml = xml[:-10]
    path, _ = make_download(tmp_path, xml)
    output = tmp_path / "sample"
    with pytest.raises(ValueError):
        cli.sample_xml(path, output)
    assert not (output / "sample_manifest.json").exists()


@pytest.mark.parametrize("kind", ["doctype", "entity", "utf16"])
def test_rejects_custom_entities_and_unsupported_encoding(tmp_path, kind):
    xml = document([page()])
    if kind == "utf16":
        xml = '<?xml version="1.0" encoding="UTF-16"?>'.encode("utf-16") + xml.decode().encode(
            "utf-16-le"
        )
    else:
        declaration = (
            b"<!DOCTYPE mediawiki>" if kind == "doctype" else b'<!ENTITY custom "expanded">'
        )
        xml = declaration + xml
    path, _ = make_download(tmp_path, xml)
    with pytest.raises(ValueError):
        cli.sample_xml(path, tmp_path / "sample")


def test_declaration_split_across_read_blocks_is_rejected():
    import io

    reader = cli._BoundedXMLReader(io.BytesIO(b"<!DOCTYPE mediawiki>"), 100)
    reader.read(4)
    with pytest.raises(ValueError, match="DTD"):
        reader.read(5)


@pytest.mark.parametrize("limit", [1, 80])
def test_decompressed_budget_is_not_compressed_file_size(tmp_path, limit):
    path, _ = make_download(tmp_path, document([page(text="正文" * 10000)]))
    output = tmp_path / "sample"
    with pytest.raises(ValueError, match="字节预算耗尽"):
        cli.sample_xml(path, output, max_xml_bytes=limit)
    assert not (output / "sample_manifest.json").exists()


def test_exact_decompressed_budget_at_eof_is_valid(tmp_path):
    xml = document([page()])
    path, _ = make_download(tmp_path, xml)
    report = cli.sample_xml(path, tmp_path / "sample", max_xml_bytes=len(xml))
    assert report["counts"]["xml_bytes_read"] == len(xml)


@pytest.mark.parametrize(
    "problem", ["missing", "not_verified", "snapshot", "path", "checksum", "bad_bz2"]
)
def test_invalid_download_or_compression_is_rejected(tmp_path, problem):
    path, dump = make_download(
        tmp_path, document([page()]), compressed=b"not bzip2" if problem == "bad_bz2" else None
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if problem == "missing":
        path.unlink()
    elif problem == "not_verified":
        manifest["status"] = "planned"
    elif problem == "snapshot":
        (path.parent / "source_config.yaml").write_bytes(b"wrong")
    elif problem == "path":
        manifest["file"]["path"] = "../escape.bz2"
    elif problem == "checksum":
        dump.write_bytes(b"X" + dump.read_bytes()[1:])
    if path.exists():
        path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "sample"
    with pytest.raises((OSError, EOFError, ValueError)):
        cli.sample_xml(path, output)
    assert not (output / "sample_manifest.json").exists()


def test_empty_eligible_sample_is_not_reported_as_success(tmp_path):
    path, _ = make_download(tmp_path, document([page(namespace=4)]))
    with pytest.raises(ValueError, match="没有可保存"):
        cli.sample_xml(path, tmp_path / "sample")


@pytest.mark.parametrize(
    "option,value",
    [
        ("max_pages", 0),
        ("max_pages", 10001),
        ("max_pages", True),
        ("max_xml_bytes", 0),
        ("max_xml_bytes", True),
    ],
)
def test_invalid_options_fail_before_reading_or_creating(tmp_path, option, value):
    output = tmp_path / "sample"
    with pytest.raises(ValueError):
        cli.sample_xml(tmp_path / "not-needed.json", output, **{option: value})
    assert not output.exists()


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "old"
    output.mkdir()
    (output / "marker").write_bytes(b"old data")
    with pytest.raises(FileExistsError):
        cli.sample_xml(tmp_path / "not-needed.json", output)
    assert (output / "marker").read_bytes() == b"old data"


def test_input_change_during_sampling_blocks_manifest(tmp_path, monkeypatch):
    path, dump = make_download(tmp_path, document([page()]))
    original = cli._iter_pages

    def mutate(*args):
        yield from original(*args)
        dump.write_bytes(dump.read_bytes() + b"changed")

    monkeypatch.setattr(cli, "_iter_pages", mutate)
    output = tmp_path / "sample"
    with pytest.raises(ValueError):
        cli.sample_xml(path, output)
    assert not (output / "sample_manifest.json").exists()


def test_cli_is_offline_and_prints_manifest(tmp_path):
    path, _ = make_download(tmp_path, document([page()]))
    output = tmp_path / "sample"
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "minigpt_cn.data.parse_xml",
            "--download-manifest",
            str(path),
            "--output-dir",
            str(output),
            "--max-pages",
            "10",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["counts"]["saved"] == 1 and not report["ready_for_training"]
    assert "RuntimeWarning" not in result.stderr

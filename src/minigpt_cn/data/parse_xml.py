"""本地小样本：流式读取已校验的MediaWiki XML，保留原始wikitext，不生成训练正文。"""

import argparse
import bz2
import hashlib
import json
import re
from pathlib import Path
from time import perf_counter
from urllib.parse import quote
from xml.etree import ElementTree as ET

from minigpt_cn.data.download import STATUS_LIMIT, _load_source, _validate_status
from minigpt_cn.data.prepare import _reject_nonfinite, _sha256

XML_CHUNK_BYTES = 16 * 1024
DEFAULT_XML_BYTES = 64 * 1024 * 1024
HOSTS = {"zhwiki": "https://zh.wikipedia.org", "zhwikisource": "https://zh.wikisource.org"}


def _small_file(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"需要普通文件，不能是符号链接：{path}")
    with path.open("rb") as file:
        raw = file.read(STATUS_LIMIT + 1)
    if len(raw) > STATUS_LIMIT:
        raise ValueError(f"元数据文件超过2MiB：{path}")
    return raw


def _dump_hashes(path: Path) -> tuple[int, str, str]:
    # 校验的是压缩字节流；最多1MiB一块读取，不把数GiB的dump读入RAM。
    sha1, sha256, size = hashlib.sha1(), hashlib.sha256(), 0
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            size += len(block)
            sha1.update(block)
            sha256.update(block)
    return size, sha1.hexdigest(), sha256.hexdigest()


def _verified_input(manifest_path: Path) -> tuple[dict, Path, dict]:
    raw = _small_file(manifest_path)
    manifest = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
    parent = manifest_path.parent
    try:
        if (
            type(manifest["format_version"]) is not int
            or manifest["format_version"] != 1
            or manifest["kind"] != "wikimedia_dump_download"
            or manifest["status"] != "verified"
        ):
            raise ValueError("需要版本1、status=verified的完整下载清单")
        source = manifest["source"]["name"]
        snapshots = {}
        for name, key in (
            ("source_config.yaml", "source_config"),
            ("dumpstatus.json", "official_status"),
        ):
            if manifest[key]["path"] != name:
                raise ValueError("快照路径必须是下载目录内的固定文件名")
            snapshots[name] = _small_file(parent / name)
            if hashlib.sha256(snapshots[name]).hexdigest() != manifest[key]["sha256"]:
                raise ValueError("下载快照的SHA-256与清单不一致")
        info, raw_config = _load_source(parent / "source_config.yaml", source)
        if raw_config != snapshots["source_config.yaml"]:
            raise RuntimeError("读取时配置快照发生变化")
        if manifest["source"] != {"name": source, **info}:
            raise ValueError("来源信息与固定配置不一致")
        status = _validate_status(info, snapshots["dumpstatus.json"])
        if any(manifest["official_status"][key] != value for key, value in status.items()):
            raise ValueError("官方状态快照与清单不一致")
        file_info = manifest["file"]
        if file_info["path"] != info["filename"]:
            raise ValueError("dump路径必须是固定文件名，不能跳出下载目录")
        dump = parent / info["filename"]
        if dump.is_symlink() or not dump.is_file():
            raise ValueError("dump必须是普通文件，不能是符号链接")
        if (
            type(file_info["bytes"]) is not int
            or file_info["bytes"] != info["bytes"]
            or file_info["sha1"] != info["sha1"]
            or dump.stat().st_size != info["bytes"]
        ):
            raise ValueError("dump大小/SHA-1信息与固定配置不一致")
        if _dump_hashes(dump) != (info["bytes"], info["sha1"], file_info["sha256"]):
            raise ValueError("本地dump的大小/SHA-1/SHA-256校验失败")
        input_info = {
            "download_manifest": str(manifest_path),
            "download_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "dump_path": str(dump),
            **file_info,
            "snapshots": {
                name: hashlib.sha256(value).hexdigest() for name, value in snapshots.items()
            },
        }
        return manifest["source"], dump, input_info
    except (KeyError, TypeError) as error:
        raise ValueError("下载清单缺少字段或结构错误") from error


class _BoundedXMLReader:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.count, self.tail = stream, limit, 0, b""

    def read(self, size):
        remaining = self.limit - self.count
        # 上限处只多探测1字节：恰好EOF可结束，否则拒绝继续解压，不写成功清单。
        block = self.stream.read(min(size, XML_CHUNK_BYTES, remaining) if remaining else 1)
        if len(block) > remaining:
            raise ValueError("解压XML字节预算耗尽；请另行评估后调整 --max-xml-bytes")
        probe = self.tail + block
        if b"\x00" in block or b"<!DOCTYPE" in probe or b"<!ENTITY" in probe:
            raise ValueError("仅接受无DTD/ENTITY声明的UTF-8 XML，不展开自定义实体")
        self.tail = probe[-16:]  # 声明跨读取块时也不能绕过检查。
        self.count += len(block)
        return block


def _required(element, tag):
    matches = element.findall(tag)
    if len(matches) != 1 or not (matches[0].text or "").strip():
        raise ValueError(f"缺失或重复的非空字段：{tag.rsplit('}', 1)[-1]}")
    return matches[0].text


def _page_record(page, ns):
    title = _required(page, ns + "title")
    namespace, page_id = int(_required(page, ns + "ns")), int(_required(page, ns + "id"))
    if namespace < 0 or page_id <= 0:
        raise ValueError("namespace必须非负，page_id必须为正整数")
    record = {"title": title, "namespace": namespace, "page_id": page_id}
    if namespace != 0:
        return "namespace", record
    if page.find(ns + "redirect") is not None:
        return "redirect", record
    revisions = page.findall(ns + "revision")
    if len(revisions) > 1:
        raise ValueError("仅支持每页一个当前revision，不接受全历史dump")
    if not revisions:
        return "missing_revision", record
    revision = revisions[0]
    revision_id = int(_required(revision, ns + "id"))
    if revision_id <= 0:
        raise ValueError("revision_id必须为正整数")
    record.update(revision_id=revision_id, timestamp=_required(revision, ns + "timestamp"))
    if (
        revision.findtext(ns + "model") != "wikitext"
        or revision.findtext(ns + "format") != "text/x-wiki"
    ):
        return "non_wikitext", record
    # XML 0.11的主槽位在revision/text；content/text可能是辅助槽位，不能用.//text取第一个。
    texts = revision.findall(ns + "text")
    if len(texts) > 1:
        raise ValueError("主槽位text重复")
    if not texts:
        return "missing_text", record
    text = texts[0]
    if "deleted" in text.attrib:
        return "deleted", record
    if "location" in text.attrib:
        return "external_text", record  # 不跟随外部内容URL，不发起下载。
    if len(text):
        raise ValueError("text必须是XML转义字符串，不能含嵌套XML元素")
    wikitext = text.text or ""
    if not wikitext.strip():
        return "empty", record
    record.update(
        wikitext=wikitext, wikitext_sha256=hashlib.sha256(wikitext.encode("utf-8")).hexdigest()
    )
    return None, record


def _iter_pages(stream, source, max_pages, max_xml_bytes, stats):
    reader = _BoundedXMLReader(stream, max_xml_bytes)
    root, ns, stack, siteinfo_seen = None, None, [], False
    try:
        for event, element in ET.iterparse(reader, events=("start", "end")):
            if event == "start":
                if root is None:
                    match = re.fullmatch(
                        r"\{(https?://www.mediawiki.org/xml/export-0\.11/)\}mediawiki", element.tag
                    )
                    if (
                        match is None
                        or element.get("version") != "0.11"
                        or element.get("{http://www.w3.org/XML/1998/namespace}lang") != "zh"
                    ):
                        raise ValueError("仅支持中文MediaWiki export-0.11根节点")
                    root, ns = element, "{" + match[1] + "}"
                stack.append(element)
                continue
            if len(stack) == 2:
                if element.tag == ns + "siteinfo":
                    if siteinfo_seen or stats["pages_scanned"]:
                        raise ValueError("siteinfo必须在page之前且仅出现一次")
                    if _required(element, ns + "dbname") != source:
                        raise ValueError("siteinfo/dbname与下载来源不一致")
                    stats["siteinfo"] = {
                        key: element.findtext(ns + key)
                        for key in ("dbname", "sitename", "base", "generator")
                    }
                    siteinfo_seen = True
                elif element.tag == ns + "page":
                    if not siteinfo_seen:
                        raise ValueError("page之前必须提供匹配来源的siteinfo")
                    stats["pages_scanned"] += 1
                    reason, record = _page_record(element, ns)
                    stats["peak_root_children"] = max(stats["peak_root_children"], len(root))
                    # clear只清正文却仍留下空page节点；remove还释放根节点的引用，避免随篇数累积。
                    # 内存约O(最大单页XML + 固定读取缓冲)，不是O(整个解压dump)。这里不是模型张量。
                    element.clear()
                    root.remove(element)
                    stack.pop()
                    stats["xml_bytes_read"] = reader.count
                    yield reason, record
                    if max_pages is not None and stats["pages_scanned"] >= max_pages:
                        stats["stop_reason"] = "page_limit"
                        return  # 小样本主动停在页边界；不声称检查了后面全部XML。
                    continue
                else:
                    raise ValueError("不支持的mediawiki直接子节点")
                element.clear()
                root.remove(element)
            stack.pop()
        if not siteinfo_seen:
            raise ValueError("缺少siteinfo")
        stats.update(
            stop_reason="end_of_xml", xml_document_complete=True, xml_bytes_read=reader.count
        )
    except ET.ParseError as error:
        raise ValueError(f"XML格式错误（已扫描{stats['pages_scanned']}页）：{error}") from error


def sample_xml(download_manifest, output_dir, *, max_pages=100, max_xml_bytes=DEFAULT_XML_BYTES):
    """仅生成有界原始wikitext样本；不清除标记、不繁转简、不划分、不训练。"""
    if type(max_pages) is not int or not 1 <= max_pages <= 10000:
        raise ValueError("max_pages必须为1～10000的整数；本入口仅用于小样本")
    if type(max_xml_bytes) is not int or max_xml_bytes < 1:
        raise ValueError("max_xml_bytes必须为正整数")
    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("--output-dir已存在，请换新目录；不覆盖旧样本")
    manifest_path = Path(download_manifest)
    if manifest_path.is_symlink():
        raise ValueError("下载清单不能是符号链接")
    started = perf_counter()
    source, dump, input_info = _verified_input(manifest_path.resolve())
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stats = {
        "pages_scanned": 0,
        "saved": 0,
        "xml_bytes_read": 0,
        "xml_document_complete": False,
        "peak_root_children": 0,
        "stop_reason": None,
        "siteinfo": None,
        "skipped": {
            key: 0
            for key in (
                "namespace",
                "redirect",
                "missing_revision",
                "non_wikitext",
                "missing_text",
                "deleted",
                "external_text",
                "empty",
            )
        },
    }
    target = output / "wikitext.jsonl"
    with bz2.open(dump, "rb") as stream, target.open("x", encoding="utf-8", newline="\n") as file:
        for reason, record in _iter_pages(stream, source["name"], max_pages, max_xml_bytes, stats):
            if reason:
                stats["skipped"][reason] += 1
                continue
            host = HOSTS[source["name"]]
            record.update(
                source={"name": source["name"], "dump_date": source["dump_date"]},
                license=source["license"],
                url=host + "/wiki/" + quote(record["title"].replace(" ", "_"), safe="/"),
                revision_url=f"{host}/w/index.php?oldid={record['revision_id']}",
                history_url=f"{host}/w/index.php?curid={record['page_id']}&action=history",
            )
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stats["saved"] += 1
    if stats["saved"] == 0:
        raise ValueError("样本没有可保存的主命名空间wikitext页面；请增加扫描页数")
    if stats["pages_scanned"] != stats["saved"] + sum(stats["skipped"].values()):
        raise RuntimeError("解析计数不平衡")
    # 完成后再次核对整份压缩输入和元数据；即使只解析前100页，也不能把变动的文件记成成功。
    _, final_dump, final_info = _verified_input(manifest_path.resolve())
    if final_dump != dump or final_info != input_info:
        raise RuntimeError("小样本解析期间下载文件/清单/快照发生变化")
    report = {
        "format_version": 1,
        "kind": "mediawiki_wikitext_sample",
        "scope": "prefix_sample",
        "ready_for_training": False,
        "input": input_info,
        "processing": {
            "algorithm": "mediawiki_export_0.11_main_wikitext_v1",
            "max_pages": max_pages,
            "max_xml_bytes": max_xml_bytes,
            "wiki_markup_cleaned": False,
            "to_simplified": False,
            "deduplication": None,
            "split": None,
        },
        "counts": stats,
        "elapsed_seconds": perf_counter() - started,
        "file": {"path": target.name, "bytes": target.stat().st_size, "sha256": _sha256(target)},
    }
    with (output / "sample_manifest.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="新样本目录，不覆盖")
    parser.add_argument("--max-pages", type=int, default=100, help="扫描页数上限，包含被过滤的页")
    parser.add_argument(
        "--max-xml-bytes", type=int, default=DEFAULT_XML_BYTES, help="解压字节预算，默认64MiB"
    )
    args = parser.parse_args(argv)
    try:
        report = sample_xml(
            args.download_manifest,
            args.output_dir,
            max_pages=args.max_pages,
            max_xml_bytes=args.max_xml_bytes,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return report
    except (OSError, EOFError, ValueError, TypeError, RuntimeError) as error:
        parser.error(f"{error}\n小样本未完成；缺少sample_manifest.json的输出不能视为成功。")


if __name__ == "__main__":
    main()

"""固定版本 Wikimedia dump 下载：默认只预览，显式 --download 才传输。"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import ssl
import sys
from datetime import datetime, timezone
from http.client import IncompleteRead
from pathlib import Path
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import yaml
from tqdm import tqdm

from minigpt_cn.data.prepare import _reject_nonfinite, _sha256

CHUNK_BYTES = 1024 * 1024
STATUS_LIMIT = 2 * CHUNK_BYTES
RESERVE_BYTES = 1024**3  # 仅给下载留1GiB余量，不是正文提取/训练所需空间的估算。
DEFAULT_CONFIG = Path("configs/wikimedia_20261001.yaml")


class _InterruptedDownload(ConnectionError):
    """仅网络中断可重试；格式、磁盘写入和校验失败不能自动重试。"""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # 不自动转向其他主机/版本；需要重新核对并固定下载地址。


def _open(url: str, timeout: float, *, offset: int = 0):
    request = Request(
        url,
        headers={
            "User-Agent": "MiniGPT-CN/0.1 (reproducible data download)",
            "Accept-Encoding": "identity",
        },
    )
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    # 保留 Python 默认 TLS 证书验证，不关闭 HTTPS 校验。
    try:
        return build_opener(_NoRedirect()).open(request, timeout=timeout)
    except HTTPError as error:
        if error.code not in (408, 429, 500, 502, 503, 504):
            raise  # 404、403、重定向、416等不靠重试掩盖。
        raise _InterruptedDownload(f"HTTP {error.code}") from error
    except (URLError, OSError) as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise  # 证书验证失败不能当普通网络抖动处理。
        raise _InterruptedDownload("网络连接中断") from error


def _load_source(path: Path, name: str) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    try:
        if type(config["format_version"]) is not int or config["format_version"] != 1:
            raise ValueError("仅支持版本1的下载配置")
        if name not in ("zhwiki", "zhwikisource"):
            raise ValueError("source 必须是 zhwiki 或 zhwikisource")
        info = config["sources"][name]
        date = info["dump_date"]
        if not isinstance(date, str) or len(date) != 8 or not date.isdigit():
            raise ValueError("dump_date 必须是有日期的 YYYYMMDD 字符串，不能是 latest")
        try:
            datetime.strptime(date, "%Y%m%d")
        except ValueError as error:
            raise ValueError("dump_date 必须是真实的 YYYYMMDD 公历日期") from error
        filename = f"{name}-{date}-pages-articles.xml.bz2"
        base = f"https://dumps.wikimedia.org/{name}/{date}/"
        if (
            info["filename"] != filename
            or info["url"] != base + filename
            or info["status_url"] != base + "dumpstatus.json"
        ):
            raise ValueError("文件名/URL 必须指向同一固定版本的官方 HTTPS pages-articles dump")
        if type(info["bytes"]) is not int or info["bytes"] <= 0:
            raise ValueError("bytes 必须是正整数")
        sha1 = info["sha1"]
        if (
            not isinstance(sha1, str)
            or len(sha1) != 40
            or any(c not in "0123456789abcdef" for c in sha1)
        ):
            raise ValueError("sha1 必须是官方40位小写十六进制摘要")
        if not isinstance(info["job"], str) or not info["job"]:
            raise ValueError("job 必须是官方任务名")
        license_info = info["license"]
        if (
            not isinstance(license_info["summary"], str)
            or not license_info["summary"]
            or license_info["review_status"] != "per_document_review_required"
            or not isinstance(license_info["references"], list)
            or not license_info["references"]
            or any(
                not isinstance(s, str) or not s.startswith("https://")
                for s in license_info["references"]
            )
        ):
            raise ValueError("必须记录许可依据，并保留逐篇许可待核对状态")
    except (KeyError, TypeError) as error:
        raise ValueError("下载配置缺失字段或结构错误") from error
    return info, raw


def _check_remote(info: dict, timeout: float) -> tuple[dict, bytes]:
    with _open(info["status_url"], timeout) as response:
        if response.status != 200:
            raise ValueError("官方状态文件必须返回 HTTP 200")
        raw = response.read(STATUS_LIMIT + 1)
    if len(raw) > STATUS_LIMIT:
        raise ValueError("官方状态文件超过2MiB，拒绝作为有效元数据")
    return _validate_status(info, raw), raw


def _validate_status(info: dict, raw: bytes) -> dict:
    """同一规则核对联网状态和旧目录快照，不以文件名相同代替版本一致。"""
    if len(raw) > STATUS_LIMIT:
        raise ValueError("官方状态文件超过2MiB，拒绝作为有效元数据")
    status = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
    try:
        job = status["jobs"][info["job"]]
        if job["status"] != "done":
            raise ValueError("官方 dump 任务尚未 done，不开始下载")
        file = job["files"][info["filename"]]
        if (
            type(file["size"]) is not int
            or file["size"] != info["bytes"]
            or file["sha1"] != info["sha1"]
            or "https://dumps.wikimedia.org" + file["url"] != info["url"]
        ):
            raise ValueError("官方元数据与固定配置不一致；先复核版本，不能自动更新配置")
        return {
            "job": info["job"],
            "status": "done",
            "updated": job.get("updated"),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    except (KeyError, TypeError) as error:
        raise ValueError("官方状态文件缺失字段或结构错误") from error


def _check_response(response, offset: int, total: int) -> None:
    if offset:
        # Range从第offset字节开始（0起始）；206必须对应[offset, total-1]，含两个端点。
        # 若服务器忽略Range返回200，直接停止，绝不把完整文件追加到旧前缀后面。
        if response.status != 206:
            raise ValueError("续传必须返回 HTTP 206；拒绝追加从头开始的响应")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if match is None or tuple(map(int, match.groups())) != (offset, total - 1, total):
            raise ValueError("Content-Range 的起点/终点/总大小与续传位置不一致")
    elif response.status != 200 or response.headers.get("Content-Range") is not None:
        raise ValueError("首次下载必须返回 HTTP 200，不能包含部分响应")
    length = response.headers.get("Content-Length")
    if length is not None and int(length) != total - offset:
        raise ValueError("HTTP Content-Length 与剩余大小不一致")
    if response.headers.get("Content-Encoding", "identity") != "identity":
        raise ValueError("下载需要原始压缩文件字节，不能额外进行 HTTP 内容编码")


def _download_file(
    info: dict, partial: Path, timeout: float, *, resume: bool = False, retries: int = 3
) -> tuple[int, str, str]:
    with partial.open("r+b" if resume else "xb") as file:
        received = file.seek(0, os.SEEK_END)  # 只在末尾追加，不截断已经下载的前缀。
        if received > info["bytes"]:
            raise ValueError("残留文件超过固定大小，不能续传")
        with tqdm(total=info["bytes"], initial=received, unit="B", unit_scale=True) as bar:
            attempts = 0
            while received < info["bytes"]:
                try:
                    with _open(info["url"], timeout, offset=received) as response:
                        _check_response(response, received, info["bytes"])
                        while True:
                            try:
                                block = response.read(
                                    min(CHUNK_BYTES, info["bytes"] - received + 1)
                                )
                            except (OSError, IncompleteRead) as error:
                                raise _InterruptedDownload("读取响应时连接中断") from error
                            if not block:
                                break
                            if received + len(block) > info["bytes"]:
                                raise ValueError("下载字节数超过固定大小，拒绝继续写入")
                            file.write(block)  # 磁盘错误不在网络重试的捕获范围内。
                            received += len(block)
                            bar.update(len(block))
                    if received != info["bytes"]:
                        raise _InterruptedDownload("响应流提前结束")
                except _InterruptedDownload as error:
                    file.flush()
                    if attempts >= retries:
                        raise _InterruptedDownload(
                            f"下载不完整：已收到{received}/{info['bytes']}字节；"
                            f"已用完{retries}次重试，可用 --download --resume 接着下载"
                        ) from error
                    attempts += 1
                    delay = min(2 ** (attempts - 1), 8)
                    tqdm.write(
                        f"{error}；{delay}秒后从{received}字节续传（重试{attempts}/{retries}）",
                        file=sys.stderr,
                    )
                    sleep(delay)
    # 从关闭后的本地文件重算摘要，而不是只对网络流计算；SHA-1核对官方，SHA-256留作本地溯源。
    sha1, sha256 = hashlib.sha1(), hashlib.sha256()
    count = 0
    with partial.open("rb") as file:
        for block in iter(lambda: file.read(CHUNK_BYTES), b""):
            count += len(block)
            sha1.update(block)
            sha256.update(block)
    if count != info["bytes"] or sha1.hexdigest() != info["sha1"]:
        raise ValueError("本地文件大小或 SHA-1 校验失败，不写成功清单")
    return count, sha1.hexdigest(), sha256.hexdigest()


def _resume_state(output: Path, info: dict, raw_config: bytes) -> tuple[int, bytes]:
    """只接受本入口留下的未完成目录；旧快照不覆盖，已成功/定名的文件不续写。"""
    names = {"source_config.yaml", "dumpstatus.json", info["filename"] + ".part"}
    if output.is_symlink() or not output.is_dir():
        raise ValueError("--resume 需要已有的普通下载目录")
    if {p.name for p in output.iterdir()} != names:
        raise ValueError("续传目录必须仅含配置/状态快照和.part；完整文件/清单/锁或其他文件禁止续写")
    if any((output / name).is_symlink() or not (output / name).is_file() for name in names):
        raise ValueError("续传文件必须是普通文件，不能是符号链接")
    if (output / "source_config.yaml").read_bytes() != raw_config:
        raise ValueError("续传配置与旧快照不一致，不能拼接不同实验/版本的数据")
    raw_status = (output / "dumpstatus.json").read_bytes()
    _validate_status(info, raw_status)
    size = (output / (info["filename"] + ".part")).stat().st_size
    if size > info["bytes"]:
        raise ValueError("残留文件超过固定大小，不能续传")
    return size, raw_status


def _publish(partial: Path, final: Path) -> None:
    if os.name == "nt":
        # Windows 的 rename 遇到已有目标会报错，不覆盖；也不要求硬链接权限。
        os.rename(partial, final)
        return
    # POSIX rename会覆盖已有目标；用原子硬链接定名，存在即报错，不复制一份大型dump。
    # 若文件系统不支持硬链接则明确失败，不回退到可能覆盖旧数据的操作。
    os.link(partial, final)
    partial.unlink()  # 仅移除本次临时名字；内容已由final保留，不删除旧文件。


def download_dump(
    config_path: str | Path,
    source: str,
    output_dir: str | Path,
    *,
    download: bool = False,
    check_remote: bool = False,
    timeout: float = 30.0,
    resume: bool = False,
    retries: int = 3,
) -> dict:
    """默认不联网、不创建目录；check_remote只读元数据；download始终核对元数据后下载。

    timeout 是每次阻塞网络操作的超时，不是整个下载的总时限。
    默认要求新目录；显式download+resume才允许接着旧.part写入。无完整清单不可视为成功。
    """
    if type(download) is not bool or type(check_remote) is not bool:
        raise TypeError("download/check_remote 必须是 bool")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout 必须是有限正数")
    if type(resume) is not bool:
        raise TypeError("resume 必须是 bool")
    if resume and not download:
        raise ValueError("--resume 必须与 --download 一起使用")
    if type(retries) is not int or not 0 <= retries <= 20:
        raise ValueError("retries 必须是0～20的整数（不含第一次请求）")
    config_path = Path(config_path).resolve()
    info, raw_config = _load_source(config_path, source)
    output = Path(output_dir)
    resume_bytes, saved_status = 0, None
    if resume:
        resume_bytes, saved_status = _resume_state(output, info, raw_config)
    elif output.exists() or output.is_symlink():
        raise FileExistsError("--output-dir 已存在，请换新目录；不覆盖旧下载")
    output = output.resolve()
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        raise ValueError("输出目录的已有父路径必须是目录，不能是普通文件")
    free = shutil.disk_usage(parent).free
    if free < info["bytes"] - resume_bytes + RESERVE_BYTES:
        raise ValueError("磁盘空间不足：至少需要剩余压缩文件大小加1GiB下载余量（不含后续处理）")
    report = {
        "format_version": 1,
        "kind": "wikimedia_dump_download",
        "status": "planned",
        "source": {"name": source, **info},
        "output_dir": str(output),
        "source_config": {
            "path": str(config_path),
            "sha256": hashlib.sha256(raw_config).hexdigest(),
        },
        "disk": {"free_bytes": free, "reserve_bytes": RESERVE_BYTES},
        "official_status": None,
        "file": None,
        "ready_for_training": False,
    }
    raw_status = None
    if check_remote or download:
        report["official_status"], raw_status = _check_remote(info, timeout)
        report["metadata_checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    if not download:
        return report
    started = perf_counter()
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        for name, raw in (("source_config.yaml", raw_config), ("dumpstatus.json", raw_status)):
            with (output / name).open("xb") as file:
                file.write(raw)
    else:
        # 联网重新核对固定版本，但清单引用原始快照，不能把新的摘要配上旧的文件。
        report["official_status"] = _validate_status(info, saved_status)
    report["transfer"] = {"resume_bytes": resume_bytes, "max_retries": retries}
    lock = output / ".download.lock"
    # 排他创建锁；并发进程不能同时追加。只清理本进程创建的锁，不删除旧数据。
    with lock.open("x"):
        pass
    try:
        return _finish_download(
            info, output, config_path, report, timeout, resume, retries, started
        )
    finally:
        lock.unlink()


def _finish_download(info, output, config_path, report, timeout, resume, retries, started):
    partial = output / (info["filename"] + ".part")
    count, sha1, sha256 = _download_file(info, partial, timeout, resume=resume, retries=retries)
    if _sha256(config_path) != report["source_config"]["sha256"]:
        raise RuntimeError("下载期间来源配置发生变化，不写成功清单")
    for name, expected in (
        ("source_config.yaml", report["source_config"]["sha256"]),
        ("dumpstatus.json", report["official_status"]["sha256"]),
    ):
        if _sha256(output / name) != expected:
            raise RuntimeError("下载期间来源快照发生变化，不写成功清单")
    _publish(partial, output / info["filename"])
    report.update(
        status="verified",
        elapsed_seconds=perf_counter() - started,
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    report["source_config"].update(path="source_config.yaml", original_path=str(config_path))
    report["official_status"]["path"] = "dumpstatus.json"
    report["file"] = {"path": info["filename"], "bytes": count, "sha1": sha1, "sha256": sha256}
    pending = output / "download_manifest.json.part"
    with pending.open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _publish(pending, output / "download_manifest.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source", choices=("zhwiki", "zhwikisource"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="新目录，不覆盖；预览不创建")
    parser.add_argument("--check-remote", action="store_true", help="仅查询小型官方状态元数据")
    parser.add_argument("--download", action="store_true", help="显式开始大型dump传输")
    parser.add_argument("--resume", action="store_true", help="配合--download，从旧目录的.part继续")
    parser.add_argument("--retries", type=int, default=3, help="网络中断重试次数，0～20，默认3")
    parser.add_argument("--timeout", type=float, default=30.0, help="单次网络阻塞超时秒数")
    args = parser.parse_args(argv)
    try:
        report = download_dump(
            args.sources_config,
            args.source,
            args.output_dir,
            download=args.download,
            check_remote=args.check_remote,
            timeout=args.timeout,
            resume=args.resume,
            retries=args.retries,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return report
    except (OSError, ValueError, TypeError, RuntimeError, yaml.YAMLError) as error:
        parser.error(
            f"{error}\n未完成下载；无完整download_manifest.json不能视为校验通过；"
            "网络中断且旧目录完整时可加 --resume，配置/响应/摘要错误须先排查。"
        )
    except KeyboardInterrupt:
        parser.exit(130, "下载已中断；保留本次诊断文件，无完整清单不可用于后续处理。\n")


if __name__ == "__main__":
    main()

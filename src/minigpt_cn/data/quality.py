"""预算子集的质量筛选和收录建议；不替用户批准许可，不改原正文。"""

import argparse
import hashlib
import json
import re
from collections import Counter
from contextlib import ExitStack
from importlib.metadata import version
from os.path import relpath
from pathlib import Path

import regex

from minigpt_cn.data.candidate_subset import _json, _records, _signature, verify_subset
from minigpt_cn.data.parse_xml import _small_file
from minigpt_cn.data.prepare import _reject_nonfinite, _sha256

POLICY = {
    "id": "zhwiki_body_quality_v2",
    "source": "zhwiki",
    "min_nonspace_chars": 100,
    "max_nonspace_chars": 50000,
    "min_han_ratio": 0.25,
    "unknown_template": "retain_warning_and_sample_review",
    "empty_parentheses": "hold_including_punctuation_only",
    "standalone_year": "hold",
    "consecutive_short_lines": "hold_at_5_nonempty_lines_le20_chars_without_sentence_punctuation",
    "other_parser_issues": "hold",
    "template_name_limit": "hold_at_16_names_or_100_chars",
    "license": "pending_separate_user_review",
    "text_transformation": None,
}
CATEGORIES = ("eligible", "held", "excluded")
STRATA = ("eligible_clean", "eligible_template_warning", "held", "excluded")
SAMPLE_PER_STRATUM = 8
HAN = regex.compile(r"\p{Script=Han}")
EMPTY_PARENTHESES = re.compile(r"\([\s,，、;；:：/／]*\)|（[\s,，、;；:：/／]*）")
YEAR_LINE = re.compile(r"(?m)^\s*\d{3,4}\s*(?:年)?\s*$")
HTML = re.compile(r"</?[A-Za-z][^>\n]{0,500}>")
SENTENCE_PUNCTUATION = re.compile(r"[。！？!?;；:：]")
# 只是可见的风险线索，不是完整版权检测；未知模板名称本来就可能被截短。
LICENSE_ALERT = re.compile(r"copyvio|copyright|fair.?use|侵[权權]|版[权權]|合理使用|^pd-")


def classify(record):
    """按规则标记候选而非断言语义正确；长度单位为非空白Unicode码点。"""
    text = record["text"]
    nonspace = sum(not c.isspace() for c in text)
    ratio = len(HAN.findall(text)) / max(nonspace, 1)
    names = record.get("unknown_templates", [])
    hints = record.get("license_hints", [])
    if not all(
        isinstance(values, list) and all(isinstance(v, str) and v.strip() for v in values)
        for values in (names, hints)
    ):
        raise ValueError("未知模板/许可线索必须是非空字符串列表")
    excluded, held = [], []
    run, max_run = 0, 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        run = run + 1 if len(line) <= 20 and not SENTENCE_PUNCTUATION.search(line) else 0
        max_run = max(max_run, run)
    if nonspace < POLICY["min_nonspace_chars"]:
        excluded.append("short_body")
    if nonspace > POLICY["max_nonspace_chars"]:
        excluded.append("long_body")
    if ratio < POLICY["min_han_ratio"]:
        excluded.append("low_han_ratio")
    if "\ufffd" in text:
        excluded.append("replacement_character")
    if HTML.search(text) or any(
        s in text for s in ("{{", "}}", "[[", "]]", "{|", "|}", "-{", "}-")
    ):
        excluded.append("residual_markup")
    if EMPTY_PARENTHESES.search(text):
        held.append("empty_parentheses")
    if YEAR_LINE.search(text):
        held.append("standalone_year")
    if max_run >= 5:
        # 连续短行可能是表格被删后剩下的标题，也可能是合法列表：只暂缓，不断言垃圾。
        held.append("consecutive_short_lines")
    if set(record["issues"]) - {"unknown_template"}:
        held.append("unexpanded_syntax")
    if len(names) >= 16 or any(len(n) >= 100 for n in names):
        held.append("template_names_possibly_truncated")
    if "unknown_template" in record["issues"] and not names:
        held.append("missing_template_names")
    if hints or any(LICENSE_ALERT.search(n.casefold()) for n in names):
        held.append("license_hint_requires_review")
    if record["source"]["name"] != POLICY["source"]:
        held.append("source_requires_separate_policy")
    # unknown_template本身不等于坏正文，也不等于所删模板没有信息价值。
    category = "excluded" if excluded else ("held" if held else "eligible")
    return {
        "category": category,
        "reasons": excluded + held,
        "nonspace_chars": nonspace,
        "han_ratio": ratio,
        "max_short_line_run": max_run,
        "body_bytes": len(text.encode("utf-8")),
        "template_warning": "unknown_template" in record["issues"],
    }


def _stratum(packet):
    q = packet["quality"]
    if q["category"] != "eligible":
        return q["category"]
    return "eligible_template_warning" if q["template_warning"] else "eligible_clean"


def _sample_key(packet):
    r = packet["document"]
    identity = [r["source"], r["page_id"], r["revision_id"], r["text_sha256"]]
    # 单独的采样域，避免复用train/valid划分或BPE采样键。
    return hashlib.sha256(b"quality-review-v1\0" + _json(identity).encode()).hexdigest()


class _Summary:
    def __init__(self):
        self.counts = Counter({c: 0 for c in CATEGORIES})
        self.bytes = Counter({c: 0 for c in CATEGORIES})
        self.reasons, self.templates = Counter(), Counter()
        self.samples = {s: [] for s in STRATA}

    def add(self, packet):
        q, r = packet["quality"], packet["document"]
        self.counts[q["category"]] += 1
        self.bytes[q["category"]] += q["body_bytes"]
        self.reasons.update(q["reasons"])
        self.templates.update(r.get("unknown_templates", []))
        samples = self.samples[_stratum(packet)]
        samples.append((_sample_key(packet), _json(packet), packet))
        samples.sort(key=lambda item: item[:2])
        del samples[SAMPLE_PER_STRATUM:]

    def selected(self):
        return [dict(stratum=s, **item[2]) for s in STRATA for item in self.samples[s]]

    def report(self):
        return {
            "documents": dict(self.counts),
            "body_bytes": dict(self.bytes),
            "reasons": dict(sorted(self.reasons.items())),
            "top_template_names": self.templates.most_common(30),
            "sample_documents": len(self.selected()),
            "sample_per_stratum": SAMPLE_PER_STRATUM,
            "approved": 0,
        }


def _subset(path):
    raw = _small_file(path)
    verify_subset(path)
    m = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
    if _small_file(path) != raw:
        raise RuntimeError("读取期间子集清单发生变化")
    return m, hashlib.sha256(raw).hexdigest()


def audit_quality(subset_manifest, output_dir):
    path, output = Path(subset_manifest).absolute(), Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("--output-dir已存在，请换新目录；不覆盖审阅结果")
    subset, manifest_hash = _subset(path)
    candidates = path.parent / "candidates.jsonl"
    signature = _signature(candidates)
    summary = _Summary()
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        files = {
            c: stack.enter_context(
                (output / f"{c}.jsonl").open("x", encoding="utf-8", newline="\n")
            )
            for c in CATEGORIES
        }
        for _, _, record in _records(candidates):
            packet = {"quality": classify(record), "document": record}
            files[packet["quality"]["category"]].write(_json(packet) + "\n")
            summary.add(packet)
    samples = summary.selected()
    with (output / "review_samples.jsonl").open("x", encoding="utf-8", newline="\n") as file:
        for packet in samples:
            file.write(_json(packet) + "\n")
    with (output / "review.md").open("x", encoding="utf-8", newline="\n") as file:
        file.write("# 质量抽检：不是逐篇审批或许可证明\n\n")
        file.write("正文预览最多1500码点；完整抽检正文在review_samples.jsonl。\n\n")
        for i, packet in enumerate(samples, 1):
            r, q = packet["document"], packet["quality"]
            file.write(f"## {i}. {r['title']} [{packet['stratum']}]\n\n")
            file.write(f"修订：{r['revision_url']}\n\n")
            file.write(f"原因：{q['reasons']}；未知模板：{r.get('unknown_templates', [])}\n\n")
            # 原文放缩进块，不把正文里的Markdown/HTML当报告指令渲染。
            preview = r["text"][:1500]
            file.write("\n".join("    " + line for line in preview.splitlines()) + "\n\n")
            if len(r["text"]) > 1500:
                file.write("（预览截短；完整正文需查看JSONL。）\n\n")
    if (
        _signature(candidates) != signature
        or _sha256(candidates) != subset["files"]["candidates.jsonl"]["sha256"]
        or _sha256(path) != manifest_hash
    ):
        raise RuntimeError("筛选期间输入发生变化；不写成功清单")
    report = {
        "format_version": 1,
        "kind": "candidate_quality_audit",
        "status": "completed",
        "scope": subset["scope"],
        "ready_for_training": False,
        "quality_reviewed": False,
        "license_reviewed": False,
        "review_method": "automated_rules_only_pending_manual_sample",
        "policy": json.loads(_json(POLICY)),
        "processing": {"code_sha256": _sha256(Path(__file__)), "regex_version": version("regex")},
        # 相对输出目录保存输入位置，Windows和WSL均可只读复核同一份产物。
        "input": {
            "path": Path(relpath(path, output.absolute())).as_posix(),
            "path_base": "quality_manifest_directory",
            "sha256": manifest_hash,
        },
        "provenance": subset["provenance"],
        "counts": summary.report(),
        "files": {
            name: {
                "path": name,
                "bytes": (output / name).stat().st_size,
                "sha256": _sha256(output / name),
            }
            for name in (*[f"{c}.jsonl" for c in CATEGORIES], "review_samples.jsonl", "review.md")
        },
    }
    _verify(output, report)
    with (output / "quality_manifest.json").open("x", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def _verify(root, m):
    if (
        type(m["format_version"]) is not int
        or m["format_version"] != 1
        or m["kind"] != "candidate_quality_audit"
        or m["status"] != "completed"
        or m["policy"] != POLICY
        or m["ready_for_training"] is not False
        or m["quality_reviewed"] is not False
        or m["license_reviewed"] is not False
        or m["review_method"] != "automated_rules_only_pending_manual_sample"
        or type(m["counts"]["approved"]) is not int
        or m["counts"]["approved"] != 0
    ):
        raise ValueError("需要原始质量审计清单，不能修改为已批准/训练就绪")
    if m["input"].get("path_base") != "quality_manifest_directory":
        raise ValueError("输入路径必须相对于质量清单目录")
    source_path = root / m["input"]["path"]
    subset, h = _subset(source_path)
    if (
        h != m["input"]["sha256"]
        or m["scope"] != subset["scope"]
        or m["provenance"] != subset["provenance"]
    ):
        raise ValueError("质量审计的输入摘要/范围/来源不一致")
    for name in (*[f"{c}.jsonl" for c in CATEGORIES], "review_samples.jsonl", "review.md"):
        info, file = m["files"][name], root / name
        if (
            info["path"] != name
            or _signature(file)[2] != info["bytes"]
            or _sha256(file) != info["sha256"]
        ):
            raise ValueError("审计文件路径/大小/摘要不一致")
    summary = _Summary()
    streams = {c: iter(_records(root / f"{c}.jsonl")) for c in CATEGORIES}
    for _, _, record in _records(source_path.parent / "candidates.jsonl"):
        packet = {"quality": classify(record), "document": record}
        row = next(streams[packet["quality"]["category"]], None)
        if row is None or row[2] != packet:
            raise ValueError("输出正文/分类与输入逐条回放不一致")
        summary.add(packet)
    if any(next(stream, None) is not None for stream in streams.values()):
        raise ValueError("输出含多余文档")
    # JSON序列化使Counter报告中的tuple/list表现与落盘JSON一致。
    if _json(summary.report()) != _json(m["counts"]):
        raise ValueError("质量计数与规则回放不一致")
    samples = [record for _, _, record in _records(root / "review_samples.jsonl")]
    if samples != summary.selected():
        raise ValueError("抽检样本不符合固定分层哈希选择")
    if (
        _sha256(source_path) != h
        or _sha256(source_path.parent / "candidates.jsonl")
        != subset["files"]["candidates.jsonl"]["sha256"]
    ):
        raise RuntimeError("复核期间原子集发生变化")
    return {"passed": True, "counts": m["counts"], "ready_for_training": False}


def verify_quality(manifest_path):
    """只读回放规则、抽检选择和输出覆盖；不读取XML或压缩包。"""
    path = Path(manifest_path).absolute()
    m = json.loads(_small_file(path).decode("utf-8"), parse_constant=_reject_nonfinite)
    return _verify(path.parent, m)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--subset-manifest", type=Path)
    mode.add_argument("--verify-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify_manifest:
            if args.output_dir:
                raise ValueError("只读复核不使用--output-dir")
            report = verify_quality(args.verify_manifest)
        else:
            if args.output_dir is None:
                raise ValueError("筛选需要--output-dir")
            m = audit_quality(args.subset_manifest, args.output_dir)
            report = {
                "output_dir": str(args.output_dir),
                "counts": m["counts"],
                "ready_for_training": False,
            }
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
        return report
    except KeyboardInterrupt:
        parser.exit(130, "已中断；没有完整且通过复核的quality_manifest.json不算成功。\n")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        parser.error(f"{error}\n质量审计未通过；不批准、不覆盖原候选。")


if __name__ == "__main__":
    main()

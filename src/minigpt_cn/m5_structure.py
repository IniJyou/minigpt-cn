"""M5局部结构的固定规则参照及严格评分；不依赖模型或训练模块。"""

import json
import re
from collections import Counter

KINDS = ("body", "heading", "definition", "theorem", "proof", "example", "other", "uncertain")
REASONS = ("mixed_roles", "ambiguous", "insufficient_context")
MARKERS = {
    "definition": r"定义|Definition",
    "theorem": r"定理|引理|命题|推论|Theorem|Lemma|Proposition|Corollary|Thm",
    "proof": r"证明|Proof|Pf",
    "example": r"例题|示例|反例|例|Example|Ex",
    "other": r"题目|习题|练习|解答|参考文献|元数据",
}
BOUNDARY = r"(?=$|[\s\d:：.．（(、，,])"
ROLE_PATTERNS = {kind: re.compile(rf"^({marker}){BOUNDARY}", re.I)
                 for kind, marker in MARKERS.items()}
ARGUMENT = re.compile(
    r"^(设|令|由|因为|假设|考虑|根据|首先|利用|取|反证|当|若|构造|计算|写成|对任意)"
    r"|因此|所以|于是|矛盾|即得|可得|故"
    r"|^(let|assume|since|by|because|for|we|if|suppose|take)\b", re.I,
)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"non-JSON constant: {value}")


def parse_json(text):
    """不修补围栏、重复键、NaN或截断回复。"""
    return json.loads(text, object_pairs_hook=_object, parse_constant=_invalid_constant)


def validate_input(value):
    if not isinstance(value, dict) or set(value) != {"blocks"}:
        raise ValueError("input must contain only blocks")
    blocks = value["blocks"]
    if not isinstance(blocks, list) or not 1 <= len(blocks) <= 8:
        raise ValueError("input requires 1..8 blocks")
    for index, block in enumerate(blocks, 1):
        if not isinstance(block, dict) or set(block) != {"block_id", "text"}:
            raise ValueError("invalid input block fields")
        if block["block_id"] != f"b{index:04d}":
            raise ValueError("input IDs must be sequential")
        if not isinstance(block["text"], str) or not block["text"].strip() or len(block["text"]) > 2000:
            raise ValueError("block text must be nonblank and <=2000 characters")
    if sum(len(block["text"]) for block in blocks) > 6000:
        raise ValueError("fragment exceeds 6000 characters")


def valid_annotation(item):
    """等价于当前schema的annotation定义；标题是否正确另计任务指标。"""
    if not isinstance(item, dict) or set(item) != {"block_id", "kind", "title", "reason"}:
        return False
    if not isinstance(item["block_id"], str) or not re.fullmatch(r"b000[1-8]", item["block_id"]):
        return False
    kind, title, reason = item["kind"], item["title"], item["reason"]
    if kind not in KINDS or (title is not None and (not isinstance(title, str) or not 1 <= len(title) <= 2000)):
        return False
    if kind == "other":
        return reason == "out_of_scope"
    if kind == "uncertain":
        return reason in REASONS
    return reason is None


def valid_prediction(value):
    return (isinstance(value, dict) and set(value) == {"annotations"}
            and isinstance(value["annotations"], list) and 1 <= len(value["annotations"]) <= 8
            and all(valid_annotation(item) for item in value["annotations"]))


def extract_title(text):
    first = text.splitlines()[0]
    heading = re.match(r"^ {0,3}#{1,6}[ \t]+(.*)$", first)
    if heading:
        value = re.sub(r"[ \t]+#+[ \t]*$", "", heading.group(1))
        if value.startswith("**") and value.endswith("**"):
            value = value[2:-2]
        return value or None
    bold = re.match(r"^\*\*(.+?)\*\*(?=：|:|$)", first)
    return bold.group(1) if bold else None


def _plain_prefix(line):
    # 仅为角色匹配移除呈现标记；不对输入或标题反缩进/改写。
    line = re.sub(r"^ {0,3}#{1,6}[ \t]+", "", line.strip())
    line = re.sub(r"^(?:[-+*]|\d+[.)])[ \t]+", "", line)
    if line.startswith("**"):
        line = line[2:].replace("**", "", 1)
    return line


def _active_lines(text):
    fence = None
    math = False
    for line in text.splitlines():
        stripped = line.strip()
        found = re.match(r"^(`{3,}|~{3,})", stripped)
        if found:
            marker = found.group(1)
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = None
            continue
        if fence is not None or stripped.startswith(">"):
            continue
        if stripped.startswith("$$"):
            if stripped == "$$" or not stripped[2:].endswith("$$"):
                math = not math
            continue
        if not math and stripped:
            yield _plain_prefix(line)


def _classify(text):
    stripped = text.lstrip()
    if re.match(r"^(?:`{3,}|~{3,}|>|\$\$)", stripped):
        return "body", None
    lines = list(_active_lines(text))
    roles = set()
    insufficient = False
    for line in lines:
        if re.match(r"^(?:其|该|上述).{0,4}证明(?:如下|见下)", line):
            insufficient = True
        if re.match(r"^(?:定义|定理|证明)(?:或|/)(?:定义|定理|证明)", line):
            return "uncertain", "ambiguous"
        if re.match(r"^(?:请证明|求证|试证明)", line):
            return "other", "out_of_scope"
        # 简单词面近似：条件叙述中的证明要求；没有语义模型或数学判断。
        if re.match(r"^(?:设|已知)", line) and re.search(r"[，。]\s*(?:证明|求证)", line):
            return "other", "out_of_scope"
        for kind, pattern in ROLE_PATTERNS.items():
            found = pattern.match(line)
            if not found:
                continue
            if kind == "other":
                return "other", "out_of_scope"
            if kind == "proof":
                rest = line[found.end():].lstrip(" :：.．")
                if not rest:
                    insufficient = True
                    continue
                if not ARGUMENT.search(rest):
                    return "other", "out_of_scope"
            roles.add(kind)
            break
    if len(roles) > 1:
        return "uncertain", "mixed_roles"
    if insufficient:
        return "uncertain", "insufficient_context"
    if roles:
        return next(iter(roles)), None
    if re.match(r"^ {0,3}#{1,6}[ \t]+", text):
        return "heading", None
    return "body", None


def predict_rules(input_value):
    """唯一推理入参是input，不接收标签、来源、review_id或数据划分。"""
    validate_input(input_value)
    annotations = []
    for block in input_value["blocks"]:
        kind, reason = _classify(block["text"])
        annotations.append({"block_id": block["block_id"], "kind": kind,
                            "title": extract_title(block["text"]), "reason": reason})
    return {"annotations": annotations}


def evaluate(responses):
    """(input, gold, raw_response)序列的固定严格口径；坏整条回复不得取得部分TP。"""
    tp, fp, fn, support = Counter(), Counter(), Counter(), Counter()
    counts = Counter({key: 0 for key in (
        "samples", "blocks", "json_valid", "schema_valid", "coverage_valid", "sample_exact",
        "kind_correct", "block_exact", "title_correct", "reason_correct", "nonnull_gold_titles",
        "nonnull_gold_reasons", "nonnull_title_correct", "nonnull_reason_correct", "invalid_kind_items",
    )})
    confusion = {kind: Counter() for kind in KINDS}
    diagnostics = []
    for input_value, gold, raw_response in responses:
        validate_input(input_value)
        ids = [block["block_id"] for block in input_value["blocks"]]
        if not valid_prediction(gold) or [a["block_id"] for a in gold["annotations"]] != ids:
            raise ValueError("invalid reference annotations")
        counts["samples"] += 1
        counts["blocks"] += len(ids)
        parsed = None
        try:
            parsed = parse_json(raw_response)
            counts["json_valid"] += 1
            parsed_ok = True
        except (ValueError, TypeError):
            parsed_ok = False
        schema_ok = parsed_ok and valid_prediction(parsed)
        coverage_ok = schema_ok and [a["block_id"] for a in parsed["annotations"]] == ids
        counts["schema_valid"] += schema_ok
        counts["coverage_valid"] += coverage_ok
        counts["sample_exact"] += coverage_ok and parsed == gold
        problems = []
        if not parsed_ok:
            problems.append("json_parse")
        elif not schema_ok:
            problems.append("schema")
        elif not coverage_ok:
            problems.append("coverage")
        errors = []
        for index, expected in enumerate(gold["annotations"]):
            actual = parsed["annotations"][index] if coverage_ok else None
            true_kind = expected["kind"]
            support[true_kind] += 1
            counts["nonnull_gold_titles"] += expected["title"] is not None
            counts["nonnull_gold_reasons"] += expected["reason"] is not None
            if actual is not None:
                predicted_kind = actual["kind"]
                confusion[true_kind][predicted_kind] += 1
                if true_kind == predicted_kind:
                    tp[true_kind] += 1
                    counts["kind_correct"] += 1
                else:
                    fn[true_kind] += 1
                    fp[predicted_kind] += 1
                for field in ("title", "reason"):
                    matched = expected[field] == actual[field]
                    counts[field + "_correct"] += matched
                    counts["nonnull_" + field + "_correct"] += matched and expected[field] is not None
                counts["block_exact"] += expected == actual
            else:
                fn[true_kind] += 1
                confusion[true_kind]["missing_or_invalid"] += 1
            fields = [field for field in ("kind", "title", "reason")
                      if actual is None or expected[field] != actual[field]]
            if fields:
                errors.append({"block_id": expected["block_id"], "fields": fields,
                               "reference": expected, "prediction": actual})
        if not coverage_ok and isinstance(parsed, dict) and isinstance(parsed.get("annotations"), list):
            # 坏整条回复中的可辨认类别项仅计FP；重复/额外编号不能抵消漏检。
            for item in parsed["annotations"]:
                if isinstance(item, dict) and item.get("kind") in KINDS:
                    fp[item["kind"]] += 1
                else:
                    counts["invalid_kind_items"] += 1
        diagnostics.append({"json_valid": parsed_ok, "schema_valid": schema_ok,
                            "coverage_valid": coverage_ok, "problems": problems,
                            "errors": errors})

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    per_class = {}
    for kind in KINDS:
        precision = ratio(tp[kind], tp[kind] + fp[kind])
        recall = ratio(tp[kind], tp[kind] + fn[kind])
        per_class[kind] = {"support": support[kind], "tp": tp[kind], "fp": fp[kind], "fn": fn[kind],
                           "precision": precision, "recall": recall,
                           "f1": ratio(2 * tp[kind], 2 * tp[kind] + fp[kind] + fn[kind])}
    rates = {key: ratio(counts[key], counts[denominator]) for key, denominator in {
        "json_valid": "samples", "schema_valid": "samples", "coverage_valid": "samples",
        "sample_exact": "samples", "kind_correct": "blocks", "block_exact": "blocks",
        "title_correct": "blocks", "reason_correct": "blocks",
        "nonnull_title_correct": "nonnull_gold_titles", "nonnull_reason_correct": "nonnull_gold_reasons",
    }.items()}
    return {"counts": dict(counts), "rates": rates, "per_class": per_class,
            "macro_f1": sum(item["f1"] for item in per_class.values()) / len(KINDS),
            "micro_f1": ratio(2 * sum(tp.values()), 2 * sum(tp.values()) + sum(fp.values()) + sum(fn.values())),
            "confusion": {kind: dict(row) for kind, row in confusion.items()}, "diagnostics": diagnostics}

"""运行已绑定摘要的M5规则参照，保留全部回复与严格校准分数。"""

import argparse
import hashlib
import json
from pathlib import Path

from minigpt_cn.m5_structure import KINDS, evaluate, parse_json, predict_rules

PROJECT = Path(__file__).resolve().parents[2]


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_rows(path, values):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def run(protocol_path, output_dir):
    protocol_path, output_dir = Path(protocol_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output exists; use a new directory")
    protocol_raw = protocol_path.read_bytes()
    protocol = parse_json(protocol_raw)
    if protocol.get("protocol_version") != "m5-rules-calibration-v1" or protocol.get("rule_version") != "m5-structure-v1.1":
        raise ValueError("unsupported baseline protocol")
    artifact_paths = {name: (PROJECT / name).resolve() for name in protocol["artifacts"]}
    actual = {name: file_sha(path) for name, path in artifact_paths.items()}
    if actual != protocol["artifacts"]:
        raise ValueError("artifact SHA256 differs from frozen protocol")
    data_name = protocol["calibration_file"]
    if data_name not in artifact_paths:
        raise ValueError("calibration file must be SHA256-bound")
    records = [parse_json(line) for line in artifact_paths[data_name].read_text(
        encoding="utf-8").splitlines()]
    if not records or len({record["review_id"] for record in records}) != len(records):
        raise ValueError("empty or duplicate review records")
    if any(record["rule_version"] != protocol["rule_version"] for record in records):
        raise ValueError("mixed rule versions")
    # 先生成所有预测；推理函数从不接收人工标签或来源元数据。
    predictions = [{"review_id": record["review_id"],
                    "response": json.dumps(predict_rules(record["input"]), ensure_ascii=False)}
                   for record in records]
    metrics = evaluate((record["input"], record["proposed_target"], prediction["response"])
                       for record, prediction in zip(records, predictions))
    diagnostics = metrics.pop("diagnostics")
    errors = []
    for record, prediction, diagnostic in zip(records, predictions, diagnostics):
        if diagnostic["problems"] or diagnostic["errors"]:
            errors.append({"review_id": record["review_id"], "input": record["input"],
                           "raw_response": prediction["response"], **diagnostic})
    # 运行期间任何已绑定输入/提示/源码变化都停止，不把混合版本写成一次结果。
    if {name: file_sha(path) for name, path in artifact_paths.items()} != actual:
        raise ValueError("bound artifact changed during run")
    if protocol_path.read_bytes() != protocol_raw:
        raise ValueError("protocol changed during run")
    output_dir.mkdir(parents=True)
    write_rows(output_dir / "predictions.jsonl", predictions)
    write_rows(output_dir / "errors.jsonl", errors)
    write_json(output_dir / "metrics.json", metrics)
    counts, rates = metrics["counts"], metrics["rates"]
    report = [
        "# M5 v1.1 规则基线：校准诊断", "",
        "固定简单规则，仅使用input；未运行Qwen或LoRA。系统提示已绑定供后续模型使用，本规则程序不通过系统提示生成。",
        "这些材料已用于制定规范，标签也由助手起草；分数不是独立人工金标、盲测或未见泛化结果。", "",
        "| 指标 | 通过/总数 | 比例 |", "| --- | ---: | ---: |",
    ]
    for label, key, denominator in (
        ("JSON解析", "json_valid", "samples"), ("输出schema", "schema_valid", "samples"),
        ("编号顺序与覆盖", "coverage_valid", "samples"), ("整片段完全匹配", "sample_exact", "samples"),
        ("块类别", "kind_correct", "blocks"), ("块全部字段", "block_exact", "blocks"),
        ("标题（包含null）", "title_correct", "blocks"),
        ("非空金标题", "nonnull_title_correct", "nonnull_gold_titles"),
        ("reason（包含null）", "reason_correct", "blocks"),
        ("非空金reason", "nonnull_reason_correct", "nonnull_gold_reasons"),
    ):
        report.append(f"| {label} | {counts[key]}/{counts[denominator]} | {rates[key]:.2%} |")
    report.extend(["", f"固定八类macro-F1={metrics['macro_f1']:.6f}，micro-F1={metrics['micro_f1']:.6f}。", "",
                   "| 类别 | 支持数 | TP | FP | FN | Precision | Recall | F1 |",
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for kind in KINDS:
        item = metrics["per_class"][kind]
        report.append(f"| {kind} | {item['support']} | {item['tp']} | {item['fp']} | {item['fn']} | "
                      f"{item['precision']:.4f} | {item['recall']:.4f} | {item['f1']:.4f} |")
    report.extend(["", "## 错误明细", "",
                   f"有错误的片段共{len(errors)}个；完整原始回复与所有错误保留于predictions.jsonl、errors.jsonl。"])
    if not errors:
        report.extend(["", "本批没有观察到字段或格式错误。这只表示固定规则在已知校准材料上匹配候选标签，不能据此宣称规则已覆盖真实笔记或LoRA有提升空间。"])
    for row in errors:
        report.extend(["", f"### {row['review_id']}", "", "格式问题：" + ", ".join(row["problems"])])
        for error in row["errors"]:
            block = next(block for block in row["input"]["blocks"] if block["block_id"] == error["block_id"])
            report.extend(["", f"{error['block_id']} 错误字段：{', '.join(error['fields'])}", "",
                           "````markdown", block["text"].replace("\r\n", "\n"), "````", "", "```json",
                           json.dumps(error, ensure_ascii=False), "```"])
    report.extend(["", "## 限制与下一步", "",
                   "格式坏整条回复按全部金标漏检计FN，其中可辨认预测类别计FP；不跳过失败样本，不自动修JSON、补块或重排。标题和reason也保留全体分母。",
                   "证明请求采用有限词面线索，没有语义判断；规则只看本块，空证明标记不会利用邻块补上下文。长公式、引文嵌套、别名及措辞变化仍可能失败。",
                   "本批proof仅3块、theorem4块，uncertain仅mixed_roles两例；来源集中且家族未完全核定。没有读取新笔记扩充通过率或调整候选金标。",
                   "后续应先核对实际错误与未覆盖变体，再单独准备绑定Qwen版本/模板的环境和未见来源评测；本轮不下载、安装或训练。", ""])
    with (output_dir / "report.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(report))
    write_json(output_dir / "run_manifest.json", {
        "protocol_sha256": hashlib.sha256(protocol_raw).hexdigest(), "artifacts": actual,
        "samples": counts["samples"], "blocks": counts["blocks"],
        "evaluation_use": "calibration_only", "model_run": False,
        "outputs": {path.name: file_sha(path) for path in sorted(output_dir.iterdir())},
    })
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT / "configs/m5_rule_baseline_v1.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    metrics = run(args.protocol, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "counts": metrics["counts"],
                      "macro_f1": metrics["macro_f1"], "model_run": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

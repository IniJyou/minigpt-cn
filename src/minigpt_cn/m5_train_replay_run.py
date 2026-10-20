"""M5固定训练原例回放：默认CPU inspect；仅两检查点28次，无训练或test入口。"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from minigpt_cn import m5_lora_run as old
from minigpt_cn.m5_greedy_run_v2 import DecodeIntegrityError, effective_config
from minigpt_cn.m5_lora_revision_run import RevisionBackend, read, rows
from minigpt_cn.m5_qwen_run import sha256_file
from minigpt_cn.m5_rule_run import PROJECT, write_json, write_rows
from minigpt_cn.m5_structure import KINDS, evaluate, parse_json

DEFINITION = "configs/m5_train_replay_v1/protocol.json"
DEFINITION_SHA = "1813c403ae0d55e58308ae74f55f40927dcec5e8ec35ff8f5f8fdc7c0289483c"
FREEZE = "logs/m5_train_replay_definition_20261019_v1/freeze_receipt.json"
SOURCES = ("src/minigpt_cn/m5_train_replay_run.py", "tests/test_m5_train_replay_run.py",
           "scripts/m5_train_replay_wsl.sh", "docs/m5-train-replay-runner.md")
CONDITIONS = ("control_step45", "candidate_step45")
PHASES = ("inspect", "replay")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def checkpoints(definition, root):
    for condition in definition["conditions"]:
        if old.file_tree(old.inside(root, condition["directory"])) != condition["files"]:
            raise DecodeIntegrityError("fixed checkpoint tree changed")


def load_execution(path, root=PROJECT):
    execution = read(path)
    bindings = execution["artifacts"]
    required = {*SOURCES, DEFINITION, FREEZE, execution["cpu_checks"], execution["cpu_input_preflight"]}
    if (execution.get("version") != "m5-train-replay-runtime-v1"
            or execution.get("definition") != DEFINITION or bindings.get(DEFINITION) != DEFINITION_SHA
            or execution.get("default_phase") != "inspect" or execution.get("allowed_phases") != list(PHASES)
            or not required <= set(bindings)):
        raise DecodeIntegrityError("unsupported or incomplete replay runtime")
    old.verify_files(root, bindings)
    definition = read(old.inside(root, DEFINITION))
    freeze = read(old.inside(root, FREEZE))
    for inherited in (freeze["artifacts"], definition["artifacts"]):
        if any(bindings.get(p) != sha for p, sha in inherited.items()):
            raise DecodeIntegrityError("missing transitive frozen artifact")
    for name in ("cpu_checks", "cpu_input_preflight"):
        if read(old.inside(root, execution[name])).get("passed") is not True:
            raise DecodeIntegrityError("CPU acceptance has not passed")
    parent = read(old.inside(root, "configs/m5_lora_v1/protocol.json"))
    if definition["model"] != parent["model"]:
        raise DecodeIntegrityError("base model differs")
    plan = copy.deepcopy(parent)
    plan["evaluation"]["generation"] = definition["generation"]
    api = read(old.inside(root, definition["generation_api_evidence"]))
    call_api = read(old.inside(root, definition["generation_call_boundary_evidence"]))
    if not api["passed"] or not call_api["passed"]:
        raise DecodeIntegrityError("generation API evidence failed")
    checkpoints(definition, root)
    return execution, definition, plan, api, call_api


def selected_records(definition, root=PROJECT):
    """按原始UTF-8行及字节定位完整记录；focus块不成为新输入。"""
    selection = read(old.inside(root, definition["selection"]))
    source = old.inside(root, definition["training_data"])
    if (selection["source_train"] != definition["training_data"]
            or sha256_file(source) != selection["source_train_sha256"]
            or selection["token_layout_source"] != definition["token_layout_evidence"]
            or sha256_file(old.inside(root, selection["token_layout_source"])) != selection["token_layout_sha256"]):
        raise DecodeIntegrityError("training source or layout identity differs")
    raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    offsets, cursor = [], 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line)
    records = []
    for order, meta in enumerate(selection["samples"], 1):
        index = meta["source_line_1based"] - 1
        line = lines[index].rstrip(b"\r\n")
        start, end = offsets[index], offsets[index] + len(line)
        if (meta["order"] != order or not meta["full_record_required"]
                or (meta["utf8_byte_start"], meta["utf8_byte_end_exclusive"]) != (start, end)
                or raw[start:end] != line or hashlib.sha256(line).hexdigest() != meta["line_utf8_sha256"]):
            raise DecodeIntegrityError("selected original UTF-8 line identity differs")
        record = parse_json(line)
        if (record["sample_id"] != meta["sample_id"] or record["source"]["group_id"] != meta["source_group_id"]
                or digest(record["input"]) != meta["input_sha256"]
                or digest(record["target"]) != meta["target_sha256"]
                or len(record["input"]["blocks"]) != meta["blocks"]):
            raise DecodeIntegrityError("selected complete record identity differs")
        # 同一严格评分器检查参考结构；仅CPU处理，不把target发给模型。
        evaluate([(record["input"], record["target"], json.dumps(record["target"]))])
        records.append(record)
    if ([r["sample_id"] for r in records] != definition["expected_sample_ids"]
            or len(records) != 14 or len({r["sample_id"] for r in records}) != 14
            or sum(len(r["input"]["blocks"]) for r in records) != 17
            or selection["summary"] != definition["selection_summary"]):
        raise DecodeIntegrityError("selected sample order or denominator differs")
    for stratum, counts in definition["scoring"]["mandatory_strata"].items():
        members = [r for r, m in zip(records, selection["samples"]) if m["stratum"] == stratum]
        if len(members) != counts["samples"] or sum(len(r["input"]["blocks"]) for r in members) != counts["blocks"]:
            raise DecodeIntegrityError("stratum denominator differs")
    if len(selection["complete_pairs"]) != 5 or any(
            p["role"] != "train" or len(p["sample_ids"]) != 2
            or not set(p["sample_ids"]) <= set(definition["expected_sample_ids"])
            for p in selection["complete_pairs"]):
        raise DecodeIntegrityError("five complete training pairs required")
    return selection, records


class ReplayBackend(RevisionBackend):
    """只复用冻结的加载/推理实现；CLI不提供训练或预检阶段。"""

    def inspect_greedy(self):
        from transformers import GenerationConfig
        from transformers.generation.utils import GenerationMixin

        # 纯配置代理：执行安装版合并逻辑，不构造模型/参数或调用generate。
        proxy = GenerationMixin()
        proxy.generation_config = GenerationConfig.from_pretrained(self.model_dir, local_files_only=True)
        prepared = effective_config(proxy, self.generation_config, self.plan["evaluation"]["generation"])
        return {"effective_mode": prepared.get_generation_mode().value,
                "do_sample": prepared.do_sample, "use_model_defaults": False,
                "model_default_do_sample": proxy.generation_config.do_sample}

    def generate(self, ids, before_forward):
        try:
            return super().generate(ids, before_forward)
        except ValueError as exc:
            # 冻结completion_record的前缀/预算异常属于完整性失败，不能计成普通空回复。
            if str(exc) in ("invalid generated token count", "generated sequence does not preserve the input prefix"):
                raise DecodeIntegrityError("completion prefix or token budget differs") from exc
            raise


def inspect_inputs(backend, definition, selection, records, root=PROJECT):
    prompt = old.inside(root, definition["system_prompt"]).read_text(encoding="utf-8")
    encoded = old.encoded_inputs(backend, prompt, records, definition["training_interface_reference"])
    lengths = [len(ids) for ids in encoded]
    if lengths != [m["verified_layout"]["input_tokens"] for m in selection["samples"]]:
        raise DecodeIntegrityError("actual tokenizer input differs from frozen layout")
    greedy = backend.inspect_greedy()
    if greedy["effective_mode"] != "greedy_search" or greedy["do_sample"] is not False or greedy["use_model_defaults"] is not False:
        raise DecodeIntegrityError("inspect effective decoding differs")
    if backend.torch.cuda.is_initialized():
        raise DecodeIntegrityError("inspect must not initialize CUDA")
    return encoded, {"phase": "inspect", "passed": True, "samples": 14, "blocks": 17,
                     "strata": definition["scoring"]["mandatory_strata"], "complete_pairs": 5,
                     "input_token_range": [min(lengths), max(lengths)],
                     "input_tokens_by_sample": dict(zip(definition["expected_sample_ids"], lengths)),
                     "effective_generation": greedy, "packages": backend.runtime["packages"],
                     "model_loaded": False, "cuda_initialized": False, "generation_calls": 0,
                     "training_steps": 0, "test_records_read": False}


class CallLedger:
    """先持久化再前向；账本损坏、删除、重复或第29次必须全局停止。"""

    def __init__(self, folder, calls):
        self.path = folder / "call_attempts.jsonl"
        self.expected, self.committed = calls, []
        if len(calls) != 28 or [c["attempt"] for c in calls] != list(range(1, 29)) or self.path.exists():
            raise DecodeIntegrityError("exactly 28 fresh planned attempts required")

    def entries(self):
        try:
            entries = rows(self.path) if self.path.exists() else []
        except (OSError, ValueError) as exc:
            raise DecodeIntegrityError("unreadable ledger") from exc
        if entries != self.committed:
            raise DecodeIntegrityError("durable ledger changed")
        return entries

    def record(self, condition, sample_id):
        index = len(self.entries())
        if index >= 28 or (condition, sample_id) != (
                self.expected[index]["condition"], self.expected[index]["sample_id"]):
            raise DecodeIntegrityError("call order or 28-attempt budget differs")
        entry = {**self.expected[index], "effective_mode": "greedy_search",
                 "do_sample": False, "use_model_defaults": False}
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise DecodeIntegrityError("attempt cannot be persisted before forward") from exc
        self.committed.append(entry)


def planned_calls(definition, selection, root):
    expected = [{"attempt": n * 14 + i + 1, "condition": name,
                 "sample_id": m["sample_id"], "stratum": m["stratum"]}
                for n, name in enumerate(CONDITIONS) for i, m in enumerate(selection["samples"])]
    if read(old.inside(root, definition["planned_call_order"]))["calls"] != expected:
        raise DecodeIntegrityError("frozen call plan differs")
    return expected


def generate_condition(backend, records, encoded, folder, condition, ledger):
    if len(records) != 14 or len(encoded) != len(records):
        raise DecodeIntegrityError("all 14 complete inputs required")
    folder.mkdir(exist_ok=False)
    predictions = []
    with (folder / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for record, ids in zip(records, encoded):
            backend.assert_inference()
            backend.check_effective()
            before = len(ledger.entries())
            try:
                prediction = backend.generate(ids, lambda: ledger.record(condition, record["sample_id"]))
            except Exception as exc:
                if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
                    raise
                prediction = {"response": "", "raw_completion": "", "generated_token_ids": [],
                              "generated_tokens": 0, "stop_reason": "error", "hit_token_limit": False,
                              "runtime_error": f"{type(exc).__name__}: {exc}"}
            if len(ledger.entries()) != before + 1:
                raise DecodeIntegrityError("generation must consume exactly one durable attempt")
            count = prediction["generated_tokens"]
            if (type(count) is not int or not 0 <= count <= 1024
                    or count != len(prediction["generated_token_ids"])):
                raise DecodeIntegrityError("generated token budget or record differs")
            prediction = {**prediction, "sample_id": record["sample_id"], "input_tokens": len(ids)}
            predictions.append(prediction)
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
    write_json(folder / "runtime.json", backend.runtime)
    return predictions


def score_records(records, predictions):
    if len(records) != len(predictions) or [r["sample_id"] for r in records] != [p["sample_id"] for p in predictions]:
        raise DecodeIntegrityError("prediction identity or full denominator differs")
    return evaluate((r["input"], r["target"], p["response"]) for r, p in zip(records, predictions))


def condition_report(records, predictions, selection):
    metrics = score_records(records, predictions)
    diagnostics = metrics.pop("diagnostics")
    samples, titles, reasons, illegal = [], {"missing": 0, "spurious": 0, "wrong_nonnull": 0}, {}, []
    for record, prediction, meta, diag in zip(records, predictions, selection["samples"], diagnostics):
        exact = diag["coverage_valid"] and not diag["errors"]
        samples.append({"sample_id": record["sample_id"], "stratum": meta["stratum"],
                        "sample_exact": exact, "focus_blocks": meta["focus_blocks"],
                        "old_presentations": meta["old_checkpoint_exact_finetuning_presentations"],
                        "new_presentations": meta["new_checkpoint_exact_finetuning_presentations"], **diag})
        try:
            parsed = parse_json(prediction["response"])
        except (ValueError, TypeError):
            parsed = None
        annotations = parsed.get("annotations") if isinstance(parsed, dict) else None
        if isinstance(annotations, list):
            illegal.extend({"sample_id": record["sample_id"], "item": a} for a in annotations
                           if not isinstance(a, dict) or a.get("kind") not in KINDS)
        if diag["coverage_valid"]:
            for gold, actual in zip(record["target"]["annotations"], annotations):
                titles["missing"] += gold["title"] is not None and actual["title"] is None
                titles["spurious"] += gold["title"] is None and actual["title"] is not None
                titles["wrong_nonnull"] += gold["title"] is not None and actual["title"] is not None and gold["title"] != actual["title"]
                key = json.dumps([gold["reason"], actual["reason"]], ensure_ascii=False)
                reasons[key] = reasons.get(key, 0) + 1
    strata = {}
    for name in ("common_train", "revision_only_train"):
        indices = [i for i, m in enumerate(selection["samples"]) if m["stratum"] == name]
        strata[name] = score_records([records[i] for i in indices], [predictions[i] for i in indices])
        strata[name].pop("diagnostics")
    exact = {s["sample_id"]: s["sample_exact"] for s in samples}
    pairs = [{"pair_id": p["pair_id"], "sample_ids": p["sample_ids"],
              "both_full_records_exact": all(exact[s] for s in p["sample_ids"])} for p in selection["complete_pairs"]]
    return {"metrics": metrics, "strata": strata, "samples": samples, "pairs": pairs,
            "title_errors_on_covered_samples": titles, "reason_confusion_on_covered_samples": reasons,
            "structurally_invalid_blocks": sum(len(r["input"]["blocks"]) for r, d in zip(records, diagnostics) if not d["coverage_valid"]),
            "illegal_kinds": illegal}


def comparison(reports, definition):
    control, candidate = (reports[n] for n in CONDITIONS)
    transitions = []
    for old_sample, new_sample in zip(control["samples"], candidate["samples"]):
        if old_sample["stratum"] == "common_train":
            transitions.append({"sample_id": old_sample["sample_id"],
                                "old_exact": old_sample["sample_exact"], "new_exact": new_sample["sample_exact"]})
    return {"diagnostic_only": True, "generation_attempts": 28, "training_steps": 0, "test_calls": 0,
            "candidate_all_14_selected_samples_exact": candidate["metrics"]["counts"]["sample_exact"] == 14,
            "common_train_paired_transitions": transitions, "interpretation": definition["interpretation"],
            "automatic_test_allowed": False, "original_selected_step": None,
            "zero_support_kinds": ["definition", "heading"]}


def run(execution_path, phase="inspect", root=PROJECT, backend_factory=ReplayBackend):
    if phase not in PHASES:
        raise ValueError("only inspect/replay; no train/test/resume/retry")
    execution_path = Path(execution_path).resolve()
    execution_sha = sha256_file(execution_path)
    execution, definition, plan, api, call_api = load_execution(execution_path, root)
    selection, records = selected_records(definition, root)
    calls = planned_calls(definition, selection, root)
    destination = old.inside(root, definition["execution_requirements"]["planned_output_directory"])
    if phase == "replay" and destination.exists():
        raise FileExistsError("replay directory already claimed; no overwrite/resume/retry")
    backend, claimed = None, False
    try:
        backend = backend_factory(plan, api, call_api)
        encoded, inspection = inspect_inputs(backend, definition, selection, records, root)
        if phase == "inspect":
            if sha256_file(execution_path) != execution_sha:
                raise DecodeIntegrityError("runtime changed during inspect")
            return inspection
        destination.mkdir(parents=True, exist_ok=False)
        claimed = True
        write_json(destination / "started.json", {"execution_sha256": execution_sha, "phase": phase})
        write_json(destination / "input_preflight.json", inspection)
        ledger = CallLedger(destination, calls)
        reports = {}
        for index, condition in enumerate(definition["conditions"]):
            if index:
                backend = backend_factory(plan, api, call_api)
                current = old.encoded_inputs(backend, old.inside(root, definition["system_prompt"]).read_text(encoding="utf-8"),
                                             records, definition["training_interface_reference"])
                if current != encoded:
                    raise DecodeIntegrityError("condition tokenizer/input changed")
                backend.inspect_greedy()
            load_execution(execution_path, root)
            if sha256_file(execution_path) != execution_sha:
                raise DecodeIntegrityError("runtime changed before model load")
            backend.load_base()
            backend.load_adapter(old.inside(root, condition["directory"]))
            folder = destination / condition["name"]
            predictions = generate_condition(backend, records, encoded, folder, condition["name"], ledger)
            reports[condition["name"]] = condition_report(records, predictions, selection)
            write_json(folder / "report.json", reports[condition["name"]])
            write_rows(folder / "all_sample_fields.jsonl", reports[condition["name"]]["samples"])
            backend.close()
        if len(ledger.entries()) != 28:
            raise DecodeIntegrityError("all 28 attempts required before sealing")
        result = comparison(reports, definition)
        write_json(destination / "result.json", result)
        load_execution(execution_path, root)
        if sha256_file(execution_path) != execution_sha:
            raise DecodeIntegrityError("runtime changed before sealing")
        old.seal(destination, execution_sha, {"status": "completed", "phase": "replay"})
        return result
    except BaseException as exc:
        if claimed:
            write_json(destination / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if backend is not None:
            backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=PROJECT / "configs/m5_train_replay_v1/runtime.json")
    parser.add_argument("--phase", choices=PHASES, default="inspect")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.execution, args.phase), ensure_ascii=False))


if __name__ == "__main__":
    main()

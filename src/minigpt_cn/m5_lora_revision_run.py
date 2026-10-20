"""M5修订对照：默认CPU inspect；独立预检及45步/87次开发对照，无test入口。"""

import argparse
import copy
import gc
import inspect
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from minigpt_cn import m5_lora_run as old
from minigpt_cn.m5_greedy_run import load_execution as load_greedy_execution
from minigpt_cn.m5_greedy_run_v2 import DecodeIntegrityError, effective_config
from minigpt_cn.m5_qwen_run import completion_record, sha256_file
from minigpt_cn.m5_rule_run import PROJECT, write_json
from minigpt_cn.m5_structure import evaluate, parse_json

SOURCES = ("src/minigpt_cn/m5_lora_revision_run.py", "tests/test_m5_lora_revision_run.py",
           "scripts/m5_lora_revision_wsl.sh", "docs/m5-lora-revision-runner.md")
CONDITIONS = (("base_diagnostic", "new_diagnostic", 24, None),
              ("control_diagnostic", "new_diagnostic", 24, "control"),
              ("candidate_diagnostic", "new_diagnostic", 24, "candidate"),
              ("candidate_original_valid", "original_valid", 15, "candidate"))
PHASES = ("inspect", "preflight", "train-compare")


def read(path):
    return parse_json(Path(path).read_bytes())


def rows(path):
    return [parse_json(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]


def load_execution(path, root=PROJECT):
    execution = read(path)
    if execution.get("version") != "m5-lora-revision-runtime-v1":
        raise ValueError("unsupported revision execution")
    bindings = execution["artifacts"]
    required = set(SOURCES) | {execution[k] for k in (
        "definition", "dataset", "data_acceptance", "data_freeze", "greedy_runtime", "cpu_checks",
        "generation_api_check", "call_boundary_api_check")}
    if not required <= set(bindings):
        raise ValueError("runner/data/CPU/API evidence must be bound")
    old.verify_files(root, bindings)
    definition = read(old.inside(root, execution["definition"]))
    dataset_path = old.inside(root, execution["dataset"])
    dataset = read(dataset_path)
    acceptance = read(old.inside(root, execution["data_acceptance"]))
    freeze = read(old.inside(root, execution["data_freeze"]))
    for inherited in (freeze["artifacts"], definition["artifacts"], dataset["artifacts"]):
        if any(bindings.get(p) != digest for p, digest in inherited.items()):
            raise ValueError("transitive artifact missing from runtime")
    if (definition.get("version") != "m5-lora-data-revision-definition-v1"
            or dataset.get("version") != "m5-lora-revision-dataset-v1"
            or dataset["definition"] != execution["definition"]
            or dataset["definition_sha256"] != sha256_file(old.inside(root, execution["definition"]))
            or not acceptance["passed"] or acceptance["dataset_sha256"] != sha256_file(dataset_path)):
        raise ValueError("definition/data acceptance differs")
    folder = old.inside(root, dataset["data_directory"])
    manifest = read(folder / "manifest.json")
    actual = old.file_tree(folder)
    actual.pop("manifest.json")
    if manifest["outputs"] != actual:
        raise ValueError("data file tree differs from manifest")
    if not read(old.inside(root, execution["cpu_checks"]))["passed"]:
        raise ValueError("CPU checks have not passed")
    _, greedy, parent, api = load_greedy_execution(old.inside(root, execution["greedy_runtime"]), root)
    if (definition["model"] != parent["model"] or definition["system_prompt"] != parent["system_prompt"]
            or definition["training_interface"] != parent["training_interface"]
            or definition["expected_packages"] != greedy["environment"]["expected_packages"]
            or definition["training_comparison"]["control"]["files"] != greedy["adapter"]["files"]):
        raise ValueError("parent model/task/control differs")
    evaluation = definition["proposed_evaluation"]
    expected_conditions = [{"model": m, "dataset": d, "samples": n} for m, d, n in (
        ("base", "new_boundary_diagnostic", 24), ("control_step45", "new_boundary_diagnostic", 24),
        ("candidate_step45", "new_boundary_diagnostic", 24), ("candidate_step45", "original_valid", 15))]
    candidate = definition["training_comparison"]["candidate"]
    if (evaluation["conditions"] != expected_conditions or evaluation["max_model_generation_calls"] != 87
            or evaluation["generation"] != greedy["generation"] or evaluation["retry"]
            or evaluation["other_checkpoints_or_seeds"] or evaluation["test_calls"] != 0
            or evaluation["automatic_test_allowed"] or not evaluation["diagnostic_only"]
            or evaluation["minimum_gates"] != parent["validation"]["minimum_gates"]
            or candidate["max_optimizer_steps"] != 45 or candidate["training_microbatches"] != 180
            or candidate["checkpoint_steps"] != [45] or not candidate["fresh_adapter"]
            or candidate["scheduler"] != parent["training"]["scheduler"]
            or candidate["scheduler"]["num_training_steps"] != 180
            or dataset["expected_supervised_token_exposures"] != 9657
            or dataset["counts"] != {"train": 84, "new_train": 48, "new_diagnostic": 24, "original_valid": 15}):
        raise ValueError("only frozen 45-step/87-call diagnostic is supported")
    history = old.inside(root, evaluation["reuse_original_valid_base_and_control"])
    old.verify_receipt(history, sha256_file(old.inside(root, execution["greedy_runtime"])))
    if api != read(old.inside(root, execution["generation_api_check"])):
        raise ValueError("generation API evidence differs")
    call_api = read(old.inside(root, execution["call_boundary_api_check"]))
    if not call_api["passed"] or call_api["generate_overrides"] != {"do_sample": False, "use_model_defaults": False}:
        raise ValueError("greedy call-boundary check differs")
    experiment = old.inside(root, execution["experiment_directory"])
    if not experiment.is_relative_to((root / "logs").resolve()):
        raise ValueError("revision output must stay under logs")
    plan = {k: copy.deepcopy(parent[k]) for k in ("model", "environment", "adapter", "training", "gpu_preflight")}
    for name in ("epochs", "optimizer_steps_per_epoch", "supervised_tokens_per_epoch", "checkpoint_epochs"):
        plan["training"].pop(name, None)
    plan["training"].update({"max_optimizer_steps": 45, "training_microbatches": 180,
                             "supervised_token_exposures_total": 9657, "checkpoint_optimizer_steps": [45]})
    plan["evaluation"] = {"generation": evaluation["generation"]}
    plan["training_interface"] = copy.deepcopy(parent["training_interface"])
    plan["expected_initial"] = candidate["expected_initial_sha256"]
    plan["schedule_path"] = dataset["training_schedule"]
    plan["schedule_sha256"] = bindings[dataset["training_schedule"]]
    return execution, definition, dataset, plan, api, call_api


def validate_groups(items, schedule):
    if len(items) != 84 or len({i["sample_id"] for i in items}) != 84:
        raise ValueError("training requires exactly 84 unique samples")
    lookup = {i["sample_id"]: i for i in items}
    order = []
    for epoch in range(3):
        shuffled = list(lookup)
        random.Random(42 + epoch).shuffle(shuffled)
        order.extend(shuffled if epoch < 2 else shuffled[:12])
    entries = schedule["groups"]
    if (len(entries) != 45 or schedule["optimizer_steps"] != 45 or schedule["training_microbatches"] != 180
            or [i for g in entries for i in g["sample_ids"]] != order):
        raise ValueError("schedule must be two epochs plus 12 samples, 45 complete groups")
    groups = []
    for step, entry in enumerate(entries, 1):
        if entry["optimizer_step"] != step or len(entry["sample_ids"]) != 4:
            raise ValueError("schedule step/group size differs")
        group = [lookup[i] for i in entry["sample_ids"]]
        if old.supervised_count(group) != entry["supervised_tokens"]:
            raise ValueError("group supervised denominator differs")
        groups.append(group)
    if sum(old.supervised_count(g) for g in groups) != schedule["supervised_token_exposures"]:
        raise ValueError("supervised exposure differs")
    return groups


def train_loop(model, items, schedule, settings, optimizer, scheduler, save, on_step,
               device="cuda:0", clock=time.monotonic):
    import torch

    groups = validate_groups(items, schedule)
    if settings["max_optimizer_steps"] != 45 or settings["supervised_token_exposures_total"] != schedule["supervised_token_exposures"]:
        raise ValueError("training budget differs")
    started, exposures = clock(), 0

    def check_stop():
        if clock() - started >= settings["max_wall_train_seconds"]:
            raise TimeoutError("training wall-clock limit reached")

    optimizer.zero_grad(set_to_none=True)
    for step, group in enumerate(groups, 1):
        check_stop()
        loss, count = old.accumulate(model, group, device, check_stop)
        old.finite_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], settings["clip_grad_norm"])
        if not torch.isfinite(norm).item():
            raise FloatingPointError("nonfinite gradient norm")
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        exposures += count
        on_step({"step": step, "epoch": (step - 1) // 21 + 1,
                 "sample_ids": [i["sample_id"] for i in group], "loss": loss, "lr": lr,
                 "supervised_tokens": count, "supervised_exposures": exposures,
                 "gradient_norm": float(norm), "elapsed_seconds": clock() - started})
        check_stop()
    save(45)
    check_stop()
    return {"completed": True, "optimizer_steps": 45, "training_microbatches": 180,
            "supervised_exposures": exposures, "complete_epochs": 2, "last_epoch_samples": 12,
            "elapsed_seconds": clock() - started}


class CallLedger:
    """每次前向前fsync；配置错误和预算/顺序错误必须越过逐条异常处理。"""

    def __init__(self, folder, records):
        self.path = folder / "call_attempts.jsonl"
        if any(len(records[split]) != count for _, split, count, _ in CONDITIONS):
            raise DecodeIntegrityError("condition sample counts differ")
        self.expected = [(name, row["sample_id"]) for name, split, count, _ in CONDITIONS for row in records[split]]
        if len(self.expected) != 87:
            raise DecodeIntegrityError("call budget must be exactly 87")

    def entries(self):
        try:
            actual = rows(self.path) if self.path.exists() else []
            if len(actual) > 87:
                raise DecodeIntegrityError("call ledger exceeds budget")
            for index, entry in enumerate(actual):
                if (entry["attempt"] != index + 1
                        or (entry["condition"], entry["sample_id"]) != self.expected[index]
                        or entry["effective_mode"] != "greedy_search"
                        or entry["do_sample"] is not False or entry["use_model_defaults"] is not False):
                    raise DecodeIntegrityError("durable call ledger differs")
        except Exception as exc:
            raise DecodeIntegrityError("unreadable durable call ledger") from exc
        return actual

    def record(self, condition, sample_id):
        index = len(self.entries())
        if index >= 87 or (condition, sample_id) != self.expected[index]:
            raise DecodeIntegrityError("call condition/order/budget differs")
        entry = {"attempt": index + 1, "condition": condition, "sample_id": sample_id,
                 "created_utc": datetime.now(timezone.utc).isoformat(), "effective_mode": "greedy_search",
                 "do_sample": False, "use_model_defaults": False}
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise DecodeIntegrityError("cannot persist call attempt before forward") from exc


class RevisionBackend(old.LoraBackend):
    def __init__(self, plan, api, call_api):
        super().__init__(plan)
        from transformers.generation.utils import GenerationMixin

        if (sha256_file(inspect.getsourcefile(type(self.generation_config))) != api["configuration_source_sha256"]
                or sha256_file(inspect.getsourcefile(GenerationMixin)) != call_api["generation_utils_sha256"]):
            raise DecodeIntegrityError("actual generation implementation differs from regression evidence")
        self.runtime.update({"generation_utils_sha256": call_api["generation_utils_sha256"],
                             "generate_overrides": {"do_sample": False, "use_model_defaults": False}})

    def check_effective(self):
        configuration = effective_config(self.model, self.generation_config, self.plan["evaluation"]["generation"])
        self.runtime.update({"effective_generation_config": configuration.to_dict(),
                             "effective_generation_mode": "greedy_search"})
        return configuration

    def assert_inference(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise DecodeIntegrityError("generation requires a fully frozen eval model")

    def load_base(self):
        super().load_base()
        self.assert_inference()
        self.check_effective()

    def load_adapter(self, path):
        super().load_adapter(path)
        self.assert_inference()
        self.check_effective()

    def close(self):
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        # inspect只构造tokenizer，不能因empty_cache触发CUDA初始化。
        if self.torch.cuda.is_initialized():
            self.torch.cuda.empty_cache()

    def generate(self, ids, before_forward):
        self.assert_inference()
        configuration = self.check_effective()
        before_forward()
        torch = self.torch
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # input_ids/attention_mask: [batch=1, input_tokens]，不截断或padding。
        inputs = torch.tensor([ids], dtype=torch.long, device="cuda:0")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         generation_config=configuration, do_sample=False,
                                         use_model_defaults=False, logits_to_keep=1)
        torch.cuda.synchronize()
        result = completion_record(self.tokenizer, ids, output[0].tolist(),
                                   self.plan["evaluation"]["generation"]["eos_token_id"], 1024)
        result["seconds"] = time.perf_counter() - started
        self.runtime.update(self.memory())
        return result

    def train(self, items, folder, expected_initial, schedule):
        from transformers import get_linear_schedule_with_warmup

        initial = self.new_adapter()
        if initial["initial_adapter_sha256"] != expected_initial or expected_initial != self.plan["expected_initial"]:
            raise ValueError("fresh training initialization differs from frozen preflight")
        self.prepare_training()
        settings = self.plan["training"]
        optimizer = self.torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
            **{k: tuple(v) if k == "betas" else v for k, v in settings["optimizer"].items() if k != "class"})
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=180)
        checkpoints = []

        def save(step):
            path = folder / "checkpoints" / f"step_{step:06d}"
            path.mkdir(parents=True, exist_ok=False)
            self.model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
            self.torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                             "rng": old.rng_state(), "step": step, "epoch": 3, "epoch_sample_offset": 12,
                             "training_schedule_sha256": self.plan["schedule_sha256"],
                             "sample_order": [i for g in schedule["groups"] for i in g["sample_ids"]]},
                            path / "training_state.pt")
            checkpoints.append({"step": step, "files": old.file_tree(path)})

        self.torch.cuda.reset_peak_memory_stats()
        with (folder / "training.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            def on_step(row):
                row.update(self.memory())
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"train step={row['step']}/45 loss={row['loss']:.6f}", flush=True)
            result = train_loop(self.model, items, schedule, settings, optimizer, scheduler, save, on_step)
        return {**result, "initial_adapter": initial, "checkpoints": checkpoints, **self.memory()}


def prepare_inputs(backend, dataset, plan, prompt, root):
    records, encoded, items = {}, {}, {}
    for split, expected, actual_split in (("train", 84, "train"), ("new_diagnostic", 24, "valid"), ("original_valid", 15, "valid")):
        mini = {"records": {actual_split: dataset["records"][split]}, "rule_version": "m5-structure-v1.2",
                "dataset_acceptance": {actual_split + "_samples": expected}}
        records[split] = old.load_records(mini, actual_split, root)
        encoded[split] = old.encoded_inputs(backend, prompt, records[split], plan["training_interface"])
        items[split] = [old.training_item(backend.tokenizer, prompt, row, plan["training_interface"]) for row in records[split]]
    schedule = read(old.inside(root, plan["schedule_path"]))
    validate_groups(items["train"], schedule)
    if schedule["supervised_token_exposures"] != 9657:
        raise ValueError("actual planned exposure must be 9657")
    return records, encoded, items["train"], schedule


def preflight_permission(experiment, execution_sha, plan):
    preflight = old.preflight_permission(experiment, execution_sha)
    return validate_preflight(preflight, plan)


def validate_preflight(preflight, plan):
    if (preflight.get("passed") is not True or preflight.get("optimizer_steps") != 0
            or preflight["initial_adapter_sha256"] != plan["expected_initial"]
            or preflight["forward_backward_pairs"] != 2 or preflight["sequence_tokens"] != 1961
            or not preflight["base_frozen"] or preflight["target_module_count"] != 196
            or preflight["trainable_parameters"] != 5046272
            or preflight["peak_reserved_bytes"] > preflight["reserved_limit_bytes"]):
        raise ValueError("revision preflight initialization/resource evidence differs")
    return preflight


def generate_condition(backend, records, encoded, folder, condition, ledger):
    folder.mkdir(exist_ok=False)
    predictions = []
    with (folder / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row, ids in zip(records, encoded):
            try:
                prediction = backend.generate(ids, lambda: ledger.record(condition, row["sample_id"]))
            except Exception as exc:
                if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
                    raise
                prediction = {"response": "", "raw_completion": "", "generated_token_ids": [],
                              "generated_tokens": 0, "stop_reason": "error", "hit_token_limit": False,
                              "runtime_error": f"{type(exc).__name__}: {exc}", "seconds": None}
            prediction = {"sample_id": row["sample_id"], "input_tokens": len(ids), **prediction}
            predictions.append(prediction)
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{condition} {len(predictions)}/{len(records)} tokens={prediction['generated_tokens']}", flush=True)
    metrics = old.evaluate_output(records, predictions, folder)
    write_json(folder / "runtime.json", backend.runtime)
    return metrics, predictions


def pair_summary(records, predictions, pairs):
    if [p["sample_id"] for p in predictions] != [r["sample_id"] for r in records]:
        raise ValueError("pair prediction order differs")
    correct = {r["sample_id"]: bool(evaluate([(r["input"], r["target"], p["response"])])["counts"]["sample_exact"])
               for r, p in zip(records, predictions)}
    selected = [p for p in pairs if p["role"] == "new_boundary_diagnostic"]
    if len(selected) != 12 or {i for p in selected for i in p["sample_ids"]} != set(correct):
        raise ValueError("diagnostic pair partition differs")
    details = [{"pair_id": p["pair_id"], "phenomenon": p["phenomenon"],
                "complete": all(correct[i] for i in p["sample_ids"])} for p in selected]
    by_phenomenon = {f"P{i}": {"pairs": sum(d["phenomenon"] == f"P{i}" for d in details),
                                 "complete": sum(d["phenomenon"] == f"P{i}" and d["complete"] for d in details)}
                    for i in range(1, 7)}
    return {"pairs": 12, "complete": sum(d["complete"] for d in details),
            "by_phenomenon": by_phenomenon, "details": details}


def comparison_result(metrics, historical, thresholds):
    base, control, candidate = [metrics[name] for name in ("base_diagnostic", "control_diagnostic", "candidate_diagnostic")]
    current = metrics["candidate_original_valid"]
    before = historical["adapter_step_000045"]
    keys = ("json_valid", "schema_valid", "coverage_valid", "block_exact", "sample_exact",
            "nonnull_title_correct", "nonnull_reason_correct")
    nonregression = {key: current["rates"][key] >= before["rates"][key] for key in keys}
    nonregression["macro_f1"] = current["macro_f1"] >= before["macro_f1"]
    directional = {"diagnostic_macro_f1_increased": candidate["macro_f1"] > control["macro_f1"],
                   "diagnostic_block_exact_increased": candidate["rates"]["block_exact"] > control["rates"]["block_exact"],
                   "original_valid_nonregression": nonregression}
    gates = {"control_diagnostic": old.quality_gates(control, base, thresholds),
             "candidate_diagnostic": old.quality_gates(candidate, base, thresholds),
             "candidate_original_valid": old.quality_gates(current, historical["base"], thresholds)}
    return {"diagnostic_only": True, "directional_improvement": all(nonregression.values())
             and directional["diagnostic_macro_f1_increased"] and directional["diagnostic_block_exact_increased"],
            "directional_checks": directional, "gates": gates,
            "quality_passed_both_development_sets": gates["candidate_diagnostic"]["passed"] and gates["candidate_original_valid"]["passed"],
            "optimizer_steps": 45, "supervised_exposures": 9657, "model_generation_calls": 87,
            "test_calls": 0, "automatic_test_allowed": False, "original_selected_step": None}


def run(execution_path, phase="inspect", root=PROJECT, backend_factory=RevisionBackend):
    if phase not in PHASES:
        raise ValueError("only inspect/preflight/train-compare; no test or retry entry")
    execution_path = Path(execution_path).resolve()
    execution_sha = sha256_file(execution_path)
    execution, definition, dataset, plan, api, call_api = load_execution(execution_path, root)
    experiment = old.inside(root, execution["experiment_directory"])
    destination = experiment / phase
    if phase != "inspect" and (destination.exists() or phase == "preflight" and experiment.exists()):
        raise FileExistsError("phase already claimed; no overwrite or retry")
    preflight = preflight_permission(experiment, execution_sha, plan) if phase == "train-compare" else None
    if phase == "train-compare" and read(experiment / "execution.json") != execution:
        raise ValueError("preflight belongs to another execution")
    backend = backend_factory(plan, api, call_api)
    claimed = False
    try:
        prompt = old.inside(root, definition["system_prompt"]).read_text(encoding="utf-8")
        records, encoded, items, schedule = prepare_inputs(backend, dataset, plan, prompt, root)
        inspection = {"phase": "inspect", "model_loaded": False,
                      "samples": {k: len(v) for k, v in records.items()},
                      "blocks": {k: sum(len(r["input"]["blocks"]) for r in v) for k, v in records.items()},
                      "input_token_ranges": {k: [min(map(len, ids)), max(map(len, ids))] for k, ids in encoded.items()},
                      "supervised_tokens_once": sum(i["target_tokens_including_eos"] for i in items),
                      "planned_optimizer_steps": 45, "planned_training_microbatches": 180,
                      "planned_supervised_exposures": schedule["supervised_token_exposures"],
                      "generation_calls": 0, "training_steps": 0, "test_records_read": False}
        if phase == "inspect":
            return inspection
        if phase == "preflight":
            experiment.mkdir(parents=True, exist_ok=False)
            write_json(experiment / "execution.json", execution)
        destination.mkdir(exist_ok=False)
        claimed = True
        write_json(destination / "started.json", {"phase": phase, "execution_sha256": execution_sha})
        write_json(destination / "input_preflight.json", inspection)
        if phase == "preflight":
            result = validate_preflight(backend.gpu_preflight(items), plan)
            write_json(destination / "gpu_preflight.json", result)
            write_json(destination / "runtime.json", backend.runtime)
        else:
            backend.load_base()
            training = backend.train(items, destination, preflight["initial_adapter_sha256"], schedule)
            if (not training.get("completed") or training["optimizer_steps"] != 45
                    or training["training_microbatches"] != 180 or training["supervised_exposures"] != 9657
                    or training["initial_adapter"]["initial_adapter_sha256"] != plan["expected_initial"]
                    or [c["step"] for c in training["checkpoints"]] != [45]):
                raise ValueError("incomplete or different training; no generation allowed")
            write_json(destination / "training_result.json", training)
            write_json(destination / "training_runtime.json", backend.runtime)
            backend.close()
            candidate = destination / "checkpoints/step_000045"
            lock = training["checkpoints"][0]["files"]
            if old.file_tree(candidate) != lock:
                raise ValueError("candidate checkpoint differs before lock")
            write_json(destination / "candidate_lock.json", {"step": 45, "files": lock,
                                                            "training_schedule_sha256": plan["schedule_sha256"]})
            pairs = rows(old.inside(root, dataset["data_directory"]) / "pairs.jsonl")
            ledger = CallLedger(destination, records)
            metrics, pair_scores = {}, {}
            for condition, split, count, adapter_role in CONDITIONS:
                if old.file_tree(candidate) != lock:
                    raise ValueError("candidate checkpoint changed after lock")
                backend = backend_factory(plan, api, call_api)
                if old.encoded_inputs(backend, prompt, records[split], plan["training_interface"]) != encoded[split]:
                    raise ValueError("condition input/tokenizer differs")
                backend.load_base()
                if adapter_role:
                    adapter = candidate if adapter_role == "candidate" else old.inside(root, definition["training_comparison"]["control"]["directory"])
                    if adapter_role == "control" and old.file_tree(adapter) != definition["training_comparison"]["control"]["files"]:
                        raise ValueError("fixed control checkpoint differs")
                    backend.load_adapter(adapter)
                metrics[condition], predictions = generate_condition(backend, records[split], encoded[split],
                                                                      destination / condition, condition, ledger)
                if split == "new_diagnostic":
                    pair_scores[condition] = pair_summary(records[split], predictions, pairs)
                backend.close()
            if len(ledger.entries()) != 87:
                raise DecodeIntegrityError("not all 87 attempts were durably recorded")
            historical = {}
            history = old.inside(root, definition["proposed_evaluation"]["reuse_original_valid_base_and_control"])
            for name in ("base", "adapter_step_000045"):
                before = rows(history / name / "predictions.jsonl")
                if [p["sample_id"] for p in before] != [r["sample_id"] for r in records["original_valid"]]:
                    raise ValueError("historical original valid IDs differ")
                score = evaluate((r["input"], r["target"], p["response"]) for r, p in zip(records["original_valid"], before))
                score.pop("diagnostics")
                if score != read(history / name / "metrics.json"):
                    raise ValueError("historical strict score differs")
                historical[name] = score
            result = comparison_result(metrics, historical, definition["proposed_evaluation"]["minimum_gates"])
            write_json(destination / "pair_summary.json", pair_scores)
            write_json(destination / "result.json", result)
        if sha256_file(execution_path) != execution_sha:
            raise ValueError("execution changed during phase")
        load_execution(execution_path, root)
        old.seal(destination, execution_sha, {"status": "completed", "phase": phase})
        return result
    except BaseException as exc:
        if claimed:
            write_json(destination / "failure.json", {"phase": phase, "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=PROJECT / "configs/m5_lora_revision_v1/runtime.json")
    parser.add_argument("--phase", choices=PHASES, default="inspect")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.execution, args.phase), ensure_ascii=False))


if __name__ == "__main__":
    main()

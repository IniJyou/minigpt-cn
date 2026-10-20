"""固定84条训练拟合诊断：默认CPU inspect，完整状态续行与固定53次评测。"""

import argparse
import contextlib
import copy
import hashlib
import inspect
import json
import math
import os
import random
import time
from pathlib import Path

from minigpt_cn import m5_lora_run as old
from minigpt_cn import m5_train_replay_run as replay
from minigpt_cn.m5_greedy_run_v2 import DecodeIntegrityError
from minigpt_cn.m5_lora_revision_run import pair_summary, read, rows
from minigpt_cn.m5_qwen_run import sha256_file
from minigpt_cn.m5_rule_run import PROJECT, write_json, write_rows

DEFINITION = "configs/m5_train_fit_v1/protocol.json"
DEFINITION_SHA = "98f80e012ae6cf84bc85c2e5b3f6d1ea46497f9a8120dec5bef2eb0b08856111"
FREEZE = "logs/m5_train_fit_definition_20261019_v1/freeze_receipt.json"
SOURCES = ("src/minigpt_cn/m5_train_fit_run.py", "tests/test_m5_train_fit_run.py",
           "scripts/m5_train_fit_wsl.sh", "docs/m5-train-fit-runner.md")
PHASES = ("inspect", "preflight", "train-evaluate")
SPLITS = ("training_replay", "new_diagnostic", "original_valid")
ADAPTER_FIELDS = ("task_type", "inference_mode", "r", "lora_alpha", "lora_dropout", "bias", "lora_bias",
                  "target_modules", "layers_to_transform", "fan_in_fan_out", "init_lora_weights",
                  "use_rslora", "use_dora", "modules_to_save")


def tree_digest(value):
    """摘要含容器、键、名称、dtype/shape/值；不把CPU/GPU存放位置当参数身份。"""
    import numpy as np
    import torch

    h = hashlib.sha256()
    def visit(item):
        h.update(type(item).__name__.encode())
        if torch.is_tensor(item):
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str((item.dtype, item.shape)).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for entry in item:
                visit(entry)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def capture_rng(cuda=False):
    import numpy as np
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if cuda else []}


def restore_rng(state, cuda=False):
    import numpy as np
    import torch

    if bool(state["cuda"]) != cuda:
        raise ValueError("RNG device scope differs")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if cuda:
        torch.cuda.set_rng_state_all(state["cuda"])


def registry(model):
    result = [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)}
              for n, p in model.named_parameters() if p.requires_grad]
    if not result or len({v["name"] for v in result}) != len(result):
        raise ValueError("empty or aliased trainable registry")
    return result


def lr_at_update(step):
    index = step - 1
    return 1e-4 * (index / 10 if index < 10 else max(0, 180 - index) / 170)


def validate_schedule(items, schedule):
    lookup = {i["sample_id"]: i for i in items}
    if len(items) != len(lookup) or len(items) != 84 or len(schedule["groups"]) != 105:
        raise ValueError("84 unique records and105 complete groups required")
    expected = []
    for epoch in range(5):
        part = list(lookup)
        random.Random(42 + epoch).shuffle(part)
        expected.extend(part)
    actual = [s for g in schedule["groups"] for s in g["sample_ids"]]
    if actual != expected or schedule["continued_groups"] != schedule["groups"][45:]:
        raise ValueError("105-step fixed order/suffix differs")
    for step, group in enumerate(schedule["groups"], 1):
        if (group["optimizer_step"] != step or len(group["sample_ids"]) != 4
                or old.supervised_count([lookup[s] for s in group["sample_ids"]]) != group["supervised_tokens"]
                or not math.isclose(group["lr_before_update"], lr_at_update(step), abs_tol=1e-15)):
            raise ValueError("group count/denominator/lr differs")
    return lookup


def validate_optimizer(model, optimizer, state, expected_registry, step, expected_lr):
    import torch

    if registry(model) != expected_registry or len(optimizer.param_groups) != 1:
        raise ValueError("named parameter identity/order differs")
    actual_params = [p for p in model.parameters() if p.requires_grad]
    if [id(p) for p in optimizer.param_groups[0]["params"]] != [id(p) for p in actual_params]:
        raise ValueError("optimizer parameter position differs from frozen constructor")
    saved = state["optimizer"]
    if (len(saved["param_groups"]) != 1 or saved["param_groups"][0]["params"] != list(range(len(actual_params)))
            or set(saved["state"]) != set(range(len(actual_params)))
            or not math.isclose(saved["param_groups"][0]["lr"], expected_lr, abs_tol=1e-15)
            or state["scheduler"]["last_epoch"] != step):
        raise ValueError("optimizer IDs/count/lr or scheduler position differs")
    for key, value in saved["param_groups"][0].items():
        if key not in ("params", "lr") and optimizer.param_groups[0].get(key) != value:
            raise ValueError("saved optimizer settings differ from original constructor")
    if (state["scheduler"].get("_step_count") != step + 1
            or state["scheduler"].get("_last_lr") != [expected_lr]):
        raise ValueError("scheduler update count or next lr differs")
    if "parameter_registry" in state and state["parameter_registry"] != expected_registry:
        raise ValueError("saved parameter names differ even if shapes match")
    for index, parameter in enumerate(actual_params):
        moments = saved["state"][index]
        if set(moments) != {"step", "exp_avg", "exp_avg_sq"} or moments["step"].item() != step:
            raise ValueError("Adam updates or state keys differ")
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = moments[key]
            if tensor.shape != parameter.shape or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
                raise ValueError("Adam moment shape/dtype/value differs")


def restore_optimizer(model, optimizer, scheduler, state, expected_registry):
    step = state["step"]
    validate_optimizer(model, optimizer, state, expected_registry, step, lr_at_update(step + 1))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if (tree_digest(optimizer.state_dict()) != tree_digest(state["optimizer"])
            or tree_digest(scheduler.state_dict()) != tree_digest(state["scheduler"])):
        raise ValueError("loaded optimizer/scheduler state differs")
    optimizer.zero_grad(set_to_none=True)
    # RNG由调用者在模型构造、预检、状态核对全部完成后最后恢复。


class AttemptLedger:
    def __init__(self, path, expected):
        self.path, self.expected, self.committed = path, expected, []
        if path.exists():
            raise DecodeIntegrityError("ledger already exists; no retry/resume")

    def entries(self):
        try:
            entries = rows(self.path) if self.path.exists() else []
        except (OSError, ValueError) as exc:
            raise DecodeIntegrityError("unreadable ledger") from exc
        if entries != self.committed:
            raise DecodeIntegrityError("durable ledger changed")
        return entries

    def record(self, identity):
        index = len(self.entries())
        if index >= len(self.expected) or identity != self.expected[index]:
            raise DecodeIntegrityError("attempt order/identity/budget differs")
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(identity) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise DecodeIntegrityError("cannot persist attempt before forward") from exc
        self.committed.append(copy.deepcopy(identity))


def accumulate_checked(model, items, device, guard, before_forward):
    import torch

    denominator, total = old.supervised_count(items), 0.0
    for item in items:
        guard()
        batch = old.collate([item], device)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()
        before_forward(item["sample_id"])
        with amp:
            loss = model(**batch, use_cache=False, logits_to_keep=0, num_items_in_batch=denominator).loss
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise FloatingPointError("nonfinite or nonscalar loss")
        total += float(loss.detach())
        loss.backward()
        del loss, batch
        guard()
    return total, denominator


def update_loop(model, items, groups, optimizer, scheduler, start_step, initial_exposure,
                before_forward, on_step, guard=lambda: None, device="cuda:0"):
    import torch

    lookup = {i["sample_id"]: i for i in items}
    if len(lookup) != len(items) or not groups:
        raise ValueError("unique items and complete groups required")
    for offset, group in enumerate(groups, start_step + 1):
        if (group["optimizer_step"] != offset or len(group["sample_ids"]) != 4
                or old.supervised_count([lookup[s] for s in group["sample_ids"]]) != group["supervised_tokens"]):
            raise ValueError("group sequence or denominator differs before updates")
    exposures = initial_exposure
    optimizer.zero_grad(set_to_none=True)
    for group in groups:
        guard()
        step = group["optimizer_step"]
        if scheduler.last_epoch != step - 1 or not math.isclose(optimizer.param_groups[0]["lr"], group["lr_before_update"], abs_tol=1e-15):
            raise ValueError("restored learning-rate position differs")
        loss, count = accumulate_checked(model, [lookup[s] for s in group["sample_ids"]], device, guard, before_forward)
        old.finite_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(norm).item():
            raise FloatingPointError("nonfinite gradient norm")
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        guard()
        exposures += count
        on_step({"step": step, "sample_ids": group["sample_ids"], "loss": loss, "lr": lr,
                 "supervised_tokens": count, "supervised_exposures": exposures, "gradient_norm": float(norm)})
    return {"completed": True, "last_step": groups[-1]["optimizer_step"], "additional_optimizer_steps": len(groups),
            "training_microbatches": len(groups) * 4, "supervised_exposures": exposures}


class ResourceGuard:
    def __init__(self, memory, total_bytes, seconds=1800, clock=time.monotonic):
        self.memory, self.total, self.seconds, self.clock = memory, total_bytes, seconds, clock
        self.started = clock()
        if total_bytes <= 1073741824:
            raise MemoryError("invalid device total")

    def __call__(self):
        if self.clock() - self.started >= self.seconds:
            raise TimeoutError("training wall-clock budget reached")
        observed = self.memory()
        if (observed.get("device_total_bytes", self.total) != self.total
                or not 0 <= observed["peak_allocated_bytes"] <= observed["peak_reserved_bytes"]
                or observed["peak_reserved_bytes"] > self.total - 1073741824):
            raise MemoryError("reserved headroom or memory reading integrity failed")
        return observed


def load_execution(path, root=PROJECT):
    execution = read(path)
    required = {*SOURCES, DEFINITION, FREEZE, execution["cpu_checks"], execution["cpu_input_preflight"]}
    bindings = execution["artifacts"]
    if (execution.get("version") != "m5-train-fit-runtime-v1" or execution.get("definition") != DEFINITION
            or bindings.get(DEFINITION) != DEFINITION_SHA or not required <= set(bindings)
            or execution.get("allowed_phases") != list(PHASES) or execution.get("default_phase") != "inspect"):
        raise ValueError("unsupported or incomplete train-fit runtime")
    old.verify_files(root, bindings)
    definition = read(old.inside(root, DEFINITION))
    for inherited in (definition["artifacts"], read(old.inside(root, FREEZE))["artifacts"]):
        if any(bindings.get(name) != sha for name, sha in inherited.items()):
            raise ValueError("missing transitive frozen artifact")
    if not read(old.inside(root, execution["cpu_checks"]))["passed"]:
        raise ValueError("CPU checks not accepted")
    proof = read(old.inside(root, execution["cpu_input_preflight"]))
    if not proof["passed"] or not proof["exact_cpu_continuation_passed"]:
        raise ValueError("exact continuation/actual input proof not accepted")
    plan = read(old.inside(root, "configs/m5_lora_v1/protocol.json"))
    plan["evaluation"]["generation"] = definition["proposed_evaluation"]["generation"]
    if old.file_tree(old.inside(root, definition["control"]["directory"])) != definition["control"]["files"]:
        raise ValueError("fixed continuation checkpoint changed")
    api = read(old.inside(root, "logs/m5_valid_error_review_20261019_v1/generation_api_check.json"))
    call_api = read(old.inside(root, "logs/m5_greedy_acceptance_20261019_v1/call_api_check.json"))
    return execution, definition, plan, api, call_api, proof


def prepare_inputs(backend, definition, root=PROJECT):
    dataset = read(old.inside(root, "configs/m5_lora_revision_v1/dataset.json"))
    replay_def = read(old.inside(root, replay.DEFINITION))
    selection, replay_records = replay.selected_records(replay_def, root)
    records = {"train": rows(old.inside(root, definition["training_data"])), "training_replay": replay_records,
               "new_diagnostic": rows(old.inside(root, dataset["records"]["new_diagnostic"])),
               "original_valid": rows(old.inside(root, dataset["records"]["original_valid"]))}
    prompt = old.inside(root, definition["system_prompt"]).read_text(encoding="utf-8")
    items = [old.training_item(backend.tokenizer, prompt, row, definition["training_interface"]) for row in records["train"]]
    layouts = read(old.inside(root, "data/processed/m5_lora_revision_candidate_v1/token_checks.json"))["layouts"]
    layouts = {entry["sample_id"]: entry for entry in layouts if entry["role"] == "augmented_train"}
    for item in items:
        expected = layouts[item["sample_id"]]
        if (item["input_tokens"] != expected["input_tokens"]
                or item["target_tokens_including_eos"] != expected["target_tokens_including_eos"]
                or len(item["input_ids"]) != expected["sequence_tokens"]):
            raise ValueError("actual training layout differs")
    schedule = read(old.inside(root, definition["planned_schedule"]))
    validate_schedule(items, schedule)
    if (sum(g["supervised_tokens"] for g in schedule["groups"]) != 23010
            or sum(g["supervised_tokens"] for g in schedule["continued_groups"]) != 13353):
        raise ValueError("supervised exposure budget differs")
    actual = rows(old.inside(root, "logs/m5_lora_revision_experiment_20261019_v1/train-compare/training.jsonl"))
    if len(actual) != 45 or any(g["sample_ids"] != a["sample_ids"] or g["supervised_tokens"] != a["supervised_tokens"]
                               or not math.isclose(g["lr_before_update"], a["lr"], abs_tol=1e-15)
                               for g, a in zip(schedule["groups"][:45], actual)):
        raise ValueError("frozen actual45-step prefix differs")
    manifest = read(old.inside(root, definition["proposed_evaluation"]["manifest"]))
    encoded, calls = {}, []
    for split, counts, condition in zip(SPLITS, ((14, 17), (24, 50), (15, 62)), manifest["conditions"]):
        members = records[split]
        if (len(members), sum(len(r["input"]["blocks"]) for r in members)) != counts:
            raise ValueError("evaluation full denominator differs")
        if condition["dataset"] != split or condition["sample_ids"] != [r["sample_id"] for r in members]:
            raise ValueError("fixed evaluation IDs differ")
        encoded[split] = old.encoded_inputs(backend, prompt, members, definition["training_interface"])
        baseline = rows(old.inside(root, condition["control_outputs"]) / "predictions.jsonl")
        score = replay.score_records(members, baseline)
        score.pop("diagnostics")
        if score != condition["control_metrics"]:
            raise ValueError("reused control output/strict metrics differ")
        for record in members:
            calls.append({"attempt": len(calls) + 1, "checkpoint": "continued_step105", "dataset": split,
                          "sample_id": record["sample_id"], "input_sha256": replay.digest(record["input"]),
                          "target_sha256": replay.digest(record["target"]), "target_supplied_to_generation": False})
    if calls != manifest["calls"] or len(calls) != 53:
        raise ValueError("53-call fixed manifest differs")
    return records, items, encoded, schedule, manifest, selection


def constructor_sources():
    from peft import get_peft_model
    from peft.tuners.lora.layer import LoraLayer
    from peft.tuners.lora.model import LoraModel
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    return {name: sha256_file(inspect.getsourcefile(cls)) for name, cls in (
        ("qwen3", Qwen3ForCausalLM), ("get_peft_model", get_peft_model), ("lora_model", LoraModel), ("lora_layer", LoraLayer))}


def meta_model(plan, model_dir):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        base = AutoModelForCausalLM.from_config(config, trust_remote_code=False).to(dtype=torch.bfloat16).eval().requires_grad_(False)
        model = get_peft_model(base, LoraConfig(**{k: plan["adapter"][k] for k in ADAPTER_FIELDS}),
                               autocast_adapter_dtype=plan["adapter"]["autocast_adapter_dtype"])
    if any(p.device.type != "meta" for p in model.parameters()):
        raise ValueError("architecture identity probe must allocate no real model parameters")
    return model


def optimizer_and_scheduler(model, plan):
    import torch
    from transformers import get_linear_schedule_with_warmup

    fields = {k: tuple(v) if k == "betas" else v for k, v in plan["training"]["optimizer"].items() if k != "class"}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], **fields)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=180)
    return optimizer, scheduler


class FitBackend(replay.ReplayBackend):
    def checkpoint_identity(self, definition, schedule, root=PROJECT):
        from peft import PeftModelForCausalLM
        from safetensors.torch import load_file
        from transformers.loss.loss_utils import ForCausalLMLoss, fixed_cross_entropy
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        torch = self.torch
        if torch.cuda.is_initialized():
            raise ValueError("CPU identity inspection must not initialize CUDA")
        evidence = read(old.inside(root, definition["installed_loss_api_evidence"]))["functions"]
        for name, function in (("peft_forward", PeftModelForCausalLM.forward), ("qwen3_forward", Qwen3ForCausalLM.forward),
                               ("causal_lm_loss", ForCausalLMLoss), ("cross_entropy", fixed_cross_entropy)):
            if sha256_file(inspect.getsourcefile(function)) != evidence[name]["file_sha256"]:
                raise ValueError("installed loss/forward implementation changed")
        before_rng = capture_rng()
        model = meta_model(self.plan, self.model_dir)
        expected = registry(model)
        if len(expected) != 392 or any(e["dtype"] != "torch.float32" for e in expected):
            raise ValueError("same original392 FP32 parameter registrations required")
        folder = old.inside(root, definition["control"]["directory"])
        if old.file_tree(folder) != definition["control"]["files"]:
            raise ValueError("source checkpoint tree differs")
        # 仅加载约20MB adapter与Adam/RNG到CPU；绝不读取基座权重张量。
        tensors = load_file(folder / "adapter_model.safetensors", device="cpu")
        state = torch.load(folder / "training_state.pt", map_location="cpu", weights_only=False)
        keys, digest = [], hashlib.sha256()
        for entry in sorted(expected, key=lambda e: e["name"]):
            name = entry["name"]
            key = name.replace(".default.", ".")
            tensor = tensors[key]
            if list(tensor.shape) != entry["shape"] or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all().item():
                raise ValueError("named adapter tensor identity differs")
            digest.update(name.encode())
            digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
            keys.append(key)
        if set(keys) != set(tensors) or digest.hexdigest() != definition["control"]["parameter_sha256"]:
            raise ValueError("trained adapter digest differs")
        optimizer, _ = optimizer_and_scheduler(model, self.plan)
        validate_optimizer(model, optimizer, state, expected, 45, lr_at_update(46))
        if (state["step"] != 45 or state["epoch"] != 3 or state["epoch_sample_offset"] != 12
                or state["sample_order"] != [s for g in schedule["groups"][:45] for s in g["sample_ids"]]
                or state["training_schedule_sha256"] != sha256_file(old.inside(root, "data/processed/m5_lora_revision_candidate_v1/training_schedule.json"))
                or set(state["rng"]) != {"python", "numpy", "torch", "cuda"} or len(state["rng"]["cuda"]) != 1):
            raise ValueError("source45-step counters/cursor/RNG differ")
        result = {"parameter_registry": expected, "registry_sha256": tree_digest(expected),
                  "adapter_sha256": digest.hexdigest(), "optimizer_sha256": tree_digest(state["optimizer"]),
                  "scheduler_sha256": tree_digest(state["scheduler"]), "rng_sha256": tree_digest(state["rng"]),
                  "constructor_sources": constructor_sources(), "optimizer_step": 45, "scheduler_last_epoch": 45,
                  "epoch": 3, "epoch_sample_offset": 12, "architecture_device": "meta",
                  "base_weights_loaded": False, "cuda_initialized": False}
        restore_rng(before_rng)
        return result

    def load_continuation(self, definition, identity, root=PROJECT):
        from safetensors.torch import load_file

        self.load_base()
        initial = self.new_adapter()
        if initial["initial_adapter_sha256"] != definition["treatment"]["expected_initialization_reference_sha256"]:
            raise ValueError("original adapter constructor/initial identity differs")
        if registry(self.model) != identity["parameter_registry"] or constructor_sources() != identity["constructor_sources"]:
            raise ValueError("actual named registration differs from meta identity proof")
        folder = old.inside(root, definition["control"]["directory"])
        if old.file_tree(folder) != definition["control"]["files"]:
            raise ValueError("checkpoint changed before restore")
        tensors = load_file(folder / "adapter_model.safetensors", device="cpu")
        with self.torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    p.copy_(tensors[name.replace(".default.", ".")])
        state = self.torch.load(folder / "training_state.pt", map_location="cpu", weights_only=False)
        if old.adapter_digest(self.model) != identity["adapter_sha256"]:
            raise ValueError("live restored weights differ")
        self.prepare_training()
        optimizer, scheduler = optimizer_and_scheduler(self.model, self.plan)
        restore_optimizer(self.model, optimizer, scheduler, state, identity["parameter_registry"])
        for key in ("optimizer", "scheduler", "rng"):
            if tree_digest(state[key]) != identity[key + "_sha256"]:
                raise ValueError("source state digest changed")
        return optimizer, scheduler, state

    def guarded_memory(self):
        return {**self.memory(), "device_total_bytes": self.torch.cuda.get_device_properties(0).total_memory}

    def continuation_preflight(self, definition, items, identity, folder, root=PROJECT):
        optimizer, scheduler, state = self.load_continuation(definition, identity, root)
        total = self.torch.cuda.get_device_properties(0).total_memory
        guard = ResourceGuard(self.guarded_memory, total)
        longest = max(items, key=lambda item: len(item["input_ids"]))
        if len(longest["input_ids"]) != 1961:
            raise ValueError("preflight longest sequence differs")
        expected = [{"probe": i + 1, "sample_id": longest["sample_id"]} for i in range(2)]
        ledger = AttemptLedger(folder / "forward_backward_attempts.jsonl", expected)
        restore_rng(state["rng"], cuda=True)
        losses = []
        for identity_row in expected:
            self.model.zero_grad(set_to_none=True)
            def before_forward(sample):
                if sample != identity_row["sample_id"]:
                    raise DecodeIntegrityError("preflight sample identity differs")
                ledger.record(identity_row)
            loss, _ = accumulate_checked(self.model, [longest], "cuda:0", guard, before_forward)
            if not old.finite_gradients(self.model):
                raise FloatingPointError("preflight has no nonzero adapter gradient")
            losses.append(loss)
        self.model.zero_grad(set_to_none=True)
        if (old.adapter_digest(self.model) != identity["adapter_sha256"]
                or tree_digest(optimizer.state_dict()) != identity["optimizer_sha256"]
                or tree_digest(scheduler.state_dict()) != identity["scheduler_sha256"]):
            raise ValueError("preflight mutated continuation weights or optimizer")
        restore_rng(state["rng"], cuda=True)
        if tree_digest(capture_rng(cuda=True)) != identity["rng_sha256"]:
            raise ValueError("preflight did not restore all RNG")
        return {"passed": True, "forward_backward_pairs": len(ledger.entries()), "optimizer_steps": 0,
                "source_step": 45, "sequence_tokens": len(longest["input_ids"]), "losses": losses,
                "checkpoint_identity": identity, **guard(), "reserved_limit_bytes": total - 1073741824}

    def continue_training(self, definition, items, schedule, identity, folder, root=PROJECT):
        optimizer, scheduler, state = self.load_continuation(definition, identity, root)
        groups = schedule["continued_groups"]
        if len(groups) != 60 or sum(g["supervised_tokens"] for g in groups) != 13353:
            raise ValueError("60-step continuation budget differs")
        expected = [{"absolute_microbatch": 181 + i, "absolute_step": 46 + i // 4, "sample_id": sample}
                    for i, sample in enumerate(s for group in groups for s in group["sample_ids"])]
        ledger = AttemptLedger(folder / "training_attempts.jsonl", expected)
        guard = ResourceGuard(self.guarded_memory, self.torch.cuda.get_device_properties(0).total_memory)
        restore_rng(state["rng"], cuda=True)
        if tree_digest(capture_rng(cuda=True)) != identity["rng_sha256"]:
            raise ValueError("first update RNG does not match saved45-step state")
        with (folder / "training.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            def before_forward(sample):
                index = len(ledger.entries())
                if index >= 240 or expected[index]["sample_id"] != sample:
                    raise DecodeIntegrityError("training microbatch identity/budget differs")
                ledger.record(expected[index])

            def on_step(entry):
                if registry(self.model) != identity["parameter_registry"]:
                    raise ValueError("trainable names changed during continuation")
                entry.update(guard())
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                print(f"continue step={entry['step']}/105 loss={entry['loss']:.6f}", flush=True)

            result = update_loop(self.model, items, groups, optimizer, scheduler, 45, 9657,
                                 before_forward, on_step, guard)
        if (len(ledger.entries()) != 240 or result["supervised_exposures"] != 23010
                or result["last_step"] != 105 or scheduler.last_epoch != 105):
            raise ValueError("incomplete continuation cannot generate")
        target = folder / "checkpoints/step_000105"
        target.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(target, safe_serialization=True, save_embedding_layers=False)
        self.torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": capture_rng(cuda=True), "step": 105, "epoch": 5, "epoch_sample_offset": 84,
            "next_cursor_zero_based_epoch": 5, "next_cursor_offset": 0,
            "parameter_registry": identity["parameter_registry"],
            "training_schedule_sha256": sha256_file(old.inside(root, definition["planned_schedule"])),
            "sample_order": [s for g in schedule["groups"] for s in g["sample_ids"]]}, target / "training_state.pt")
        return {**result, "checkpoint_files": old.file_tree(target), "checkpoint_step": 105, **guard()}


def inspection(backend, definition, prepared, identity):
    records, items, encoded, schedule, _, _ = prepared
    if backend.torch.cuda.is_initialized():
        raise ValueError("inspect must not initialize CUDA")
    greedy = backend.inspect_greedy()
    if greedy["effective_mode"] != "greedy_search" or greedy["do_sample"] is not False or greedy["use_model_defaults"] is not False:
        raise DecodeIntegrityError("inspect greedy merge differs")
    return {"passed": True, "phase": "inspect", "base_weights_loaded": False, "architecture_device": "meta",
            "adapter_and_optimizer_read_on_cpu": True, "cuda_initialized": False,
            "samples": {k: len(v) for k, v in records.items()},
            "blocks": {k: sum(len(r["input"]["blocks"]) for r in v) for k, v in records.items()},
            "evaluation_input_ranges": {k: [min(map(len, v)), max(map(len, v))] for k, v in encoded.items()},
            "training_sequence_max": max(len(i["input_ids"]) for i in items),
            "training_target_tokens_once": sum(i["target_tokens_including_eos"] for i in items),
            "planned_groups": len(schedule["groups"]), "planned_additional_updates": 60,
            "planned_additional_supervised_tokens": 13353, "planned_generation_calls": 53,
            "checkpoint_identity": identity, "packages": backend.runtime["packages"],
            "effective_generation": greedy, "real_training_steps": 0, "generation_calls": 0, "test_records_read": False}


def generate_set(backend, records, encoded, folder, split, ledger, expected):
    if len(records) != len(encoded):
        raise DecodeIntegrityError("full generation denominator required")
    folder.mkdir(exist_ok=False)
    predictions = []
    with (folder / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for record, ids in zip(records, encoded):
            backend.assert_inference()
            backend.check_effective()
            before = len(ledger.entries())
            if before >= 53 or expected[before]["dataset"] != split or expected[before]["sample_id"] != record["sample_id"]:
                raise DecodeIntegrityError("generation sample/order/53-call budget differs")
            try:
                prediction = backend.generate(ids, lambda index=before: ledger.record(expected[index]))
            except Exception as exc:
                if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
                    raise
                prediction = {"response": "", "raw_completion": "", "generated_token_ids": [], "generated_tokens": 0,
                              "stop_reason": "error", "hit_token_limit": False, "runtime_error": f"{type(exc).__name__}: {exc}"}
            if (len(ledger.entries()) != before + 1 or type(prediction["generated_tokens"]) is not int
                    or not 0 <= prediction["generated_tokens"] <= 1024
                    or len(prediction["generated_token_ids"]) != prediction["generated_tokens"]):
                raise DecodeIntegrityError("generation ledger or token budget differs")
            prediction = {**prediction, "sample_id": record["sample_id"], "input_tokens": len(ids)}
            predictions.append(prediction)
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
    return predictions


def report_set(records, predictions, split, selection, root):
    if split == "training_replay":
        return replay.condition_report(records, predictions, selection)
    score = replay.score_records(records, predictions)
    diagnostics = score.pop("diagnostics")
    samples, title_errors, reason_confusion, illegal = [], {"missing": 0, "spurious": 0, "wrong_nonnull": 0}, {}, []
    for record, prediction, diag in zip(records, predictions, diagnostics):
        samples.append({"sample_id": record["sample_id"], "sample_exact": diag["coverage_valid"] and not diag["errors"], **diag})
        try:
            parsed = read_response(prediction["response"])
        except (ValueError, TypeError):
            parsed = None
        values = parsed.get("annotations") if isinstance(parsed, dict) else None
        if isinstance(values, list):
            illegal.extend({"sample_id": record["sample_id"], "item": a} for a in values
                           if not isinstance(a, dict) or a.get("kind") not in replay.KINDS)
        if diag["coverage_valid"]:
            for gold, actual in zip(record["target"]["annotations"], values):
                title_errors["missing"] += gold["title"] is not None and actual["title"] is None
                title_errors["spurious"] += gold["title"] is None and actual["title"] is not None
                title_errors["wrong_nonnull"] += gold["title"] is not None and actual["title"] is not None and gold["title"] != actual["title"]
                key = json.dumps([gold["reason"], actual["reason"]], ensure_ascii=False)
                reason_confusion[key] = reason_confusion.get(key, 0) + 1
    result = {"metrics": score, "samples": samples, "illegal_kinds": illegal,
              "title_errors_on_covered_samples": title_errors, "reason_confusion_on_covered_samples": reason_confusion}
    if split == "new_diagnostic":
        result["pairs"] = pair_summary(records, predictions, rows(old.inside(root, "data/processed/m5_lora_revision_candidate_v1/pairs.jsonl")))
    return result


def read_response(value):
    from minigpt_cn.m5_structure import parse_json

    return parse_json(value)


def comparison(reports, definition, root=PROJECT):
    baseline = {c["dataset"]: c["control_metrics"] for c in definition["proposed_evaluation"]["conditions"]}
    current = reports["training_replay"]
    before_groups = read(old.inside(root, definition["proposed_evaluation"]["conditions"][0]["control_outputs"]) / "report.json")["strata"]
    improvement = {key: current["metrics"]["counts"][key] > baseline["training_replay"]["counts"][key]
                   for key in ("sample_exact", "block_exact")}
    for group in before_groups:
        for key in ("sample_exact", "block_exact"):
            improvement[group + ":" + key] = current["strata"][group]["counts"][key] >= before_groups[group]["counts"][key]
    development, gates, deltas = {}, {}, {}
    for split in SPLITS:
        score, previous = reports[split]["metrics"], baseline[split]
        deltas[split] = {key: score["rates"][key] - value for key, value in previous["rates"].items()}
        deltas[split]["macro_f1"] = score["macro_f1"] - previous["macro_f1"]
        if split != "training_replay":
            development[split] = {key: score["macro_f1"] >= previous["macro_f1"] if key == "macro_f1"
                                 else score["rates"][key] >= previous["rates"][key]
                                 for key in definition["criteria"]["development_nonregression_metrics"]}
            base = read(old.inside(root, definition["proposed_evaluation"]["reused_base_metrics"][split]))
            gates[split] = old.quality_gates(score, base, definition["criteria"]["development_absolute_quality_gates"])
    return {"diagnostic_only": True, "additional_optimizer_steps": 60, "generation_calls": 53, "test_calls": 0,
            "training_fit_checks": improvement, "training_fit_improved": all(improvement.values()),
            "selected_all_14_fit": current["metrics"]["counts"]["sample_exact"] == 14,
            "development_nonregression": development, "development_both_nonregression": all(all(v.values()) for v in development.values()),
            "development_quality_gates": gates, "metric_deltas": deltas,
            "automatic_test_allowed": False, "original_selected_step": None}


def run(execution_path, phase="inspect", root=PROJECT, backend_factory=FitBackend):
    if phase not in PHASES:
        raise ValueError("only inspect/preflight/train-evaluate; no test/retry/resume")
    execution_path = Path(execution_path).resolve()
    execution_sha = sha256_file(execution_path)
    execution, definition, plan, api, call_api, proof = load_execution(execution_path, root)
    experiment = old.inside(root, definition["future_runtime_requirements"]["planned_output_directory"])
    destination = experiment / phase
    if phase != "inspect" and (destination.exists() or phase == "preflight" and experiment.exists()):
        raise FileExistsError("exclusive phase already claimed; no overwrite/retry/resume")
    if phase == "train-evaluate":
        old.verify_receipt(experiment / "preflight", execution_sha)
        accepted = read(experiment / "preflight/gpu_preflight.json")
        if (not accepted["passed"] or accepted["optimizer_steps"] != 0 or accepted["forward_backward_pairs"] != 2
                or accepted["checkpoint_identity"] != proof["checkpoint_identity"]
                or accepted["peak_reserved_bytes"] > accepted["reserved_limit_bytes"]):
            raise ValueError("valid standalone continuation preflight required")
    backend, claimed = None, False
    try:
        backend = backend_factory(plan, api, call_api)
        prepared = prepare_inputs(backend, definition, root)
        records, items, encoded, schedule, manifest, selection = prepared
        identity = backend.checkpoint_identity(definition, schedule, root)
        if identity != proof["checkpoint_identity"]:
            raise ValueError("actual source45-step identity differs from bound CPU evidence")
        report = inspection(backend, definition, prepared, identity)
        if sha256_file(execution_path) != execution_sha:
            raise ValueError("runtime changed during inspect")
        if phase == "inspect":
            return report
        if phase == "preflight":
            experiment.mkdir(parents=True, exist_ok=False)
            write_json(experiment / "execution.json", execution)
        elif read(experiment / "execution.json") != execution:
            raise ValueError("preflight execution identity differs")
        destination.mkdir(exist_ok=False)
        claimed = True
        write_json(destination / "started.json", {"phase": phase, "execution_sha256": execution_sha})
        write_json(destination / "input_preflight.json", report)
        load_execution(execution_path, root)
        if phase == "preflight":
            result = backend.continuation_preflight(definition, items, identity, destination, root)
            write_json(destination / "gpu_preflight.json", result)
        else:
            training = backend.continue_training(definition, items, schedule, identity, destination, root)
            if (training["last_step"] != 105 or training["additional_optimizer_steps"] != 60
                    or training["training_microbatches"] != 240 or training["supervised_exposures"] != 23010):
                raise ValueError("complete60-step continuation required before generation")
            write_json(destination / "training_result.json", training)
            candidate = destination / "checkpoints/step_000105"
            lock = training["checkpoint_files"]
            if old.file_tree(candidate) != lock:
                raise ValueError("step105 tree differs before lock")
            write_json(destination / "candidate_lock.json", {"step": 105, "files": lock})
            backend.close()
            backend = backend_factory(plan, api, call_api)
            prompt = old.inside(root, definition["system_prompt"]).read_text(encoding="utf-8")
            for split in SPLITS:
                if old.encoded_inputs(backend, prompt, records[split], definition["training_interface"]) != encoded[split]:
                    raise ValueError("generation inputs/tokenizer changed")
            load_execution(execution_path, root)
            backend.load_base()
            backend.load_adapter(candidate)
            expected = [{**call, "effective_mode": "greedy_search", "do_sample": False, "use_model_defaults": False}
                        for call in manifest["calls"]]
            ledger = AttemptLedger(destination / "call_attempts.jsonl", expected)
            reports = {}
            for split in SPLITS:
                if old.file_tree(candidate) != lock:
                    raise ValueError("candidate changed between sets")
                predictions = generate_set(backend, records[split], encoded[split], destination / split, split, ledger, expected)
                reports[split] = report_set(records[split], predictions, split, selection, root)
                write_json(destination / split / "report.json", reports[split])
                write_rows(destination / split / "all_sample_fields.jsonl", reports[split]["samples"])
            if len(ledger.entries()) != 53:
                raise DecodeIntegrityError("all53 attempts required before sealing")
            write_json(destination / "generation_runtime.json", backend.runtime)
            result = comparison(reports, definition, root)
            write_json(destination / "result.json", result)
            if old.file_tree(candidate) != lock:
                raise ValueError("candidate changed before seal")
        load_execution(execution_path, root)
        if sha256_file(execution_path) != execution_sha:
            raise ValueError("runtime changed before seal")
        backend.close()
        old.seal(destination, execution_sha, {"status": "completed", "phase": phase})
        return result
    except BaseException as exc:
        if claimed:
            write_json(destination / "failure.json", {"phase": phase, "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if backend is not None:
            backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=PROJECT / "configs/m5_train_fit_v1/runtime.json")
    parser.add_argument("--phase", choices=PHASES, default="inspect")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.execution, args.phase), ensure_ascii=False))


if __name__ == "__main__":
    main()

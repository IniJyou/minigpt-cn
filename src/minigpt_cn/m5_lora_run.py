"""独立M5 LoRA入口：只读inspect、GPU预检、训练/valid、锁定后test。"""

import argparse
import contextlib
import gc
import hashlib
import json
import random
import time
from pathlib import Path

from minigpt_cn.m5_qwen_run import QwenBackend, messages_for, sha256_file
from minigpt_cn.m5_rule_run import PROJECT, write_json, write_rows
from minigpt_cn.m5_structure import (
    evaluate,
    extract_title,
    parse_json,
    predict_rules,
    valid_prediction,
    validate_input,
)


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escapes the project")
    return path


def verify_files(root, expected):
    for name, digest in expected.items():
        if sha256_file(inside(root, name)) != digest:
            raise ValueError(f"artifact SHA256 differs: {name}")


def load_execution(path, root=PROJECT):
    path = Path(path).resolve()
    execution = parse_json(path.read_bytes())
    if execution.get("version") != "m5-lora-runtime-v1":
        raise ValueError("unsupported execution protocol")
    verify_files(root, execution["artifacts"])
    plan_path = inside(root, execution["plan"])
    if sha256_file(plan_path) != execution["plan_sha256"]:
        raise ValueError("plan SHA256 differs")
    plan = parse_json(plan_path.read_bytes())
    if plan.get("version") != "m5-lora-experiment-v1" or plan["rule_version"] != "m5-structure-v1.2":
        raise ValueError("unsupported experiment/rule version")
    # 原规范只读；新增依赖使用本执行协议，不修改旧基座的完整包清单。
    verify_files(root, plan["artifacts"])
    data = parse_json(inside(root, plan["dataset_protocol"]).read_bytes())
    base = parse_json(inside(root, plan["base_protocol"]).read_bytes())
    verify_files(root, data["artifacts"])
    verify_files(root, base["artifacts"])
    required = set(plan["records"].values()) | {plan["system_prompt"]}
    model = inside(root, plan["model"]["model_directory"])
    required |= {(model / name).relative_to(root).as_posix() for name in base["model_files"]}
    if not required <= set(execution["artifacts"]):
        raise ValueError("model, all splits and prompt must be SHA256-bound")
    if {p.name for p in model.iterdir() if p.is_file()} != set(base["model_files"]):
        raise ValueError("unexpected/missing base model sidecar")
    experiment = inside(root, execution["experiment_directory"])
    if not experiment.is_relative_to((root / "logs").resolve()):
        raise ValueError("experiment must be inside project logs")
    return execution, plan


def load_records(plan, split, root=PROJECT):
    records = [parse_json(line) for line in inside(root, plan["records"][split]).read_text(
        encoding="utf-8").splitlines()]
    expected = plan["dataset_acceptance"][f"{split}_samples"]
    if len(records) != expected or len({r["sample_id"] for r in records}) != expected:
        raise ValueError("sample count/IDs differ")
    for row in records:
        if row["schema_version"] != plan["rule_version"] or row["split"] != split:
            raise ValueError("record version/split differs")
        validate_input(row["input"])
        target = row["target"]
        if not valid_prediction(target):
            raise ValueError("invalid reference")
        if [a["block_id"] for a in target["annotations"]] != [b["block_id"] for b in row["input"]["blocks"]]:
            raise ValueError("reference coverage differs")
        if any(a["title"] != extract_title(b["text"]) for a, b in zip(
                target["annotations"], row["input"]["blocks"])):
            raise ValueError("reference title differs")
    return records


def training_item(tokenizer, prompt, row, interface):
    messages = messages_for(prompt, row["input"])
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    target = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
    ids = tokenizer.encode(prefix + target + "<|im_end|>", add_special_tokens=False)
    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("target tokenization changes the prefix")
    tail = ids[len(prefix_ids):]
    if not tail or tail[-1] != interface["eos_token_id"] or tail.count(interface["eos_token_id"]) != 1:
        raise ValueError("target must end with exactly one EOS")
    if tokenizer.decode(tail[:-1], skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) != target:
        raise ValueError("target JSON token roundtrip differs")
    if (len(prefix_ids) > interface["max_input_tokens"]
            or len(tail) > interface["max_target_tokens_including_eos"]
            or len(ids) > interface["max_sequence_tokens"]):
        raise ValueError("training token budget exceeded; no truncation")
    return {"sample_id": row["sample_id"], "input_ids": ids,
            "labels": [-100] * len(prefix_ids) + tail, "input_tokens": len(prefix_ids),
            "target_tokens_including_eos": len(tail)}


def collate(items, device, pad_token_id=151643):
    import torch

    if not items:
        raise ValueError("empty batch")
    length = max(len(item["input_ids"]) for item in items)
    ids, labels, masks = [], [], []
    for item in items:
        tokens, targets = item["input_ids"], item["labels"]
        if len(tokens) != len(targets) or not tokens:
            raise ValueError("token/label shape differs")
        pad = length - len(tokens)
        ids.append(tokens + [pad_token_id] * pad)
        labels.append(targets + [-100] * pad)
        masks.append([1] * len(tokens) + [0] * pad)
    # input_ids/labels/attention_mask: [B, L]；labels尚未移位。
    return {key: torch.tensor(value, dtype=torch.long, device=device)
            for key, value in (("input_ids", ids), ("labels", labels), ("attention_mask", masks))}


def supervised_count(items):
    count = sum(sum(token != -100 for token in item["labels"][1:]) for item in items)
    if count <= 0:
        raise ValueError("accumulation group has no supervised tokens")
    return count


def accumulate(model, items, device, check_stop=lambda: None):
    import torch

    denominator = supervised_count(items)
    total = 0.0
    for item in items:
        check_stop()
        batch = collate([item], device)
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()
        with amp:
            loss = model(**batch, use_cache=False, logits_to_keep=0,
                         num_items_in_batch=denominator).loss
        if loss.ndim != 0 or not torch.isfinite(loss).item():
            raise FloatingPointError("nonfinite or nonscalar loss")
        total += float(loss.detach())
        # CausalLM已用整组监督token数N作分母；不再除以梯度累积次数。
        loss.backward()
        del loss, batch
        check_stop()
    return total, denominator


def finite_gradients(model):
    import torch

    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise FloatingPointError("missing or nonfinite adapter gradients")
    return any(torch.count_nonzero(g).item() for g in gradients)


def train_loop(model, items, settings, optimizer, scheduler, save, on_step, device="cuda:0",
               clock=time.monotonic):
    import torch

    group_size = settings["gradient_accumulation_steps"]
    if not items or len(items) % group_size:
        raise ValueError("all records must form complete accumulation groups")
    if len(items) // group_size != settings["optimizer_steps_per_epoch"]:
        raise ValueError("optimizer steps per epoch differ")
    started, step, exposures = clock(), 0, 0

    def check_stop():
        if clock() - started >= settings["max_wall_train_seconds"]:
            raise TimeoutError("training wall-clock limit reached")

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(settings["epochs"]):
        order = list(range(len(items)))
        random.Random(settings["seed"] + epoch).shuffle(order)
        for offset in range(0, len(order), group_size):
            check_stop()
            if step >= settings["max_optimizer_steps"]:
                raise ValueError("unexpected extra optimizer step")
            group = [items[i] for i in order[offset:offset + group_size]]
            loss, count = accumulate(model, group, device, check_stop)
            finite_gradients(model)
            parameters = [p for p in model.parameters() if p.requires_grad]
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings["clip_grad_norm"])
            if not torch.isfinite(norm).item():
                raise FloatingPointError("nonfinite gradient norm")
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            exposures += count
            on_step({"step": step, "epoch": epoch + 1, "loss": loss, "lr": lr,
                     "supervised_tokens": count, "supervised_exposures": exposures,
                     "input_token_positions": sum(len(i["input_ids"]) for i in group),
                     "gradient_norm": float(norm), "elapsed_seconds": clock() - started})
            check_stop()
        if epoch + 1 in settings["checkpoint_epochs"]:
            save(step, epoch + 1, [items[i]["sample_id"] for i in order])
            check_stop()
    if step != settings["max_optimizer_steps"] or exposures != settings["supervised_token_exposures_total"]:
        raise ValueError("training completion counters differ")
    return {"completed": True, "optimizer_steps": step, "supervised_exposures": exposures,
            "elapsed_seconds": clock() - started}


def quality_gates(metrics, base_metrics, thresholds):
    keys = {"json_rate": "json_valid", "schema_rate": "schema_valid",
            "coverage_rate": "coverage_valid", "block_all_fields_rate": "block_exact",
            "sample_exact_rate": "sample_exact", "non_null_title_exact_rate": "nonnull_title_correct",
            "non_null_reason_exact_rate": "nonnull_reason_correct"}
    observed = {key: metrics["rates"][value] for key, value in keys.items()}
    observed["macro_f1"] = metrics["macro_f1"]
    observed["macro_f1_gain_over_same_prompt_base"] = metrics["macro_f1"] - base_metrics["macro_f1"]
    checks = {key: observed[key] >= minimum for key, minimum in thresholds.items()}
    return {"passed": all(checks.values()), "observed": observed, "checks": checks}


def choose_checkpoint(candidates, base_metrics, thresholds):
    reviewed = [{**c, "gates": quality_gates(c["metrics"], base_metrics, thresholds)} for c in candidates]
    eligible = [c for c in reviewed if c["gates"]["passed"]]
    eligible.sort(key=lambda c: (-c["metrics"]["macro_f1"], -c["metrics"]["rates"]["block_exact"],
                                 -c["metrics"]["rates"]["sample_exact"], c["step"]))
    return {"accepted": bool(eligible), "selected_step": eligible[0]["step"] if eligible else None,
            "candidates": reviewed}


def file_tree(folder):
    files = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlinks are not accepted in run artifacts")
        if path.is_file():
            files[path.relative_to(folder).as_posix()] = sha256_file(path)
    return files


def seal(folder, execution_sha, details):
    write_json(folder / "receipt.json", {"execution_sha256": execution_sha,
                                       "outputs": file_tree(folder), **details})


def verify_receipt(folder, execution_sha):
    receipt = parse_json((folder / "receipt.json").read_bytes())
    actual = file_tree(folder)
    actual.pop("receipt.json")
    if (receipt.get("execution_sha256") != execution_sha or receipt.get("status") != "completed"
            or receipt.get("outputs") != actual):
        raise ValueError("phase receipt/artifact integrity differs")
    return receipt


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    import numpy as np
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def adapter_digest(model):
    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def check_adapter(model, config):
    import torch

    layers = [name for name, module in model.named_modules()
              if hasattr(module, "lora_A") and hasattr(module, "lora_B")]
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if len(layers) != config["expected_target_module_count"]:
        raise ValueError("injected module count differs")
    if any(not any(part in name.split(".") for part in ("lora_A", "lora_B")) for name, _ in trainable):
        raise ValueError("base/non-adapter parameter is trainable")
    if sum(p.numel() for _, p in trainable) != config["expected_trainable_parameters"]:
        raise ValueError("trainable parameter count differs")
    if any(p.dtype != torch.float32 for _, p in trainable):
        raise ValueError("adapter parameters must be FP32")
    if any(torch.count_nonzero(p).item() for name, p in trainable if ".lora_B." in name):
        raise ValueError("initial LoRA B is not zero")
    return {"target_module_count": len(layers), "trainable_parameters": sum(p.numel() for _, p in trainable),
            "initial_adapter_sha256": adapter_digest(model), "base_frozen": True}


class LoraBackend(QwenBackend):
    def __init__(self, plan):
        environment, model = plan["environment"], plan["model"]
        super().__init__({"environment_prefix": environment["prefix"],
                          "python_version": environment["python_version"],
                          "expected_packages": environment["expected_packages_after_user_install"],
                          "model_directory": model["model_directory"],
                          "chat_template_sha256": model["chat_template_sha256"],
                          "generation": plan["evaluation"]["generation"], "seed_per_sample": 42})
        self.plan = plan

    def load_base(self):
        self.torch.use_deterministic_algorithms(self.plan["training"]["deterministic_algorithms"])
        super().load()

    def new_adapter(self):
        from peft import LoraConfig, get_peft_model

        seed_all(self.plan["training"]["seed"])
        config = self.plan["adapter"]
        fields = ("task_type", "inference_mode", "r", "lora_alpha", "lora_dropout", "bias", "lora_bias",
                  "target_modules", "layers_to_transform", "fan_in_fan_out", "init_lora_weights",
                  "use_rslora", "use_dora", "modules_to_save")
        self.model = get_peft_model(self.model, LoraConfig(**{k: config[k] for k in fields}),
                                    autocast_adapter_dtype=config["autocast_adapter_dtype"])
        result = check_adapter(self.model, config)
        self.runtime["initial_adapter"] = result
        self.runtime["lora_run"] = True
        return result

    def prepare_training(self):
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            "use_reentrant": False, "preserve_rng_state": True})
        self.model.train()

    def load_adapter(self, folder):
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, str(folder), is_trainable=False,
                                               local_files_only=True).eval().requires_grad_(False)
        self.runtime["lora_run"] = True
        self.runtime["adapter_directory"] = str(folder)
        self.runtime["adapter_files"] = file_tree(folder)

    def close(self):
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        self.torch.cuda.empty_cache()

    def memory(self):
        self.torch.cuda.synchronize()
        return {"peak_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": self.torch.cuda.max_memory_reserved()}

    def gpu_preflight(self, items):
        self.load_base()
        initial = self.new_adapter()
        self.prepare_training()
        state = rng_state()
        longest = max(items, key=lambda item: len(item["input_ids"]))
        probes = []
        self.torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(self.plan["gpu_preflight"]["max_forward_backward_pairs"]):
                self.model.zero_grad(set_to_none=True)
                loss, count = accumulate(self.model, [longest], "cuda:0")
                if not finite_gradients(self.model):
                    raise FloatingPointError("all adapter gradients are zero")
                probes.append({"loss": loss, "supervised_tokens": count})
        finally:
            self.model.zero_grad(set_to_none=True)
            restore_rng(state)
        if adapter_digest(self.model) != initial["initial_adapter_sha256"]:
            raise ValueError("preflight modified initial adapter")
        memory = self.memory()
        limit = self.torch.cuda.get_device_properties(0).total_memory - 1073741824
        if memory["peak_reserved_bytes"] > limit:
            raise MemoryError("GPU preflight reserved-memory headroom failed")
        return {"passed": True, **initial, **memory, "reserved_limit_bytes": limit,
                "source_sample_id": longest["sample_id"], "sequence_tokens": len(longest["input_ids"]),
                "forward_backward_pairs": len(probes), "optimizer_steps": 0, "probes": probes}

    def train(self, items, folder, expected_initial):
        from transformers import get_linear_schedule_with_warmup

        initial = self.new_adapter()
        if initial["initial_adapter_sha256"] != expected_initial:
            raise ValueError("training initialization differs from preflight")
        self.prepare_training()
        settings = self.plan["training"]
        options = settings["optimizer"]
        optimizer = self.torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            **{k: tuple(v) if k == "betas" else v for k, v in options.items() if k != "class"})
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=180)
        checkpoints = []

        def save(step, epoch, order):
            destination = folder / "checkpoints" / f"step_{step:06d}"
            destination.mkdir(parents=True, exist_ok=False)
            self.model.save_pretrained(destination, safe_serialization=True, save_embedding_layers=False)
            self.torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                             "rng": rng_state(), "step": step, "epoch": epoch, "sample_order": order},
                            destination / "training_state.pt")
            checkpoints.append({"step": step, "directory": destination.name, "files": file_tree(destination)})

        self.torch.cuda.reset_peak_memory_stats()
        with (folder / "training.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            def on_step(row):
                row.update(self.memory())
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"train step={row['step']}/180 loss={row['loss']:.6f}", flush=True)

            result = train_loop(self.model, items, settings, optimizer, scheduler, save, on_step)
        result.update({"initial_adapter": initial, "checkpoints": checkpoints, **self.memory()})
        return result


def encoded_inputs(backend, prompt, records, interface):
    encoded = [backend.encode(messages_for(prompt, r["input"])) for r in records]
    if any(not ids or len(ids) > interface["max_input_tokens"] for ids in encoded):
        raise ValueError("input budget exceeded; no truncation")
    if interface["max_input_tokens"] + interface["max_target_tokens_including_eos"] > backend.context_limit:
        raise ValueError("input/output budget exceeds context")
    return encoded


def evaluate_output(records, predictions, folder):
    if len(records) != len(predictions):
        raise ValueError("all samples must remain in evaluation denominators")
    metrics = evaluate((r["input"], r["target"], p["response"]) for r, p in zip(records, predictions))
    diagnostics = metrics.pop("diagnostics")
    write_rows(folder / "errors.jsonl", [
        {"sample_id": r["sample_id"], "input": r["input"], "raw_response": p["response"], **d}
        for r, p, d in zip(records, predictions, diagnostics) if d["problems"] or d["errors"]])
    write_json(folder / "metrics.json", metrics)
    return metrics


def evaluate_model(backend, prompt, records, folder):
    folder.mkdir(parents=True, exist_ok=False)
    encoded = encoded_inputs(backend, prompt, records, backend.plan["training_interface"])
    predictions = []
    with (folder / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row, ids in zip(records, encoded):
            try:
                prediction = backend.generate(list(ids))
            except Exception as exc:
                prediction = {"response": "", "raw_completion": "", "generated_token_ids": [],
                              "generated_tokens": 0, "stop_reason": "error", "hit_token_limit": False,
                              "runtime_error": f"{type(exc).__name__}: {exc}", "seconds": None}
                if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
                    stream.write(json.dumps({"sample_id": row["sample_id"], **prediction}) + "\n")
                    stream.flush()
                    raise
            prediction = {"sample_id": row["sample_id"], "input_tokens": len(ids), **prediction}
            predictions.append(prediction)
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{folder.name} {len(predictions)}/{len(records)} "
                  f"tokens={prediction['generated_tokens']}", flush=True)
    metrics = evaluate_output(records, predictions, folder)
    write_json(folder / "runtime.json", backend.runtime)
    return metrics


def evaluate_rule(records, folder):
    folder.mkdir()
    predictions = [{"sample_id": r["sample_id"], "response": json.dumps(predict_rules(r["input"]),
                                                                      ensure_ascii=False)} for r in records]
    write_rows(folder / "predictions.jsonl", predictions)
    return evaluate_output(records, predictions, folder)


def preflight_permission(experiment, execution_sha):
    verify_receipt(experiment / "preflight", execution_sha)
    preflight = parse_json((experiment / "preflight/gpu_preflight.json").read_bytes())
    if preflight.get("passed") is not True or preflight.get("optimizer_steps") != 0:
        raise ValueError("GPU preflight was not accepted")
    return preflight


def test_permission(experiment, execution_sha, max_steps):
    preflight_permission(experiment, execution_sha)
    verify_receipt(experiment / "train-valid", execution_sha)
    selection = parse_json((experiment / "train-valid/selection.json").read_bytes())
    training = parse_json((experiment / "train-valid/training_result.json").read_bytes())
    if not selection["accepted"] or not training["completed"] or training["optimizer_steps"] != max_steps:
        raise ValueError("test requires completed training and accepted locked validation")
    step = selection["selected_step"]
    candidate = next(c for c in selection["candidates"] if c["step"] == step)
    checkpoint = experiment / "train-valid/checkpoints" / f"step_{step:06d}"
    if file_tree(checkpoint) != candidate["checkpoint_files"]:
        raise ValueError("selected checkpoint differs from lock")
    return selection, checkpoint


def run(execution_path, phase="inspect", root=PROJECT, backend_factory=LoraBackend):
    execution_path = Path(execution_path).resolve()
    execution_sha = sha256_file(execution_path)
    execution, plan = load_execution(execution_path, root)
    experiment = inside(root, execution["experiment_directory"])
    if phase not in ("inspect", "preflight", "train-valid", "test"):
        raise ValueError("unsupported phase")
    destination = experiment / phase
    selection, checkpoint = None, None
    if phase == "test":
        # 在读取test记录、tokenize或构造backend之前检查锁和前置结果。
        selection, checkpoint = test_permission(experiment, execution_sha, plan["training"]["max_optimizer_steps"])
    elif phase == "train-valid":
        preflight_permission(experiment, execution_sha)
    if phase != "inspect" and (destination.exists() or phase == "preflight" and experiment.exists()):
        raise FileExistsError("phase already claimed; no overwrite or retry")
    splits = ("test",) if phase == "test" else ("train", "valid")
    records = {split: load_records(plan, split, root) for split in splits}
    prompt = inside(root, plan["system_prompt"]).read_text(encoding="utf-8")
    backend = backend_factory(plan)
    encoded = {split: encoded_inputs(backend, prompt, rr, plan["training_interface"])
               for split, rr in records.items()}
    items = [training_item(backend.tokenizer, prompt, r, plan["training_interface"])
             for r in records.get("train", [])]
    if items and sum(i["target_tokens_including_eos"] for i in items) != plan["training"]["supervised_tokens_per_epoch"]:
        raise ValueError("supervised training token count differs")
    if phase == "inspect":
        return {"phase": phase, "model_loaded": False,
                "samples": {s: len(rr) for s, rr in records.items()},
                "input_token_ranges": {s: [min(map(len, ids)), max(map(len, ids))] for s, ids in encoded.items()},
                "supervised_tokens": sum(i["target_tokens_including_eos"] for i in items)}
    if phase == "preflight":
        experiment.mkdir(parents=True, exist_ok=False)
        write_json(experiment / "execution.json", execution)
    elif parse_json((experiment / "execution.json").read_bytes()) != execution:
        raise ValueError("experiment belongs to a different execution")
    destination.mkdir(exist_ok=False)
    write_json(destination / "started.json", {"phase": phase, "execution_sha256": execution_sha})
    try:
        if phase == "preflight":
            result = backend.gpu_preflight(items)
            if not result.get("passed") or result.get("optimizer_steps") != 0:
                raise ValueError("GPU preflight did not pass without optimizer steps")
            write_json(destination / "gpu_preflight.json", result)
        elif phase == "train-valid":
            backend.load_base()
            base_metrics = evaluate_model(backend, prompt, records["valid"], destination / "valid_base")
            evaluate_rule(records["valid"], destination / "valid_rule")
            preflight = parse_json((experiment / "preflight/gpu_preflight.json").read_bytes())
            training = backend.train(items, destination, preflight["initial_adapter_sha256"])
            if (not training.get("completed")
                    or training.get("optimizer_steps") != plan["training"]["max_optimizer_steps"]
                    or training.get("supervised_exposures") != plan["training"]["supervised_token_exposures_total"]
                    or [c["step"] for c in training["checkpoints"]] != plan["training"]["checkpoint_optimizer_steps"]):
                raise ValueError("training is incomplete; validation selection is forbidden")
            write_json(destination / "training_result.json", training)
            backend.close()
            candidates = []
            for step in plan["training"]["checkpoint_optimizer_steps"]:
                adapter = destination / "checkpoints" / f"step_{step:06d}"
                files = file_tree(adapter)
                if files != next(c["files"] for c in training["checkpoints"] if c["step"] == step):
                    raise ValueError("checkpoint changed before validation")
                backend = backend_factory(plan)
                backend.load_base()
                backend.load_adapter(adapter)
                metrics = evaluate_model(backend, prompt, records["valid"], destination / f"valid_step_{step:06d}")
                candidates.append({"step": step, "metrics": metrics, "checkpoint_files": files})
                backend.close()
            result = choose_checkpoint(candidates, base_metrics, plan["validation"]["minimum_gates"])
            result["base_metrics"] = base_metrics
            write_json(destination / "selection.json", result)
        else:
            backend.load_base()
            base_metrics = evaluate_model(backend, prompt, records["test"], destination / "base")
            backend.close()
            backend = backend_factory(plan)
            backend.load_base()
            backend.load_adapter(checkpoint)
            metrics = evaluate_model(backend, prompt, records["test"], destination / "adapter")
            evaluate_rule(records["test"], destination / "rule")
            result = {"selected_step": selection["selected_step"],
                      "gates": quality_gates(metrics, base_metrics, plan["validation"]["minimum_gates"]),
                      "base_metrics": base_metrics, "adapter_metrics": metrics}
            write_json(destination / "result.json", result)
        if sha256_file(execution_path) != execution_sha:
            raise ValueError("execution protocol changed during phase")
        load_execution(execution_path, root)
        seal(destination, execution_sha, {"status": "completed", "phase": phase})
        return result
    except BaseException as exc:
        write_json(destination / "failure.json", {"phase": phase, "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=PROJECT / "configs/m5_lora_v1/runtime.json")
    parser.add_argument("--phase", choices=("inspect", "preflight", "train-valid", "test"), default="inspect")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.execution, args.phase), ensure_ascii=False))


if __name__ == "__main__":
    main()

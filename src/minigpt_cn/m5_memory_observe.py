"""有界资源观测：默认纯文件inspect，probe最多6更新/25前后向，无生成或权重保存。"""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
CONTRACT = "logs/m5_memory_observability_definition_20261019_v1/observation_contract.json"
FREEZE = "logs/m5_memory_observability_definition_20261019_v1/freeze_receipt.json"
OLD_RUNTIME = "configs/m5_train_fit_v1/runtime.json"
RUNTIME = "configs/m5_memory_observe_v1/runtime.json"
OUTPUT = "logs/m5_memory_observe_experiment_20261019_v1"
SOURCES = ("src/minigpt_cn/m5_memory_observe.py", "tests/test_m5_memory_observe.py",
           "scripts/m5_memory_observe_wsl.sh", "docs/m5-memory-observe-runner.md")
PREDICATES = ("device_total_changed", "peak_relation_invalid", "reserved_headroom_exceeded")
MARGIN = 1073741824


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact outside project")
    return path


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class EvidenceWriteError(RuntimeError):
    """写入失败即停；异常链保留原模型/资源异常。"""


class Journal:
    def __init__(self, path, context, sync=os.fsync):
        self.path, self.context, self.sync = Path(path), context, sync
        self.committed = []
        with self.path.open("x", encoding="utf-8"):
            pass

    def emit(self, event, primary=None, **fields):
        if event not in ("guard_enter", "guard_result"):
            self.context["stage"] = event
        row = {**copy.deepcopy(self.context), "event_id": len(self.committed) + 1, "event": event,
               "utc": datetime.now(timezone.utc).isoformat(), "monotonic_ns": time.monotonic_ns(), **fields}
        try:
            existing = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
            if existing != self.committed:
                raise OSError("durable observation journal changed")
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                self.sync(stream.fileno())
        except BaseException as exc:
            print(f"evidence write failed: {type(exc).__name__}: {exc}; primary={primary!r}", file=sys.stderr)
            raise EvidenceWriteError("cannot persist observation; no next model operation") from (primary or exc)
        self.committed.append(copy.deepcopy(row))
        return row["event_id"]


def predicates(observed, total):
    tests = (lambda: observed.get("device_total_bytes", total) != total,
             lambda: not 0 <= observed["peak_allocated_bytes"] <= observed["peak_reserved_bytes"],
             lambda: observed["peak_reserved_bytes"] > total - MARGIN)
    results, errors = {}, {}
    for name, test in zip(PREDICATES, tests):
        try:
            results[name] = bool(test())
        except Exception as exc:
            results[name], errors[name] = None, exc
    # 保持旧短路判定的异常优先级，同时记录其余可判定条件。
    for name in PREDICATES:
        if name in errors:
            return results, errors[name]
        if results[name]:
            return results, MemoryError("reserved headroom or memory reading integrity failed")
    return results, None


class MemorySampler:
    """与冻结memory相同的同步/读取顺序；状态记录仅操作CPU字典。"""

    def __init__(self, cuda):
        self.cuda = cuda
        self.reset_status()

    def reset_status(self):
        self.status = {"synchronize_status": "not_started", "memory_read_status": "not_started",
                       "partial_observed": {}}

    def __call__(self):
        self.status["synchronize_status"] = "entered"
        self.cuda.synchronize()
        self.status["synchronize_status"] = "returned"
        values = self.status["partial_observed"]
        for name, getter in (("peak_allocated_bytes", self.cuda.max_memory_allocated),
                             ("peak_reserved_bytes", self.cuda.max_memory_reserved),
                             ("device_total_bytes", lambda: self.cuda.get_device_properties(0).total_memory)):
            self.status["memory_read_status"] = name + ":entered"
            values[name] = getter()
            self.status["memory_read_status"] = name + ":returned"
        self.status["memory_read_status"] = "returned"
        return dict(values)


class ObservedGuard:
    def __init__(self, memory, total, journal, clock=time.monotonic):
        self.memory, self.total, self.journal, self.clock = memory, total, journal, clock
        self.started, self.samples, self.last_valid = clock(), 0, None
        if total <= MARGIN:
            raise MemoryError("invalid device total")

    def __call__(self, stage):
        self.journal.context["stage"] = stage
        self.journal.emit("guard_enter")
        if hasattr(self.memory, "reset_status"):
            self.memory.reset_status()
        observed, results = None, dict.fromkeys(PREDICATES)
        elapsed = self.clock() - self.started
        try:
            if elapsed >= 1800:
                raise TimeoutError("training wall-clock budget reached")
            if self.samples >= 69:
                raise ValueError("guard sample budget exhausted")
            self.samples += 1
            observed = copy.deepcopy(self.memory())
            results, error = predicates(observed, self.total)
            if error is not None:
                raise error
        except BaseException as exc:
            self.journal.emit("guard_result", primary=exc, passed=False,
                              guard_elapsed_seconds=elapsed, observed=observed,
                              expected_total_bytes=self.total, reserved_limit_bytes=self.total - MARGIN,
                              predicate_results=results, last_valid_event_id=self.last_valid,
                              **copy.deepcopy(getattr(self.memory, "status", {})),
                              exception_type=type(exc).__name__, exception_message=str(exc),
                              traceback=traceback.format_exc())
            raise
        self.last_valid = self.journal.emit(
            "guard_result", passed=True, guard_elapsed_seconds=elapsed, observed=observed,
            expected_total_bytes=self.total, reserved_limit_bytes=self.total - MARGIN,
            predicate_results=results, **copy.deepcopy(getattr(self.memory, "status", {})))
        return observed


def bounded_loop(model, items, groups, optimizer, scheduler, guard, journal, before_forward, on_step,
                 device="cuda:0"):
    import torch

    from minigpt_cn import m5_train_fit_run as fit

    lookup = {item["sample_id"]: item for item in items}
    if len(lookup) != len(items) or len(groups) != 7:
        raise ValueError("unique items and exactly seven full groups required")
    for step, group in enumerate(groups, 46):
        if (group["optimizer_step"] != step or len(group["sample_ids"]) != 4
                or fit.old.supervised_count([lookup[s] for s in group["sample_ids"]]) != group["supervised_tokens"]):
            raise ValueError("full group order/count/denominator differs")
    ctx, attempts, updates, exposure = journal.context, 0, 0, 9657
    optimizer.zero_grad(set_to_none=True)
    for group in groups:
        step = group["optimizer_step"]
        ctx.update(absolute_step=step, absolute_microbatch=None, sample_id=None, microbatch_in_group=None,
                   input_tokens=None, target_tokens_including_eos=None, sequence_tokens=None,
                   group_denominator=group["supervised_tokens"], optimizer_step_entered=False,
                   optimizer_step_returned=False, scheduler_step_returned=False)
        guard("group_start")
        if scheduler.last_epoch != step - 1 or not math.isclose(
                optimizer.param_groups[0]["lr"], group["lr_before_update"], abs_tol=1e-15):
            raise ValueError("restored learning-rate position differs")
        full_items = [lookup[s] for s in group["sample_ids"]]
        denominator, loss_sum = fit.old.supervised_count(full_items), 0.0
        for position, item in enumerate(full_items, 1):
            if attempts >= 25 or updates > 6:
                raise ValueError("bounded probe would exceed model budget")
            ctx.update(absolute_microbatch=181 + attempts, sample_id=item["sample_id"],
                       microbatch_in_group=position, input_tokens=item.get("input_tokens"),
                       target_tokens_including_eos=item.get("target_tokens_including_eos"),
                       sequence_tokens=len(item["input_ids"]))
            guard("before_collate")
            batch = fit.old.collate([item], device)
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else contextlib.nullcontext()
            before_forward(item["sample_id"])
            attempts += 1
            journal.emit("forward_enter")
            with amp:
                # input_ids/labels: [1,T]；分母来自完整四条组，末尾n08仍用417。
                loss = model(**batch, use_cache=False, logits_to_keep=0, num_items_in_batch=denominator).loss
            journal.emit("forward_returned")
            if loss.ndim != 0 or not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite or nonscalar loss")
            loss_sum += float(loss.detach())
            journal.emit("backward_enter")
            loss.backward()
            journal.emit("backward_returned")
            del loss, batch
            guard("after_backward_delete")
            if attempts == 25:
                if step != 52 or position != 1 or updates != 6:
                    raise ValueError("partial-group stop boundary differs")
                return {"status": "bounded_probe_complete_no_reproduction", "optimizer_updates": updates,
                        "forward_backward_attempts": attempts, "guard_samples": guard.samples,
                        "last_completed_absolute_step": 51, "supervised_exposures": exposure,
                        "partial_group_denominator": denominator, "step52_optimizer_executed": False,
                        "generation_calls": 0, "test_calls": 0, "training_complete": False}
        fit.old.finite_gradients(model)
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(norm).item():
            raise FloatingPointError("nonfinite gradient norm")
        lr = optimizer.param_groups[0]["lr"]
        ctx["optimizer_step_entered"] = True
        journal.emit("optimizer_enter")
        optimizer.step()
        ctx["optimizer_step_returned"] = True
        ctx["last_optimizer_step_returned"] = step
        journal.emit("optimizer_returned")
        journal.emit("scheduler_enter")
        scheduler.step()
        ctx["scheduler_step_returned"] = True
        journal.emit("scheduler_returned")
        optimizer.zero_grad(set_to_none=True)
        updates += 1
        ctx["last_completed_absolute_step"] = step
        guard("after_optimizer_zero_grad")
        exposure += denominator
        observed = guard("on_step")  # 冻结入口on_step内部原有的第二次检查。
        on_step({"step": step, "sample_ids": group["sample_ids"], "loss": loss_sum, "lr": lr,
                 "supervised_tokens": denominator, "supervised_exposures": exposure,
                 "gradient_norm": float(norm), **observed})
    raise ValueError("probe failed to reach fixed partial-group stop")


def inspect_runtime(path, root=PROJECT):
    execution = read(path)
    bindings = execution["artifacts"]
    required = {*SOURCES, CONTRACT, FREEZE, OLD_RUNTIME, execution["cpu_checks"]}
    if (execution.get("version") != "m5-memory-observe-runtime-v1"
            or execution.get("allowed_phases") != ["inspect", "probe"]
            or execution.get("default_phase") != "inspect" or execution.get("output_directory") != OUTPUT
            or not required <= set(bindings)):
        raise ValueError("unsupported or incomplete observation runtime")
    for name, expected in bindings.items():
        if sha(inside(root, name)) != expected:
            raise ValueError("bound artifact changed: " + name)
    frozen = read(root / FREEZE)
    if any(bindings.get(name) != value for name, value in frozen["artifacts"].items()):
        raise ValueError("missing transitive artifact")
    proof = read(inside(root, execution["cpu_checks"]))
    if not proof["passed"] or not proof["exact_cpu_trajectory_passed"]:
        raise ValueError("CPU observation evidence required")
    contract = read(root / CONTRACT)
    if sha(root / OLD_RUNTIME) != contract["old_runtime_sha256"]:
        raise ValueError("old runtime changed")
    proposal = contract["future_probe_proposal_not_authorized"]
    if (proposal["max_optimizer_updates"] != 6 or proposal["max_forward_backward_attempts"] != 25
            or proposal["max_guard_samples_if_no_failure"] != 69
            or proposal["first_partial_group_denominator"] != 417
            or proposal["generation_calls"] != 0 or proposal["extra_preflight_calls"] != 0):
        raise ValueError("probe contract differs")
    definition = read(root / "configs/m5_train_fit_v1/protocol.json")
    schedule = read(inside(root, definition["planned_schedule"]))
    layouts = {row["sample_id"]: row for row in
               read(root / "data/processed/m5_lora_revision_candidate_v1/token_checks.json")["layouts"]
               if row["role"] == "augmented_train"}
    groups = schedule["continued_groups"][:7]
    expected = [{"absolute_microbatch": 181 + i, "absolute_step": 46 + i // 4, "sample_id": sample}
                for i, sample in enumerate(s for g in groups for s in g["sample_ids"])][:25]
    reviewed = read(root / "logs/m5_memory_observability_definition_20261019_v1/review.json")["sample_layouts"]
    if (len(groups) != 7 or len(expected) != 25 or groups[-1]["supervised_tokens"] != 417
            or expected[-1]["sample_id"] != "m5-sft-v1-real-n08"):
        raise ValueError("fixed observation suffix differs")
    for wanted, prior in zip(expected, reviewed):
        layout = layouts[wanted["sample_id"]]
        if (any(prior[k] != v for k, v in wanted.items())
                or any(prior[k] != layout[k] for k in ("input_tokens", "target_tokens_including_eos", "sequence_tokens"))):
            raise ValueError("reviewed layout/order differs")
    return {"passed": True, "phase": "inspect", "runtime_sha256": sha(path), "bound_artifacts": len(bindings),
            "groups": groups, "expected_attempts": expected, "layouts": [layouts[r["sample_id"]] for r in expected],
            "max_optimizer_updates": 6, "max_forward_backward_attempts": 25, "max_guard_samples": 69,
            "partial_group_denominator": 417, "model_weights_loaded": False, "cuda_initialized": False,
            "generation_calls": 0, "test_calls": 0, "output_created": False}


def run(path, phase="inspect", root=PROJECT, backend_factory=None):
    if phase not in ("inspect", "probe"):
        raise ValueError("only inspect/probe; no train/evaluate/resume/retry")
    destination = inside(root, OUTPUT)
    if phase == "probe" and destination.exists():
        raise FileExistsError("exclusive probe already claimed; no overwrite/retry/resume")
    report = inspect_runtime(path, root)
    if phase == "inspect":
        return report
    from minigpt_cn import m5_train_fit_run as fit

    _, definition, plan, api, call_api, proof = fit.load_execution(root / OLD_RUNTIME, root)
    backend, claimed, journal = None, False, None
    try:
        backend = (backend_factory or fit.FitBackend)(plan, api, call_api)
        prepared = fit.prepare_inputs(backend, definition, root)
        _, items, _, schedule, _, _ = prepared
        identity = backend.checkpoint_identity(definition, schedule, root)
        if identity != proof["checkpoint_identity"]:
            raise ValueError("source step45 identity differs")
        # 再核对实际编码；本入口不能只信布局文件而执行不同的序列。
        lookup = {item["sample_id"]: item for item in items}
        for expected, layout in zip(report["expected_attempts"], report["layouts"]):
            item = lookup[expected["sample_id"]]
            if (item["input_tokens"] != layout["input_tokens"]
                    or item["target_tokens_including_eos"] != layout["target_tokens_including_eos"]
                    or len(item["input_ids"]) != layout["sequence_tokens"]):
                raise ValueError("actual probe input differs")
        if sha(path) != report["runtime_sha256"]:
            raise ValueError("runtime changed before probe")
        destination.mkdir(parents=True, exist_ok=False)
        claimed = True
        write(destination / "started.json", {"phase": "probe", "runtime_sha256": report["runtime_sha256"]})
        write(destination / "input_inspect.json", report)
        context = {"phase": "probe", "device": "cuda:0", "stage": "restore",
                   "last_completed_absolute_step": 45, "last_optimizer_step_returned": 45}
        journal = Journal(destination / "observations.jsonl", context)
        journal.emit("restore_enter")
        optimizer, scheduler, state = backend.load_continuation(definition, identity, root)
        journal.emit("restore_returned")
        sampler = MemorySampler(backend.torch.cuda)
        guard = ObservedGuard(sampler, backend.torch.cuda.get_device_properties(0).total_memory, journal)
        ledger = fit.AttemptLedger(destination / "training_attempts.jsonl", report["expected_attempts"])
        fit.restore_rng(state["rng"], cuda=True)
        if fit.tree_digest(fit.capture_rng(cuda=True)) != identity["rng_sha256"]:
            raise ValueError("first update RNG differs")
        with (destination / "training.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            def before_forward(sample):
                index = len(ledger.entries())
                if index >= 25 or report["expected_attempts"][index]["sample_id"] != sample:
                    raise ValueError("probe attempt identity/budget differs")
                ledger.record(report["expected_attempts"][index])

            def on_step(entry):
                if fit.registry(backend.model) != identity["parameter_registry"]:
                    raise ValueError("trainable registration changed")
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(f"observe step={entry['step']}/51 loss={entry['loss']:.6f}", flush=True)

            result = bounded_loop(backend.model, items, report["groups"], optimizer, scheduler, guard,
                                  journal, before_forward, on_step)
        if len(ledger.entries()) != 25 or guard.samples != 69 or scheduler.last_epoch != 51:
            raise ValueError("incomplete bounded observation")
        if sha(path) != report["runtime_sha256"]:
            raise ValueError("runtime changed during probe")
        journal.emit("bounded_probe_end", result=result)
        write(destination / "result.json", result)
        closing, backend = backend, None
        closing.close()
        files = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
        write(destination / "receipt.json", {"status": result["status"], "phase": "probe",
                                              "runtime_sha256": report["runtime_sha256"], "outputs": files})
        return result
    except BaseException as exc:
        details = {"phase": "probe", "error_type": type(exc).__name__, "error": str(exc),
                   "traceback": traceback.format_exc(), "context": copy.deepcopy(journal.context) if journal else None,
                   "generation_calls": 0, "training_complete": False}
        if claimed:
            try:
                write(destination / "failure.json", details)
            except BaseException as write_error:
                print(f"failure record unavailable: {write_error!r}; primary={exc!r}", file=sys.stderr)
        raise
    finally:
        if backend is not None:
            primary = sys.exc_info()[1]
            try:
                backend.close()
            except BaseException as cleanup_error:
                print(f"cleanup failed: {cleanup_error!r}; primary={primary!r}", file=sys.stderr)
                if primary is None:
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=PROJECT / RUNTIME)
    parser.add_argument("--phase", choices=("inspect", "probe"), default="inspect")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.execution, args.phase), ensure_ascii=False))


if __name__ == "__main__":
    main()

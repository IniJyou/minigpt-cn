"""独立子进程的单项allocator候选；默认标准库inspect，显式probe另行授权。"""

import argparse
import copy
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
import warnings
from pathlib import Path

from minigpt_cn import m5_memory_observe as observe

PROJECT = Path(__file__).resolve().parents[2]
FREEZE = "logs/m5_memory_observe_probe_20261019_v1/freeze_receipt.json"
API_PROOF = "logs/m5_allocator_runner_20261019_v1/local_api.json"
RUNTIME = "configs/m5_allocator_v1/runtime.json"
OUTPUT = "logs/m5_allocator_experiment_20261019_v1"
CANDIDATE = "expandable_segments:True"
ENV_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF",
            "PYTORCH_NO_CUDA_MEMORY_CACHING", "PYTORCH_NO_HIP_MEMORY_CACHING")
SOURCES = ("src/minigpt_cn/m5_allocator_run.py", "tests/test_m5_allocator_run.py",
           "scripts/m5_allocator_wsl.sh", "docs/m5-allocator-runner.md")


def environment_plan(environ):
    inherited = {key: environ.get(key) for key in ENV_KEYS}
    if inherited["PYTORCH_CUDA_ALLOC_CONF"] not in (None, "", CANDIDATE):
        raise ValueError("inherited CUDA allocator configuration conflicts with single candidate")
    if any(inherited[key] not in (None, "") for key in ENV_KEYS[1:]):
        raise ValueError("inherited allocator alias/cache switch conflicts; explicit resolution required")
    child = dict(environ)
    child["PYTORCH_CUDA_ALLOC_CONF"] = CANDIDATE
    return child, {"inherited": inherited, "child": {key: child.get(key) for key in ENV_KEYS},
                   "changed_keys": [key for key in ENV_KEYS if environ.get(key) != child.get(key)],
                   "candidate": CANDIDATE, "allocator_verified": False,
                   "support_status": "not_checked_no_cuda_in_inspect"}


def verify_local_api(proof):
    if sys.prefix != proof["environment_prefix"]:
        raise ValueError("dedicated M5 environment required")
    distribution = importlib.metadata.distribution("torch")
    if distribution.version != proof["torch_version"] or distribution.version != "2.8.0+cu126":
        raise ValueError("fixed Torch2.8 build required")
    for relative, identity in proof["files"].items():
        if observe.sha(Path(distribution.locate_file(relative))) != identity["sha256"]:
            raise ValueError("installed allocator source changed: " + relative)


def allocator_evidence(torch, destination, require_segment):
    """只在显式GPU worker中调用；额外两次配置快照独立于69次原guard。"""
    report = {"candidate": CANDIDATE, "phase": "after_restore" if require_segment else "before_load",
              "verified": False, "support_status": "unknown", "backend": None,
              "allocator_settings": None, "device0_segment_count": None, "expandable_segment_count": None}
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*expandable_segments.*not supported.*")
            backend = torch.cuda.memory.get_allocator_backend()
            report["backend"] = backend
            if backend != "native":
                raise RuntimeError("candidate requires verified native allocator; no backend fallback")
            snapshot = torch.cuda.memory._snapshot()
        settings = snapshot["allocator_settings"]
        report["allocator_settings"] = copy.deepcopy(settings)
        if (settings.get("PYTORCH_CUDA_ALLOC_CONF") != CANDIDATE
                or settings.get("expandable_segments") is not True):
            raise RuntimeError("expandable_segments not effective/supported; no fallback")
        segments = [segment for segment in snapshot["segments"] if segment["device"] == 0]
        expandable = [segment for segment in segments if segment.get("is_expandable") is True
                      and segment.get("total_size", 0) > 0]
        report.update(device0_segment_count=len(segments), expandable_segment_count=len(expandable))
        if require_segment and not expandable:
            raise RuntimeError("no actual expandable segment after restore; effectiveness unproven")
        report.update(verified=True, support_status="segment_use_verified" if require_segment else "configuration_accepted")
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        observe.write(destination, report)
        raise
    observe.write(destination, report)
    return report


def inspect_runtime(path, root=PROJECT, environ=None):
    execution = observe.read(path)
    bindings = execution["artifacts"]
    required = {*SOURCES, FREEZE, API_PROOF, observe.RUNTIME, execution["cpu_checks"]}
    if (execution.get("version") != "m5-allocator-runtime-v1"
            or execution.get("default_phase") != "inspect" or execution.get("allowed_phases") != ["inspect", "probe"]
            or execution.get("output_directory") != OUTPUT or execution.get("allocator_candidate") != CANDIDATE
            or execution.get("configuration_snapshots") != 2 or not required <= set(bindings)):
        raise ValueError("unsupported or incomplete allocator runtime")
    for relative, expected in bindings.items():
        if observe.sha(observe.inside(root, relative)) != expected:
            raise ValueError("bound artifact changed: " + relative)
    previous = observe.read(root / FREEZE)
    if any(bindings.get(name) != expected for name, expected in previous["artifacts"].items()):
        raise ValueError("missing transitive allocator evidence")
    proof = observe.read(observe.inside(root, execution["cpu_checks"]))
    if not proof["passed"] or not proof["child_isolation_passed"] or not proof["frozen_loop_reused"]:
        raise ValueError("CPU allocator evidence missing")
    baseline = observe.read(root / "logs/m5_memory_observe_runner_20261019_v1/default_inspect_windows.json")
    if (baseline["runtime_sha256"] != observe.sha(root / observe.RUNTIME)
            or baseline["partial_group_denominator"] != 417
            or (baseline["max_optimizer_updates"], baseline["max_forward_backward_attempts"], baseline["max_guard_samples"]) != (6, 25, 69)):
        raise ValueError("fixed bounded baseline differs")
    _, plan = environment_plan(os.environ if environ is None else environ)
    return {**baseline, "runtime_sha256": observe.sha(path), "bound_artifacts": len(bindings),
            "baseline_runtime_sha256": baseline["runtime_sha256"], "allocator_environment": plan,
            "configuration_snapshots": 2, "allocator_verified": False, "phase": "inspect"}


def launch(path, root=PROJECT, executor=None):
    destination = observe.inside(root, OUTPUT)
    if destination.exists():
        raise FileExistsError("exclusive allocator experiment already claimed; no retry/resume")
    report = inspect_runtime(path, root)
    child_env, plan = environment_plan(os.environ)
    # 只改变传给子进程的副本；配置在导入Torch之前生效，父进程和全局设置不变。
    command = [sys.executable, "-B", "-m", "minigpt_cn.m5_allocator_run", "--worker", "--phase", "probe",
               "--execution", str(Path(path).resolve())]
    destination.mkdir(parents=True, exist_ok=False)
    observe.write(destination / "launch.json", {"runtime_sha256": report["runtime_sha256"],
                                              "environment": plan, "command": command, "parent_pid": os.getpid()})
    try:
        result = (executor or subprocess.run)(command, cwd=root, env=child_env, check=False)
    except BaseException as exc:
        observe.write(destination / "launch_failure.json", {"error_type": type(exc).__name__, "error": str(exc),
                                                          "traceback": traceback.format_exc(), "retry": False})
        raise
    if result.returncode == 0:
        try:
            verify_worker_receipt(destination, report["runtime_sha256"])
        except BaseException as exc:
            observe.write(destination / "launch_result.json", {"child_returncode": 0, "retry": False,
                                                              "worker_receipt_verified": False, "error": str(exc)})
            raise
    observe.write(destination / "launch_result.json", {"child_returncode": result.returncode, "retry": False,
                                                      "worker_receipt_verified": result.returncode == 0})
    return {"child_returncode": result.returncode, "output_directory": OUTPUT,
            "training_complete": False, "generation_calls": 0}


def verify_worker_receipt(destination, runtime_sha):
    receipt = observe.read(destination / "receipt.json")
    files = {p.name: observe.sha(p) for p in destination.iterdir()
             if p.is_file() and p.name not in ("receipt.json", "launch_result.json")}
    if (receipt["status"] != "bounded_probe_complete_no_reproduction"
            or receipt["phase"] != "allocator_probe" or receipt["runtime_sha256"] != runtime_sha
            or receipt["outputs"] != files or (destination / "failure.json").exists()):
        raise ValueError("worker receipt/artifact integrity differs")
    result = observe.read(destination / "result.json")
    if (result["optimizer_updates"] != 6 or result["forward_backward_attempts"] != 25
            or result["guard_samples"] != 69 or result["partial_group_denominator"] != 417
            or result["training_complete"] or result["generation_calls"] or result["test_calls"]
            or result["step52_optimizer_executed"] or result["allocator_candidate"] != CANDIDATE
            or result["allocator_verified"] is not True):
        raise ValueError("worker result differs from bounded candidate")
    effective = observe.read(destination / "allocator_after_restore.json")
    if effective["verified"] is not True or effective["support_status"] != "segment_use_verified":
        raise ValueError("worker did not prove effective allocator")


def assert_fresh_worker():
    if any(name in sys.modules for name in ("torch", "transformers", "peft")):
        raise RuntimeError("worker must start before model library imports")
    _, plan = environment_plan(os.environ)
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") != CANDIDATE or plan["changed_keys"]:
        raise RuntimeError("worker candidate was not set before startup")


def run_worker(path, root=PROJECT, backend_factory=None):
    assert_fresh_worker()
    report = inspect_runtime(path, root)
    destination = observe.inside(root, OUTPUT)
    launch_info = observe.read(destination / "launch.json")
    if (launch_info["runtime_sha256"] != report["runtime_sha256"]
            or launch_info["environment"]["child"] != {key: os.environ.get(key) for key in ENV_KEYS}):
        raise ValueError("launch/runtime/environment identity differs")
    observe.write(destination / "worker_started.json", {"runtime_sha256": report["runtime_sha256"],
                                                       "worker_pid": os.getpid(), "parent_pid": os.getppid()})
    backend, journal = None, None
    try:
        verify_local_api(observe.read(root / API_PROOF))
        from minigpt_cn import m5_train_fit_run as fit

        _, definition, plan, api, call_api, proof = fit.load_execution(root / observe.OLD_RUNTIME, root)
        backend = (backend_factory or fit.FitBackend)(plan, api, call_api)
        if backend.torch.__version__ != "2.8.0+cu126":
            raise ValueError("actual imported Torch build differs")
        prepared = fit.prepare_inputs(backend, definition, root)
        _, items, _, schedule, _, _ = prepared
        identity = backend.checkpoint_identity(definition, schedule, root)
        if identity != proof["checkpoint_identity"]:
            raise ValueError("source step45 identity differs")
        lookup = {item["sample_id"]: item for item in items}
        for expected, layout in zip(report["expected_attempts"], report["layouts"]):
            item = lookup[expected["sample_id"]]
            if (item["input_tokens"] != layout["input_tokens"]
                    or item["target_tokens_including_eos"] != layout["target_tokens_including_eos"]
                    or len(item["input_ids"]) != layout["sequence_tokens"]):
                raise ValueError("actual candidate layout differs")
        context = {"phase": "allocator_probe", "device": "cuda:0", "stage": "allocator_before_load",
                   "last_completed_absolute_step": 45, "last_optimizer_step_returned": 45}
        journal = observe.Journal(destination / "observations.jsonl", context)
        journal.emit("allocator_check_before_load")
        allocator_evidence(backend.torch, destination / "allocator_before_load.json", False)
        journal.emit("restore_enter")
        optimizer, scheduler, state = backend.load_continuation(definition, identity, root)
        journal.emit("restore_returned")
        guard = observe.ObservedGuard(observe.MemorySampler(backend.torch.cuda),
                                      backend.torch.cuda.get_device_properties(0).total_memory, journal)
        journal.emit("allocator_check_after_restore")
        effective = allocator_evidence(backend.torch, destination / "allocator_after_restore.json", True)
        ledger = fit.AttemptLedger(destination / "training_attempts.jsonl", report["expected_attempts"])
        fit.restore_rng(state["rng"], cuda=True)
        if fit.tree_digest(fit.capture_rng(cuda=True)) != identity["rng_sha256"]:
            raise ValueError("restored RNG differs before first attempt")
        if observe.sha(path) != report["runtime_sha256"]:
            raise ValueError("runtime changed before candidate updates")
        with (destination / "training.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            def before_forward(sample):
                index = len(ledger.entries())
                if index >= 25 or report["expected_attempts"][index]["sample_id"] != sample:
                    raise ValueError("candidate attempt identity/budget differs")
                ledger.record(report["expected_attempts"][index])

            def on_step(entry):
                if fit.registry(backend.model) != identity["parameter_registry"]:
                    raise ValueError("candidate trainable registration changed")
                stream.write(json.dumps(entry) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                print(f"allocator candidate step={entry['step']}/51 loss={entry['loss']:.6f}", flush=True)

            # 原循环原样复用：6更新/25条/69采样，末条n08仍用整组417分母。
            result = observe.bounded_loop(backend.model, items, report["groups"], optimizer, scheduler,
                                          guard, journal, before_forward, on_step)
        if len(ledger.entries()) != 25 or guard.samples != 69 or scheduler.last_epoch != 51:
            raise ValueError("candidate did not reach exact bounded boundary")
        if observe.sha(path) != report["runtime_sha256"]:
            raise ValueError("runtime changed during candidate")
        result.update(allocator_candidate=CANDIDATE, allocator_verified=effective["verified"])
        journal.emit("bounded_probe_end", result=result)
        closing, backend = backend, None
        closing.close()
        observe.write(destination / "result.json", result)
        # launch_result由父进程在子进程退出后写；回执只封存worker创建的文件。
        files = {p.name: observe.sha(p) for p in destination.iterdir() if p.is_file()}
        observe.write(destination / "receipt.json", {"status": result["status"], "phase": "allocator_probe",
                                                    "runtime_sha256": report["runtime_sha256"], "outputs": files})
        return result
    except BaseException as exc:
        try:
            observe.write(destination / "failure.json", {
                "phase": "allocator_probe", "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(), "context": copy.deepcopy(journal.context) if journal else None,
                "training_complete": False, "generation_calls": 0, "retry": False})
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


def run(path, phase="inspect", root=PROJECT):
    if phase not in ("inspect", "probe"):
        raise ValueError("only inspect/probe; no training/evaluation/retry/resume")
    return inspect_runtime(path, root) if phase == "inspect" else launch(path, root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=PROJECT / RUNTIME)
    parser.add_argument("--phase", choices=("inspect", "probe"), default="inspect")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker and args.phase != "probe":
        raise ValueError("internal worker requires explicit probe")
    result = run_worker(args.execution) if args.worker else run(args.execution, args.phase)
    print(json.dumps(result, ensure_ascii=False))
    if result.get("child_returncode", 0):
        raise SystemExit(result["child_returncode"])


if __name__ == "__main__":
    main()

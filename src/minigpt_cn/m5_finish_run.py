"""M5一次性收尾：通过候选观测后，原完整状态续行60步并固定评测53次。"""

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from minigpt_cn import m5_allocator_run as allocator

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = "logs/m5_finish_experiment_20261020_v1"
RUNTIME = "configs/m5_finish_v1/runtime.json"
AUDIT = "logs/m5_closeout_20261020_v1"
SOURCES = ("src/minigpt_cn/m5_finish_run.py", "tests/test_m5_finish_run.py")


def append(path, value):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def backend_class(base, destination):
    class FinishBackend(base):
        def load_continuation(self, *args, **kwargs):
            optimizer, scheduler, state = super().load_continuation(*args, **kwargs)
            allocator.allocator_evidence(self.torch, destination / "allocator_after_restore.json", True)
            self.finish_updates, self.finish_samples = 0, 0
            self.finish_total = self.torch.cuda.get_device_properties(0).total_memory
            with (destination / "optimizer_returns.jsonl").open("x"):
                pass
            with (destination / "memory_samples.jsonl").open("x"):
                pass
            def after_update(optimizer, args, kwargs):
                self.finish_updates += 1
                append(destination / "optimizer_returns.jsonl", {"absolute_step": 45 + self.finish_updates})
            self.finish_hook = optimizer.register_step_post_hook(after_update)
            return optimizer, scheduler, state

        def guarded_memory(self):
            observed = super().guarded_memory()
            self.finish_samples += 1
            results, error = allocator.observe.predicates(observed, self.finish_total)
            append(destination / "memory_samples.jsonl", {
                "sample": self.finish_samples, "optimizer_updates_returned": self.finish_updates,
                "expected_total_bytes": self.finish_total,
                "observed": observed, "predicate_results": results,
                "predicate_error": str(error) if error else None})
            # 原ResourceGuard使用并判定的就是这份读数，不额外查询或放宽门槛。
            return observed
    return FinishBackend


def validate_training(training, updates):
    if (training["last_step"] != 105 or training["additional_optimizer_steps"] != 60
            or training["training_microbatches"] != 240 or training["supervised_exposures"] != 23010
            or updates != [{"absolute_step": step} for step in range(46, 106)]):
        raise ValueError("complete60 updates/240 microbatches required before generation")


def inspect_runtime(path, root=PROJECT):
    runtime = allocator.observe.read(path)
    required = {*SOURCES, allocator.RUNTIME, f"{AUDIT}/scope.json", f"{AUDIT}/allocator_probe_acceptance.json",
                runtime["cpu_checks"]}
    if (runtime.get("version") != "m5-finish-runtime-v1" or runtime.get("output_directory") != OUTPUT
            or runtime.get("budget") != {"optimizer_updates": 60, "microbatches": 240, "generation_calls": 53, "test_calls": 0}
            or not required <= set(runtime["artifacts"])):
        raise ValueError("unsupported final runtime")
    for relative, digest in runtime["artifacts"].items():
        if allocator.observe.sha(allocator.observe.inside(root, relative)) != digest:
            raise ValueError("final bound artifact changed: " + relative)
    proof = allocator.observe.read(root / runtime["cpu_checks"])
    if not proof["passed"]:
        raise ValueError("final CPU evidence missing")
    probe = allocator.observe.read(root / f"{AUDIT}/allocator_probe_acceptance.json")
    if not probe["passed"] or not probe["all_guards_passed"] or not probe["allocator_verified"]:
        raise ValueError("accepted bounded allocator probe required")
    allocator.verify_worker_receipt(root / allocator.OUTPUT, probe["runtime_sha256"])
    _, environment = allocator.environment_plan(os.environ)
    return {"passed": True, "phase": "inspect", "runtime_sha256": allocator.observe.sha(path),
            "environment": environment, "budget": runtime["budget"], "cuda_initialized": False,
            "base_weights_loaded": False}


def worker(path, root=PROJECT):
    allocator.assert_fresh_worker()
    checked = inspect_runtime(path, root)
    folder = allocator.observe.inside(root, OUTPUT)
    launch = allocator.observe.read(folder / "launch.json")
    if launch["runtime_sha256"] != checked["runtime_sha256"]:
        raise ValueError("final launch identity differs")
    allocator.observe.write(folder / "worker_started.json", {"pid": os.getpid(), "runtime_sha256": checked["runtime_sha256"]})
    backend = None
    try:
        allocator.verify_local_api(allocator.observe.read(root / allocator.API_PROOF))
        from minigpt_cn import m5_train_fit_run as fit

        _, definition, plan, api, call_api, proof = fit.load_execution(root / allocator.observe.OLD_RUNTIME, root)
        backend = backend_class(fit.FitBackend, folder)(plan, api, call_api)
        prepared = fit.prepare_inputs(backend, definition, root)
        records, items, encoded, schedule, manifest, selection = prepared
        identity = backend.checkpoint_identity(definition, schedule, root)
        if identity != proof["checkpoint_identity"]:
            raise ValueError("final source45 identity differs")
        allocator.allocator_evidence(backend.torch, folder / "allocator_before_load.json", False)
        # 原冻结FitBackend.continue_training原样继承，包含原60组、RNG和资源/墙钟检查。
        training = backend.continue_training(definition, items, schedule, identity, folder, root)
        validate_training(training, fit.rows(folder / "optimizer_returns.jsonl"))
        allocator.observe.write(folder / "training_result.json", training)
        target = folder / "checkpoints/step_000105"
        lock = training["checkpoint_files"]
        if fit.old.file_tree(target) != lock:
            raise ValueError("final checkpoint differs before evaluation")
        allocator.observe.write(folder / "candidate_lock.json", {"step": 105, "files": lock})
        closing, backend = backend, None
        closing.close()
        backend = fit.FitBackend(plan, api, call_api)
        prompt = (root / definition["system_prompt"]).read_text(encoding="utf-8")
        for split in fit.SPLITS:
            if fit.old.encoded_inputs(backend, prompt, records[split], definition["training_interface"]) != encoded[split]:
                raise ValueError("final evaluation encoding differs")
        backend.load_base()
        backend.load_adapter(target)
        expected = [{**call, "effective_mode": "greedy_search", "do_sample": False, "use_model_defaults": False}
                    for call in manifest["calls"]]
        ledger = fit.AttemptLedger(folder / "call_attempts.jsonl", expected)
        reports = {}
        for split in fit.SPLITS:
            if fit.old.file_tree(target) != lock:
                raise ValueError("final candidate changed between sets")
            predictions = fit.generate_set(backend, records[split], encoded[split], folder / split,
                                           split, ledger, expected)
            reports[split] = fit.report_set(records[split], predictions, split, selection, root)
            allocator.observe.write(folder / split / "report.json", reports[split])
            fit.write_rows(folder / split / "all_sample_fields.jsonl", reports[split]["samples"])
            print(f"final evaluation completed: {split}", flush=True)
        if len(ledger.entries()) != 53:
            raise ValueError("exact53 final calls required")
        result = fit.comparison(reports, definition, root)
        allocator.observe.write(folder / "result.json", result)
        allocator.observe.write(folder / "generation_runtime.json", backend.runtime)
        if fit.old.file_tree(target) != lock or allocator.observe.sha(path) != checked["runtime_sha256"]:
            raise ValueError("final artifacts changed before seal")
        closing, backend = backend, None
        closing.close()
        fit.old.seal(folder, checked["runtime_sha256"], {"status": "completed", "phase": "m5-finish",
                                                      "test_calls": 0, "generation_calls": 53})
        return result
    except BaseException as exc:
        try:
            allocator.observe.write(folder / "failure.json", {"phase": "m5-finish", "error_type": type(exc).__name__,
                                                              "error": str(exc), "traceback": traceback.format_exc(),
                                                              "retry": False})
        except BaseException as write_error:
            print(f"final failure record unavailable: {write_error!r}; primary={exc!r}", file=sys.stderr)
        raise
    finally:
        if backend is not None:
            primary = sys.exc_info()[1]
            try:
                backend.close()
            except BaseException:
                if primary is None:
                    raise


def run(path, phase="inspect", root=PROJECT, executor=None):
    if phase not in ("inspect", "finish"):
        raise ValueError("only inspect/finish; no retry/resume")
    folder = allocator.observe.inside(root, OUTPUT)
    if phase == "finish" and folder.exists():
        raise FileExistsError("single final attempt already claimed")
    checked = inspect_runtime(path, root)
    if phase == "inspect":
        return checked
    environment, plan = allocator.environment_plan(os.environ)
    folder.mkdir(parents=True, exist_ok=False)
    allocator.observe.write(folder / "launch.json", {"runtime_sha256": checked["runtime_sha256"], "environment": plan})
    command = [sys.executable, "-B", "-m", "minigpt_cn.m5_finish_run", "--worker", "--execution", str(Path(path).resolve())]
    completed = (executor or subprocess.run)(command, cwd=root, env=environment, check=False)
    allocator.observe.write(folder / "launch_result.json", {"child_returncode": completed.returncode, "retry": False})
    return {"child_returncode": completed.returncode, "output_directory": OUTPUT}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=PROJECT / RUNTIME)
    parser.add_argument("--phase", choices=("inspect", "finish"), default="inspect")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    result = worker(args.execution) if args.worker else run(args.execution, args.phase)
    print(json.dumps(result, ensure_ascii=False))
    if result.get("child_returncode", 0):
        raise SystemExit(result["child_returncode"])


if __name__ == "__main__":
    main()

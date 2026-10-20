"""仅CPU验证子进程隔离、配置生效门槛和有界调用委托；不调用CUDA。"""

import ast
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from minigpt_cn import m5_allocator_run as candidate

ROOT = Path(__file__).resolve().parents[1]
TOTAL = 3 * candidate.observe.MARGIN


@pytest.mark.parametrize("inherited", [None, "", candidate.CANDIDATE])
def test_environment_copy_changes_only_candidate(inherited):
    parent = {"UNRELATED": "keep"}
    if inherited is not None:
        parent["PYTORCH_CUDA_ALLOC_CONF"] = inherited
    original = dict(parent)
    child, plan = candidate.environment_plan(parent)
    assert parent == original and child is not parent
    assert child["UNRELATED"] == "keep" and child["PYTORCH_CUDA_ALLOC_CONF"] == candidate.CANDIDATE
    assert plan["inherited"]["PYTORCH_CUDA_ALLOC_CONF"] == inherited
    assert plan["changed_keys"] == ([] if inherited == candidate.CANDIDATE else ["PYTORCH_CUDA_ALLOC_CONF"])
    assert not plan["allocator_verified"]


@pytest.mark.parametrize("key,value", [
    ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False"),
    ("PYTORCH_CUDA_ALLOC_CONF", "backend:native,expandable_segments:True"),
    ("PYTORCH_CUDA_ALLOC_CONF", " expandable_segments:True"),
    ("PYTORCH_ALLOC_CONF", candidate.CANDIDATE), ("PYTORCH_HIP_ALLOC_CONF", "anything"),
    ("PYTORCH_NO_CUDA_MEMORY_CACHING", "1"), ("PYTORCH_NO_CUDA_MEMORY_CACHING", "0"),
    ("PYTORCH_NO_HIP_MEMORY_CACHING", "1")])
def test_ambiguous_or_additional_config_rejected(key, value):
    environ = {key: value}
    with pytest.raises(ValueError):
        candidate.environment_plan(environ)
    assert environ == {key: value}


def test_real_cpu_child_sees_setting_before_model_import_without_parent_mutation():
    before = dict(os.environ)
    clean = {key: value for key, value in before.items() if key not in candidate.ENV_KEYS}
    child, _ = candidate.environment_plan(clean)
    program = ("import os,sys,json; print(json.dumps({'config':os.environ['PYTORCH_CUDA_ALLOC_CONF'],"
               "'model_imported':any(n in sys.modules for n in ('torch','transformers','peft'))}))")
    done = subprocess.run([sys.executable, "-B", "-c", program], env=child, capture_output=True,
                          text=True, check=True)
    assert json.loads(done.stdout) == {"config": candidate.CANDIDATE, "model_imported": False}
    assert dict(os.environ) == before


class FakeMemory:
    def __init__(self, damage=None):
        self.damage, self.calls = damage, []

    def get_allocator_backend(self):
        self.calls.append("backend")
        return "cudaMallocAsync" if self.damage == "backend" else "native"

    def _snapshot(self):
        self.calls.append("snapshot")
        if self.damage == "warning":
            warnings.warn("expandable_segments not supported on this platform", UserWarning, stacklevel=1)
        if self.damage == "error":
            raise RuntimeError("snapshot error")
        settings = {"PYTORCH_CUDA_ALLOC_CONF": candidate.CANDIDATE, "expandable_segments": True}
        if self.damage == "false":
            settings["expandable_segments"] = False
        elif self.damage == "string_true":
            settings["expandable_segments"] = "True"
        elif self.damage == "different_config":
            settings["PYTORCH_CUDA_ALLOC_CONF"] += ",max_split_size_mb:128"
        elif self.damage == "missing":
            del settings["expandable_segments"]
        segment = {"device": 0, "is_expandable": True, "total_size": 2097152}
        if self.damage == "not_expandable":
            segment["is_expandable"] = False
        elif self.damage == "other_device":
            segment["device"] = 1
        return {"allocator_settings": settings, "segments": [] if self.damage == "empty" else [segment]}


@pytest.mark.parametrize("damage", ["backend", "warning", "error", "false", "string_true", "different_config", "missing"])
def test_unknown_or_unsupported_allocator_stops_with_evidence(tmp_path, damage):
    memory = FakeMemory(damage)
    torch = SimpleNamespace(cuda=SimpleNamespace(memory=memory))
    path = tmp_path / "evidence.json"
    with pytest.raises((RuntimeError, UserWarning)):
        candidate.allocator_evidence(torch, path, False)
    report = candidate.observe.read(path)
    assert not report["verified"] and report["error_type"] and report["traceback"]
    assert memory.calls == (["backend"] if damage == "backend" else ["backend", "snapshot"])


@pytest.mark.parametrize("damage", ["empty", "not_expandable", "other_device"])
def test_parsed_setting_does_not_prove_actual_segment(tmp_path, damage):
    memory = FakeMemory(damage)
    torch = SimpleNamespace(cuda=SimpleNamespace(memory=memory))
    before = candidate.allocator_evidence(torch, tmp_path / "before.json", False)
    assert before["support_status"] == "configuration_accepted"
    with pytest.raises(RuntimeError, match="no actual expandable"):
        candidate.allocator_evidence(torch, tmp_path / "after.json", True)
    assert not candidate.observe.read(tmp_path / "after.json")["verified"]


def test_valid_allocator_evidence_and_exclusive_write(tmp_path):
    torch = SimpleNamespace(cuda=SimpleNamespace(memory=FakeMemory()))
    path = tmp_path / "effective.json"
    report = candidate.allocator_evidence(torch, path, True)
    assert report["verified"] and report["expandable_segment_count"] == 1
    assert report["support_status"] == "segment_use_verified"
    with pytest.raises(FileExistsError):
        candidate.allocator_evidence(torch, path, True)


@pytest.mark.parametrize("damage", ["prefix", "version", "hash"])
def test_installed_api_identity_is_required(tmp_path, monkeypatch, damage):
    path = tmp_path / "memory.py"
    path.write_text("fixed allocator interface", encoding="utf-8")
    proof = {"environment_prefix": sys.prefix, "torch_version": "2.8.0+cu126",
             "files": {"memory.py": {"sha256": candidate.observe.sha(path)}}}
    distribution = SimpleNamespace(version="2.8.0+cu126", locate_file=lambda relative: tmp_path / relative)
    monkeypatch.setattr(candidate.importlib.metadata, "distribution", lambda _: distribution)
    if damage == "prefix":
        proof["environment_prefix"] = "another environment"
    elif damage == "version":
        distribution.version = "2.14.0+cu126"
    else:
        proof["files"]["memory.py"]["sha256"] = "bad"
    with pytest.raises(ValueError):
        candidate.verify_local_api(proof)


@pytest.mark.parametrize("damage", ["exit", "spawn_error", "false_success"])
def test_launch_single_attempt_failure_and_parent_environment(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(candidate, "inspect_runtime", lambda *args: {"runtime_sha256": "runtime"})
    for key in candidate.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    before, calls = dict(os.environ), []
    def executor(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"] == candidate.CANDIDATE
        assert kwargs["env"] is not os.environ and not kwargs["check"]
        assert "--worker" in command and command[0] == sys.executable
        if damage == "spawn_error":
            raise OSError("spawn failed")
        return SimpleNamespace(returncode=3 if damage == "exit" else 0)
    if damage == "exit":
        assert candidate.launch(tmp_path / "runtime", tmp_path, executor)["child_returncode"] == 3
    else:
        with pytest.raises((OSError, FileNotFoundError)):
            candidate.launch(tmp_path / "runtime", tmp_path, executor)
    assert len(calls) == 1 and dict(os.environ) == before
    with pytest.raises(FileExistsError):
        candidate.launch(tmp_path / "runtime", tmp_path, executor)
    assert len(calls) == 1


def test_launch_conflict_rejected_before_output_or_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, "inspect_runtime", lambda *args: {})
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    with pytest.raises(ValueError):
        candidate.launch(tmp_path / "missing", tmp_path, lambda *args: pytest.fail("no child"))
    assert not (tmp_path / candidate.OUTPUT).exists()


def test_default_cli_and_forbidden_phases(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(candidate, "run", lambda path, phase: calls.append(phase) or {"passed": True})
    candidate.main([])
    assert calls == ["inspect"] and json.loads(capsys.readouterr().out)["passed"]
    with pytest.raises(ValueError, match="internal worker"):
        candidate.main(["--worker"])


@pytest.mark.parametrize("phase", ["train", "evaluate", "preflight", "resume", "retry", "test"])
def test_forbidden_phase_before_io(tmp_path, phase):
    with pytest.raises(ValueError):
        candidate.run(tmp_path / "absent", phase, tmp_path)


def test_default_worker_requires_fresh_process(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(RuntimeError, match="before model library"):
        candidate.assert_fresh_worker()


@pytest.mark.parametrize("setting", [None, "expandable_segments:False", candidate.CANDIDATE])
def test_fresh_worker_checks_startup_environment(monkeypatch, setting):
    for name in ("torch", "transformers", "peft"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    for key in candidate.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    if setting is not None:
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", setting)
    if setting == candidate.CANDIDATE:
        candidate.assert_fresh_worker()
    else:
        with pytest.raises((RuntimeError, ValueError)):
            candidate.assert_fresh_worker()


def test_candidate_delegates_training_math_to_frozen_loop():
    module = ast.parse(Path(candidate.__file__).read_text(encoding="utf-8"))
    calls = [ast.unparse(node.func) for node in ast.walk(module) if isinstance(node, ast.Call)]
    assert calls.count("observe.bounded_loop") == 1
    assert not any(call.endswith((".backward", ".generate", ".save_pretrained", ".empty_cache")) for call in calls)


@pytest.fixture
def cpu_worker(tmp_path, monkeypatch):
    from minigpt_cn import m5_train_fit_run as fit

    for key in candidate.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", candidate.CANDIDATE)
    baseline = candidate.observe.read(ROOT / "logs/m5_memory_observe_runner_20261019_v1/default_inspect_windows.json")
    report = {**baseline, "runtime_sha256": "candidate-runtime"}
    runtime = tmp_path / "runtime.json"
    candidate.observe.write(runtime, {})
    report["runtime_sha256"] = candidate.observe.sha(runtime)
    destination = tmp_path / candidate.OUTPUT
    destination.mkdir(parents=True)
    _, plan = candidate.environment_plan(os.environ)
    candidate.observe.write(destination / "launch.json", {"runtime_sha256": report["runtime_sha256"], "environment": plan})
    items = [{**layout, "input_ids": [1] * layout["sequence_tokens"]} for layout in report["layouts"]]
    identity = {"rng_sha256": "rng", "parameter_registry": ["fixed names"]}
    memory = FakeMemory()
    cuda = SimpleNamespace(memory=memory, get_device_properties=lambda _: SimpleNamespace(total_memory=TOTAL))
    backend = SimpleNamespace(torch=SimpleNamespace(cuda=cuda, __version__="2.8.0+cu126"), model=object())
    state = {"loads": 0, "loops": 0, "closed": 0}
    scheduler = SimpleNamespace(last_epoch=45)
    backend.checkpoint_identity = lambda *args: identity
    def load(*args):
        state["loads"] += 1
        return object(), scheduler, {"rng": "rng"}
    def close():
        state["closed"] += 1
    backend.load_continuation, backend.close = load, close
    monkeypatch.setattr(candidate, "assert_fresh_worker", lambda: None)
    monkeypatch.setattr(candidate, "verify_local_api", lambda _: None)
    monkeypatch.setattr(candidate, "inspect_runtime", lambda *args: report)
    # API文件内容也只作CPU替身，真实文件由受保护inspect和绑定链另外验收。
    api = tmp_path / candidate.API_PROOF
    api.parent.mkdir(parents=True)
    candidate.observe.write(api, {})
    monkeypatch.setattr(fit, "load_execution", lambda *args: (None, {}, {}, {}, {}, {"checkpoint_identity": identity}))
    monkeypatch.setattr(fit, "prepare_inputs", lambda *args: ({}, items, {}, {}, None, None))
    monkeypatch.setattr(fit, "restore_rng", lambda *args, **kwargs: None)
    monkeypatch.setattr(fit, "capture_rng", lambda *args, **kwargs: "rng")
    monkeypatch.setattr(fit, "tree_digest", lambda _: "rng")
    monkeypatch.setattr(fit, "registry", lambda _: identity["parameter_registry"])
    def loop(model, actual_items, groups, optimizer, actual_scheduler, guard, journal, before, on_step):
        state["loops"] += 1
        assert actual_items is items and groups == report["groups"] and groups[-1]["supervised_tokens"] == 417
        for entry in report["expected_attempts"]:
            before(entry["sample_id"])
        for group in groups[:6]:
            on_step({"step": group["optimizer_step"], "loss": 1.0})
        guard.samples = 69
        actual_scheduler.last_epoch = 51
        return {"status": "bounded_probe_complete_no_reproduction", "optimizer_updates": 6,
                "forward_backward_attempts": 25, "guard_samples": 69, "partial_group_denominator": 417,
                "supervised_exposures": 10765, "training_complete": False,
                "step52_optimizer_executed": False, "generation_calls": 0, "test_calls": 0}
    monkeypatch.setattr(candidate.observe, "bounded_loop", loop)
    return runtime, destination, memory, backend, state, report


@pytest.mark.parametrize("damage", [None, "before", "after"])
def test_worker_effect_gate_before_model_or_updates(cpu_worker, tmp_path, damage):
    runtime, destination, memory, backend, state, report = cpu_worker
    if damage == "before":
        memory.damage = "false"
    elif damage == "after":
        memory.damage = "empty"
    if damage:
        with pytest.raises(RuntimeError):
            candidate.run_worker(runtime, tmp_path, lambda *args: backend)
        failure = candidate.observe.read(destination / "failure.json")
        assert failure["phase"] == "allocator_probe" and not failure["training_complete"]
        assert state["loads"] == (1 if damage == "after" else 0)
        assert state["loops"] == 0 and not (destination / "receipt.json").exists()
    else:
        result = candidate.run_worker(runtime, tmp_path, lambda *args: backend)
        assert result["allocator_verified"] and state["loads"] == state["loops"] == 1
        candidate.verify_worker_receipt(destination, report["runtime_sha256"])
        assert len((destination / "training_attempts.jsonl").read_text().splitlines()) == 25
        assert len((destination / "training.jsonl").read_text().splitlines()) == 6
        assert memory.calls == ["backend", "snapshot", "backend", "snapshot"]
        assert not (destination / "checkpoints").exists()
    assert state["closed"] == 1
    with pytest.raises(FileExistsError):
        candidate.run_worker(runtime, tmp_path, lambda *args: pytest.fail("no retry"))


@pytest.mark.parametrize("damage", ["file", "runtime", "budget", "effective"])
def test_parent_rejects_corrupted_worker_receipt(cpu_worker, tmp_path, damage):
    runtime, destination, _, backend, _, report = cpu_worker
    candidate.run_worker(runtime, tmp_path, lambda *args: backend)
    if damage == "file":
        (destination / "training.jsonl").write_text("tampered", encoding="utf-8")
    elif damage == "runtime":
        report["runtime_sha256"] = "other"
    else:
        name = "result.json" if damage == "budget" else "allocator_after_restore.json"
        content = candidate.observe.read(destination / name)
        content["optimizer_updates" if damage == "budget" else "verified"] = 7 if damage == "budget" else False
        (destination / name).write_text(json.dumps(content), encoding="utf-8")
        # 重算文件摘要仍不能使错误预算/生效状态过关。
        receipt = candidate.observe.read(destination / "receipt.json")
        receipt["outputs"][name] = candidate.observe.sha(destination / name)
        (destination / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError):
        candidate.verify_worker_receipt(destination, report["runtime_sha256"])

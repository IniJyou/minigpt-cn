"""CPU精确续行与执行边界；非零dropout及checkpointing，不读取基座权重。"""

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from minigpt_cn import m5_train_fit_run as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def seed():
    torch.manual_seed(19)
    np.random.seed(19)
    random.seed(19)


class CheckpointedDropout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.randn(32, 8), requires_grad=False)
        self.first = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
        self.second = torch.nn.Parameter(torch.randn(8, 8) * 0.1)
        self.output = torch.nn.Parameter(torch.randn(8, 32), requires_grad=False)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, input_ids, labels, num_items_in_batch, **kwargs):
        def block(value):
            return self.dropout(torch.tanh(value @ self.first)) @ self.second
        hidden = checkpoint(block, self.frozen[input_ids], use_reentrant=False, preserve_rng_state=True)
        logits = hidden @ self.output
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1),
                                                 ignore_index=-100, reduction="sum") / num_items_in_batch
        return SimpleNamespace(loss=loss)


def tiny_material():
    items = [{"sample_id": f"t{i:02d}", "input_ids": [1, 3 + i % 20, 4, 5, 2],
              "labels": [-100, -100, 4, 5, 2]} for i in range(84)]
    groups = []
    for epoch in range(5):
        part = list(items)
        random.Random(42 + epoch).shuffle(part)
        for offset in range(0, 84, 4):
            groups.append({"optimizer_step": len(groups) + 1, "sample_ids": [i["sample_id"] for i in part[offset:offset + 4]],
                           "supervised_tokens": 12, "lr_before_update": runner.lr_at_update(len(groups) + 1)})
    return items, {"groups": groups, "continued_groups": groups[45:]}


def optimizers(model):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4,
                                  foreach=False, fused=False, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: s / 10 if s < 10 else max(0, 180 - s) / 170)
    return optimizer, scheduler


def save_payload(model, optimizer, scheduler, step):
    return copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "rng": runner.capture_rng(), "step": step, "parameter_registry": runner.registry(model),
                          "epoch": step // 21 + 1, "epoch_sample_offset": step % 21 * 4})


def run_part(model, optimizer, scheduler, items, groups, step, exposure):
    calls, updates = [], []
    def before(sample):
        calls.append((sample, random.random(), float(np.random.random())))
    result = runner.update_loop(model, items, groups, optimizer, scheduler, step, exposure,
                                before, updates.append, device="cpu")
    return result, calls, updates


def test_nonzero_dropout_checkpointing_exact_105_step_continuation(tmp_path):
    items, schedule = tiny_material()
    runner.validate_schedule(items, schedule)
    seed()
    full = CheckpointedDropout().train()
    optimizer, scheduler = optimizers(full)
    full_result, full_calls, full_updates = run_part(full, optimizer, scheduler, items, schedule["groups"], 0, 0)
    expected = save_payload(full, optimizer, scheduler, 105)
    seed()
    prefix = CheckpointedDropout().train()
    optimizer, scheduler = optimizers(prefix)
    _, prefix_calls, _ = run_part(prefix, optimizer, scheduler, items, schedule["groups"][:45], 0, 0)
    state = save_payload(prefix, optimizer, scheduler, 45)
    assert state["epoch"] == 3 and state["epoch_sample_offset"] == 12
    path = tmp_path / "tiny_checkpoint.pt"
    torch.save(state, path)
    restored = CheckpointedDropout().train()
    restored.load_state_dict(state["model"])
    optimizer, scheduler = optimizers(restored)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    runner.restore_optimizer(restored, optimizer, scheduler, loaded, state["parameter_registry"])
    assert restored.dropout.p == 0.3 and restored.training
    runner.restore_rng(loaded["rng"])
    result, calls, updates = run_part(restored, optimizer, scheduler, items, schedule["continued_groups"], 45, 45 * 12)
    actual = save_payload(restored, optimizer, scheduler, 105)
    assert runner.tree_digest(actual) == runner.tree_digest(expected)
    assert prefix_calls + calls == full_calls and updates == full_updates[45:]
    assert result["additional_optimizer_steps"] == 60 and result["training_microbatches"] == 240
    assert result["supervised_exposures"] == full_result["supervised_exposures"] == 1260
    assert torch.equal(restored.frozen, prefix.frozen) and torch.equal(restored.output, prefix.output)
    assert not torch.equal(restored.first, prefix.first)


@pytest.fixture
def prefix_state():
    items, schedule = tiny_material()
    seed()
    model = CheckpointedDropout()
    optimizer, scheduler = optimizers(model)
    run_part(model, optimizer, scheduler, items, schedule["groups"][:45], 0, 0)
    return model, optimizer, scheduler, save_payload(model, optimizer, scheduler, 45)


@pytest.mark.parametrize("damage", ["names", "live_order", "missing_state", "step", "lr", "scheduler", "moments", "settings", "extra_scheduler_step"])
def test_optimizer_restore_rejects_bad_identity_before_updates(prefix_state, damage):
    model, _, _, state = prefix_state
    optimizer, scheduler = optimizers(model)
    expected = runner.registry(model)
    if damage == "names":
        state["parameter_registry"].reverse()  # 两参数shape相同，仍须拒绝。
    elif damage == "live_order":
        optimizer.param_groups[0]["params"].reverse()
    elif damage == "missing_state":
        state["optimizer"]["state"].pop(1)
    elif damage == "step":
        state["optimizer"]["state"][0]["step"] += 1
    elif damage == "lr":
        state["optimizer"]["param_groups"][0]["lr"] = 0.0
    elif damage == "scheduler":
        state["scheduler"]["last_epoch"] = 44
    elif damage == "moments":
        state["optimizer"]["state"][0]["exp_avg"].fill_(float("nan"))
    elif damage == "settings":
        state["optimizer"]["param_groups"][0]["betas"] = (0.8, 0.999)
    else:
        state["scheduler"]["_step_count"] += 1
    with pytest.raises(ValueError):
        runner.restore_optimizer(model, optimizer, scheduler, state, expected)
    assert not optimizer.state


@pytest.mark.parametrize("damage", ["order", "count", "denominator", "lr", "suffix"])
def test_fixed_schedule_damage_rejected(damage):
    items, schedule = tiny_material()
    if damage == "order":
        schedule["groups"][0]["sample_ids"].reverse()
    elif damage == "count":
        items.pop()
    elif damage == "denominator":
        schedule["groups"][1]["supervised_tokens"] += 1
    elif damage == "lr":
        schedule["groups"][1]["lr_before_update"] = 0
    else:
        schedule["continued_groups"] = schedule["groups"][46:]
    with pytest.raises(ValueError):
        runner.validate_schedule(items, schedule)


@pytest.mark.parametrize("kind", ["order", "lr", "nonfinite", "resource"])
def test_continuation_failure_before_optimizer_update(prefix_state, kind):
    model, optimizer, scheduler, _ = prefix_state
    items, schedule = tiny_material()
    groups = schedule["continued_groups"]
    before = runner.tree_digest(model.state_dict())
    def guard():
        return None
    if kind == "order":
        groups[0]["optimizer_step"] = 47
    elif kind == "lr":
        optimizer.param_groups[0]["lr"] = 0
    elif kind == "nonfinite":
        model.first.register_hook(lambda grad: grad * float("nan"))
    else:
        def guard():
            raise MemoryError("over headroom")
    with pytest.raises((ValueError, FloatingPointError, MemoryError)):
        runner.update_loop(model, items, groups, optimizer, scheduler, 45, 540, lambda _: None,
                           lambda _: pytest.fail("must not update"), guard, device="cpu")
    assert runner.tree_digest(model.state_dict()) == before


@pytest.mark.parametrize("damage", ["reserved", "allocated", "total", "timeout"])
def test_resource_and_wall_stop(damage):
    memory = {"peak_allocated_bytes": 10, "peak_reserved_bytes": 20, "device_total_bytes": 3 * 1024**3}
    ticks = iter([0, 1801 if damage == "timeout" else 0])
    guard = runner.ResourceGuard(lambda: memory, 3 * 1024**3, clock=lambda: next(ticks))
    if damage == "reserved":
        memory["peak_reserved_bytes"] = 3 * 1024**3
    elif damage == "allocated":
        memory["peak_allocated_bytes"] = 30
    elif damage == "total":
        memory["device_total_bytes"] += 1
    with pytest.raises((MemoryError, TimeoutError)):
        guard()


@pytest.mark.parametrize("count", [2, 240, 53])
def test_persistent_budget_and_next_attempt_rejected(tmp_path, monkeypatch, count):
    expected = [{"attempt": i + 1} for i in range(count)]
    ledger = runner.AttemptLedger(tmp_path / "attempts.jsonl", expected)
    syncs = []
    monkeypatch.setattr(runner.os, "fsync", lambda fd: syncs.append(fd))
    for row in expected:
        ledger.record(row)
    assert len(ledger.entries()) == len(syncs) == count
    with pytest.raises(runner.DecodeIntegrityError):
        ledger.record({"attempt": count + 1})


@pytest.mark.parametrize("damage", ["delete", "change", "fsync"])
def test_ledger_damage_is_fatal(tmp_path, monkeypatch, damage):
    path = tmp_path / "attempts.jsonl"
    ledger = runner.AttemptLedger(path, [{"n": 1}, {"n": 2}])
    ledger.record({"n": 1})
    if damage == "delete":
        path.unlink()
    elif damage == "change":
        path.write_text('{}\n', encoding="utf-8")
    else:
        def failed(fd):
            raise OSError("disk failed")
        monkeypatch.setattr(runner.os, "fsync", failed)
    with pytest.raises(runner.DecodeIntegrityError):
        ledger.record({"n": 2})


@pytest.mark.parametrize("phase", ["test", "train", "resume", "retry"])
def test_forbidden_phase_before_any_io(tmp_path, phase):
    with pytest.raises(ValueError, match="only inspect"):
        runner.run(tmp_path / "absent", phase, tmp_path)


def test_cli_defaults_to_inspect(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runner, "run", lambda path, phase: calls.append(phase) or {"passed": True})
    runner.main([])
    assert calls == ["inspect"] and json.loads(capsys.readouterr().out)["passed"]


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    definition = runner.read(ROOT / runner.DEFINITION)
    plan = runner.read(ROOT / "configs/m5_lora_v1/protocol.json")
    manifest = runner.read(ROOT / definition["proposed_evaluation"]["manifest"])
    replay_def = runner.read(ROOT / runner.replay.DEFINITION)
    selection, training_replay = runner.replay.selected_records(replay_def, ROOT)
    dataset = runner.read(ROOT / "configs/m5_lora_revision_v1/dataset.json")
    records = {"train": runner.rows(ROOT / definition["training_data"]), "training_replay": training_replay,
               "new_diagnostic": runner.rows(ROOT / dataset["records"]["new_diagnostic"]),
               "original_valid": runner.rows(ROOT / dataset["records"]["original_valid"])}
    for relative in (definition["system_prompt"], "data/processed/m5_lora_revision_candidate_v1/pairs.jsonl",
                     definition["proposed_evaluation"]["conditions"][0]["control_outputs"] + "/report.json",
                     *definition["proposed_evaluation"]["reused_base_metrics"].values()):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / relative).read_bytes())
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text('{"fixture":true}', encoding="utf-8")
    items, schedule = tiny_material()
    encoded = {split: [[i] for i in range(len(records[split]))] for split in runner.SPLITS}
    identity = {"fake_named_identity": True}
    state = SimpleNamespace(events=[], failure=None, incomplete=False, corrupt=False, changed_inputs=False,
                            root=tmp_path, definition=definition, runtime_path=runtime_path)
    monkeypatch.setattr(runner, "load_execution", lambda path, root: (
        {"fixture": True}, definition, plan, {}, {}, {"passed": True, "checkpoint_identity": identity}))
    monkeypatch.setattr(runner, "prepare_inputs", lambda *args: (records, items, encoded, schedule, manifest, selection))
    monkeypatch.setattr(runner, "inspection", lambda *args: {"passed": True, "generation_calls": 0})
    def encoded_inputs(backend, prompt, values, interface):
        return [[i + int(state.changed_inputs)] for i in range(len(values))]
    monkeypatch.setattr(runner.old, "encoded_inputs", encoded_inputs)
    class Backend:
        def __init__(self, *args):
            self.runtime = {"model_loaded": False}
            self.set_index = 0
            state.events.append("construct")
        def checkpoint_identity(self, *args):
            return identity
        def close(self):
            state.events.append("close")
        def continuation_preflight(self, definition, items, actual, folder, root):
            state.events.append("preflight")
            return {"passed": True, "forward_backward_pairs": 2, "optimizer_steps": 0,
                    "checkpoint_identity": identity, "peak_reserved_bytes": 1, "reserved_limit_bytes": 2}
        def continue_training(self, definition, items, schedule, actual, folder, root):
            state.events.append("train")
            if state.failure:
                raise state.failure
            target = folder / "checkpoints/step_000105"
            target.mkdir(parents=True)
            (target / "fake_weight.txt").write_text("tiny fixture", encoding="utf-8")
            return {"last_step": 105, "additional_optimizer_steps": 59 if state.incomplete else 60,
                    "training_microbatches": 240, "supervised_exposures": 23010,
                    "checkpoint_files": runner.old.file_tree(target)}
        def load_base(self):
            state.events.append("load")
        def load_adapter(self, path):
            state.events.append("adapter")
        def assert_inference(self):
            return None
        def check_effective(self):
            return None
        def generate(self, ids, before_forward):
            before_forward()
            split = runner.SPLITS[self.set_index]
            record = records[split][ids[0]]
            if ids[0] == len(records[split]) - 1:
                self.set_index += 1
            state.events.append("generate")
            if state.corrupt:
                raise RuntimeError("ordinary synthetic error")
            return {"response": json.dumps(record["target"]), "generated_token_ids": [1], "generated_tokens": 1}
    state.backend = Backend
    return state


def execute(state, phase):
    return runner.run(state.runtime_path, phase, state.root, state.backend)


def test_default_workflow_inspect_creates_nothing(workflow):
    assert execute(workflow, "inspect")["passed"]
    assert workflow.events == ["construct", "close"]
    assert not (workflow.root / workflow.definition["future_runtime_requirements"]["planned_output_directory"]).exists()


def test_train_requires_standalone_preflight_before_backend_construction(workflow):
    with pytest.raises(FileNotFoundError):
        execute(workflow, "train-evaluate")
    assert not workflow.events


def test_complete_preflight_and_60_step_53_call_controller(workflow):
    state = workflow
    assert execute(state, "preflight")["optimizer_steps"] == 0
    result = execute(state, "train-evaluate")
    assert result["training_fit_improved"] and result["development_both_nonregression"]
    assert state.events.count("train") == 1 and state.events.count("generate") == 53
    folder = state.root / state.definition["future_runtime_requirements"]["planned_output_directory"]
    runner.old.verify_receipt(folder / "train-evaluate", runner.sha256_file(state.runtime_path))
    assert len(runner.rows(folder / "train-evaluate/call_attempts.jsonl")) == 53
    for phase in ("preflight", "train-evaluate"):
        with pytest.raises(FileExistsError):
            execute(state, phase)


@pytest.mark.parametrize("damage", ["incomplete", "oom", "inputs"])
def test_failure_preserved_no_generation_or_success_receipt(workflow, damage):
    state = workflow
    execute(state, "preflight")
    if damage == "incomplete":
        state.incomplete = True
    elif damage == "oom":
        state.failure = MemoryError("synthetic OOM")
    else:
        state.changed_inputs = True
    with pytest.raises((ValueError, MemoryError)):
        execute(state, "train-evaluate")
    folder = state.root / state.definition["future_runtime_requirements"]["planned_output_directory"] / "train-evaluate"
    assert (folder / "failure.json").exists() and not (folder / "receipt.json").exists()
    assert "generate" not in state.events


def test_ordinary_generation_failure_keeps53_full_denominators(workflow):
    state = workflow
    execute(state, "preflight")
    state.corrupt = True
    result = execute(state, "train-evaluate")
    assert result["generation_calls"] == 53 and not result["training_fit_improved"]
    folder = state.root / state.definition["future_runtime_requirements"]["planned_output_directory"] / "train-evaluate"
    for split, count, blocks in zip(runner.SPLITS, (14, 24, 15), (17, 50, 62)):
        report = runner.read(folder / split / "report.json")
        assert report["metrics"]["counts"]["samples"] == count
        assert report["metrics"]["counts"]["blocks"] == blocks
        assert report["metrics"]["counts"]["block_exact"] == 0


def test_identity_change_stops_before_claim_or_load(workflow, monkeypatch):
    monkeypatch.setattr(workflow.backend, "checkpoint_identity", lambda *args: {"changed": True})
    with pytest.raises(ValueError, match="identity"):
        execute(workflow, "preflight")
    assert "load" not in workflow.events and "preflight" not in workflow.events


def test_runtime_change_before_seal_preserves_failure(workflow, monkeypatch):
    state = workflow
    execute(state, "preflight")
    original = runner.comparison
    def changed(*args):
        state.runtime_path.write_text('{"changed":true}', encoding="utf-8")
        return original(*args)
    monkeypatch.setattr(runner, "comparison", changed)
    with pytest.raises(ValueError, match="runtime changed"):
        execute(state, "train-evaluate")
    folder = state.root / state.definition["future_runtime_requirements"]["planned_output_directory"] / "train-evaluate"
    assert (folder / "failure.json").exists() and not (folder / "receipt.json").exists()


def test_restoring_cpu_rng_rejects_saved_cuda_scope():
    state = runner.capture_rng()
    state["cuda"] = [torch.ones(3, dtype=torch.uint8)]
    with pytest.raises(ValueError, match="scope"):
        runner.restore_rng(state)


@pytest.fixture
def runtime_files(tmp_path, monkeypatch):
    files = {runner.DEFINITION: {"artifacts": {}, "proposed_evaluation": {"generation": {"do_sample": False}},
                                  "control": {"directory": "checkpoint", "files": {}}},
             runner.FREEZE: {"artifacts": {}}, "cpu.json": {"passed": True},
             "inputs.json": {"passed": True, "exact_cpu_continuation_passed": True},
             "configs/m5_lora_v1/protocol.json": {"evaluation": {}},
             "logs/m5_valid_error_review_20261019_v1/generation_api_check.json": {},
             "logs/m5_greedy_acceptance_20261019_v1/call_api_check.json": {}}
    files.update({name: {"fixture": True} for name in runner.SOURCES})
    for relative, value in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(runner, "DEFINITION_SHA", runner.sha256_file(tmp_path / runner.DEFINITION))
    runtime = {"version": "m5-train-fit-runtime-v1", "definition": runner.DEFINITION,
               "cpu_checks": "cpu.json", "cpu_input_preflight": "inputs.json", "default_phase": "inspect",
               "allowed_phases": list(runner.PHASES), "artifacts": {p: runner.sha256_file(tmp_path / p) for p in files}}
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(runtime), encoding="utf-8")
    return tmp_path, path, runtime


def test_runtime_loader_requires_bound_cpu_exact_continuation(runtime_files):
    root, path, _ = runtime_files
    _, _, plan, _, _, proof = runner.load_execution(path, root)
    assert proof["exact_cpu_continuation_passed"] and plan["evaluation"]["generation"] == {"do_sample": False}


@pytest.mark.parametrize("damage", ["missing_source", "digest", "definition", "phase", "cpu_failed", "not_exact", "transitive"])
def test_runtime_rejects_bad_cpu_or_source_evidence(runtime_files, damage):
    root, path, runtime = runtime_files
    if damage == "missing_source":
        runtime["artifacts"].pop(runner.SOURCES[0])
    elif damage == "digest":
        runtime["artifacts"]["inputs.json"] = "0" * 64
    elif damage == "definition":
        runtime["artifacts"][runner.DEFINITION] = "0" * 64
    elif damage == "phase":
        runtime["default_phase"] = "train-evaluate"
    elif damage == "cpu_failed":
        (root / "cpu.json").write_text('{"passed":false}', encoding="utf-8")
        runtime["artifacts"]["cpu.json"] = runner.sha256_file(root / "cpu.json")
    elif damage == "not_exact":
        (root / "inputs.json").write_text('{"passed":true,"exact_cpu_continuation_passed":false}', encoding="utf-8")
        runtime["artifacts"]["inputs.json"] = runner.sha256_file(root / "inputs.json")
    else:
        (root / runner.FREEZE).write_text('{"artifacts":{"unbound":"abc"}}', encoding="utf-8")
        runtime["artifacts"][runner.FREEZE] = runner.sha256_file(root / runner.FREEZE)
    path.write_text(json.dumps(runtime), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_execution(path, root)


def test_preflight_receipt_tamper_blocks_train_before_construct(workflow):
    state = workflow
    execute(state, "preflight")
    folder = state.root / state.definition["future_runtime_requirements"]["planned_output_directory"]
    (folder / "preflight/gpu_preflight.json").write_text('{}', encoding="utf-8")
    previous = state.events.count("construct")
    with pytest.raises(ValueError):
        execute(state, "train-evaluate")
    assert state.events.count("construct") == previous


def test_missing_forward_ledger_callback_is_global_failure(tmp_path):
    expected = [{"dataset": "training_replay", "sample_id": "a"}]
    ledger = runner.AttemptLedger(tmp_path / "calls.jsonl", expected)
    backend = SimpleNamespace(assert_inference=lambda: None, check_effective=lambda: None,
        generate=lambda ids, before: {"response": "", "generated_tokens": 0, "generated_token_ids": []})
    with pytest.raises(runner.DecodeIntegrityError):
        runner.generate_set(backend, [{"sample_id": "a"}], [[1]], tmp_path / "generation",
                            "training_replay", ledger, expected)
    assert not ledger.committed

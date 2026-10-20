"""修订入口CPU边界测试：小张量和替身，不读取Qwen权重或真实候选正文。"""

import copy
import json
import random
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from minigpt_cn import m5_lora_revision_run as runner


def tiny_items():
    return [{"sample_id": f"t{i:02d}", "input_ids": [1, 2, 3], "labels": [-100, -100, 3],
             "input_tokens": 2, "target_tokens_including_eos": 1} for i in range(84)]


def schedule_for(items):
    ids = [i["sample_id"] for i in items]
    order = []
    for epoch in range(3):
        part = list(ids)
        random.Random(42 + epoch).shuffle(part)
        order.extend(part if epoch < 2 else part[:12])
    return {"optimizer_steps": 45, "training_microbatches": 180, "supervised_token_exposures": 180,
            "groups": [{"optimizer_step": i // 4 + 1, "sample_ids": order[i:i + 4], "supervised_tokens": 4}
                       for i in range(0, 180, 4)]}


class TinyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.frozen = torch.nn.Parameter(torch.tensor(9.0), requires_grad=False)

    def forward(self, labels, num_items_in_batch, **kwargs):
        count = (labels[:, 1:] != -100).sum()
        return SimpleNamespace(loss=self.weight.square() * count / num_items_in_batch)


def settings():
    return {"max_optimizer_steps": 45, "supervised_token_exposures_total": 180,
            "max_wall_train_seconds": 60, "clip_grad_norm": 1.0}


def test_exact_partial_epoch_stops_at_45_and_uses_original_lr_prefix():
    model = TinyLoss()
    optimizer = torch.optim.SGD([model.weight], lr=0.01)

    def factor(step):
        return step / 10 if step < 10 else (180 - step) / 170

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    saves, steps = [], []
    items = tiny_items()
    schedule = schedule_for(items)
    result = runner.train_loop(model, items, schedule, settings(), optimizer, scheduler,
                               saves.append, steps.append, device="cpu")
    assert result["optimizer_steps"] == 45 and result["training_microbatches"] == 180
    assert result["supervised_exposures"] == 180 and saves == [45]
    assert [s["epoch"] for s in steps] == [1] * 21 + [2] * 21 + [3] * 3
    assert [s["sample_ids"] for s in steps] == [g["sample_ids"] for g in schedule["groups"]]
    assert [s["lr"] for s in steps] == pytest.approx([0.01 * factor(i) for i in range(45)])
    assert model.frozen.item() == 9.0 and model.frozen.grad is None
    assert 0 < model.weight.item() < 0.25


@pytest.mark.parametrize("damage", ["missing_group", "reordered", "wrong_denominator", "duplicate_id", "wrong_step"])
def test_invalid_schedule_rejected_before_any_update(damage):
    items = tiny_items()
    schedule = schedule_for(items)
    if damage == "missing_group":
        schedule["groups"].pop()
    elif damage == "reordered":
        schedule["groups"][0]["sample_ids"].reverse()
    elif damage == "wrong_denominator":
        schedule["groups"][0]["supervised_tokens"] += 1
    elif damage == "duplicate_id":
        items[-1]["sample_id"] = items[0]["sample_id"]
    else:
        schedule["groups"][0]["optimizer_step"] = 2
    model = TinyLoss()
    optimizer = torch.optim.SGD([model.weight], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    with pytest.raises(ValueError):
        runner.train_loop(model, items, schedule, settings(), optimizer, scheduler,
                          lambda _: pytest.fail("save"), lambda _: pytest.fail("update"), device="cpu")
    assert model.weight.item() == 0.25


def test_deadline_prevents_first_optimizer_update():
    items, model = tiny_items(), TinyLoss()
    optimizer = torch.optim.SGD([model.weight], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    ticks = iter([0, 61])
    with pytest.raises(TimeoutError):
        runner.train_loop(model, items, schedule_for(items), settings(), optimizer, scheduler,
                          lambda _: pytest.fail("save"), lambda _: pytest.fail("update"), device="cpu",
                          clock=lambda: next(ticks))
    assert model.weight.item() == 0.25


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_nonfinite_loss_or_gradient_stops_before_update(failure):
    items, model = tiny_items(), TinyLoss()
    if failure == "loss":
        model.weight.data.fill_(float("nan"))
    else:
        model.weight.register_hook(lambda value: value * float("inf"))
    optimizer = torch.optim.SGD([model.weight], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    with pytest.raises(FloatingPointError):
        runner.train_loop(model, items, schedule_for(items), settings(), optimizer, scheduler,
                          lambda _: pytest.fail("save"), lambda _: pytest.fail("update"), device="cpu")
    assert not optimizer.state


def record(sample):
    return {"sample_id": sample, "input": {"blocks": [{"block_id": "b0001", "text": "普通正文"}]},
            "target": {"annotations": [{"block_id": "b0001", "kind": "body", "title": None, "reason": None}]}}


def eval_records():
    return {"new_diagnostic": [record(f"d{i:02d}") for i in range(24)],
            "original_valid": [record(f"v{i:02d}") for i in range(15)]}


def test_ledger_persists_87_sequential_attempts_and_rejects_88(tmp_path, monkeypatch):
    ledger = runner.CallLedger(tmp_path, eval_records())
    syncs = []
    monkeypatch.setattr(runner.os, "fsync", lambda fd: syncs.append(fd))
    for condition, sample in ledger.expected:
        ledger.record(condition, sample)
    assert len(ledger.entries()) == len(syncs) == 87
    with pytest.raises(runner.DecodeIntegrityError):
        ledger.record(*ledger.expected[-1])
    assert len(ledger.entries()) == 87


@pytest.mark.parametrize("damage", ["wrong_condition", "wrong_id", "corrupt_json", "missing_key", "write_failure"])
def test_ledger_failure_is_fatal_before_forward(tmp_path, monkeypatch, damage):
    ledger = runner.CallLedger(tmp_path, eval_records())
    if damage == "corrupt_json":
        ledger.path.write_text("{", encoding="utf-8")
    elif damage == "missing_key":
        ledger.path.write_text("{}\n", encoding="utf-8")
    elif damage == "write_failure":
        def fail_sync(_):
            raise OSError("synthetic disk failure")
        monkeypatch.setattr(runner.os, "fsync", fail_sync)
    with pytest.raises(runner.DecodeIntegrityError):
        ledger.record("control_diagnostic" if damage == "wrong_condition" else "base_diagnostic",
                      "not-the-next-id" if damage == "wrong_id" else "d00")
    assert not issubclass(runner.DecodeIntegrityError, Exception)


def test_incomplete_or_wrong_condition_counts_rejected(tmp_path):
    records = eval_records()
    records["new_diagnostic"].pop()
    with pytest.raises(runner.DecodeIntegrityError):
        runner.CallLedger(tmp_path, records)


class PredictionBackend:
    runtime = {"synthetic": True}
    failure = None

    def generate(self, ids, before_forward):
        if self.failure == "integrity":
            raise runner.DecodeIntegrityError("synthetic merge differs")
        before_forward()
        if self.failure == "ordinary":
            raise RuntimeError("synthetic row failure")
        if self.failure == "oom":
            raise MemoryError("synthetic oom")
        response = json.dumps(record("x")["target"])
        return {"response": response, "raw_completion": response, "generated_token_ids": [1],
                "generated_tokens": 1, "runtime_error": None}


def test_ordinary_prediction_errors_keep_full_denominator(tmp_path):
    records = eval_records()
    ledger = runner.CallLedger(tmp_path, records)
    backend = PredictionBackend()
    backend.failure = "ordinary"
    metrics, predictions = runner.generate_condition(backend, records["new_diagnostic"], [[1]] * 24,
                                                     tmp_path / "base", "base_diagnostic", ledger)
    assert metrics["counts"]["samples"] == metrics["counts"]["blocks"] == 24
    assert metrics["counts"]["block_exact"] == 0 and len(predictions) == len(ledger.entries()) == 24
    assert all(p["response"] == "" for p in predictions)


@pytest.mark.parametrize("failure,attempts", [("integrity", 0), ("oom", 1)])
def test_integrity_and_oom_stop_globally_without_success_metrics(tmp_path, failure, attempts):
    records = eval_records()
    ledger = runner.CallLedger(tmp_path, records)
    backend = PredictionBackend()
    backend.failure = failure
    with pytest.raises(runner.DecodeIntegrityError if failure == "integrity" else MemoryError):
        runner.generate_condition(backend, records["new_diagnostic"], [[1]] * 24,
                                  tmp_path / "base", "base_diagnostic", ledger)
    assert len(ledger.entries()) == attempts and not (tmp_path / "base/metrics.json").exists()


def test_generate_wires_actual_false_overrides_and_records_before_forward():
    events, calls = [], []
    backend = runner.RevisionBackend.__new__(runner.RevisionBackend)
    expected = {"do_sample": False, "num_beams": 1, "eos_token_id": [151645]}

    class Configuration(SimpleNamespace):
        def get_generation_mode(self):
            return SimpleNamespace(value="sample" if self.do_sample else "greedy_search")

        def to_dict(self):
            return vars(self)

    config = Configuration(**expected)

    def prepare(configuration, **kwargs):
        assert kwargs == {"use_model_defaults": False, "do_sample": False}
        events.append("prepare")
        return copy.deepcopy(configuration), {}

    def generate(**kwargs):
        assert events[-1] == "ledger"
        calls.append(kwargs)
        return [SimpleNamespace(tolist=lambda: [42, 99, 151645])]

    cuda = SimpleNamespace(manual_seed_all=lambda _: None, synchronize=lambda: None,
                           max_memory_allocated=lambda: 10, max_memory_reserved=lambda: 20)
    backend.torch = SimpleNamespace(manual_seed=lambda _: None, cuda=cuda, long="long",
                                    tensor=lambda *a, **k: "synthetic-input", ones_like=lambda _: "mask",
                                    inference_mode=nullcontext)
    backend.plan = {"evaluation": {"generation": expected}}
    backend.runtime = {}
    backend.generation_config = config
    backend.model = SimpleNamespace(training=False, parameters=lambda: [SimpleNamespace(requires_grad=False)],
                                    _prepare_generation_config=prepare, generate=generate)
    backend.tokenizer = SimpleNamespace(decode=lambda ids, **kw: "x" + ("<eos>" if 151645 in ids else ""))
    result = backend.generate([42], lambda: events.append("ledger"))
    assert result["response"] == "x" and result["generated_tokens"] == 2
    assert events == ["prepare", "ledger"] and len(calls) == 1
    assert calls[0]["do_sample"] is False and calls[0]["use_model_defaults"] is False
    assert calls[0]["logits_to_keep"] == 1 and "labels" not in calls[0]


def score(macro=0.6, block=0.6):
    return {"macro_f1": macro, "rates": {"json_valid": 1.0, "schema_valid": 1.0, "coverage_valid": 1.0,
                                         "block_exact": block, "sample_exact": 0.5,
                                         "nonnull_title_correct": 0.6, "nonnull_reason_correct": 0.6}}


def gates():
    return {"json_rate": 1.0, "schema_rate": 1.0, "coverage_rate": 1.0, "block_all_fields_rate": 0.8,
            "sample_exact_rate": 0.5, "macro_f1": 0.75, "macro_f1_gain_over_same_prompt_base": 0.15,
            "non_null_title_exact_rate": 0.8, "non_null_reason_exact_rate": 0.8}


def test_directional_improvement_does_not_replace_absolute_gates_or_mix_base_scores():
    metrics = {"base_diagnostic": score(0.3), "control_diagnostic": score(0.6),
               "candidate_diagnostic": score(0.7, 0.7), "candidate_original_valid": score(0.65, 0.65)}
    historical = {"base": score(0.1), "adapter_step_000045": score(0.6)}
    result = runner.comparison_result(metrics, historical, gates())
    assert result["directional_improvement"] and not result["quality_passed_both_development_sets"]
    assert result["gates"]["candidate_diagnostic"]["observed"]["macro_f1_gain_over_same_prompt_base"] == pytest.approx(0.4)
    assert result["gates"]["candidate_original_valid"]["observed"]["macro_f1_gain_over_same_prompt_base"] == pytest.approx(0.55)
    assert result["automatic_test_allowed"] is False and result["original_selected_step"] is None
    metrics["candidate_original_valid"]["rates"]["nonnull_title_correct"] = 0.5
    assert not runner.comparison_result(metrics, historical, gates())["directional_improvement"]


def pair_records():
    return [{"pair_id": f"pair{i}", "phenomenon": f"P{i // 2 + 1}",
             "role": "new_boundary_diagnostic", "sample_ids": [f"d{2 * i:02d}", f"d{2 * i + 1:02d}"]} for i in range(12)]


def test_pair_requires_both_members_and_reports_all_six_phenomena():
    records = eval_records()["new_diagnostic"]
    predictions = [{"sample_id": r["sample_id"], "response": json.dumps(r["target"])} for r in records]
    predictions[0]["response"] = "{}"
    result = runner.pair_summary(records, predictions, pair_records())
    assert result["complete"] == 11 and result["pairs"] == 12
    assert result["by_phenomenon"]["P1"] == {"pairs": 2, "complete": 1}
    assert set(result["by_phenomenon"]) == {f"P{i}" for i in range(1, 7)}
    with pytest.raises(ValueError, match="order"):
        runner.pair_summary(records, predictions[::-1], pair_records())


@pytest.mark.parametrize("phase", ["test", "train", "run", "eval", "train-valid", "resume"])
def test_forbidden_phase_rejected_before_io_or_backend(tmp_path, phase):
    with pytest.raises(ValueError, match="no test"):
        runner.run(tmp_path / "does-not-exist", phase, tmp_path,
                   backend_factory=lambda *_: pytest.fail("backend"))
    assert list(tmp_path.iterdir()) == []


def test_cli_defaults_to_inspect_and_rejects_test(monkeypatch):
    seen = []
    monkeypatch.setattr(runner, "run", lambda path, phase: seen.append(phase) or {})
    runner.main([])
    assert seen == ["inspect"]
    with pytest.raises(SystemExit):
        runner.main(["--phase", "test"])
    assert seen == ["inspect"]


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_loader_rejects_unsupported_version_and_missing_bindings(tmp_path):
    path = tmp_path / "runtime.json"
    put(path, {"version": "other"})
    with pytest.raises(ValueError, match="unsupported"):
        runner.load_execution(path, tmp_path)
    runtime = {"version": "m5-lora-revision-runtime-v1", "artifacts": {},
               **{k: k + ".json" for k in ("definition", "dataset", "data_acceptance", "data_freeze", "greedy_runtime",
                                           "cpu_checks", "generation_api_check", "call_boundary_api_check")}}
    put(path, runtime)
    with pytest.raises(ValueError, match="must be bound"):
        runner.load_execution(path, tmp_path)


class FakeBackend(PredictionBackend):
    context_limit = 8192
    seen = []
    incomplete = False

    def __init__(self, plan, *args):
        self.plan = plan
        self.loaded = False
        self.runtime = {"synthetic": True, "model_loaded": False}

    def encode(self, messages):
        return [42]

    def load_base(self):
        self.loaded = True
        self.runtime["model_loaded"] = True
        self.seen.append("load_base")

    def load_adapter(self, path):
        assert self.loaded and Path(path).is_dir()
        self.seen.append("load_adapter")

    def close(self):
        self.loaded = False

    def gpu_preflight(self, items):
        self.seen.append("preflight")
        return {"passed": True, "optimizer_steps": 0, "initial_adapter_sha256": "init",
                "forward_backward_pairs": 2, "sequence_tokens": 1961, "base_frozen": True,
                "target_module_count": 196, "trainable_parameters": 5046272,
                "peak_reserved_bytes": 100, "reserved_limit_bytes": 200}

    def train(self, items, folder, expected_initial, schedule):
        assert self.loaded and expected_initial == "init"
        self.seen.append("train")
        checkpoint = folder / "checkpoints/step_000045"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"synthetic")
        return {"completed": not self.incomplete, "optimizer_steps": 45, "training_microbatches": 180,
                "supervised_exposures": 9657, "initial_adapter": {"initial_adapter_sha256": "init"},
                "checkpoints": [{"step": 45, "files": runner.old.file_tree(checkpoint)}]}

    def generate(self, ids, before_forward):
        assert self.loaded
        self.seen.append("generate")
        return super().generate(ids, before_forward)


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    (tmp_path / "logs/control").mkdir(parents=True)
    (tmp_path / "logs/control/adapter_model.safetensors").write_bytes(b"old synthetic control")
    (tmp_path / "prompt.txt").write_text("system", encoding="utf-8")
    execution = {"experiment_directory": "logs/revision"}
    path = tmp_path / "runtime.json"
    put(path, execution)
    definition = {"system_prompt": "prompt.txt",
                  "training_comparison": {"control": {"directory": "logs/control",
                      "files": runner.old.file_tree(tmp_path / "logs/control")}},
                  "proposed_evaluation": {"reuse_original_valid_base_and_control": "logs/history", "minimum_gates": gates()}}
    dataset = {"data_directory": "data"}
    plan = {"expected_initial": "init", "schedule_sha256": "schedule", "training_interface": {
        "max_input_tokens": 4096, "max_target_tokens_including_eos": 1024}}
    records = {"train": [record(f"t{i}") for i in range(84)], **eval_records()}
    encoded = {k: [[42]] * len(v) for k, v in records.items()}
    items = tiny_items()
    for name in ("base", "adapter_step_000045"):
        folder = tmp_path / "logs/history" / name
        folder.mkdir(parents=True)
        predictions = [{"sample_id": r["sample_id"], "response": json.dumps(r["target"])} for r in records["original_valid"]]
        (folder / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in predictions) + "\n", encoding="utf-8")
        metrics = runner.evaluate((r["input"], r["target"], p["response"]) for r, p in zip(records["original_valid"], predictions))
        metrics.pop("diagnostics")
        put(folder / "metrics.json", metrics)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/pairs.jsonl").write_text("\n".join(json.dumps(p) for p in pair_records()), encoding="utf-8")
    bundle = (execution, definition, dataset, plan, {}, {})
    monkeypatch.setattr(runner, "load_execution", lambda *args: bundle)
    monkeypatch.setattr(runner, "prepare_inputs", lambda *args: (records, encoded, items, {"supervised_token_exposures": 9657}))
    FakeBackend.seen = []
    FakeBackend.incomplete = False
    FakeBackend.failure = None
    return tmp_path, path, plan


def test_inspect_never_claims_directory_loads_weights_or_generates(workflow):
    root, path, _ = workflow
    result = runner.run(path, root=root, backend_factory=FakeBackend)
    assert result["model_loaded"] is False and result["test_records_read"] is False
    assert result["samples"] == {"train": 84, "new_diagnostic": 24, "original_valid": 15}
    assert result["planned_supervised_exposures"] == 9657 and FakeBackend.seen == []
    assert not (root / "logs/revision").exists()


def test_training_requires_valid_preflight_before_backend(workflow):
    root, path, _ = workflow
    with pytest.raises(FileNotFoundError):
        runner.run(path, "train-compare", root, FakeBackend)
    assert FakeBackend.seen == [] and not (root / "logs/revision").exists()


def test_complete_synthetic_workflow_locks_candidate_then_runs_exactly_87(workflow):
    root, path, _ = workflow
    runner.run(path, "preflight", root, FakeBackend)
    result = runner.run(path, "train-compare", root, FakeBackend)
    destination = root / "logs/revision/train-compare"
    assert result["model_generation_calls"] == FakeBackend.seen.count("generate") == 87
    assert FakeBackend.seen.index("train") < FakeBackend.seen.index("generate")
    assert result["test_calls"] == 0 and result["automatic_test_allowed"] is False
    assert len(runner.rows(destination / "call_attempts.jsonl")) == 87
    assert (destination / "candidate_lock.json").is_file()
    assert not (root / "logs/revision/test").exists()
    runner.old.verify_receipt(destination, runner.sha256_file(path))
    before = list(FakeBackend.seen)
    with pytest.raises(FileExistsError):
        runner.run(path, "train-compare", root, FakeBackend)
    assert before == FakeBackend.seen


def test_incomplete_training_preserves_failure_and_prohibits_generation_and_retry(workflow):
    root, path, _ = workflow
    runner.run(path, "preflight", root, FakeBackend)
    FakeBackend.incomplete = True
    with pytest.raises(ValueError, match="incomplete"):
        runner.run(path, "train-compare", root, FakeBackend)
    folder = root / "logs/revision/train-compare"
    assert (folder / "failure.json").exists() and not (folder / "receipt.json").exists()
    assert "generate" not in FakeBackend.seen
    with pytest.raises(FileExistsError):
        runner.run(path, "train-compare", root, FakeBackend)


@pytest.mark.parametrize("key,value", [("initial_adapter_sha256", "wrong"), ("optimizer_steps", 1),
                                       ("forward_backward_pairs", 3), ("peak_reserved_bytes", 201)])
def test_failed_preflight_gate_blocks_training(workflow, key, value):
    root, path, plan = workflow
    result = FakeBackend(plan).gpu_preflight([])
    result[key] = value
    with pytest.raises(ValueError, match="preflight"):
        runner.validate_preflight(result, plan)


def test_unexpected_existing_preflight_directory_is_never_overwritten(workflow):
    root, path, _ = workflow
    (root / "logs/revision").mkdir()
    with pytest.raises(FileExistsError):
        runner.run(path, "preflight", root, FakeBackend)
    assert FakeBackend.seen == []

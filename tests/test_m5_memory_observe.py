"""观测故障与CPU轨迹对照；不读取真实权重，不调用CUDA或generate。"""

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.checkpoint import checkpoint

from minigpt_cn import m5_memory_observe as observer
from minigpt_cn import m5_train_fit_run as fit

ROOT = Path(__file__).resolve().parents[1]
TOTAL = 3 * observer.MARGIN


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def journal(tmp_path):
    return observer.Journal(tmp_path / "observations.jsonl", {"phase": "cpu", "last_completed_absolute_step": 45})


def safe_memory():
    return {"peak_allocated_bytes": 10, "peak_reserved_bytes": 20, "device_total_bytes": TOTAL}


@pytest.mark.parametrize("mask", range(8))
def test_all_predicate_combinations_and_frozen_decision(tmp_path, mask):
    values = safe_memory()
    if mask & 1:
        values["device_total_bytes"] += 1
    if mask & 2:
        values["peak_allocated_bytes"] = -1
    if mask & 4:
        values["peak_reserved_bytes"] = TOTAL
    log = journal(tmp_path)
    count = []
    def memory():
        count.append(1)
        return values
    guard = observer.ObservedGuard(memory, TOTAL, log)
    if mask:
        with pytest.raises(MemoryError):
            guard("test")
        with pytest.raises(MemoryError):
            fit.ResourceGuard(lambda: values, TOTAL)()
    else:
        assert guard("test") == fit.ResourceGuard(lambda: values, TOTAL)()
    result = log.committed[-1]
    assert result["predicate_results"] == dict(zip(observer.PREDICATES, [bool(mask & x) for x in (1, 2, 4)]))
    assert result["observed"] == values and len(count) == 1
    values["peak_reserved_bytes"] = 0
    assert result["observed"]["peak_reserved_bytes"] != 0


@pytest.mark.parametrize("damage", ["equal_limit", "missing_total", "allocated_exceeds_reserved", "missing_peak", "short_circuit"])
def test_boundaries_missing_values_and_exception_priority(tmp_path, damage):
    values = safe_memory()
    if damage == "equal_limit":
        values["peak_reserved_bytes"] = TOTAL - observer.MARGIN
    elif damage == "missing_total":
        del values["device_total_bytes"]
    elif damage == "allocated_exceeds_reserved":
        values["peak_allocated_bytes"] = 21
    else:
        del values["peak_allocated_bytes"]
        if damage == "short_circuit":
            values["device_total_bytes"] += 1
    log = journal(tmp_path)
    guard = observer.ObservedGuard(lambda: values, TOTAL, log)
    if damage in ("equal_limit", "missing_total"):
        assert guard("test") == values
    else:
        kind = KeyError if damage == "missing_peak" else MemoryError
        with pytest.raises(kind):
            guard("test")
        with pytest.raises(kind):
            fit.ResourceGuard(lambda: values, TOTAL)()
        if damage in ("missing_peak", "short_circuit"):
            assert log.committed[-1]["predicate_results"]["peak_relation_invalid"] is None


class FakeCuda:
    def __init__(self, damage=None):
        self.damage, self.calls = damage, []

    def call(self, name, result):
        self.calls.append(name)
        if name == self.damage:
            raise RuntimeError(name + " failed")
        return result

    def synchronize(self):
        return self.call("sync", None)

    def max_memory_allocated(self):
        return self.call("allocated", 10)

    def max_memory_reserved(self):
        return self.call("reserved", 20)

    def get_device_properties(self, device):
        assert device == 0
        return self.call("total", SimpleNamespace(total_memory=TOTAL))


@pytest.mark.parametrize("damage", [None, "sync", "allocated", "reserved", "total"])
def test_sampler_exact_order_partial_read_and_no_reread(tmp_path, damage):
    cuda, log = FakeCuda(damage), journal(tmp_path)
    guard = observer.ObservedGuard(observer.MemorySampler(cuda), TOTAL, log)
    order = ["sync", "allocated", "reserved", "total"]
    if damage is None:
        assert guard("test") == safe_memory()
        assert log.committed[-1]["synchronize_status"] == "returned"
    else:
        with pytest.raises(RuntimeError, match=damage):
            guard("test")
        assert log.committed[-1]["observed"] is None
        assert all(value is None for value in log.committed[-1]["predicate_results"].values())
        assert log.committed[-1]["last_valid_event_id"] is None
        order = order[:order.index(damage) + 1]
    assert cuda.calls == order


def test_timeout_records_unknown_without_sampling(tmp_path):
    log = journal(tmp_path)
    times = iter([0, 1800])
    guard = observer.ObservedGuard(lambda: pytest.fail("must not sample"), TOTAL, log, clock=lambda: next(times))
    with pytest.raises(TimeoutError):
        guard("timeout")
    assert guard.samples == 0 and log.committed[-1]["observed"] is None


@pytest.mark.parametrize("damage", ["fsync", "tamper", "delete"])
def test_journal_persistence_failure_stops_and_preserves_primary(tmp_path, damage, capsys):
    log = journal(tmp_path)
    log.emit("prior")
    if damage == "fsync":
        def fail(fd):
            raise OSError("disk fsync")
        log.sync = fail
    elif damage == "tamper":
        log.path.write_text('{}\n', encoding="utf-8")
    else:
        log.path.unlink()
    primary = MemoryError("primary resource error")
    with pytest.raises(observer.EvidenceWriteError) as caught:
        log.emit("error", primary=primary)
    assert caught.value.__cause__ is primary
    assert "primary resource error" in capsys.readouterr().err
    taken = tmp_path / "taken.jsonl"
    observer.Journal(taken, {})
    with pytest.raises(FileExistsError):
        observer.Journal(taken, {})


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.randn(32, 4), requires_grad=False)
        self.first = torch.nn.Parameter(torch.randn(4, 4) * 0.1)
        self.second = torch.nn.Parameter(torch.randn(4, 4) * 0.1)
        self.head = torch.nn.Parameter(torch.randn(4, 32), requires_grad=False)
        self.dropout = torch.nn.Dropout(0.05)
        self.denominators = []

    def forward(self, input_ids, labels, num_items_in_batch, **kwargs):
        self.denominators.append(num_items_in_batch)
        def block(value):
            return self.dropout(torch.tanh(value @ self.first)) @ self.second
        hidden = checkpoint(block, self.base[input_ids], use_reentrant=False, preserve_rng_state=True)
        logits = hidden @ self.head
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1),
                                                 ignore_index=-100, reduction="sum") / num_items_in_batch
        return SimpleNamespace(loss=loss)


def make_optimizers(model):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4,
                                  foreach=False, fused=False, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: s / 10 if s < 10 else max(0, 180 - s) / 170)
    return optimizer, scheduler


@pytest.fixture(scope="module")
def saved45():
    layouts = observer.read(ROOT / "data/processed/m5_lora_revision_candidate_v1/token_checks.json")["layouts"]
    items = []
    for layout in layouts:
        if layout["role"] != "augmented_train":
            continue
        length, targets = layout["sequence_tokens"], layout["target_tokens_including_eos"]
        tokens = [1 + i % 30 for i in range(length)]
        items.append({**layout, "input_ids": tokens, "labels": [-100] * (length - targets) + tokens[-targets:]})
    schedule = observer.read(ROOT / "logs/m5_train_fit_definition_20261019_v1/planned_schedule.json")
    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    model = TinyModel().train()
    optimizer, scheduler = make_optimizers(model)
    fit.update_loop(model, items, schedule["groups"][:45], optimizer, scheduler, 0, 0,
                    lambda _: None, lambda _: None, device="cpu")
    return items, schedule, copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), \
        copy.deepcopy(scheduler.state_dict()), fit.capture_rng()


def restore(saved):
    items, schedule, weights, adam, rates, rng = saved
    model = TinyModel().train()
    model.load_state_dict(weights)
    optimizer, scheduler = make_optimizers(model)
    optimizer.load_state_dict(copy.deepcopy(adam))
    scheduler.load_state_dict(copy.deepcopy(rates))
    fit.restore_rng(rng)
    return model, optimizer, scheduler, items, schedule["continued_groups"][:7]


def snapshot(model, optimizer, scheduler):
    return fit.tree_digest({"weights": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "rng": fit.capture_rng(),
                            "gradients": [p.grad for p in model.parameters()]})


def test_exact_cpu_original_vs_observed_25_and_417(saved45, tmp_path):
    model, optimizer, scheduler, items, groups = restore(saved45)
    calls, entries, counter = [], [], []
    class Boundary(Exception):
        pass
    def before(sample):
        calls.append((sample, random.random(), float(np.random.random())))
    def guard():
        counter.append(1)
        if len(counter) == 69:
            raise Boundary()
    def on_step(row):
        guard()  # 原入口日志回调的额外检查。
        entries.append(row)
    with pytest.raises(Boundary):
        fit.update_loop(model, items, groups, optimizer, scheduler, 45, 9657, before, on_step, guard, device="cpu")
    expected, expected_denominators = snapshot(model, optimizer, scheduler), model.denominators
    baseline = copy.deepcopy(model.base)
    model, optimizer, scheduler, items, groups = restore(saved45)
    log = journal(tmp_path)
    observed = observer.ObservedGuard(safe_memory, TOTAL, log)
    actual_calls, actual_entries = [], []
    def actual_before(sample):
        actual_calls.append((sample, random.random(), float(np.random.random())))
    result = observer.bounded_loop(model, items, groups, optimizer, scheduler, observed, log,
                                   actual_before, actual_entries.append, device="cpu")
    assert snapshot(model, optimizer, scheduler) == expected
    assert actual_calls == calls and len(calls) == 25
    assert model.denominators == expected_denominators and model.denominators[-1] == 417
    assert len(entries) == len(actual_entries) == 6
    assert all(all(row[k] == other[k] for k in row) for row, other in zip(entries, actual_entries))
    assert result["guard_samples"] == 69 and result["optimizer_updates"] == 6
    assert result["supervised_exposures"] == 10765 and not result["training_complete"]
    assert scheduler.last_epoch == 51 and torch.equal(model.base, baseline)
    assert any(not torch.equal(model.state_dict()[k], saved45[2][k]) for k in ("first", "second"))
    assert log.context["optimizer_step_entered"] is False  # step52禁止更新。


@pytest.mark.parametrize("failure", ["optimizer_internal", "scheduler_internal", "after_optimizer_guard", "journal_before_forward"])
def test_update_boundary_and_no_next_operation(saved45, tmp_path, failure):
    model, optimizer, scheduler, items, groups = restore(saved45)
    before = copy.deepcopy(model.first)
    calls, log = [], journal(tmp_path)
    count = []
    def memory():
        count.append(1)
        value = safe_memory()
        if failure == "after_optimizer_guard" and len(count) == 10:
            value["peak_reserved_bytes"] = TOTAL
        return value
    guard = observer.ObservedGuard(memory, TOTAL, log)
    if failure == "optimizer_internal":
        original = optimizer.step
        def failed_step():
            original()
            raise RuntimeError("optimizer internal after mutation")
        optimizer.step = failed_step
    elif failure == "scheduler_internal":
        def failed_scheduler():
            raise RuntimeError("scheduler internal")
        scheduler.step = failed_scheduler
    elif failure == "journal_before_forward":
        def failed_sync(fd):
            raise OSError("fsync failed")
        log.sync = failed_sync
    with pytest.raises((RuntimeError, MemoryError)):
        observer.bounded_loop(model, items, groups, optimizer, scheduler, guard, log,
                              calls.append, lambda _: pytest.fail("no logged complete update"), device="cpu")
    if failure == "journal_before_forward":
        assert not calls and torch.equal(model.first, before)
    else:
        assert len(calls) == 4 and not torch.equal(model.first, before)
        assert log.context["optimizer_step_entered"]
        assert log.context["optimizer_step_returned"] == (failure != "optimizer_internal")
        assert log.context["scheduler_step_returned"] == (failure == "after_optimizer_guard")


@pytest.mark.parametrize("damage", ["group_count", "step", "denominator", "duplicate"])
def test_bad_group_rejected_before_forward(saved45, tmp_path, damage):
    model, optimizer, scheduler, items, groups = restore(saved45)
    items, groups = copy.deepcopy(items), copy.deepcopy(groups)
    if damage == "group_count":
        groups.pop()
    elif damage == "step":
        groups[0]["optimizer_step"] += 1
    elif damage == "denominator":
        groups[-1]["supervised_tokens"] = 161
    else:
        items.append(items[0])
    log = journal(tmp_path)
    guard = observer.ObservedGuard(safe_memory, TOTAL, log)
    with pytest.raises(ValueError):
        observer.bounded_loop(model, items, groups, optimizer, scheduler, guard, log,
                              lambda _: pytest.fail("no forward"), lambda _: None, device="cpu")
    assert not log.committed


@pytest.mark.parametrize("phase", ["train", "evaluate", "resume", "retry", "test"])
def test_forbidden_phase_before_io(tmp_path, phase):
    with pytest.raises(ValueError):
        observer.run(tmp_path / "absent", phase, tmp_path)


def test_existing_output_refused_before_loading_runtime(tmp_path):
    (tmp_path / observer.OUTPUT).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        observer.run(tmp_path / "absent", "probe", tmp_path)


def test_cli_defaults_to_inspect(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(observer, "run", lambda path, phase: calls.append(phase) or {"passed": True})
    observer.main([])
    assert calls == ["inspect"] and json.loads(capsys.readouterr().out)["passed"]


def test_runtime_missing_bindings_fails_without_model(tmp_path):
    path = tmp_path / "runtime.json"
    observer.write(path, {"artifacts": {}, "cpu_checks": "missing"})
    with pytest.raises(ValueError):
        observer.run(path, root=tmp_path, backend_factory=lambda *args: pytest.fail("no backend"))


@pytest.mark.parametrize("failure", [False, True, "cleanup"])
def test_cpu_workflow_seals_bounded_probe_or_preserves_failure(saved45, tmp_path, monkeypatch, failure, capsys):
    model, optimizer, scheduler, items, groups = restore(saved45)
    runtime = tmp_path / "runtime.json"
    observer.write(runtime, {"fake_cpu_workflow": True})
    old_path = tmp_path / observer.OLD_RUNTIME
    old_path.parent.mkdir(parents=True)
    observer.write(old_path, {})
    expected = observer.read(ROOT / "logs/m5_memory_observability_definition_20261019_v1/review.json")["sample_layouts"]
    identities = [{key: row[key] for key in ("absolute_step", "absolute_microbatch", "sample_id")} for row in expected]
    report = {"passed": True, "runtime_sha256": observer.sha(runtime), "expected_attempts": identities,
              "layouts": expected, "groups": groups}
    identity = {"rng_sha256": fit.tree_digest(saved45[5]), "parameter_registry": fit.registry(model)}
    cuda, closed = FakeCuda(), []
    if failure:
        def reserved():
            value = cuda.call("reserved", 20)
            return TOTAL if cuda.calls.count("reserved") == 69 else value
        cuda.max_memory_reserved = reserved
    class Backend:
        torch = SimpleNamespace(cuda=cuda)
        def __init__(self, *args):
            self.model = model
        def checkpoint_identity(self, *args):
            return identity
        def load_continuation(self, *args):
            return optimizer, scheduler, {"rng": saved45[5]}
        def close(self):
            closed.append(1)
            if failure == "cleanup":
                raise RuntimeError("cleanup failed")
    original_restore, original_capture, original_loop = fit.restore_rng, fit.capture_rng, observer.bounded_loop
    monkeypatch.setattr(observer, "inspect_runtime", lambda *args: report)
    monkeypatch.setattr(fit, "load_execution", lambda *args: (None, {}, {}, {}, {}, {"checkpoint_identity": identity}))
    monkeypatch.setattr(fit, "prepare_inputs", lambda *args: ({}, items, {}, {}, None, None))
    monkeypatch.setattr(fit, "restore_rng", lambda rng, cuda=False: original_restore(rng))
    monkeypatch.setattr(fit, "capture_rng", lambda cuda=False: original_capture())
    monkeypatch.setattr(observer, "bounded_loop", lambda *args: original_loop(*args, device="cpu"))
    if failure:
        with pytest.raises(MemoryError):
            observer.run(runtime, "probe", tmp_path, Backend)
    else:
        result = observer.run(runtime, "probe", tmp_path, Backend)
        assert result["status"] == "bounded_probe_complete_no_reproduction"
    folder = tmp_path / observer.OUTPUT
    attempts = [json.loads(line) for line in (folder / "training_attempts.jsonl").read_text().splitlines()]
    assert attempts == identities and len(closed) == 1
    assert len((folder / "training.jsonl").read_text().splitlines()) == 6
    assert not (folder / "checkpoints").exists() and not (folder / "call_attempts.jsonl").exists()
    if failure:
        saved_failure = observer.read(folder / "failure.json")
        assert saved_failure["error_type"] == "MemoryError"
        assert saved_failure["context"]["stage"] == "after_backward_delete"
        assert not (folder / "receipt.json").exists()
        events = [json.loads(line) for line in (folder / "observations.jsonl").read_text().splitlines()]
        assert events[-1]["observed"]["peak_reserved_bytes"] == TOTAL
        if failure == "cleanup":
            assert "primary=MemoryError" in capsys.readouterr().err
    else:
        receipt = observer.read(folder / "receipt.json")
        assert receipt["status"] == "bounded_probe_complete_no_reproduction"
        assert receipt["outputs"] == {p.name: observer.sha(p) for p in folder.iterdir() if p.name != "receipt.json"}
    with pytest.raises(FileExistsError):
        observer.run(runtime, "probe", tmp_path, Backend)

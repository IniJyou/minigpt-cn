"""固定回放的CPU验收；仅小型本地原例/替身，不读取真实权重或test正文。"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minigpt_cn import m5_train_replay_run as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def material(tmp_path):
    definition = runner.read(ROOT / runner.DEFINITION)
    selection = runner.read(ROOT / definition["selection"])
    for relative in (definition["selection"], definition["training_data"], definition["token_layout_evidence"],
                     definition["planned_call_order"], definition["system_prompt"]):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    selection, records = runner.selected_records(definition, tmp_path)
    return tmp_path, definition, selection, records


def replace_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_original_record_selection_and_fixed_call_plan(material):
    root, definition, selection, records = material
    assert len(records) == 14 and sum(len(r["input"]["blocks"]) for r in records) == 17
    assert len(records[1]["input"]["blocks"]) == 3
    assert selection["samples"][7]["new_checkpoint_exact_finetuning_presentations"] == 3
    calls = runner.planned_calls(definition, selection, root)
    assert [c["condition"] for c in calls] == [runner.CONDITIONS[0]] * 14 + [runner.CONDITIONS[1]] * 14


@pytest.mark.parametrize("field,value", [
    ("utf8_byte_start", 1), ("source_line_1based", 1), ("line_utf8_sha256", "0" * 64),
    ("target_sha256", "0" * 64), ("input_sha256", "0" * 64),
    ("blocks", 1), ("source_group_id", "other-family"), ("full_record_required", False),
    ("stratum", "revision_only_train"),
])
def test_selection_identity_damage_rejected(material, field, value):
    root, definition, selection, _ = material
    selection["samples"][0][field] = value
    replace_json(root / definition["selection"], selection)
    with pytest.raises(runner.DecodeIntegrityError):
        runner.selected_records(definition, root)


def test_source_byte_change_rejected(material):
    root, definition, _, _ = material
    with (root / definition["training_data"]).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(runner.DecodeIntegrityError):
        runner.selected_records(definition, root)


def test_order_change_rejected(material):
    root, definition, selection, _ = material
    selection["samples"].reverse()
    replace_json(root / definition["selection"], selection)
    with pytest.raises(runner.DecodeIntegrityError):
        runner.selected_records(definition, root)


def test_ledger_28_fsync_then_reject_29(material, monkeypatch):
    root, definition, selection, _ = material
    ledger = runner.CallLedger(root, runner.planned_calls(definition, selection, root))
    syncs = []
    monkeypatch.setattr(runner.os, "fsync", lambda fd: syncs.append(fd))
    for call in ledger.expected:
        ledger.record(call["condition"], call["sample_id"])
    assert len(ledger.entries()) == len(syncs) == 28
    with pytest.raises(runner.DecodeIntegrityError):
        ledger.record(ledger.expected[-1]["condition"], ledger.expected[-1]["sample_id"])


@pytest.mark.parametrize("damage", ["delete", "truncate", "reorder", "append"])
def test_ledger_external_damage_global_stop(material, damage):
    root, definition, selection, _ = material
    ledger = runner.CallLedger(root, runner.planned_calls(definition, selection, root))
    ledger.record(ledger.expected[0]["condition"], ledger.expected[0]["sample_id"])
    if damage == "delete":
        ledger.path.unlink()
    elif damage == "truncate":
        ledger.path.write_text("", encoding="utf-8")
    elif damage == "reorder":
        changed = copy.deepcopy(ledger.committed)
        changed[0]["sample_id"] = "changed"
        ledger.path.write_text(json.dumps(changed[0]) + "\n", encoding="utf-8")
    else:
        with ledger.path.open("a", encoding="utf-8") as stream:
            stream.write("{}\n")
    with pytest.raises(runner.DecodeIntegrityError):
        ledger.record(ledger.expected[1]["condition"], ledger.expected[1]["sample_id"])


def test_fsync_failure_and_wrong_order_never_commit(material, monkeypatch):
    root, definition, selection, _ = material
    ledger = runner.CallLedger(root, runner.planned_calls(definition, selection, root))
    with pytest.raises(runner.DecodeIntegrityError):
        ledger.record(ledger.expected[1]["condition"], ledger.expected[1]["sample_id"])
    assert not ledger.path.exists()
    def failed(fd):
        raise OSError("disk failure")
    monkeypatch.setattr(runner.os, "fsync", failed)
    with pytest.raises(runner.DecodeIntegrityError):
        ledger.record(ledger.expected[0]["condition"], ledger.expected[0]["sample_id"])
    assert not ledger.committed


@pytest.fixture
def execution(material, monkeypatch):
    root, definition, selection, records = material
    runtime = root / "runtime.json"
    replace_json(runtime, {"fixture": True})
    loads, events = [], []
    def loader(path, base):
        loads.append(len(loads))
        return {}, definition, {}, {}, {}
    monkeypatch.setattr(runner, "load_execution", loader)
    state = SimpleNamespace(failure=None, skip_callback=False, bad_tokens=False, wrong_length=False,
                            wrong_mode=False, gpu_initialized=False, events=events, loads=loads,
                            root=root, definition=definition, selection=selection, records=records,
                            runtime=runtime)
    class Backend:
        def __init__(self, *args):
            self.runtime = {"packages": {f"package-{i}": "1" for i in range(43)}}
            self.context_limit = 40960
            self.torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: state.gpu_initialized))
            self.index = 0
            events.append("construct")
        def encode(self, messages):
            assert len(messages) == 2 and [m["role"] for m in messages] == ["system", "user"]
            assert json.loads(messages[1]["content"]) == records[self.index]["input"]
            count = selection["samples"][self.index]["verified_layout"]["input_tokens"]
            self.index += 1
            return [self.index - 1] * (count + int(state.wrong_length))
        def inspect_greedy(self):
            return {"effective_mode": "sample" if state.wrong_mode else "greedy_search",
                    "do_sample": False, "use_model_defaults": False}
        def load_base(self):
            assert (root / definition["execution_requirements"]["planned_output_directory"] / "started.json").exists()
            events.append("load")
        def load_adapter(self, path):
            events.append("adapter")
        def assert_inference(self):
            return None
        def check_effective(self):
            return None
        def generate(self, ids, before_forward):
            if not state.skip_callback:
                before_forward()
            events.append("generate")
            if state.failure:
                raise state.failure
            return {"response": json.dumps(records[ids[0]]["target"]), "generated_token_ids": [1],
                    "generated_tokens": 1025 if state.bad_tokens else 1, "stop_reason": "eos"}
        def close(self):
            events.append("close")
    state.backend = Backend
    return state


def test_default_inspect_does_not_claim_directory_load_generate_or_init_cuda(execution):
    s = execution
    report = runner.run(s.runtime, root=s.root, backend_factory=s.backend)
    assert report["samples"] == 14 and report["blocks"] == 17 and report["complete_pairs"] == 5
    assert report["input_token_range"] == [1267, 1321]
    assert s.events == ["construct", "close"]
    assert not (s.root / s.definition["execution_requirements"]["planned_output_directory"]).exists()


@pytest.mark.parametrize("field", ["wrong_length", "wrong_mode", "gpu_initialized"])
def test_inspect_drift_prevents_model_load_or_output(execution, field):
    setattr(execution, field, True)
    with pytest.raises(runner.DecodeIntegrityError):
        runner.run(execution.runtime, root=execution.root, backend_factory=execution.backend)
    assert "load" not in execution.events


def test_exact_28_success_and_reports_with_original_context(execution):
    s = execution
    result = runner.run(s.runtime, "replay", s.root, s.backend)
    folder = s.root / s.definition["execution_requirements"]["planned_output_directory"]
    assert result["candidate_all_14_selected_samples_exact"] and len(result["common_train_paired_transitions"]) == 4
    assert s.events.count("generate") == 28 and s.events.count("load") == 2
    runner.old.verify_receipt(folder, runner.sha256_file(s.runtime))
    report = runner.read(folder / "candidate_step45/report.json")
    assert report["metrics"]["counts"]["block_exact"] == 17
    assert report["strata"]["common_train"]["counts"]["blocks"] == 7
    assert report["strata"]["revision_only_train"]["counts"]["blocks"] == 10
    assert len(report["pairs"]) == sum(p["both_full_records_exact"] for p in report["pairs"]) == 5
    assert report["metrics"]["per_class"]["definition"]["support"] == 0
    assert report["metrics"]["macro_f1"] == 0.75
    before = s.events.count("load")
    with pytest.raises(FileExistsError):
        runner.run(s.runtime, "replay", s.root, s.backend)
    assert s.events.count("load") == before


def test_ordinary_generation_errors_keep_all_denominators(execution):
    s = execution
    s.failure = RuntimeError("synthetic generation error")
    runner.run(s.runtime, "replay", s.root, s.backend)
    folder = s.root / s.definition["execution_requirements"]["planned_output_directory"]
    report = runner.read(folder / "candidate_step45/report.json")
    assert report["metrics"]["counts"]["samples"] == 14 and report["metrics"]["counts"]["blocks"] == 17
    assert report["structurally_invalid_blocks"] == 17 and report["metrics"]["counts"]["block_exact"] == 0
    assert len(runner.rows(folder / "call_attempts.jsonl")) == 28


@pytest.mark.parametrize("failure", [MemoryError("synthetic OOM"), runner.DecodeIntegrityError("configuration changed")])
def test_fatal_errors_leave_failure_no_receipt_and_no_retry(execution, failure):
    s = execution
    s.failure = failure
    with pytest.raises(type(failure)):
        runner.run(s.runtime, "replay", s.root, s.backend)
    folder = s.root / s.definition["execution_requirements"]["planned_output_directory"]
    assert (folder / "failure.json").exists() and not (folder / "receipt.json").exists()
    assert s.events.count("generate") == 1


@pytest.mark.parametrize("field", ["skip_callback", "bad_tokens"])
def test_generation_contract_or_token_budget_failure_global_stop(execution, field):
    s = execution
    setattr(s, field, True)
    with pytest.raises(runner.DecodeIntegrityError):
        runner.run(s.runtime, "replay", s.root, s.backend)
    assert s.events.count("generate") == 1


def test_changed_checkpoint_before_candidate_load_stops(execution, monkeypatch):
    s = execution
    original = runner.load_execution
    def changed(path, root):
        if s.events.count("load") == 1:
            raise runner.DecodeIntegrityError("checkpoint changed")
        return original(path, root)
    monkeypatch.setattr(runner, "load_execution", changed)
    with pytest.raises(runner.DecodeIntegrityError):
        runner.run(s.runtime, "replay", s.root, s.backend)
    assert s.events.count("load") == 1 and s.events.count("generate") == 14


def test_runtime_change_after_calls_prevents_success_receipt(execution, monkeypatch):
    s = execution
    original = runner.comparison
    def changed(*args):
        replace_json(s.runtime, {"changed": True})
        return original(*args)
    monkeypatch.setattr(runner, "comparison", changed)
    with pytest.raises(runner.DecodeIntegrityError):
        runner.run(s.runtime, "replay", s.root, s.backend)
    folder = s.root / s.definition["execution_requirements"]["planned_output_directory"]
    assert (folder / "failure.json").exists() and not (folder / "receipt.json").exists()


@pytest.mark.parametrize("damage", ["illegal_kind", "missing_block", "invalid_json", "extra_key"])
def test_structural_failures_score_entire_multiblock_sample_zero(material, damage):
    _, _, selection, records = material
    predictions = [{"sample_id": r["sample_id"], "response": json.dumps(r["target"])} for r in records]
    value = copy.deepcopy(records[1]["target"])
    if damage == "illegal_kind":
        value["annotations"][0]["kind"] = "lemma"
    elif damage == "missing_block":
        value["annotations"].pop()
    elif damage == "extra_key":
        value["unexpected"] = True
    predictions[1]["response"] = "bad JSON" if damage == "invalid_json" else json.dumps(value)
    report = runner.condition_report(records, predictions, selection)
    assert report["metrics"]["counts"]["blocks"] == 17
    assert report["metrics"]["counts"]["block_exact"] == 14
    assert report["structurally_invalid_blocks"] == 3
    assert bool(report["illegal_kinds"]) == (damage == "illegal_kind")


def test_pair_strictness_and_field_reports(material):
    _, _, selection, records = material
    predictions = [{"sample_id": r["sample_id"], "response": json.dumps(r["target"])} for r in records]
    changed = copy.deepcopy(records[4]["target"])
    changed["annotations"][0]["title"] = None
    predictions[4]["response"] = json.dumps(changed)
    report = runner.condition_report(records, predictions, selection)
    assert report["title_errors_on_covered_samples"]["missing"] == 1
    assert sum(p["both_full_records_exact"] for p in report["pairs"]) == 4
    assert report["samples"][4]["errors"][0]["fields"] == ["title"]


@pytest.mark.parametrize("phase", ["train", "test", "resume", "preflight"])
def test_forbidden_phase_rejected_before_io(phase, tmp_path):
    with pytest.raises(ValueError, match="only inspect/replay"):
        runner.run(tmp_path / "absent.json", phase, tmp_path)


def test_cli_default_inspect(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(runner, "run", lambda path, phase: calls.append(phase) or {"passed": True})
    runner.main([])
    assert calls == ["inspect"] and json.loads(capsys.readouterr().out)["passed"]


@pytest.mark.parametrize("message", ["invalid generated token count", "generated sequence does not preserve the input prefix"])
def test_frozen_completion_integrity_errors_are_fatal(monkeypatch, message):
    def failure(*args):
        raise ValueError(message)
    monkeypatch.setattr(runner.RevisionBackend, "generate", failure)
    backend = object.__new__(runner.ReplayBackend)
    with pytest.raises(runner.DecodeIntegrityError):
        backend.generate([1], lambda: None)


@pytest.fixture
def runtime_material(tmp_path, monkeypatch):
    definition = {"model": {"model_directory": "tiny"}, "generation": {"do_sample": False},
                  "artifacts": {}, "conditions": [], "generation_api_evidence": "api.json",
                  "generation_call_boundary_evidence": "call_api.json"}
    files = {runner.DEFINITION: definition, runner.FREEZE: {"artifacts": {}},
             "checks.json": {"passed": True}, "inputs.json": {"passed": True},
             "api.json": {"passed": True}, "call_api.json": {"passed": True},
             "configs/m5_lora_v1/protocol.json": {"model": definition["model"], "evaluation": {}}}
    files.update({source: {"cpu_fixture": True} for source in runner.SOURCES})
    for relative, value in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        replace_json(path, value)
    monkeypatch.setattr(runner, "DEFINITION_SHA", runner.sha256_file(tmp_path / runner.DEFINITION))
    runtime = {"version": "m5-train-replay-runtime-v1", "definition": runner.DEFINITION,
               "default_phase": "inspect", "allowed_phases": list(runner.PHASES),
               "cpu_checks": "checks.json", "cpu_input_preflight": "inputs.json",
               "artifacts": {p: runner.sha256_file(tmp_path / p) for p in files}}
    path = tmp_path / "runtime.json"
    replace_json(path, runtime)
    return tmp_path, path, runtime


def test_runtime_loader_transitive_binding(runtime_material):
    root, path, _ = runtime_material
    _, definition, plan, _, _ = runner.load_execution(path, root)
    assert plan["evaluation"]["generation"] == definition["generation"]


@pytest.mark.parametrize("damage", ["missing_source", "digest", "phase", "definition", "source_content", "cpu_failed", "transitive"])
def test_runtime_loader_rejects_unaccepted_or_changed_evidence(runtime_material, damage):
    root, path, runtime = runtime_material
    if damage == "missing_source":
        runtime["artifacts"].pop(runner.SOURCES[0])
    elif damage == "digest":
        runtime["artifacts"]["inputs.json"] = "0" * 64
    elif damage == "phase":
        runtime["default_phase"] = "replay"
    elif damage == "definition":
        runtime["artifacts"][runner.DEFINITION] = "0" * 64
    elif damage == "source_content":
        (root / runner.SOURCES[0]).write_text("changed", encoding="utf-8")
    elif damage == "cpu_failed":
        replace_json(root / "checks.json", {"passed": False})
        runtime["artifacts"]["checks.json"] = runner.sha256_file(root / "checks.json")
    else:
        replace_json(root / runner.FREEZE, {"artifacts": {"unbound.json": "0" * 64}})
        runtime["artifacts"][runner.FREEZE] = runner.sha256_file(root / runner.FREEZE)
    replace_json(path, runtime)
    with pytest.raises((ValueError, runner.DecodeIntegrityError)):
        runner.load_execution(path, root)

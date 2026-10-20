"""纯合成数据/替身验证30次诊断边界，不读取私人笔记或真实权重。"""

import copy
import json
from types import SimpleNamespace

import pytest

from minigpt_cn import m5_greedy_run as runner
from minigpt_cn.m5_structure import evaluate


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return name


def records():
    result = []
    for i in range(15):
        count = 5 if i < 2 else 4
        blocks = [{"block_id": f"b{j + 1:04d}", "text": f"正文{i}:{j}"} for j in range(count)]
        result.append({"schema_version": "m5-structure-v1.2", "sample_id": f"valid{i}", "split": "valid",
                       "source": {"secret": "SOURCE_NOT_MODEL_INPUT"}, "input": {"blocks": blocks},
                       "target": {"annotations": [{"block_id": b["block_id"], "kind": "body", "title": None, "reason": None}
                                                   for b in blocks]}})
    return result


def make_workspace(root, completed=True, accepted=False, bad_generation=False):
    root.mkdir(exist_ok=True)
    put(root, "model/config.json", {})
    (root / "prompt.txt").write_text("system", encoding="utf-8")
    (root / "valid.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records()), encoding="utf-8")
    for split in ("train", "test"):
        (root / f"{split}.jsonl").write_text("MUST_NOT_PARSE_THIS_AS_RECORDS", encoding="utf-8")
    inputs = {name: runner.sha256_file(root / name) for name in (
        "model/config.json", "prompt.txt", "valid.jsonl", "train.jsonl", "test.jsonl")}
    put(root, "dataset.json", {"artifacts": inputs})
    put(root, "base.json", {"artifacts": inputs, "model_files": ["config.json"]})
    thresholds = {"json_rate": 1.0, "schema_rate": 1.0, "coverage_rate": 1.0,
                  "block_all_fields_rate": 0.8, "sample_exact_rate": 0.5, "macro_f1": 0.75,
                  "macro_f1_gain_over_same_prompt_base": 0.15,
                  "non_null_title_exact_rate": 0.8, "non_null_reason_exact_rate": 0.8}
    parent = {"version": "m5-lora-experiment-v1", "rule_version": "m5-structure-v1.2",
              "dataset_protocol": "dataset.json", "base_protocol": "base.json", "artifacts": inputs,
              "records": {s: s + ".jsonl" for s in ("train", "valid", "test")},
              "dataset_acceptance": {"valid_samples": 15}, "system_prompt": "prompt.txt",
              "model": {"model_directory": "model"}, "training": {"supervised_token_exposures_total": 48920},
              "validation": {"minimum_gates": thresholds}}
    put(root, "parent.json", parent)
    parent_artifacts = {**inputs, **{name: runner.sha256_file(root / name) for name in ("parent.json", "dataset.json", "base.json")}}
    put(root, "parent_runtime.json", {"version": "m5-lora-runtime-v1", "artifacts": parent_artifacts,
                                      "experiment_directory": "logs/parent"})
    parent_sha = runner.sha256_file(root / "parent_runtime.json")
    preflight = root / "logs/parent/preflight"
    preflight.mkdir(parents=True)
    put(root, "logs/parent/preflight/gpu_preflight.json", {"passed": True, "optimizer_steps": 0})
    runner.seal(preflight, parent_sha, {"status": "completed", "phase": "preflight"})
    phase = root / "logs/parent/train-valid"
    adapter_name = "logs/parent/train-valid/checkpoints/step_000045"
    put(root, adapter_name + "/adapter_model.safetensors", "fake adapter")
    files = runner.file_tree(root / adapter_name)
    put(root, "logs/parent/train-valid/training_result.json", {"completed": completed, "optimizer_steps": 180, "supervised_exposures": 48920})
    put(root, "logs/parent/train-valid/selection.json", {"accepted": accepted, "selected_step": 45 if accepted else None,
                                                       "candidates": [{"step": 45, "checkpoint_files": files}]})
    for name in ("sampled_base", "sampled_adapter"):
        predictions = [{"sample_id": r["sample_id"], "response": "{}"} for r in records()]
        path = root / name
        path.mkdir()
        (path / "predictions.jsonl").write_text("\n".join(json.dumps(p) for p in predictions), encoding="utf-8")
    runner.seal(phase, parent_sha, {"status": "completed", "phase": "train-valid"})
    generation = {"do_sample": False, "num_beams": 1, "temperature": None, "top_p": None,
                  "top_k": None, "min_p": None, "max_new_tokens": 1024}
    put(root, "api.json", {"greedy_generation": generation})
    put(root, "cpu.json", {"passed": True})
    for name in runner.SOURCES:
        put(root, name, "synthetic source binding")
    artifacts = {p.relative_to(root).as_posix(): runner.sha256_file(p) for p in root.rglob("*") if p.is_file()}
    plan = {"version": "m5-greedy-diagnostic-v1", "artifacts": artifacts,
            "parent_runtime": "parent_runtime.json", "parent_runtime_sha256": parent_sha,
            "parent_plan": "parent.json", "parent_plan_sha256": runner.sha256_file(root / "parent.json"),
            "adapter": {"step": 45, "files": files}, "model": parent["model"], "system_prompt": "prompt.txt",
            "valid_records": "valid.jsonl", "valid_samples": 15, "valid_blocks": 62,
            "generation": {**generation, "do_sample": bad_generation}, "max_input_tokens": 4096, "max_new_tokens": 1024,
            "scoring": {"minimum_gates": thresholds}, "experiment_directory": "logs/greedy",
            "conditions": [{"name": "base", "adapter_directory": None, "samples": 15},
                           {"name": "adapter_step_000045", "adapter_directory": adapter_name, "samples": 15}],
            "limits": {"max_model_generation_calls": 30, "training_steps": 0, "test_model_calls": 0,
                       "retry": False, "other_checkpoints_or_seeds": False},
            "historical_sampled_references": {"base": "sampled_base", "adapter_step_000045": "sampled_adapter"}}
    put(root, "plan.json", plan)
    runtime = {"version": "m5-greedy-runtime-v1", "plan": "plan.json", "plan_sha256": runner.sha256_file(root / "plan.json"),
               "cpu_checks": "cpu.json", "generation_api_check": "api.json",
               "artifacts": {**artifacts, "plan.json": runner.sha256_file(root / "plan.json")}}
    put(root, "runtime.json", runtime)
    return root / "runtime.json"


class FakeBackend:
    context_limit = 8192
    instances = []
    calls = 0
    error = None
    oversize_last = False

    def __init__(self, plan, api):
        self.runtime = {"synthetic": True}
        self.loaded = False
        self.seen = []
        self.instances.append(self)

    def encode(self, messages):
        assert "SOURCE_NOT_MODEL_INPUT" not in json.dumps(messages)
        assert "annotations" not in messages[1]["content"]
        self.seen.append(copy.deepcopy(messages))
        raw = messages[1]["content"]
        if self.oversize_last and "正文14:" in raw:
            return [1] * 4097
        return [ord(char) for char in raw]

    def load_base(self):
        assert len(self.seen) == 15
        self.loaded = True

    def load_adapter(self, path):
        assert path.name == "step_000045" and self.loaded

    def generate(self, ids):
        assert self.loaded
        type(self).calls += 1
        if self.error is not None and self.calls == 1:
            raise self.error
        value = json.loads("".join(chr(i) for i in ids))
        response = json.dumps({"annotations": [{"block_id": b["block_id"], "kind": "body", "title": None, "reason": None}
                                                for b in value["blocks"]]})
        return {"response": response, "raw_completion": response, "generated_token_ids": [1],
                "generated_tokens": 1, "runtime_error": None, "hit_token_limit": False, "seconds": 0.01}

    def close(self):
        self.loaded = False


@pytest.fixture(autouse=True)
def reset_backend():
    FakeBackend.instances, FakeBackend.calls = [], 0
    FakeBackend.error, FakeBackend.oversize_last = None, False


def test_inspect_default_no_model_or_output_and_no_train_test_parse(tmp_path):
    path = make_workspace(tmp_path)
    value = runner.run(path, root=tmp_path, backend_factory=FakeBackend)
    assert value["samples"] == 15 and value["blocks"] == 62 and not value["model_loaded"]
    assert FakeBackend.calls == 0 and not any(b.loaded for b in FakeBackend.instances)
    assert not (tmp_path / "logs/greedy").exists()


def test_exact_30_calls_allow_unaccepted_parent_without_releasing_test(tmp_path, monkeypatch):
    path = make_workspace(tmp_path)
    monkeypatch.setattr(runner, "quality_gates", lambda *args: {"passed": True})
    result = runner.run(path, "run", tmp_path, FakeBackend)
    assert result["model_generation_calls"] == FakeBackend.calls == 30
    assert result["gates"]["passed"] and result["automatic_test_allowed"] is False
    assert result["training_steps"] == result["test_calls"] == 0
    output = tmp_path / "logs/greedy"
    runner.verify_receipt(output, runner.sha256_file(path))
    assert not (tmp_path / "logs/parent/test").exists()
    for name in ("base", "adapter_step_000045"):
        metrics = json.loads((output / name / "metrics.json").read_text())
        assert metrics["counts"]["samples"] == 15 and metrics["counts"]["blocks"] == 62


@pytest.mark.parametrize("phase", ["train", "test", "preflight", "train-valid"])
def test_no_other_phase_even_before_protocol_reads(tmp_path, phase):
    with pytest.raises(ValueError, match="no train/test"):
        runner.run(tmp_path / "does-not-exist.json", phase, tmp_path, FakeBackend)
    assert not FakeBackend.instances


@pytest.mark.parametrize("options", [{"completed": False}, {"accepted": True}, {"bad_generation": True}])
def test_parent_state_or_decoder_mismatch_before_backend(tmp_path, options):
    path = make_workspace(tmp_path, **options)
    with pytest.raises(ValueError):
        runner.run(path, "run", tmp_path, FakeBackend)
    assert not FakeBackend.instances and not (tmp_path / "logs/greedy").exists()


@pytest.mark.parametrize("name", ["valid.jsonl", "test.jsonl", "src/minigpt_cn/m5_greedy_run.py",
                                   "logs/parent/train-valid/selection.json",
                                   "logs/parent/train-valid/checkpoints/step_000045/adapter_model.safetensors"])
def test_any_bound_corruption_stops_before_backend(tmp_path, name):
    path = make_workspace(tmp_path)
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        runner.run(path, "run", tmp_path, FakeBackend)
    assert not FakeBackend.instances


def test_extra_model_sidecar_rejected(tmp_path):
    path = make_workspace(tmp_path)
    (tmp_path / "model/extra").write_bytes(b"sidecar")
    with pytest.raises(ValueError, match="sidecar"):
        runner.run(path, "run", tmp_path, FakeBackend)


def test_existing_destination_rejected_before_backend(tmp_path):
    path = make_workspace(tmp_path)
    (tmp_path / "logs/greedy").mkdir()
    with pytest.raises(FileExistsError):
        runner.run(path, "run", tmp_path, FakeBackend)
    assert not FakeBackend.instances


def test_all_inputs_checked_before_load_and_no_truncation(tmp_path):
    path = make_workspace(tmp_path)
    FakeBackend.oversize_last = True
    with pytest.raises(ValueError, match="budget"):
        runner.run(path, "run", tmp_path, FakeBackend)
    assert len(FakeBackend.instances[0].seen) == 15 and not FakeBackend.instances[0].loaded
    assert not (tmp_path / "logs/greedy").exists()


def test_non_oom_error_keeps_denominator_and_never_retries(tmp_path):
    path = make_workspace(tmp_path)
    FakeBackend.error = RuntimeError("synthetic failure")
    runner.run(path, "run", tmp_path, FakeBackend)
    metrics = json.loads((tmp_path / "logs/greedy/base/metrics.json").read_text())
    predictions = (tmp_path / "logs/greedy/base/predictions.jsonl").read_text().splitlines()
    assert FakeBackend.calls == 30 and len(predictions) == 15
    assert metrics["counts"]["samples"] == 15 and metrics["counts"]["blocks"] == 62
    assert metrics["counts"]["json_valid"] == 14 and json.loads(predictions[0])["response"] == ""


class OutOfMemoryError(RuntimeError):
    pass


@pytest.mark.parametrize("error", [MemoryError("oom"), OutOfMemoryError("cuda oom")])
def test_oom_preserves_failure_stops_globally_and_refuses_retry(tmp_path, error):
    path = make_workspace(tmp_path)
    FakeBackend.error = error
    with pytest.raises(type(error)):
        runner.run(path, "run", tmp_path, FakeBackend)
    output = tmp_path / "logs/greedy"
    assert FakeBackend.calls == 1 and (output / "failure.json").exists()
    assert not (output / "receipt.json").exists() and not (output / "adapter_step_000045").exists()
    with pytest.raises(FileExistsError):
        runner.run(path, "run", tmp_path, FakeBackend)
    assert FakeBackend.calls == 1


def test_pairing_counts_invalid_whole_sample_without_partial_credit():
    row = records()[0]
    before = {"sample_id": row["sample_id"], "response": json.dumps(row["target"])}
    shortened = {"annotations": row["target"]["annotations"][:-1]}
    after = {"sample_id": row["sample_id"], "response": json.dumps(shortened)}
    result = runner.paired_changes([row], [before], [after])
    assert result["fewer_exact_blocks"] == 1 and result["samples"][0]["block_exact_delta"] == -5
    assert evaluate([(row["input"], row["target"], after["response"])])["counts"]["block_exact"] == 0
    with pytest.raises(ValueError, match="IDs"):
        runner.paired_changes([row], [{**before, "sample_id": "another"}], [after])


@pytest.mark.parametrize("training,requires_grad", [(True, False), (False, True)])
def test_real_backend_rejects_train_mode_or_unfrozen_parameters(training, requires_grad):
    backend = runner.GreedyBackend.__new__(runner.GreedyBackend)
    backend.model = SimpleNamespace(training=training, parameters=lambda: [SimpleNamespace(requires_grad=requires_grad)])
    with pytest.raises(ValueError, match="fully frozen"):
        backend.assert_frozen()
    assert not hasattr(runner.GreedyBackend, "train")


def test_cli_rejects_test(tmp_path):
    with pytest.raises(SystemExit):
        runner.main(["--phase", "test", "--execution", str(tmp_path / "no.json")])

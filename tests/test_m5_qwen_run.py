"""Qwen入口的CPU边界测试；只用合成输入和替身，不读模型/个人笔记。"""

import copy
import json
import tempfile
from pathlib import Path

import pytest

from minigpt_cn.m5_qwen_run import (
    completion_record,
    installed_packages,
    messages_for,
    run,
    sha256_file,
)
from minigpt_cn.m5_rule_run import PROJECT


@pytest.fixture
def local_tmp():
    root = PROJECT / "logs"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m5-qwen-test-", dir=root) as name:
        yield Path(name)


def annotation(kind="body", title=None):
    return {"block_id": "b0001", "kind": kind, "title": title, "reason": None}


def make_protocol(tmp, count=1):
    model = tmp / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    prompt = tmp / "prompt.txt"
    prompt.write_text("task-only", encoding="utf-8")
    data = tmp / "samples.jsonl"
    rows = [{"review_id": f"s{i}", "rule_version": "m5-structure-v1.1",
             "input": {"blocks": [{"block_id": "b0001", "text": "普通正文"}]},
             "source": "SECRET_SOURCE", "proposed_target": {
                 "annotations": [annotation("heading", "SECRET_REFERENCE")]}}
            for i in range(count)]
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    def relative(path):
        return path.relative_to(PROJECT).as_posix()

    settings = {
        "protocol_version": "m5-qwen-calibration-v1.1", "rule_version": "m5-structure-v1.1",
        "model_directory": relative(model), "model_files": ["config.json"],
        "system_prompt": relative(prompt),
        "data_files": {"calibration": {"path": relative(data), "samples": count, "blocks": count}},
        "max_input_tokens": 20, "generation": {"max_new_tokens": 10},
        "artifacts": {relative(p): sha256_file(p) for p in [prompt, data, model / "config.json"]},
    }
    path = tmp / "protocol.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    return path, prompt, settings


class FakeBackend:
    context_limit = 64

    def __init__(self, protocol):
        self.runtime = {"synthetic_backend": True}
        self.loaded = False
        self.seen = []

    def encode(self, messages):
        self.seen.append(copy.deepcopy(messages))
        return [1, 2, 3]

    def load(self):
        self.loaded = True

    def generate(self, ids):
        assert self.loaded and ids == [1, 2, 3]
        response = json.dumps({"annotations": [annotation()]})
        return {"response": response, "raw_completion": response + "<eos>",
                "generated_token_ids": [11, 99], "generated_tokens": 2,
                "stop_reason": "eos", "hit_token_limit": False,
                "runtime_error": None, "seconds": 0.0}


class FakeTokenizer:
    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join({1: "PROMPT", 2: "```json\n", 3: "{}", 4: "<think>", 99: "<eos>"}[i]
                       for i in ids)


def test_environment_inventory_excludes_source_metadata(local_tmp, monkeypatch):
    site = local_tmp / "site"
    source = local_tmp / "source"
    for parent, name, version in [(site, "torch", "2.8.0"), (source, "torch", "999.0")]:
        metadata = parent / f"{name}-{version}.dist-info" / "METADATA"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(f"Name: {name}\nVersion: {version}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.setattr("sysconfig.get_paths", lambda: {"purelib": str(site), "platlib": str(site)})
    assert installed_packages() == {"torch": "2.8.0"}


def test_only_generated_tokens_scored_without_repair():
    row = completion_record(FakeTokenizer(), [1], [1, 2, 3, 4, 99], [99], 5)
    assert row["response"] == "```json\n{}<think>"
    assert row["raw_completion"] == "```json\n{}<think><eos>"
    assert row["generated_token_ids"] == [2, 3, 4, 99]
    assert row["stop_reason"] == "eos" and not row["hit_token_limit"]


def test_limit_keeps_incomplete_json_in_denominator():
    row = completion_record(FakeTokenizer(), [1], [1, 2, 3], [99], 2)
    assert row["response"] == "```json\n{}"
    assert row["hit_token_limit"] and row["stop_reason"] == "length"


@pytest.mark.parametrize("sequence", [[2, 3], [1], [1, 2, 3, 4, 99]])
def test_invalid_generation_prefix_or_length(sequence):
    with pytest.raises(ValueError):
        completion_record(FakeTokenizer(), [1], sequence, [99], 2)


def test_serialization_preserves_text_and_accepts_no_metadata():
    value = {"blocks": [{"block_id": "b0001", "text": "公式\\alpha\r\n```\n代码\n```"}]}
    before = copy.deepcopy(value)
    messages = messages_for("system", value)
    assert json.loads(messages[1]["content"]) == value == before
    assert messages[0] == {"role": "system", "content": "system"}
    with pytest.raises(ValueError):
        messages_for("system", {**value, "reference": "forbidden"})


def test_reference_is_not_forwarded_and_cannot_be_copied(local_tmp):
    protocol, _, _ = make_protocol(local_tmp)
    backend = FakeBackend({})
    output = local_tmp / "run"
    metrics = run(protocol, output, backend_factory=lambda _: backend)
    assert metrics["counts"]["kind_correct"] == 0
    assert "SECRET" not in json.dumps(backend.seen)
    assert metrics["counts"]["samples"] == 1
    assert json.loads((output / "run_manifest.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        run(protocol, output, backend_factory=lambda _: pytest.fail("should not load backend"))


def test_artifact_change_rejected_before_output_or_backend(local_tmp):
    protocol, prompt, _ = make_protocol(local_tmp)
    prompt.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        run(protocol, local_tmp / "run", backend_factory=lambda _: pytest.fail("backend"))
    assert not (local_tmp / "run").exists()


def test_unbound_model_sidecar_rejected(local_tmp):
    protocol, _, settings = make_protocol(local_tmp)
    (PROJECT / settings["model_directory"] / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected"):
        run(protocol, local_tmp / "run", backend_factory=lambda _: pytest.fail("backend"))


@pytest.mark.parametrize("input_limit,output_limit", [(2, 10), (60, 10)])
def test_all_budgets_checked_before_model_load(local_tmp, input_limit, output_limit):
    protocol, _, settings = make_protocol(local_tmp)
    settings["max_input_tokens"] = input_limit
    settings["generation"]["max_new_tokens"] = output_limit
    protocol.write_text(json.dumps(settings))
    backend = FakeBackend({})
    with pytest.raises(ValueError, match="budget"):
        run(protocol, local_tmp / "run", backend_factory=lambda _: backend)
    assert not backend.loaded and not (local_tmp / "run").exists()


def test_generation_exception_is_scored_and_preserved(local_tmp):
    protocol, _, _ = make_protocol(local_tmp, count=2)

    class FailingBackend(FakeBackend):
        def generate(self, ids):
            raise RuntimeError("synthetic generation failure")

    output = local_tmp / "run"
    result = run(protocol, output, backend_factory=FailingBackend)
    assert result["counts"]["samples"] == result["counts"]["blocks"] == 2
    assert result["counts"]["json_valid"] == result["counts"]["kind_correct"] == 0
    assert result["per_class"]["heading"]["fn"] == 2
    assert len((output / "errors.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "run_manifest.json").read_text())["runtime_errors"] == 2


def test_midrun_mutation_preserves_raw_output_but_rejects_results(local_tmp):
    protocol, prompt, _ = make_protocol(local_tmp)

    class MutatingBackend(FakeBackend):
        def generate(self, ids):
            prompt.write_text("changed", encoding="utf-8")
            return super().generate(ids)

    output = local_tmp / "run"
    with pytest.raises(ValueError, match="changed during run"):
        run(protocol, output, backend_factory=MutatingBackend)
    assert (output / "predictions.jsonl").exists() and (output / "failure.json").exists()
    assert not (output / "metrics.json").exists()

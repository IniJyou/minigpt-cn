"""LoRA入口CPU边界测试；合成数据/替身，不读个人笔记或Qwen权重。"""

import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from minigpt_cn import m5_lora_run as runner


@pytest.fixture
def workspace():
    with tempfile.TemporaryDirectory(prefix="m5-lora-test-", dir=runner.PROJECT / "logs") as name:
        root = Path(name)
        (root / "logs").mkdir()
        yield root


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return json.dumps(messages, ensure_ascii=False) + "|assistant|"

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        end = "<|im_end|>"
        if text.endswith(end):
            return [ord(c) + 1 for c in text[:-len(end)]] + [151645]
        return [ord(c) + 1 for c in text]

    def decode(self, ids, **kwargs):
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "".join(chr(i - 1) for i in ids)


def record(split="train"):
    return {"schema_version": "m5-structure-v1.2", "sample_id": split + "1", "split": split,
            "source": {"source_id": "SECRET_SOURCE", "group_id": split, "synthetic": True},
            "input": {"blocks": [{"block_id": "b0001", "text": "普通正文"}]},
            "target": {"annotations": [{"block_id": "b0001", "kind": "body", "title": None, "reason": None}]}}


def interface():
    return {"eos_token_id": 151645, "pad_token_id": 151643, "max_input_tokens": 4096,
            "max_target_tokens_including_eos": 1024, "max_sequence_tokens": 5120}


def thresholds():
    return {"json_rate": 1.0, "schema_rate": 1.0, "coverage_rate": 1.0,
            "block_all_fields_rate": 0.8, "sample_exact_rate": 0.5, "macro_f1": 0.1,
            "macro_f1_gain_over_same_prompt_base": 0.1,
            "non_null_title_exact_rate": 0.0, "non_null_reason_exact_rate": 0.0}


def make_execution(root):
    def put(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return name

    prompt = root / "prompt.txt"
    prompt.write_text("system", encoding="utf-8")
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    paths = {}
    for split in ("train", "valid", "test"):
        paths[split] = put(split + ".jsonl", record(split))
    artifacts = {p: runner.sha256_file(root / p) for p in ["prompt.txt", "model/config.json", *paths.values()]}
    dataset = put("dataset.json", {"artifacts": artifacts})
    base = put("base.json", {"artifacts": artifacts, "model_files": ["config.json"]})
    item = runner.training_item(Tokenizer(), "system", record(), interface())
    count = item["target_tokens_including_eos"]
    plan = {"version": "m5-lora-experiment-v1", "rule_version": "m5-structure-v1.2",
            "dataset_protocol": dataset, "base_protocol": base, "artifacts": artifacts,
            "records": paths, "system_prompt": "prompt.txt", "model": {"model_directory": "model"},
            "dataset_acceptance": {s + "_samples": 1 for s in paths},
            "training_interface": interface(), "training": {"supervised_tokens_per_epoch": count,
                "supervised_token_exposures_total": count * 2, "max_optimizer_steps": 2,
                "checkpoint_optimizer_steps": [1, 2]},
            "validation": {"minimum_gates": thresholds()}}
    put("plan.json", plan)
    execution = {"version": "m5-lora-runtime-v1", "plan": "plan.json",
                 "plan_sha256": runner.sha256_file(root / "plan.json"), "experiment_directory": "logs/experiment",
                 "artifacts": artifacts}
    put("execution.json", execution)
    return root / "execution.json", plan


class Backend:
    context_limit = 8192
    seen = []
    failures = False

    def __init__(self, plan):
        self.plan, self.tokenizer = plan, Tokenizer()
        self.runtime = {"synthetic": True}
        self.adapter, self.loaded = False, False

    def encode(self, messages):
        self.seen.append(copy.deepcopy(messages))
        return self.tokenizer.encode(self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    def load_base(self):
        self.loaded = True
        self.runtime["model_loaded"] = True

    def load_adapter(self, path):
        self.adapter = True

    def generate(self, ids):
        assert self.loaded
        response = json.dumps(record()["target"]) if self.adapter and not self.failures else ""
        return {"response": response, "raw_completion": response, "generated_token_ids": [42],
                "generated_tokens": 1, "runtime_error": None}

    def gpu_preflight(self, items):
        return {"passed": True, "optimizer_steps": 0, "initial_adapter_sha256": "init"}

    def train(self, items, folder, expected_initial):
        assert expected_initial == "init"
        checkpoints = []
        for step in self.plan["training"]["checkpoint_optimizer_steps"]:
            path = folder / "checkpoints" / f"step_{step:06d}"
            path.mkdir(parents=True)
            (path / "adapter_model.safetensors").write_bytes(b"synthetic adapter")
            checkpoints.append({"step": step, "files": runner.file_tree(path)})
        return {"completed": True, "optimizer_steps": 2, "checkpoints": checkpoints,
                "supervised_exposures": self.plan["training"]["supervised_token_exposures_total"]}

    def close(self):
        self.loaded = False


def test_supervision_masks_only_prefix_and_preserves_target_eos():
    item = runner.training_item(Tokenizer(), "system", record(), interface())
    n = item["input_tokens"]
    assert item["labels"][:n] == [-100] * n
    assert item["labels"][n:] == item["input_ids"][n:]
    assert item["labels"][-1] == 151645
    assert "SECRET_SOURCE" not in Tokenizer().decode(item["input_ids"][:n],
        skip_special_tokens=False, clean_up_tokenization_spaces=False)


@pytest.mark.parametrize("key", ["max_input_tokens", "max_target_tokens_including_eos", "max_sequence_tokens"])
def test_token_budget_never_truncates(key):
    limits = interface()
    limits[key] = 1
    with pytest.raises(ValueError, match="budget"):
        runner.training_item(Tokenizer(), "system", record(), limits)


def test_target_prefix_mismatch_rejected():
    class BadTokenizer(Tokenizer):
        def encode(self, text, **kwargs):
            result = super().encode(text, **kwargs)
            if text.endswith("<|im_end|>"):
                result[0] += 1
            return result
    with pytest.raises(ValueError, match="prefix"):
        runner.training_item(BadTokenizer(), "system", record(), interface())


def test_collator_right_pads_masks_and_does_not_shift():
    items = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
             {"input_ids": [4, 5], "labels": [-100, 5]}]
    batch = runner.collate(items, "cpu")
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 151643]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert runner.supervised_count(items) == 3


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 6)
        self.linear = torch.nn.Linear(6, 10)

    def forward(self, input_ids, attention_mask, labels, num_items_in_batch=None, **kwargs):
        logits = self.linear(self.embedding(input_ids))
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 10),
            labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
        denominator = num_items_in_batch or (labels[:, 1:] != -100).sum()
        return SimpleNamespace(loss=loss / denominator)


def small_items():
    return [{"sample_id": "a", "input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
            {"sample_id": "b", "input_ids": [5, 6, 7], "labels": [-100, -100, 7]}]


def test_accumulated_loss_and_gradients_match_single_token_mean():
    torch.manual_seed(42)
    model = TinyLM()
    full = copy.deepcopy(model)
    loss, count = runner.accumulate(model, small_items(), "cpu")
    batch = runner.collate(small_items(), "cpu", pad_token_id=0)
    expected = full(**batch).loss
    expected.backward()
    assert count == 3 and loss == pytest.approx(expected.item(), abs=1e-6)
    for a, b in zip(model.parameters(), full.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-5)


def train_settings():
    return {"gradient_accumulation_steps": 2, "optimizer_steps_per_epoch": 1, "epochs": 2,
            "seed": 42, "max_optimizer_steps": 2, "max_wall_train_seconds": 60,
            "clip_grad_norm": 1.0, "checkpoint_epochs": [1, 2], "supervised_token_exposures_total": 6}


def test_training_updates_only_complete_groups_and_saves_fixed_steps():
    model = TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    saves, steps = [], []
    result = runner.train_loop(model, small_items(), train_settings(), optimizer, scheduler,
                               lambda step, epoch, order: saves.append((step, epoch, order)), steps.append, device="cpu")
    assert result["optimizer_steps"] == 2 and result["supervised_exposures"] == 6
    assert [(s, e) for s, e, _ in saves] == [(1, 1), (2, 2)]
    assert [r["supervised_tokens"] for r in steps] == [3, 3]


def test_deadline_stops_before_update():
    model = TinyLM()
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ticks = iter([0.0, 61.0])
    with pytest.raises(TimeoutError):
        runner.train_loop(model, small_items(), train_settings(), optimizer, scheduler,
                          lambda *_: pytest.fail("save"), lambda _: pytest.fail("update"),
                          device="cpu", clock=lambda: next(ticks))
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value)


def test_nonfinite_loss_stops_before_update():
    class BadLM(TinyLM):
        def forward(self, **kwargs):
            return SimpleNamespace(loss=self.linear.weight.sum() * float("nan"))
    model = BadLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    with pytest.raises(FloatingPointError):
        runner.train_loop(model, small_items(), train_settings(), optimizer, scheduler,
                          lambda *_: pytest.fail("save"), lambda _: pytest.fail("step"), device="cpu")


def test_receipt_rejects_tampered_output(workspace):
    folder = workspace / "phase"
    folder.mkdir()
    (folder / "selection.json").write_text('{"accepted":false}')
    runner.seal(folder, "execution", {"status": "completed"})
    runner.verify_receipt(folder, "execution")
    (folder / "selection.json").write_text('{"accepted":true}')
    with pytest.raises(ValueError, match="integrity"):
        runner.verify_receipt(folder, "execution")


def test_inspect_has_no_model_or_output_and_does_not_read_test(workspace, monkeypatch):
    execution, _ = make_execution(workspace)
    original = runner.load_records
    def load(plan, split, root):
        assert split != "test"
        return original(plan, split, root)
    monkeypatch.setattr(runner, "load_records", load)
    result = runner.run(execution, root=workspace, backend_factory=Backend)
    assert result["model_loaded"] is False and not (workspace / "logs/experiment").exists()
    assert "SECRET_SOURCE" not in json.dumps(Backend.seen)


def test_tampered_bound_input_rejected_before_backend(workspace):
    execution, _ = make_execution(workspace)
    (workspace / "prompt.txt").write_text("changed")
    with pytest.raises(ValueError, match="SHA256"):
        runner.run(execution, root=workspace, backend_factory=lambda _: pytest.fail("backend"))


def test_unknown_base_sidecar_is_rejected(workspace):
    execution, _ = make_execution(workspace)
    (workspace / "model/adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="sidecar"):
        runner.run(execution, root=workspace, backend_factory=lambda _: pytest.fail("backend"))


def test_test_is_guarded_before_reading_or_constructing_backend(workspace, monkeypatch):
    execution, _ = make_execution(workspace)
    monkeypatch.setattr(runner, "load_records", lambda *_: pytest.fail("read test"))
    with pytest.raises(FileNotFoundError):
        runner.run(execution, "test", root=workspace, backend_factory=lambda _: pytest.fail("backend"))


def test_fake_phased_flow_selects_earlier_checkpoint_and_test_once(workspace):
    execution, _ = make_execution(workspace)
    runner.run(execution, "preflight", root=workspace, backend_factory=Backend)
    with pytest.raises(FileExistsError):
        runner.run(execution, "preflight", root=workspace, backend_factory=lambda _: pytest.fail("backend"))
    selection = runner.run(execution, "train-valid", root=workspace, backend_factory=Backend)
    assert selection["accepted"] and selection["selected_step"] == 1
    result = runner.run(execution, "test", root=workspace, backend_factory=Backend)
    assert result["gates"]["passed"]
    with pytest.raises(FileExistsError):
        runner.run(execution, "test", root=workspace, backend_factory=lambda _: pytest.fail("backend"))


def test_no_validation_pass_leaves_test_unread(workspace, monkeypatch):
    class BadBackend(Backend):
        failures = True
    execution, _ = make_execution(workspace)
    runner.run(execution, "preflight", root=workspace, backend_factory=BadBackend)
    selection = runner.run(execution, "train-valid", root=workspace, backend_factory=BadBackend)
    assert not selection["accepted"]
    monkeypatch.setattr(runner, "load_records", lambda *_: pytest.fail("test read"))
    with pytest.raises(ValueError, match="accepted"):
        runner.run(execution, "test", root=workspace, backend_factory=lambda _: pytest.fail("backend"))


def test_checkpoint_mutation_prevents_test(workspace):
    execution, _ = make_execution(workspace)
    runner.run(execution, "preflight", root=workspace, backend_factory=Backend)
    runner.run(execution, "train-valid", root=workspace, backend_factory=Backend)
    (workspace / "logs/experiment/train-valid/checkpoints/step_000001/adapter_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        runner.run(execution, "test", root=workspace, backend_factory=lambda _: pytest.fail("backend"))


def test_evaluation_never_accepts_a_shorter_prediction_list(workspace):
    with pytest.raises(ValueError, match="denominators"):
        runner.evaluate_output([record("valid")], [], workspace)


def test_failed_preflight_cannot_publish_a_receipt(workspace):
    class BadPreflight(Backend):
        def gpu_preflight(self, items):
            return {"passed": False, "optimizer_steps": 0}
    execution, _ = make_execution(workspace)
    with pytest.raises(ValueError, match="preflight"):
        runner.run(execution, "preflight", root=workspace, backend_factory=BadPreflight)
    folder = workspace / "logs/experiment/preflight"
    assert (folder / "failure.json").exists() and not (folder / "receipt.json").exists()


def test_incomplete_training_cannot_select_or_evaluate_checkpoints(workspace):
    class Incomplete(Backend):
        def train(self, items, folder, expected_initial):
            return {"completed": False}
        def load_adapter(self, path):
            pytest.fail("incomplete run must not evaluate checkpoint")
    execution, _ = make_execution(workspace)
    runner.run(execution, "preflight", root=workspace, backend_factory=Backend)
    with pytest.raises(ValueError, match="incomplete"):
        runner.run(execution, "train-valid", root=workspace, backend_factory=Incomplete)
    assert not (workspace / "logs/experiment/train-valid/selection.json").exists()
    assert not (workspace / "logs/experiment/train-valid/receipt.json").exists()


def test_generation_errors_remain_in_all_denominators(workspace):
    class FailedGeneration(Backend):
        def generate(self, ids):
            raise RuntimeError("synthetic failure")
    backend = FailedGeneration({"training_interface": interface()})
    backend.load_base()
    rr = [record("valid"), record("valid")]
    rr[1]["sample_id"] = "valid2"
    metrics = runner.evaluate_model(backend, "system", rr, workspace / "eval")
    assert metrics["counts"]["samples"] == metrics["counts"]["blocks"] == 2
    assert metrics["counts"]["json_valid"] == 0
    assert metrics["per_class"]["body"]["fn"] == 2
    assert len((workspace / "eval/predictions.jsonl").read_text().splitlines()) == 2


def test_project_path_escape_is_rejected(workspace):
    with pytest.raises(ValueError, match="escapes"):
        runner.inside(workspace, "../outside")


def metric(f1=0.8, block=0.9, sample=0.8):
    return {"macro_f1": f1, "rates": {"json_valid": 1.0, "schema_valid": 1.0, "coverage_valid": 1.0,
            "block_exact": block, "sample_exact": sample, "nonnull_title_correct": 1.0,
            "nonnull_reason_correct": 1.0}}


def test_selection_uses_all_gates_before_unrounded_ranking():
    gates = thresholds()
    gates["macro_f1"] = 0.75
    candidates = [{"step": 90, "metrics": metric(0.80001)}, {"step": 45, "metrics": metric(0.8)}]
    invalid = {"step": 180, "metrics": metric(1.0)}
    invalid["metrics"]["rates"]["json_valid"] = 0.9
    result = runner.choose_checkpoint(candidates + [invalid], metric(0.0), gates)
    assert result["selected_step"] == 90
    assert not result["candidates"][-1]["gates"]["passed"]

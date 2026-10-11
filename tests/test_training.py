"""D阶段：原作业更新公式、完整验证、项目训练及真实恢复回归。"""

import json
import math
import random
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from minigpt_cn.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from minigpt_cn.data.tokens import get_batch, read_tokens
from minigpt_cn.evaluation.evaluate import evaluate
from minigpt_cn.evaluation.evaluate import main as evaluate_main
from minigpt_cn.model import MiniGPT
from minigpt_cn.tokenizer import BPETokenizer
from minigpt_cn.training.checkpoint import load_checkpoint, save_checkpoint
from minigpt_cn.training.optim import AdamW, clip_gradient_norm, lr_cosine_schedule
from minigpt_cn.training.train import main as train_main
from minigpt_cn.training.train import train


@pytest.fixture(autouse=True)
def isolate_randomness():
    threads, python_state, numpy_state = (
        torch.get_num_threads(),
        random.getstate(),
        np.random.get_state(),
    )
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield
    torch.set_num_threads(threads)
    random.setstate(python_state)
    np.random.set_state(numpy_state)


@pytest.fixture
def run(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    BPETokenizer(vocab, [], ["<|endoftext|>"]).save(tokenizer)
    train_path, valid_path = tmp_path / "train.bin", tmp_path / "valid.bin"
    np.tile([1, 2, 3, 4], 16).astype("<u2").tofile(train_path)
    np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3], dtype="<u2").tofile(valid_path)
    return RunConfig(
        ModelConfig(257, 4, 1, 2, 8, 16, dropout=0.1),
        TrainingConfig(
            batch_size=2,
            max_steps=4,
            learning_rate=0.02,
            min_learning_rate=0.002,
            warmup_steps=0,
            lr_decay_steps=10,
            device="cpu",
            precision="float32",
            log_interval=1,
            eval_interval=2,
            save_interval=2,
        ),
        DataConfig(str(train_path), str(valid_path), str(tokenizer)),
    )


def assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            assert_nested_equal(a, e)
    else:
        assert actual == expected


def test_adamw_matches_assignment_formula_and_zero_lr_is_finite():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    optimizer = AdamW([parameter], lr=0.01, betas=(0.9, 0.95), eps=0.1, weight_decay=0.2)
    expected, m, v = parameter.detach().clone(), torch.zeros(3), torch.zeros(3)
    for step, gradient in enumerate(
        (torch.tensor([0.2, 0.0, -0.3]), torch.tensor([-0.1, 0.5, 0.0])), 1
    ):
        parameter.grad = gradient
        m = 0.9 * m + 0.1 * gradient
        v = 0.95 * v + 0.05 * gradient.square()
        expected = expected * (1 - 0.01 * 0.2) - 0.01 * math.sqrt(1 - 0.95**step) / (
            1 - 0.9**step
        ) * m / (v.sqrt() + 0.1)
        optimizer.step()
        torch.testing.assert_close(parameter, expected)
    optimizer.param_groups[0]["lr"] = 0
    parameter.grad.zero_()
    before = parameter.detach().clone()
    optimizer.step()
    assert torch.isfinite(parameter).all()
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert optimizer.state[parameter]["step"] == 3


def test_optimizer_rejects_low_precision_and_norm_clip_is_global():
    low_precision = torch.nn.Parameter(torch.ones(2, dtype=torch.float16))
    low_precision.grad = torch.zeros_like(low_precision)
    with pytest.raises(ValueError, match="FP32"):
        AdamW([low_precision]).step()
    first, second = torch.nn.Parameter(torch.ones(1)), torch.nn.Parameter(torch.ones(1))
    first.grad, second.grad = torch.tensor([3.0]), torch.tensor([4.0])
    assert clip_gradient_norm([first, second], 2.0) == 5.0
    torch.testing.assert_close(first.grad, torch.tensor([3.0 * 2 / (5 + 1e-6)]))
    torch.testing.assert_close(second.grad, torch.tensor([4.0 * 2 / (5 + 1e-6)]))
    assert lr_cosine_schedule(0, 1, 0.1, 2, 10) == 0
    assert lr_cosine_schedule(2, 1, 0.1, 2, 10) == 1
    assert lr_cosine_schedule(10, 1, 0.1, 2, 10) == 0.1
    assert lr_cosine_schedule(20, 1, 0.1, 0, 10) == 0.1


def test_fixed_batch_overfit_decreases_loss(run):
    model = MiniGPT(replace(run.model, dropout=0))
    optimizer = AdamW(model.parameters(), lr=0.02)
    x, y = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 1]])
    initial = model(x, y).loss.item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = model(x, y).loss
        assert torch.isfinite(loss)
        loss.backward()
        clip_gradient_norm(model.parameters(), 1)
        optimizer.step()
    assert model(x, y).loss.item() < initial * 0.1
    assert all(torch.isfinite(p).all() for p in model.parameters())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_actual_training_resume_matches_continuous_including_dropout(run, tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CPU CI 不需要 GPU")
    run = replace(
        run,
        training=replace(
            run.training, device=device, precision="bfloat16" if device == "cuda" else "float32"
        ),
    )
    continuous, first, resumed = tmp_path / "continuous", tmp_path / "first", tmp_path / "resumed"
    train(run, continuous)
    train(run, first, stop_after=2)
    result = train(run, resumed, resume=first / "ckpt_final.pt")
    assert result["iteration"] == 4
    expected = torch.load(continuous / "ckpt_final.pt", weights_only=True)
    actual = torch.load(resumed / "ckpt_final.pt", weights_only=True)
    for key in ("model_state_dict", "optimizer", "sampler_state", "rng_state"):
        assert_nested_equal(actual[key], expected[key])
    assert actual["iteration"] == 4
    assert (
        actual["run_state"]["elapsed_seconds"]
        > torch.load(first / "ckpt_final.pt", weights_only=True)["run_state"]["elapsed_seconds"]
    )
    assert (first / "step_000002.pt").is_file() and not (first / "step_000003.pt").exists()
    # 共享参数只出现一次；优化器每个参数实际更新次数与 checkpoint 一致。
    assert all(state["step"] == 4 for state in actual["optimizer"]["state"].values())
    assert all(value.dtype == torch.float32 for value in actual["model_state_dict"].values())
    assert all(
        value.dtype == torch.float32
        for state in actual["optimizer"]["state"].values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    model = load_checkpoint(resumed / "ckpt_final.pt").model
    assert len(actual["optimizer"]["state"]) == len(list(model.parameters()))
    expected_rows = [
        json.loads(line) for line in (continuous / "metrics.jsonl").read_text().splitlines()
    ]
    actual_rows = [
        json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()
    ]
    for step in (3, 4):
        e = next(row for row in expected_rows if row["event"] == "train" and row["step"] == step)
        a = next(row for row in actual_rows if row["event"] == "train" and row["step"] == step)
        assert (a["loss"], a["lr"]) == (e["loss"], e["lr"])


def test_full_validation_covers_tail_weights_by_targets_and_restores_mode(run):
    model = MiniGPT(run.model).train()
    tokens = read_tokens(run.data.valid_path, run.model.vocab_size)
    calls, grad_modes = [], []

    def observe(module, inputs):
        calls.append(inputs[0].shape[1])
        grad_modes.append(torch.is_grad_enabled())

    handle = model.register_forward_pre_hook(observe)
    report = evaluate(model, tokens)
    handle.remove()
    assert report["targets"] == 10 and calls == [4, 4, 2]
    assert not any(grad_modes) and model.training
    model.eval()
    weighted = 0
    with torch.no_grad():
        for start in (0, 4, 8):
            end = min(start + 4, 10)
            x = torch.tensor(tokens[start:end].astype(np.int64))[None]
            y = torch.tensor(tokens[start + 1 : end + 1].astype(np.int64))[None]
            weighted += model(x, y).loss.item() * (end - start)
    assert report["loss"] == weighted / 10
    assert report["ppl"] == math.exp(report["loss"])


def test_bad_file_format_and_sampling_bounds(run, tmp_path):
    bad = tmp_path / "bad.bin"
    for content in (b"a", b"\x93NUMPY", b"\xff\xff\x00\x00"):
        bad.write_bytes(content)
        with pytest.raises(ValueError):
            read_tokens(bad, 257)
    generator = torch.Generator().manual_seed(7)
    tokens = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    x, y = get_batch(tokens, 2, 4, "cpu", generator=generator)
    assert x.dtype == y.dtype == torch.long
    assert x.tolist() == [[1, 2, 3, 4]] * 2 and y.tolist() == [[2, 3, 4, 5]] * 2
    with pytest.raises(ValueError, match="长度"):
        get_batch(tokens, 2, 5, "cpu", generator=generator)


def test_resume_boundary_and_changed_settings_data_are_rejected(run, tmp_path):
    first = tmp_path / "first"
    train(run, first, stop_after=2)
    checkpoint = first / "ckpt_final.pt"
    unchanged = checkpoint.read_bytes()
    target = tmp_path / "noop"
    assert (
        train(replace(run, training=replace(run.training, max_steps=1)), target, resume=checkpoint)[
            "iteration"
        ]
        == 2
    )
    assert not target.exists() and checkpoint.read_bytes() == unchanged
    with pytest.raises(ValueError, match="仅允许"):
        train(
            replace(run, training=replace(run.training, weight_decay=0)), target, resume=checkpoint
        )
    altered = read_tokens(run.data.train_path, 257).copy()
    altered[0] = 10
    altered.tofile(run.data.train_path)
    with pytest.raises(ValueError, match="校验和"):
        train(run, target, resume=checkpoint)
    assert not target.exists()


def test_reject_inference_resume_existing_output_and_nan_final(run, tmp_path, monkeypatch):
    inference = tmp_path / "inference.pt"
    save_checkpoint(
        inference, MiniGPT(run.model), tokenizer_path=run.data.tokenizer_path, iteration=5000
    )
    with pytest.raises(ValueError, match="训练 checkpoint"):
        train(run, tmp_path / "resume", resume=inference)
    with pytest.raises(FileExistsError):
        train(run, tmp_path)
    original_step = AdamW.step

    def poison(optimizer):
        original_step(optimizer)
        with torch.no_grad():
            optimizer.param_groups[0]["params"][0].fill_(float("nan"))

    monkeypatch.setattr(AdamW, "step", poison)
    broken = tmp_path / "nan"
    with pytest.raises(FloatingPointError, match="更新后参数"):
        train(run, broken)
    assert not (broken / "ckpt_final.pt").exists()
    assert not list(broken.glob("*.pt"))


def test_cli_uses_run_yaml_and_evaluates_saved_configuration(run, tmp_path):
    config = tmp_path / "run.yaml"
    import yaml

    config.write_text(yaml.safe_dump(asdict(run)), encoding="utf-8")
    out = tmp_path / "cli"
    result = train_main(["--config", str(config), "--out-dir", str(out), "--stop-after", "2"])
    assert result["iteration"] == 2
    report_path = tmp_path / "eval.json"
    report = evaluate_main(
        [
            "--checkpoint",
            str(out / "ckpt_final.pt"),
            "--data",
            run.data.valid_path,
            "--output",
            str(report_path),
        ]
    )
    assert report["targets"] == 10 and report["iteration"] == 2
    assert json.loads(report_path.read_text())["loss"] == result["validation"]["loss"]


@pytest.mark.parametrize("failure", ["loss", "gradient", "validation", "optimizer"])
def test_nonfinite_checks_abort_without_normal_final(run, tmp_path, monkeypatch, failure):
    forward = MiniGPT.forward
    step = AdamW.step

    def poison_forward(model, *args, **kwargs):
        output = forward(model, *args, **kwargs)
        if failure == "validation" or (failure == "loss" and model.training):
            output.loss = output.loss * float("nan")
        elif failure == "gradient" and model.training:
            output.loss.register_hook(lambda gradient: gradient * float("nan"))
        return output

    def poison_optimizer(optimizer):
        step(optimizer)
        first = optimizer.param_groups[0]["params"][0]
        optimizer.state[first]["exp_avg"].fill_(float("nan"))

    if failure == "optimizer":
        monkeypatch.setattr(AdamW, "step", poison_optimizer)
    else:
        monkeypatch.setattr(MiniGPT, "forward", poison_forward)
    out = tmp_path / "failed"
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        train(run, out)
    assert not (out / "ckpt_final.pt").exists()

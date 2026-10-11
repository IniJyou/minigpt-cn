"""由作业生成测试接入项目，补充 greedy、CLI、UTF-8 报告与异常恢复。"""

import json
import math

import pytest
import torch

from minigpt_cn.config import ModelConfig
from minigpt_cn.inference.generate import _decode_report, _top_p_filter, generate, main
from minigpt_cn.model import MiniGPT, ModelOutput
from minigpt_cn.tokenizer import BPETokenizer
from minigpt_cn.training.checkpoint import file_sha256, save_checkpoint


@pytest.fixture(autouse=True)
def isolated_random_state():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield
    torch.set_num_threads(threads)


@pytest.fixture
def model():
    # CPU/FP32，真实小模型；dropout 非零，可以检查推理时是否确实关闭。
    return MiniGPT(ModelConfig(4, 4, 1, 2, 8, 16, dropout=0.5))


@pytest.fixture
def prompt():
    return torch.tensor([[1, 2]], dtype=torch.long)  # [batch=1,sequence=2]。


def fixed_forward(values):
    def forward(ids):
        logits = torch.tensor(values, dtype=torch.float32, device=ids.device)
        return ModelOutput(logits=logits.expand(*ids.shape, len(values)))

    return forward


def test_zero_tokens_preserves_prompt_with_independent_copy(model, prompt):
    original = prompt.clone()
    result = generate(model, prompt, 0)
    assert torch.equal(result, original) and torch.equal(prompt, original)
    assert result.data_ptr() != prompt.data_ptr() and model.training


def test_generation_length_range_and_original_prompt(model, prompt):
    original = prompt.clone()
    result = generate(model, prompt, 3)
    assert result.shape == (1, 5) and result.dtype == torch.long
    assert torch.equal(result[:, :2], original) and torch.equal(prompt, original)
    assert ((result >= 0) & (result < 4)).all()


def test_eos_stops_after_one_token(model, prompt, monkeypatch):
    monkeypatch.setattr(model, "forward", fixed_forward([-torch.inf, -torch.inf, -torch.inf, 0]))
    assert generate(model, prompt, 3, eos_token_id=3).tolist() == [[1, 2, 3]]


def test_long_prompt_only_crops_model_input_not_output(model):
    prompt = torch.tensor([[1, 2, 3, 0, 1, 2]])
    lengths = []
    handle = model.register_forward_pre_hook(
        lambda module, inputs: lengths.append(inputs[0].shape[1])
    )
    try:
        result = generate(model, prompt, 3)
    finally:
        handle.remove()
    assert lengths == [4, 4, 4] and result.shape == (1, 9)
    assert torch.equal(result[:, :6], prompt)


@pytest.mark.parametrize("p,mask", [(0.5, [False, True, True]), (0.6, [False, False, True])])
def test_top_p_equal_and_crossing_boundary(p, mask):
    logits = torch.tensor([[0.5, 0.25, 0.25]]).log()
    before = logits.clone()
    actual = _top_p_filter(logits, p)
    mask = torch.tensor([mask])
    assert torch.equal(torch.isneginf(actual), mask)
    torch.testing.assert_close(actual[~mask], before[~mask], rtol=0, atol=0)
    assert torch.equal(logits, before)


def test_top_p_scatter_back_to_original_vocabulary_order():
    logits = torch.tensor([[0.25, 0.5, 0.25]]).log()
    assert torch.isneginf(_top_p_filter(logits, 0.5)).tolist() == [[True, False, True]]


def test_forward_no_grad_dropout_off_and_original_mode_restored(model, prompt):
    modes = []
    handle = model.register_forward_hook(
        lambda module, inputs, output: modes.append((torch.is_grad_enabled(), module.training))
    )
    try:
        with torch.enable_grad():
            generate(model, prompt, 3)
    finally:
        handle.remove()
    assert modes == [(False, False)] * 3 and model.training
    model.eval()
    generate(model, prompt, 1)
    assert not model.training


@pytest.mark.parametrize(
    "temperature,expected", [(1.0, [1 / 5, 4 / 5, 0, 0]), (2.0, [1 / 3, 2 / 3, 0, 0])]
)
def test_temperature_exact_sampling_probabilities(
    model, prompt, monkeypatch, temperature, expected
):
    captured = []
    monkeypatch.setattr(model, "forward", fixed_forward([0, math.log(4), -torch.inf, -torch.inf]))

    def capture(probs, num_samples, generator=None):
        captured.append(probs.clone())
        assert num_samples == 1 and probs.dtype == torch.float32
        return torch.ones((1, 1), dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", capture)
    generate(model, prompt, 1, temperature=temperature)
    torch.testing.assert_close(captured[0], torch.tensor([expected]))


def test_greedy_uses_argmax_without_sampling_or_consuming_rng(model, prompt, monkeypatch):
    monkeypatch.setattr(model, "forward", fixed_forward([0, 2, 3, 1]))

    def forbidden(*args, **kwargs):
        raise AssertionError("greedy 不应调用 multinomial")

    monkeypatch.setattr(torch, "multinomial", forbidden)
    generator = torch.Generator().manual_seed(17)
    state = generator.get_state().clone()
    global_state = torch.get_rng_state().clone()
    actual = generate(
        model, prompt, 3, greedy=True, temperature=0.1, top_p=0.1, generator=generator
    )
    assert actual.tolist() == [[1, 2, 2, 2, 2]]
    assert torch.equal(generator.get_state(), state) and torch.equal(
        torch.get_rng_state(), global_state
    )


def test_seeded_sampler_repeats_and_does_not_consume_global_rng(model, prompt):
    state = torch.get_rng_state().clone()
    a = generate(model, prompt, 8, generator=torch.Generator().manual_seed(31))
    b = generate(model, prompt, 8, generator=torch.Generator().manual_seed(31))
    assert torch.equal(a, b) and torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"prompt_ids": [[1, 2]]}, TypeError),
        ({"prompt_ids": torch.tensor([1, 2])}, ValueError),
        ({"prompt_ids": torch.tensor([[-1, 2]])}, ValueError),
        ({"prompt_ids": torch.tensor([[1, 4]])}, ValueError),
        ({"prompt_ids": torch.empty((1, 0), dtype=torch.long)}, ValueError),
        ({"prompt_ids": torch.tensor([[1, 2], [2, 1]])}, ValueError),
        ({"prompt_ids": torch.tensor([[1.0, 2.0]])}, TypeError),
        ({"prompt_ids": torch.empty((1, 2), dtype=torch.long, device="meta")}, ValueError),
        ({"max_new_tokens": 1.5}, TypeError),
        ({"max_new_tokens": True}, TypeError),
        ({"max_new_tokens": -1}, ValueError),
        ({"eos_token_id": -1}, ValueError),
        ({"eos_token_id": 4}, ValueError),
        ({"eos_token_id": True}, TypeError),
        ({"temperature": 0}, ValueError),
        ({"temperature": -1}, ValueError),
        ({"temperature": float("nan")}, ValueError),
        ({"temperature": float("inf")}, ValueError),
        ({"top_p": 0}, ValueError),
        ({"top_p": 1.1}, ValueError),
        ({"top_p": float("nan")}, ValueError),
        ({"top_p": float("inf")}, ValueError),
        ({"greedy": 1}, TypeError),
        ({"use_cache": 1}, TypeError),
        ({"generator": 1}, TypeError),
        ({"precision": "float16"}, ValueError),
    ],
)
def test_invalid_inputs_fail_clearly(model, prompt, overrides, error):
    kwargs = {"prompt_ids": prompt, "max_new_tokens": 1, **overrides}
    with pytest.raises(error):
        generate(model, **kwargs)


@pytest.mark.parametrize(
    "values", [[float("nan"), 0, 0, 0], [float("inf"), 0, 0, 0], [-torch.inf] * 4]
)
def test_invalid_logits_abort_and_restore_training_mode(model, prompt, monkeypatch, values):
    monkeypatch.setattr(model, "forward", fixed_forward(values))
    with pytest.raises(FloatingPointError, match="logits"):
        generate(model, prompt, 1)
    assert model.training


@pytest.fixture
def checkpoint(tmp_path):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"<|endoftext|>"
    tokenizer = tmp_path / "tokenizer.json"
    BPETokenizer(vocab, [], ["<|endoftext|>"]).save(tokenizer)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, MiniGPT(ModelConfig(257, 4, 1, 2, 8, 16)), tokenizer_path=tokenizer, iteration=7
    )
    return path, tokenizer


@pytest.mark.parametrize("use_cache", [False, True])
def test_cli_reads_checkpoint_config_and_preserves_source(checkpoint, tmp_path, capsys, use_cache):
    path, _ = checkpoint
    before = file_sha256(path)
    output = tmp_path / "sample.json"
    report = main(
        [
            "--checkpoint",
            str(path),
            "--prompt",
            "中文😀",
            "--greedy",
            "--max-new-tokens",
            "3",
            "--output",
            str(output),
            *(["--use-cache"] if use_cache else []),
        ]
    )
    assert report["iteration"] == 7 and report["model_config"]["vocab_size"] == 257
    assert report["generated_tokens"] <= 3 and report["use_cache"] is use_cache
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert file_sha256(path) == before and "续写" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main(["--checkpoint", str(path), "--prompt", "中文", "--output", str(output)])
    assert error.value.code == 2 and file_sha256(path) == before


def test_cli_sampling_seed_reproduces_raw_ids(checkpoint):
    path, _ = checkpoint
    args = [
        "--checkpoint",
        str(path),
        "--prompt",
        "中文",
        "--no-eos",
        "--max-new-tokens",
        "8",
        "--temperature",
        "0.8",
        "--top-p",
        "0.9",
        "--seed",
        "42",
    ]
    a, b = main(args), main(args)
    assert a["full_ids"] == b["full_ids"] and a["generated_tokens"] == 8


@pytest.mark.parametrize(
    "extra",
    [
        ["--prompt", ""],
        ["--prompt", "中", "--eos-token", "未知EOS"],
        ["--prompt", "中", "--seed", "-1"],
    ],
)
def test_cli_rejects_empty_prompt_invalid_eos_and_seed(checkpoint, extra):
    path, _ = checkpoint
    with pytest.raises(SystemExit) as error:
        main(["--checkpoint", str(path), *extra])
    assert error.value.code == 2


def test_cli_rejects_different_tokenizer(checkpoint):
    path, tokenizer = checkpoint
    tokenizer.write_text(tokenizer.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["--checkpoint", str(path), "--prompt", "中文"])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "data,invalid,text",
    [(b"\xff", True, "�"), ("中".encode(), False, "中"), ("�".encode(), False, "�")],
)
def test_utf8_bytes_assembled_before_decoding_and_actual_invalid_flag(
    checkpoint, data, invalid, text
):
    _, tokenizer_path = checkpoint
    tokenizer = BPETokenizer.load(tokenizer_path)
    assert _decode_report(tokenizer, list(data)) == (text, invalid)


def test_cli_retains_and_reports_invalid_utf8(checkpoint, tmp_path, monkeypatch, capsys):
    path, _ = checkpoint

    def invalid_generate(model, prompt_ids, *args, **kwargs):
        return torch.cat((prompt_ids, torch.tensor([[255]])), dim=1)

    monkeypatch.setattr("minigpt_cn.inference.generate.generate", invalid_generate)
    report = main(
        ["--checkpoint", str(path), "--prompt", "中", "--output", str(tmp_path / "bad.json")]
    )
    assert report["continuation"] == "�" and report["generated_ids"] == [255]
    assert report["full_utf8_invalid"] and report["continuation_utf8_invalid"]
    assert report["replacement_character_count"] == 1
    assert "UTF-8" in capsys.readouterr().err


def test_cuda_bfloat16_generation_keeps_parameters_fp32(model, prompt):
    if not torch.cuda.is_available():
        pytest.skip("CPU CI 不依赖 GPU")
    model.cuda()
    prompt = prompt.cuda()
    # 默认 GPU 写法 cuda 与参数 cuda:0 应正确匹配，不能误报不同设备。
    ids = generate(
        model,
        prompt,
        3,
        precision="bfloat16",
        generator=torch.Generator(device="cuda").manual_seed(42),
    )
    assert ids.shape == (1, 5) and ids.dtype == torch.long and ids.device.type == "cuda"
    assert all(p.dtype == torch.float32 for p in model.parameters()) and model.training
    with pytest.raises(ValueError, match="同一设备"):
        generate(model, prompt, 1, generator=torch.Generator(device="cpu"))

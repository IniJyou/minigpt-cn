"""F：缓存/非方形mask/RoPE/滑动窗口与独立推理状态；CPU测试不依赖语料。"""

import json

import pytest
import torch

from minigpt_cn.config import ModelConfig
from minigpt_cn.inference.benchmark import benchmark
from minigpt_cn.inference.benchmark import main as benchmark_main
from minigpt_cn.inference.generate import generate
from minigpt_cn.model import MiniGPT, ModelOutput
from minigpt_cn.tokenizer import BPETokenizer
from minigpt_cn.training.checkpoint import save_checkpoint


@pytest.fixture(autouse=True)
def isolated_state():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        yield
    torch.set_num_threads(threads)


@pytest.fixture
def model():
    return MiniGPT(ModelConfig(32, 8, 2, 2, 16, 32, dropout=0.2)).eval()


@torch.no_grad()
def test_prefill_and_chunked_logits_loss_shape_match_full_model(model):
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    targets = inputs.roll(-1, dims=-1)
    full = model(inputs, targets)
    parts, cache, weighted_loss = [], None, 0
    for start, end in ((0, 2), (2, 3), (3, 6)):
        output = model(inputs[:, start:end], targets[:, start:end], use_cache=True, kv_cache=cache)
        assert output.logits.shape == (2, end - start, 32)
        assert output.loss.ndim == 0 and len(output.cache) == 2
        for k, v in output.cache:
            assert k.shape == v.shape == (2, 2, end, 8)
            assert not k.requires_grad and not v.requires_grad
        parts.append(output.logits)
        weighted_loss += output.loss.item() * (end - start)
        cache = output.cache
    torch.testing.assert_close(torch.cat(parts, dim=1), full.logits, rtol=1e-5, atol=1e-6)
    assert weighted_loss / 6 == pytest.approx(full.loss.item(), abs=1e-6)
    assert full.cache is None


def test_prefill_preserves_exact_logits_state_keys_and_cache_is_detached(model):
    inputs = torch.tensor([[1, 2, 3]])
    keys = set(model.state_dict())
    expected = model(inputs).logits
    actual = model(inputs, use_cache=True)
    torch.testing.assert_close(actual.logits, expected, rtol=0, atol=0)
    assert set(model.state_dict()) == keys and not any("cache" in name for name in keys)
    assert all(not tensor.requires_grad for pair in actual.cache for tensor in pair)
    assert model.token_embeddings.W is model.lm_head.W


@torch.no_grad()
def test_cache_contains_rotated_keys_and_old_cache_is_not_modified(model):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    old = model(ids[:, :3], use_cache=True).cache
    snapshot = tuple((k.clone(), v.clone()) for k, v in old)
    output = model(ids[:, 3:], use_cache=True, kv_cache=old)
    for pair, before, after in zip(old, snapshot, output.cache):
        for value, copied, new_value in zip(pair, before, after):
            torch.testing.assert_close(value, copied, rtol=0, atol=0)
            torch.testing.assert_close(new_value[:, :, :3], copied, rtol=0, atol=0)
    first = model.layers[0]
    normalized = first.ln1(model.token_embeddings(ids))
    k = first.attn.k_proj(normalized).reshape(1, 5, 2, 8).transpose(1, 2)
    expected = first.attn.rope(k)
    torch.testing.assert_close(output.cache[0][0], expected, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_non_square_mask_new_future_token_cannot_affect_previous_query(model):
    prefix = torch.tensor([[1, 2, 3]])
    cache = model(prefix, use_cache=True).cache
    a = model(torch.tensor([[4, 5]]), use_cache=True, kv_cache=cache).logits
    b = model(torch.tensor([[4, 20]]), use_cache=True, kv_cache=cache).logits
    torch.testing.assert_close(a[:, :1], b[:, :1], rtol=0, atol=0)
    expected = model(torch.tensor([[1, 2, 3, 4, 5]])).logits[:, 3:]
    torch.testing.assert_close(a, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "fault",
    [
        "layers",
        "container",
        "pair",
        "tensor",
        "rank",
        "batch",
        "heads",
        "head_dim",
        "length",
        "empty",
        "kv_shape",
        "dtype",
        "integer",
        "device",
    ],
)
def test_invalid_cache_has_clear_exception(model, fault):
    good = model(torch.tensor([[1, 2]]), use_cache=True).cache
    bad = list(good)
    k, v = good[0]
    error = ValueError
    if fault == "layers":
        bad = good[:-1]
    elif fault == "container":
        pass  # list 不允许冒充按层排列的 tuple。
    elif fault == "pair":
        bad[0] = (k,)
    elif fault == "tensor":
        bad[0] = (None, v)
        error = TypeError
    elif fault in {"rank", "batch", "heads", "head_dim", "length", "empty", "kv_shape"}:
        shapes = {
            "rank": (1, 2, 8),
            "batch": (2, 2, 2, 8),
            "heads": (1, 1, 2, 8),
            "head_dim": (1, 2, 2, 4),
            "length": (1, 2, 1, 8),
            "empty": (1, 2, 0, 8),
            "kv_shape": (1, 2, 2, 8),
        }
        wrong = torch.zeros(shapes[fault])
        bad[0] = (wrong, v[:, :, :1]) if fault == "kv_shape" else (wrong, wrong.clone())
    elif fault in {"dtype", "integer"}:
        dtype = torch.float64 if fault == "dtype" else torch.long
        bad[0] = (k.to(dtype), v.to(dtype))
        error = TypeError
    elif fault == "device":
        bad[0] = (k.to("meta"), v.to("meta"))
    if fault != "container":
        bad = tuple(bad)
    with pytest.raises(error, match="Cache"):
        model(torch.tensor([[3]]), use_cache=True, kv_cache=bad)


def test_cache_requires_eval_explicit_flag_and_capacity(model):
    ids = torch.tensor([[1, 2]])
    cache = model(ids, use_cache=True).cache
    with pytest.raises(ValueError, match="use_cache"):
        model(ids, kv_cache=cache)
    with pytest.raises(TypeError, match="bool"):
        model(ids, use_cache=1)
    with pytest.raises(ValueError, match="eval"):
        model.train()(ids, use_cache=True)
    model.eval()
    full = model(torch.ones((1, 8), dtype=torch.long), use_cache=True).cache
    with pytest.raises(ValueError, match="context_length"):
        model(ids, use_cache=True, kv_cache=full)


@pytest.mark.parametrize("prompt_length", [2, 7, 8, 11])
def test_greedy_matches_across_window_and_only_new_tokens_are_processed(model, prompt_length):
    ids = torch.arange(prompt_length).remainder(32)[None]
    expected = generate(model, ids, 12, greedy=True)
    calls = []

    def observe(module, args, kwargs):
        calls.append((args[0].shape[1], kwargs.get("kv_cache") is not None))

    handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        actual = generate(model, ids, 12, greedy=True, use_cache=True)
    finally:
        handle.remove()
    assert torch.equal(actual, expected) and torch.equal(actual[:, :prompt_length], ids)
    incremental = min(11, 8 - min(prompt_length, 8))
    assert calls == [(min(prompt_length, 8), False)] + [(1, True)] * incremental + [(8, False)] * (
        11 - incremental
    )


def test_independent_prompts_cache_never_leaks_and_modes_are_restored(model):
    model.train()
    a, b = torch.tensor([[1, 2]]), torch.tensor([[20, 21, 22]])
    first = generate(model, a, 4, greedy=True, use_cache=True)
    generate(model, b, 4, greedy=True, use_cache=True)
    second = generate(model, a, 4, greedy=True, use_cache=True)
    assert torch.equal(first, second) and model.training
    assert generate(model, a, 0, use_cache=True).tolist() == a.tolist()


def test_cached_eos_and_greedy_no_rng(model, monkeypatch):
    calls = []

    def eos_forward(ids, **kwargs):
        calls.append(ids.shape)
        logits = torch.full((*ids.shape, 32), -torch.inf)
        logits[..., 31] = 0
        return ModelOutput(logits)

    monkeypatch.setattr(model, "forward", eos_forward)
    state = torch.get_rng_state().clone()
    result = generate(
        model, torch.tensor([[1, 2]]), 10, eos_token_id=31, greedy=True, use_cache=True
    )
    assert result.tolist() == [[1, 2, 31]] and len(calls) == 1
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_cuda_cache_greedy_and_chunk_logits_close(model, precision):
    if not torch.cuda.is_available():
        pytest.skip("CPU CI 不依赖 CUDA")
    model.cuda()
    ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bfloat16"),
    ):
        expected = model(ids).logits[:, 3:]
        cache = model(ids[:, :3], use_cache=True).cache
        output = model(ids[:, 3:], use_cache=True, kv_cache=cache)
    torch.testing.assert_close(
        output.logits,
        expected,
        rtol=0.04 if precision == "bfloat16" else 1e-5,
        atol=0.002 if precision == "bfloat16" else 1e-6,
    )
    assert cache[0][0].dtype == (torch.bfloat16 if precision == "bfloat16" else torch.float32)
    a = generate(model, ids[:, :2], 12, greedy=True, precision=precision)
    b = generate(model, ids[:, :2], 12, greedy=True, precision=precision, use_cache=True)
    assert torch.equal(a, b)


def test_different_autocast_precision_cache_is_rejected(model):
    cache = model(torch.tensor([[1, 2]]), use_cache=True).cache
    wrong = tuple((k.to(torch.bfloat16), v.to(torch.bfloat16)) for k, v in cache)
    with pytest.raises(TypeError, match="dtype"):
        model(torch.tensor([[3]]), use_cache=True, kv_cache=wrong)


def test_benchmark_reports_equal_ids_window_counts_and_cpu_memory(model):
    result = benchmark(model, torch.tensor([[1, 2]]), max_new_tokens=10, repeats=1, warmup=0)
    assert result["greedy_ids_equal"] and result["status"] == "passed"
    assert result["cached"]["incremental_calls"] == 6 and result["cached"]["window_rebuilds"] == 3
    assert result["cached"]["kv_cache_bytes"] == 2 * 2 * 1 * 2 * 8 * 8 * 4
    assert result["cached"]["peak_allocated_bytes"] is None
    assert result["cached"]["first_token_seconds"] > 0 and result["total_speedup"] > 0


@pytest.mark.parametrize("kwargs", [{"max_new_tokens": 1}, {"repeats": 0}, {"warmup": -1}])
def test_benchmark_rejects_invalid_workload(model, kwargs):
    with pytest.raises(ValueError):
        benchmark(model, torch.tensor([[1, 2]]), **kwargs)


def test_benchmark_cli_loads_checkpoint_and_saves_json(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    vocab = {i: bytes([i]) for i in range(256)}
    BPETokenizer(vocab, []).save(tokenizer)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint, MiniGPT(ModelConfig(256, 8, 1, 2, 8, 16)), tokenizer_path=tokenizer, iteration=3
    )
    output = tmp_path / "benchmark.json"
    report = benchmark_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--prompt",
            "hi",
            "--max-new-tokens",
            "3",
            "--repeats",
            "1",
            "--warmup",
            "0",
            "--output",
            str(output),
        ]
    )
    assert report["iteration"] == 3 and report["greedy_ids_equal"]
    assert json.loads(output.read_text(encoding="utf-8")) == report

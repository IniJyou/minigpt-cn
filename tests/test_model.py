"""不依赖作业仓库、真实语料或 GPU 的核心回归。"""

import io
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from minigpt_cn.config import ModelConfig, load_model_config
from minigpt_cn.model import MiniGPT, ModelOutput
from minigpt_cn.model.attention import attention_reference, scaled_dot_product_attention
from minigpt_cn.model.components import Dropout, RMSNorm, SwiGLU
from minigpt_cn.model.rope import RotaryPositionalEmbedding


@pytest.fixture(autouse=True)
def isolated_random_state():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return ModelConfig(32, 8, 2, 2, 8, 16)


def test_reference_and_vectorized_attention_agree():
    q, k, v = [torch.randn(2, 2, 4, 6, requires_grad=True) for _ in range(3)]
    mask = torch.ones(4, 4, dtype=torch.bool).tril()
    actual = scaled_dot_product_attention(q, k, v, mask)
    torch.testing.assert_close(actual, attention_reference(q, k, v, mask))
    actual.sum().backward()
    assert all(torch.isfinite(tensor.grad).all() for tensor in (q, k, v))


@pytest.mark.parametrize("component", ["rmsnorm", "rope", "swiglu"])
def test_components_preserve_shape_and_have_finite_gradients(component):
    if component == "rope":
        layer = RotaryPositionalEmbedding(10000.0, 4, 8)
        inputs = torch.randn(2, 2, 3, 4, requires_grad=True)
    else:
        layer = RMSNorm(8) if component == "rmsnorm" else SwiGLU(8, 16)
        inputs = torch.randn(2, 3, 8, requires_grad=True)
    result = layer(inputs)
    assert result.shape == inputs.shape
    result.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters())


def test_rope_offset_matches_slice_of_longer_sequence():
    layer = RotaryPositionalEmbedding(10000.0, 4, 8)
    inputs = torch.randn(2, 2, 8, 4)
    full = layer(inputs)
    torch.testing.assert_close(layer(inputs[:, :, 3:5], offset=3), full[:, :, 3:5])
    for length in (1, 4, 8):
        assert layer(inputs[:, :, :length]).shape == inputs[:, :, :length].shape
    with pytest.raises(ValueError, match="context_length"):
        layer(inputs[:, :, :2], offset=7)


def test_logits_loss_and_causal_behavior(config):
    model = MiniGPT(replace(config, dropout=0.1)).eval()
    inputs = torch.tensor([[1, 2, 3, 4, 5]])
    changed = inputs.clone()
    changed[:, 3:] = torch.tensor([[6, 7]])
    original = model(inputs, targets=inputs)
    assert isinstance(original, ModelOutput)
    assert original.logits.shape == (1, 5, 32)
    assert original.loss.ndim == 0 and torch.isfinite(original.loss)
    assert original.cache is None
    torch.testing.assert_close(model(changed).logits[:, :3], original.logits[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(model(inputs).logits, original.logits, rtol=0, atol=0)
    original.loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model(inputs).loss is None


def test_shared_gradient_is_sum_of_both_paths_and_parameter_is_unique(config):
    tied = MiniGPT(config)
    untied = MiniGPT(replace(config, tie_embeddings=False))
    untied.load_state_dict(tied.state_dict(), strict=True)
    inputs, targets = torch.tensor([[1, 2, 3]]), torch.tensor([[2, 3, 4]])
    tied(inputs, targets).loss.backward()
    untied(inputs, targets).loss.backward()
    shared = tied.token_embeddings.W
    assert shared is tied.lm_head.W
    torch.testing.assert_close(shared.grad, untied.token_embeddings.W.grad + untied.lm_head.W.grad)
    optimizer = torch.optim.SGD(tied.parameters(), lr=0.01)
    assert sum(parameter is shared for parameter in optimizer.param_groups[0]["params"]) == 1
    tied_count = sum(parameter.numel() for parameter in tied.parameters())
    assert sum(parameter.numel() for parameter in untied.parameters()) - tied_count == 32 * 8


def test_checkpoint_reload_preserves_sharing_and_logits(config):
    model = MiniGPT(config).eval()
    inputs = torch.tensor([[1, 2]])
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = MiniGPT(config).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    assert restored.token_embeddings.W is restored.lm_head.W
    torch.testing.assert_close(restored(inputs).logits, model(inputs).logits, rtol=0, atol=0)


def test_shared_embedding_initialization_uses_configured_small_std(config):
    model = MiniGPT(config)
    weights = model.token_embeddings.W.detach()
    assert weights.abs().max() <= 3 * config.embedding_init_std
    assert 0.015 < weights.std().item() < 0.025
    assert model.lm_head.W is model.token_embeddings.W


def test_formal_parameter_count():
    config = load_model_config(Path(__file__).resolve().parents[1] / "configs/minigpt_33m.yaml")
    model = MiniGPT(config, device="meta")
    assert sum(parameter.numel() for parameter in model.parameters()) == 33_104_384


def test_dropout_preserves_expectation_and_eval_is_identity():
    layer = Dropout(0.25)
    inputs = torch.ones(10000, requires_grad=True)
    result = layer(inputs)
    assert (result == 0).any() and (result != 0).any()
    assert abs(result.mean().item() - 1.0) < 0.03
    result.sum().backward()
    torch.testing.assert_close(inputs.grad, result)
    torch.testing.assert_close(layer.eval()(inputs), inputs, rtol=0, atol=0)
    state = torch.get_rng_state()
    Dropout(0.0)(inputs)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("bad_shape", [(1, 2, 4), (2, 2, 4, 3)])
def test_bad_attention_shapes_raise(bad_shape):
    q = torch.zeros(1, 2, 4, 4)
    with pytest.raises(ValueError, match="Q/K"):
        scaled_dot_product_attention(q, torch.zeros(bad_shape), q)
    with pytest.raises(ValueError, match="至少"):
        scaled_dot_product_attention(q, q, q, torch.zeros(4, 4, dtype=torch.bool))


@pytest.mark.parametrize(
    "inputs,error",
    [
        (torch.ones(2), TypeError),
        (torch.ones(2, dtype=torch.long), ValueError),
        (torch.empty(1, 0, dtype=torch.long), ValueError),
        (torch.zeros(1, 9, dtype=torch.long), ValueError),
        (torch.tensor([[-1]]), ValueError),
        (torch.tensor([[32]]), ValueError),
    ],
)
def test_bad_model_inputs_raise(config, inputs, error):
    with pytest.raises(error):
        MiniGPT(config)(inputs)


def test_bad_targets_and_training_cache_are_explicit(config):
    model, inputs = MiniGPT(config), torch.tensor([[1, 2]])
    with pytest.raises(ValueError, match="形状"):
        model(inputs, torch.tensor([[1]]))
    with pytest.raises(ValueError, match="eval"):
        model(inputs, use_cache=True)


@pytest.mark.parametrize("field", ["rope_theta", "rms_norm_eps", "embedding_init_std"])
@pytest.mark.parametrize("value", [0, float("nan"), True])
def test_new_numerical_config_fields_are_validated(config, field, value):
    with pytest.raises(ValueError, match=field):
        replace(config, **{field: value})

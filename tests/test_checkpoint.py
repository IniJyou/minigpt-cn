"""C阶段：自带配置、共享关系、真实步数、RNG/采样和旧 pilot 导入。"""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from minigpt_cn.config import DataConfig, ModelConfig, TrainingConfig, load_run_config
from minigpt_cn.model import MiniGPT
from minigpt_cn.tokenizer import BPETokenizer
from minigpt_cn.training.checkpoint import (
    file_sha256,
    import_legacy_checkpoint,
    load_checkpoint,
    main,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def isolated_state():
    threads, python_state, numpy_state = (
        torch.get_num_threads(),
        random.getstate(),
        np.random.get_state(),
    )
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        yield
    torch.set_num_threads(threads)
    random.setstate(python_state)
    np.random.set_state(numpy_state)


@pytest.fixture
def sample(tmp_path):
    vocab = {index: bytes([index]) for index in range(256)}
    vocab[256] = b"<|endoftext|>"
    tokenizer_path = tmp_path / "tokenizer.json"
    BPETokenizer(vocab, [], ["<|endoftext|>"]).save(tokenizer_path)
    config = ModelConfig(257, 8, 1, 2, 8, 16, dropout=0.1)
    return MiniGPT(config), tokenizer_path, tmp_path / "checkpoint.pt"


def test_inference_checkpoint_rebuilds_model_and_preserves_shared_identity(sample):
    model, tokenizer, path = sample
    model.eval()
    inputs = torch.tensor([[1, 2, 3]])
    expected = model(inputs).logits
    save_checkpoint(path, model, tokenizer_path=tokenizer, iteration=60)
    loaded = load_checkpoint(path)
    assert loaded.iteration == 60
    assert loaded.model.config == model.config
    assert loaded.model.token_embeddings.W is loaded.model.lm_head.W
    assert not loaded.model.training and not loaded.resume_supported
    torch.testing.assert_close(loaded.model(inputs).logits, expected, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, tokenizer_path=tokenizer, iteration=40)
    with pytest.raises(ValueError, match="步数"):
        save_checkpoint(path, model, tokenizer_path=tokenizer, iteration=40, overwrite=True)
    assert load_checkpoint(path).iteration == 60


def test_same_shape_but_different_rope_configuration_is_rejected(sample):
    model, tokenizer, path = sample
    save_checkpoint(path, model, tokenizer_path=tokenizer, iteration=0)
    with pytest.raises(ValueError, match="配置不匹配"):
        load_checkpoint(path, expected_model_config=replace(model.config, rope_theta=20000.0))


def test_wrong_tokenizer_checksum_and_vocabulary_are_rejected(sample):
    model, tokenizer, path = sample
    save_checkpoint(path, model, tokenizer_path=tokenizer, iteration=0)
    tokenizer.write_text(tokenizer.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="校验和"):
        load_checkpoint(path)
    with pytest.raises(ValueError, match="词表"):
        save_checkpoint(
            path.parent / "wrong.pt",
            MiniGPT(replace(model.config, vocab_size=258)),
            tokenizer_path=tokenizer,
            iteration=0,
        )


def _update(model, optimizer, sampler):
    sequence = torch.randint(0, 257, (2, 5), generator=sampler)
    device = next(model.parameters()).device
    sequence = sequence.to(device)
    optimizer.zero_grad()
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        loss = model(sequence[:, :-1], sequence[:, 1:]).loss
    loss.backward()
    optimizer.step()
    return sequence, loss.detach()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training_checkpoint_replays_sampling_dropout_and_next_update(sample, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA 恢复测试需要 GPU；CPU CI 不依赖 GPU")
    model, tokenizer, path = sample
    model.to(device)
    training = TrainingConfig(
        device=device, precision="bfloat16" if device == "cuda" else "float32"
    )
    data = DataConfig("train.bin", "valid.bin", str(tokenizer))
    sampler = torch.Generator().manual_seed(11)
    # 原作业 AdamW 在D阶段接入；这里用原生优化器单独验收恢复机制，不更改训练算法。
    optimizer = torch.optim.AdamW(model.parameters(), lr=training.learning_rate, weight_decay=0.1)
    _update(model, optimizer, sampler)
    save_checkpoint(
        path,
        model,
        tokenizer_path=tokenizer,
        iteration=1,
        optimizer=optimizer,
        training_config=training,
        data_config=data,
        sampler=sampler,
    )
    expected_random = (random.random(), np.random.random())
    expected_tokens, expected_loss = _update(model, optimizer, sampler)
    restored = MiniGPT(model.config, device=device)
    wrong_optimizer = torch.optim.AdamW(list(restored.parameters())[::-1])
    with pytest.raises(ValueError, match="顺序"):
        load_checkpoint(path, model=restored, optimizer=wrong_optimizer)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.9, weight_decay=0.9)
    restored_sampler = torch.Generator().manual_seed(99)
    loaded = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        restore_rng=True,
        sampler=restored_sampler,
    )
    assert loaded.iteration == 1 and loaded.resume_supported
    assert loaded.training_config == training and loaded.data_config == data
    assert loaded.optimizer_settings[0]["weight_decay"] == 0.1
    assert loaded.optimizer_settings[0]["lr"] == training.learning_rate
    assert (random.random(), np.random.random()) == expected_random
    actual_tokens, actual_loss = _update(restored, restored_optimizer, restored_sampler)
    assert torch.equal(actual_tokens, expected_tokens)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_legacy_import_never_overwrites_original_or_invents_rng(sample):
    model, tokenizer, path = sample
    source = path.parent / "legacy.pt"
    model.eval()
    torch.save({"model_state_dict": model.state_dict(), "optimizer": {}, "iteration": 5000}, source)
    original_hash = file_sha256(source)
    with pytest.raises(ValueError, match="覆盖"):
        import_legacy_checkpoint(
            source, source, model_config=model.config, tokenizer_path=tokenizer
        )
    import_legacy_checkpoint(source, path, model_config=model.config, tokenizer_path=tokenizer)
    assert file_sha256(source) == original_hash
    loaded = load_checkpoint(path)
    assert loaded.iteration == 5000 and not loaded.resume_supported
    assert loaded.training_config is None
    inputs = torch.tensor([[1, 2]])
    torch.testing.assert_close(loaded.model(inputs).logits, model(inputs).logits, rtol=0, atol=0)
    with pytest.raises(ValueError, match="精确恢复"):
        load_checkpoint(path, restore_rng=True)


@pytest.mark.parametrize("iteration", [-1, True, 1.5])
def test_invalid_iteration_is_rejected(sample, iteration):
    model, tokenizer, path = sample
    with pytest.raises(ValueError, match="iteration"):
        save_checkpoint(path, model, tokenizer_path=tokenizer, iteration=iteration)
    assert not path.exists()


def test_bad_version_conflicting_shared_weights_and_nan_are_rejected(sample):
    model, tokenizer, path = sample
    save_checkpoint(path, model, tokenizer_path=tokenizer, iteration=0)
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 2
    torch.save(payload, path)
    with pytest.raises(ValueError, match="版本"):
        load_checkpoint(path)
    payload["format_version"] = 1
    payload["model_state_dict"]["lm_head.W"] = payload["model_state_dict"]["lm_head.W"].clone() + 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="共享"):
        load_checkpoint(path)
    with torch.no_grad():
        model.token_embeddings.W[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        save_checkpoint(path.parent / "nan.pt", model, tokenizer_path=tokenizer, iteration=0)


def test_inspection_cli_reports_actual_configuration(sample, capsys):
    model, tokenizer, path = sample
    save_checkpoint(path, model, tokenizer_path=tokenizer, iteration=7)
    main(["inspect", str(path)])
    output = capsys.readouterr().out
    assert '"iteration": 7' in output and '"resume_supported": false' in output


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("max_steps", True),
        ("log_interval", 0),
        ("eval_interval", -1),
        ("save_interval", 0),
        ("learning_rate", float("nan")),
        ("eps", 0),
        ("betas", [0.9, 1.0]),
        ("precision", "float16"),
        ("warmup_steps", 100),
    ],
)
def test_training_configuration_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})
    with pytest.raises(ValueError, match="seed"):
        TrainingConfig(seed=2**32)


def test_run_config_resolves_paths_from_yaml_directory():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    run = load_run_config(root / "configs/pilot_run.yaml")
    assert run.model.vocab_size == 2048 and run.training.max_steps == 100
    assert run.training.betas == (0.9, 0.95)
    assert run.data.tokenizer_path == str(
        root.parent / "cs336-assignment1-basics/data/processed/tokenizer_2048.json"
    )

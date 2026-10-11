"""验收工具自身的回归：原始二进制校验、尾块覆盖和按 token 加权。"""

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from minigpt_cn.config import ModelConfig
from minigpt_cn.evaluation.verify_pilot import compare_validation, read_tokens
from minigpt_cn.model import MiniGPT
from minigpt_cn.model.components import cross_entropy


@pytest.mark.parametrize(
    "contents",
    [
        b"\x00\x00",
        b"\x00\x00\x00",
        b"\x93NUMPY" + bytes(6),
        b"\x00\x00\x05\x00",
    ],
)
def test_bad_token_files_are_rejected(tmp_path, contents):
    path = tmp_path / "tokens.bin"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        read_tokens(path, vocab_size=5)


def test_uint16_token_file_loads(tmp_path):
    path = tmp_path / "tokens.bin"
    np.array([0, 1, 4], dtype="<u2").tofile(path)
    assert read_tokens(path, vocab_size=5).tolist() == [0, 1, 4]


def test_full_validation_includes_tail_and_weights_by_target_count():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = ModelConfig(5, 3, 1, 2, 8, 16)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            project = MiniGPT(config).eval()
        lengths = []

        class TensorReference(nn.Module):
            def forward(self, inputs):
                lengths.append(inputs.shape[1])
                return project(inputs).logits

        tokens = np.array([1, 2, 3, 4, 2, 1, 0, 3], dtype="<u2")
        report = compare_validation(
            TensorReference(), project, tokens, cross_entropy, "cpu", "float32"
        )
        assert lengths == [3, 3, 1]
        assert report["targets"] == 7
        assert report["max_logits_abs_error"] == 0
        total_nll = 0.0
        with torch.no_grad():
            for start in (0, 3, 6):
                x = torch.tensor(tokens[start : min(start + 3, 7)].astype(np.int64)).unsqueeze(0)
                y = torch.tensor(tokens[start + 1 : min(start + 4, 8)].astype(np.int64))
                total_nll += F.cross_entropy(
                    project(x).logits.squeeze(0), y, reduction="sum"
                ).item()
        assert report["project_loss"] == pytest.approx(total_nll / 7, abs=1e-6)
    finally:
        torch.set_num_threads(previous_threads)

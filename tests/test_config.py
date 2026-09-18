"""模型配置的单元测试。"""

from pathlib import Path

import pytest

from minigpt_cn.config import ModelConfig, load_model_config

# 每个测试从同一份最小合法配置开始，只改动当前要验证的字段。
VALID_MODEL = {
    "vocab_size": 1024,
    "context_length": 64,
    "n_layers": 2,
    "n_heads": 4,
    "d_model": 128,
    "d_ff": 336,
}


def test_smoke_config_loads() -> None:
    """仓库内置的 smoke 配置应该能够直接加载。"""

    project_root = Path(__file__).resolve().parents[1]
    config = load_model_config(project_root / "configs" / "smoke.yaml")

    assert config.n_layers == 2
    assert config.d_model == 128
    assert config.d_model // config.n_heads == 32


def test_formal_config_loads() -> None:
    """正式配置的关键尺寸应与项目计划一致。"""

    project_root = Path(__file__).resolve().parents[1]
    config = load_model_config(project_root / "configs" / "minigpt_33m.yaml")

    assert config.vocab_size == 16000
    assert config.context_length == 512
    assert config.n_layers == 8


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vocab_size", 0),
        ("context_length", 0),
        ("n_layers", -1),
        ("n_heads", 0),
        ("d_model", -128),
        ("d_ff", 0),
    ],
)
def test_positive_dimensions_are_required(field: str, value: int) -> None:
    """所有描述模型大小的整数都必须大于零。"""

    values = VALID_MODEL | {field: value}

    with pytest.raises(ValueError, match="must be positive"):
        ModelConfig(**values)


def test_d_model_must_be_divisible_by_n_heads() -> None:
    """每个 attention head 必须获得相同数量的隐藏维度。"""

    values = VALID_MODEL | {"n_heads": 3}

    with pytest.raises(ValueError, match="must be divisible"):
        ModelConfig(**values)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, True])
def test_dropout_range_is_validated(dropout: object) -> None:
    """Dropout 是左闭右开的概率值：[0, 1)。"""

    values = VALID_MODEL | {"dropout": dropout}

    with pytest.raises(ValueError, match="dropout"):
        ModelConfig(**values)


def test_missing_model_section_is_rejected() -> None:
    """缺少顶层 model 映射时，应给出清晰错误而不是稍后失败。"""

    # 使用仓库内的固定 fixture，避免测试依赖系统临时目录的权限状态。
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "tests" / "fixtures" / "missing_model.yaml"

    with pytest.raises(ValueError, match="must contain a 'model' mapping"):
        load_model_config(config_path)

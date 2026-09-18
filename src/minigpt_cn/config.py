"""项目配置的定义、校验和 YAML 加载逻辑。

训练任务通常耗时且昂贵，因此应在创建模型之前尽早发现配置错误。
本模块目前只定义模型结构配置；训练和数据配置会在对应阶段加入。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Decoder-only Transformer 的结构配置。

    ``frozen=True`` 让配置对象不可变，避免训练过程中被意外修改。
    """

    # Token 与序列相关参数。
    vocab_size: int
    context_length: int

    # Transformer 主干的宽度和深度。
    n_layers: int
    n_heads: int
    d_model: int
    d_ff: int

    # 架构选项。先把允许值限制在项目计划覆盖的范围内。
    norm: str = "rmsnorm"
    position_encoding: str = "rope"
    activation: str = "swiglu"
    tie_embeddings: bool = True
    dropout: float = 0.0
    bias: bool = False

    def __post_init__(self) -> None:
        """在配置创建后立即检查不变量。"""

        positive_values = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
        }

        for name, value in positive_values.items():
            # bool 是 int 的子类，因此需要显式排除 True/False。
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        # 多头注意力会把 d_model 均分给每个 head，必须能够整除。
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads, "
                f"got d_model={self.d_model}, n_heads={self.n_heads}"
            )

        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")

        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be a boolean")

        if not isinstance(self.bias, bool):
            raise ValueError("bias must be a boolean")

        if self.norm != "rmsnorm":
            raise ValueError(f"unsupported norm: {self.norm}")

        if self.position_encoding not in {"rope", "learned"}:
            raise ValueError(
                f"unsupported position encoding: {self.position_encoding}"
            )

        if self.activation != "swiglu":
            raise ValueError(f"unsupported activation: {self.activation}")


def load_model_config(path: str | Path) -> ModelConfig:
    """从 YAML 文件加载并校验模型配置。

    Args:
        path: YAML 配置文件路径。文件必须包含顶层 ``model`` 映射。

    Returns:
        已完成校验、不可变的 :class:`ModelConfig`。

    Raises:
        ValueError: YAML 结构错误、缺少字段或字段值非法。
    """

    config_path = Path(path)

    try:
        with config_path.open(encoding="utf-8") as file:
            # safe_load 只读取普通 YAML 数据，不构造任意 Python 对象。
            raw_config: Any = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {config_path}: {error}") from error

    if not isinstance(raw_config, dict):
        raise ValueError("configuration root must be a mapping")

    model_config = raw_config.get("model")
    if not isinstance(model_config, dict):
        raise ValueError("configuration must contain a 'model' mapping")

    try:
        return ModelConfig(**model_config)
    except TypeError as error:
        # dataclass 在缺少必填字段或出现未知字段时抛出 TypeError；
        # 对调用者统一转换为“配置错误”更容易理解。
        raise ValueError(f"invalid model configuration: {error}") from error

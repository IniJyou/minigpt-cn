"""项目配置的定义、校验和 YAML 加载逻辑。

训练任务通常耗时且昂贵，因此应在创建模型之前尽早发现配置错误。
模型、训练和数据配置集中在此；checkpoint 保存这些配置，评估/生成无需再手写结构。
"""

import math
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
    # pilot checkpoint 不含这些值，必须显式记录，不能仅靠权重形状推断。
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    embedding_init_std: float = 0.02

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
            raise ValueError(f"unsupported position encoding: {self.position_encoding}")

        if self.activation != "swiglu":
            raise ValueError(f"unsupported activation: {self.activation}")

        if self.position_encoding == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE head_dim must be even")

        for name in ("rope_theta", "rms_norm_eps", "embedding_init_std"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")


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


@dataclass(frozen=True)
class TrainingConfig:
    """训练运行配置。训练终点与余弦调度终点分开，续训不会隐式改变学习率曲线。"""

    batch_size: int = 8
    max_steps: int = 100
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 5
    lr_decay_steps: int = 100
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    device: str = "cuda"
    precision: str = "bfloat16"
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 50

    def __post_init__(self):
        for name in (
            "batch_size",
            "max_steps",
            "lr_decay_steps",
            "log_interval",
            "eval_interval",
            "save_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("warmup_steps", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.warmup_steps >= self.lr_decay_steps:
            raise ValueError("warmup_steps must be less than lr_decay_steps")
        if self.seed >= 2**32:
            raise ValueError("seed must be less than 2**32 for NumPy compatibility")
        for name in ("learning_rate", "min_learning_rate", "eps", "weight_decay", "max_grad_norm"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name in {"learning_rate", "eps", "max_grad_norm"} and value == 0):
                raise ValueError(f"{name} is outside its valid range")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate must not exceed learning_rate")
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 2:
            raise ValueError("betas must contain two values")
        if any(
            isinstance(beta, bool) or not isinstance(beta, (int, float)) or not 0 <= beta < 1
            for beta in self.betas
        ):
            raise ValueError("betas must lie in [0, 1)")
        # YAML 的序列读取为 list；转成 tuple，保证 frozen 配置内部也不可变。
        object.__setattr__(self, "betas", tuple(self.betas))
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16; pure FP16 is not supported")


@dataclass(frozen=True)
class DataConfig:
    """无文件头 uint16 数据路径与配套 tokenizer；配置加载不读取大语料。"""

    train_path: str
    valid_path: str
    tokenizer_path: str

    def __post_init__(self):
        for name in ("train_path", "valid_path", "tokenizer_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty path string")


@dataclass(frozen=True)
class RunConfig:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig


def load_run_config(path: str | Path) -> RunConfig:
    """加载完整运行配置；所有相对路径相对于 YAML 文件目录，而不是终端当前目录。"""
    path = Path(path).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    if set(raw) - {"model", "model_config", "training", "data"}:
        raise ValueError("unknown run configuration section")
    if ("model" in raw) == ("model_config" in raw):
        raise ValueError("provide exactly one of model or model_config")
    try:
        if "model_config" in raw:
            if not isinstance(raw["model_config"], str):
                raise ValueError("model_config must be a path string")
            model = load_model_config(path.parent / raw["model_config"])
        else:
            model = ModelConfig(**raw["model"])
        training = TrainingConfig(**raw.get("training", {}))
        data = DataConfig(**raw["data"])
    except (TypeError, KeyError) as error:
        raise ValueError(f"invalid run configuration: {error}") from error
    resolved = {
        name: str((path.parent / getattr(data, name)).resolve())
        for name in ("train_path", "valid_path", "tokenizer_path")
    }
    return RunConfig(model, training, DataConfig(**resolved))

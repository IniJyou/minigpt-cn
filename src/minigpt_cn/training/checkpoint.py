"""项目 checkpoint：旧作业保存/加载逻辑的配置化扩展，不改模型计算。

只加载可信来源的文件，并显式使用 weights_only=True。
新格式保留 model_state_dict / optimizer / iteration 名称，新增版本、配置和恢复状态。
"""

import argparse
import hashlib
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from minigpt_cn.config import DataConfig, ModelConfig, TrainingConfig, load_model_config
from minigpt_cn.model import MiniGPT
from minigpt_cn.tokenizer import BPETokenizer

FORMAT_VERSION = 1


@dataclass
class LoadedCheckpoint:
    model: MiniGPT
    tokenizer: BPETokenizer
    iteration: int
    training_config: TrainingConfig | None
    data_config: DataConfig | None
    optimizer_settings: list[dict]
    resume_supported: bool
    run_state: dict | None = None


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_iteration(iteration):
    # iteration 是真实“已完成更新数”，不是本次命令要求的训练终点。
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration 必须为非负整数")


def _tokenizer_info(path, vocab_size):
    tokenizer = BPETokenizer.load(path)
    if tokenizer.vocab_size != vocab_size or set(tokenizer.vocab) != set(range(vocab_size)):
        raise ValueError("tokenizer 词表大小/连续 ID 必须与模型一致")
    return tokenizer, {"vocab_size": vocab_size, "sha256": file_sha256(path)}


def capture_rng_state():
    # NumPy ndarray 转成普通 list，使整个 checkpoint 可被 weights_only=True 安全读取。
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state(),
        # CPU 运行不能因为保存 checkpoint 就额外初始化 CUDA。
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _optimizer_name(optimizer):
    cls = type(optimizer)
    return f"{cls.__module__}.{cls.__qualname__}"


def _parameter_names(optimizer, model):
    # 优化器 state 按参数位置匹配，不能只检查“来自同一个模型”。
    # 保存稳定的参数名称，拒绝重排参数后把 Q 的动量错误加载给 K。
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return [
        [names[id(parameter)] for parameter in group["params"]] for group in optimizer.param_groups
    ]


def _settings(optimizer_state):
    # 恢复会覆盖新建优化器的参数组，因此必须记录真正生效的 lr/betas/eps/weight_decay。
    return (
        []
        if optimizer_state is None
        else [
            {key: value for key, value in group.items() if key != "params"}
            for group in optimizer_state["param_groups"]
        ]
    )


def _load_weights(model, state):
    if set(state) != set(model.state_dict()):
        raise ValueError("checkpoint 权重键与模型结构不匹配")
    if any(state[name].shape != value.shape for name, value in model.state_dict().items()):
        raise ValueError("checkpoint 权重形状与模型不匹配")
    # 同名双键不能掩盖冲突：共享模型后加载的键会覆盖同一个 Parameter。
    if model.config.tie_embeddings and not torch.equal(
        state["token_embeddings.W"], state["lm_head.W"]
    ):
        raise ValueError("checkpoint 的共享 embedding/LM head 权重不一致")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("checkpoint 权重含 NaN/Inf")
    model.load_state_dict(state, strict=True)


def _write_checkpoint(path, payload, tokenizer_path, overwrite):
    path = Path(path).resolve()
    if path == Path(tokenizer_path).resolve():
        raise ValueError("checkpoint 不能覆盖 tokenizer 文件")
    if path.exists() and not overwrite:
        raise FileExistsError(f"输出已存在，拒绝覆盖：{path}")
    if path.exists():
        previous = torch.load(path, map_location="cpu", weights_only=True)
        if (
            previous.get("format_version") != FORMAT_VERSION
            or previous.get("model_config") != payload["model_config"]
        ):
            raise ValueError("只能覆盖同模型的新格式 checkpoint，请为新实验使用新目录")
        if previous["iteration"] > payload["iteration"]:
            raise ValueError("拒绝把 checkpoint 已完成步数写小")
    try:
        relative = Path(os.path.relpath(Path(tokenizer_path).resolve(), path.parent)).as_posix()
    except ValueError:  # Windows 不同盘符无法表示相对路径，加载时可显式覆盖 tokenizer 路径。
        relative = Path(tokenizer_path).resolve().as_posix()
    payload["tokenizer"]["path"] = relative
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写同目录临时文件，完整写完再替换目标，避免中断留下半个正常 checkpoint。
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)  # 只清理本次创建的临时文件，不删除任何目录。
    return path


def save_checkpoint(
    path,
    model,
    *,
    tokenizer_path,
    iteration,
    optimizer=None,
    training_config=None,
    data_config=None,
    sampler=None,
    overwrite=False,
    run_state=None,
):
    """保存真实完成步数；训练保存还需 optimizer、配置和 CPU 采样发生器。"""
    _check_iteration(iteration)
    _, tokenizer_info = _tokenizer_info(tokenizer_path, model.config.vocab_size)
    if (
        data_config is not None
        and Path(data_config.tokenizer_path).resolve() != Path(tokenizer_path).resolve()
    ):
        raise ValueError("数据配置的 tokenizer 路径与实际保存文件不一致")
    if not all(parameter.dtype == torch.float32 for parameter in model.parameters()):
        raise ValueError("当前训练格式要求 FP32 参数；BF16 仅用于前向 autocast")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("拒绝保存 NaN/Inf 模型")
    if optimizer is not None:
        if training_config is None or data_config is None or sampler is None:
            raise ValueError("训练 checkpoint 必须同时保存训练/数据配置和采样发生器")
        if sampler.device.type != "cpu":
            raise ValueError("数据采样发生器必须在 CPU")
        model_ids = {id(parameter) for parameter in model.parameters()}
        optimizer_parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        if len(optimizer_parameters) != len({id(parameter) for parameter in optimizer_parameters}):
            raise ValueError("优化器参数重复，共享参数不能更新两次")
        if any(id(parameter) not in model_ids for parameter in optimizer_parameters):
            raise ValueError("优化器参数必须来自保存的模型")
        if not all(
            torch.isfinite(value).all()
            for state in optimizer.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        ):
            raise ValueError("拒绝保存 NaN/Inf 优化器状态")
    payload = {
        "format_version": FORMAT_VERSION,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "optimizer_name": None if optimizer is None else _optimizer_name(optimizer),
        "optimizer_parameter_names": None
        if optimizer is None
        else _parameter_names(optimizer, model),
        "iteration": iteration,
        "training_config": None if training_config is None else asdict(training_config),
        "data_config": None if data_config is None else asdict(data_config),
        "tokenizer": tokenizer_info,
        "rng_state": capture_rng_state(),
        "sampler_state": None if sampler is None else sampler.get_state(),
        "resume_supported": optimizer is not None,
        # D阶段：语料校验和、累计耗时和训练来源；旧C文件没有此项仍可用于推理。
        "run_state": run_state,
    }
    return _write_checkpoint(path, payload, tokenizer_path, overwrite)


def load_checkpoint(
    path,
    *,
    device="cpu",
    tokenizer_path=None,
    expected_model_config=None,
    model=None,
    optimizer=None,
    restore_rng=False,
    sampler=None,
):
    """未传 model 时按保存配置创建 eval 模型；训练恢复传现有模型/优化器/采样器。"""
    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    version = payload.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != FORMAT_VERSION:
        raise ValueError("不支持的 checkpoint 版本；旧 pilot 必须先显式 import-legacy")
    config = ModelConfig(**payload["model_config"])
    _check_iteration(payload["iteration"])
    if expected_model_config is not None and config != expected_model_config:
        raise ValueError("checkpoint 模型配置不匹配，包括 RoPE theta、dropout 和初始化配置")
    if model is not None and model.config != config:
        raise ValueError("提供的模型与 checkpoint 模型配置不匹配")
    tokenizer_path = (
        path.parent / payload["tokenizer"]["path"]
        if tokenizer_path is None
        else Path(tokenizer_path)
    )
    tokenizer, info = _tokenizer_info(tokenizer_path, config.vocab_size)
    if info["sha256"] != payload["tokenizer"]["sha256"]:
        raise ValueError("tokenizer 校验和不匹配，请使用保存时的同一版本")
    training = (
        None if payload["training_config"] is None else TrainingConfig(**payload["training_config"])
    )
    data = None if payload["data_config"] is None else DataConfig(**payload["data_config"])
    resumable = payload.get("resume_supported") is True and all(
        payload.get(name) is not None
        for name in ("optimizer", "training_config", "data_config", "rng_state", "sampler_state")
    )
    if optimizer is not None:
        if model is None or not resumable:
            raise ValueError("恢复优化器需要提供模型，且 checkpoint 必须支持训练恢复")
        if _optimizer_name(optimizer) != payload["optimizer_name"]:
            raise ValueError("优化器类型不匹配，不能把不同实现的状态静默混用")
        model_ids = {id(parameter) for parameter in model.parameters()}
        if any(
            id(parameter) not in model_ids
            for group in optimizer.param_groups
            for parameter in group["params"]
        ):
            raise ValueError("优化器参数必须来自提供的模型")
        if _parameter_names(optimizer, model) != payload.get("optimizer_parameter_names"):
            raise ValueError("优化器参数组的顺序与 checkpoint 不匹配")
    if restore_rng:
        if not resumable or optimizer is None or sampler is None:
            raise ValueError("精确恢复需要训练 checkpoint、优化器与采样发生器；旧 pilot 不支持")
        if sampler.device.type != "cpu":
            raise ValueError("数据采样发生器必须在 CPU")
        cuda_states = payload["rng_state"]["torch_cuda"]
        if cuda_states and (
            not torch.cuda.is_available() or len(cuda_states) > torch.cuda.device_count()
        ):
            raise ValueError("当前 CUDA 设备不足以恢复原随机状态；推理加载不需要 restore_rng")
    if model is None:
        model = MiniGPT(config, device=device, dtype=torch.float32).eval()
    _load_weights(model, payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_rng:
        # 必须在模型初始化/权重和优化器重载之后恢复，初始化本身会消耗随机数。
        restore_rng_state(payload["rng_state"])
        sampler.set_state(payload["sampler_state"])
    effective_state = optimizer.state_dict() if optimizer is not None else payload["optimizer"]
    return LoadedCheckpoint(
        model,
        tokenizer,
        payload["iteration"],
        training,
        data,
        _settings(effective_state),
        resumable,
        payload.get("run_state"),
    )


def import_legacy_checkpoint(source, output, *, model_config, tokenizer_path):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("不能覆盖原作业 checkpoint")
    legacy = torch.load(source, map_location="cpu", weights_only=True)
    if "format_version" in legacy:
        raise ValueError("已经是项目格式，不需要再次 import-legacy")
    _check_iteration(legacy["iteration"])
    _, tokenizer_info = _tokenizer_info(tokenizer_path, model_config.vocab_size)
    model = MiniGPT(model_config, dtype=torch.float32)
    _load_weights(model, legacy["model_state_dict"])
    # 旧 pilot 缺模型配置、RNG 和采样状态：显式提供结构，只导入推理权重。
    # 不用“当前进程的 RNG”冒充5000步时的状态，也不伪造旧实验的 TrainingConfig。
    payload = {
        "format_version": FORMAT_VERSION,
        "model_config": asdict(model_config),
        "model_state_dict": model.state_dict(),
        "iteration": legacy["iteration"],
        "optimizer": None,
        "optimizer_name": None,
        "optimizer_parameter_names": None,
        "training_config": None,
        "data_config": None,
        "tokenizer": tokenizer_info,
        "rng_state": None,
        "sampler_state": None,
        "resume_supported": False,
        "source": {"kind": "legacy-pilot-inference", "sha256": file_sha256(source)},
    }
    return _write_checkpoint(output, payload, tokenizer_path, overwrite=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    legacy = commands.add_parser("import-legacy", help="原 pilot -> 新推理格式，不覆盖原文件")
    legacy.add_argument("--source", type=Path, required=True)
    legacy.add_argument("--output", type=Path, required=True)
    legacy.add_argument("--model-config", type=Path, required=True)
    legacy.add_argument("--tokenizer", type=Path, required=True)
    inspect = commands.add_parser("inspect", help="加载并校验新格式，打印实际配置")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--tokenizer", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    try:
        if args.command == "import-legacy":
            output = import_legacy_checkpoint(
                args.source,
                args.output,
                model_config=load_model_config(args.model_config),
                tokenizer_path=args.tokenizer,
            )
            print(f"已保存：{output}\n仅用于推理；原 checkpoint 未修改，旧实验不支持精确续训。")
        else:
            loaded = load_checkpoint(args.path, tokenizer_path=args.tokenizer)
            print(
                json.dumps(
                    {
                        "format_version": FORMAT_VERSION,
                        "iteration": loaded.iteration,
                        "model_config": asdict(loaded.model.config),
                        "training_config": None
                        if loaded.training_config is None
                        else asdict(loaded.training_config),
                        "data_config": None
                        if loaded.data_config is None
                        else asdict(loaded.data_config),
                        "vocab_size": loaded.tokenizer.vocab_size,
                        "unique_parameters": sum(
                            parameter.numel() for parameter in loaded.model.parameters()
                        ),
                        "shared_parameter": loaded.model.token_embeddings.W
                        is loaded.model.lm_head.W,
                        "resume_supported": loaded.resume_supported,
                        "optimizer_settings": loaded.optimizer_settings,
                        "run_state": loaded.run_state,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

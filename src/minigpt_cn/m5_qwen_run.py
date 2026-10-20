"""离线Qwen结构标注：固定协议、全量原始响应、沿用严格校准评分。"""

import argparse
import hashlib
import json
import os
import platform
import sys
import sysconfig
import time
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

from minigpt_cn.m5_rule_run import PROJECT, write_json, write_rows
from minigpt_cn.m5_structure import KINDS, evaluate, parse_json, valid_prediction, validate_input


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_packages():
    # PYTHONPATH中的本项目egg-info属于已绑定源码，不冒充虚拟环境安装包。
    paths = sysconfig.get_paths()
    locations = sorted({paths["purelib"], paths["platlib"]})
    return {d.metadata["Name"]: d.version for d in distributions(path=locations)}


def offline_environment():
    sys.dont_write_bytecode = True
    cache = PROJECT / "data/raw/m5_hf_cache"
    os.environ.update({
        "HF_HOME": str(cache), "HF_HUB_CACHE": str(cache / "hub"),
        "HF_XET_CACHE": str(cache / "xet"), "HF_ASSETS_CACHE": str(cache / "assets"),
        "HF_DATASETS_CACHE": str(cache / "datasets"), "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    })


def messages_for(prompt, input_value):
    """推理只拿完整提示和input；不接收参考标注、来源或review_id。"""
    validate_input(input_value)
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(
                input_value, ensure_ascii=False, separators=(",", ":"))}]


def completion_record(tokenizer, prompt_ids, sequence, eos_ids, max_new_tokens):
    """只切除输入token和一个终止EOS；其他特殊token、空白、围栏全部保留。"""
    if sequence[:len(prompt_ids)] != prompt_ids:
        raise ValueError("generated sequence does not preserve the input prefix")
    new_ids = sequence[len(prompt_ids):]
    if not new_ids or len(new_ids) > max_new_tokens:
        raise ValueError("invalid generated token count")
    stopped_eos = new_ids[-1] in eos_ids
    content_ids = new_ids[:-1] if stopped_eos else new_ids
    decode = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
    return {"response": tokenizer.decode(content_ids, **decode),
            "raw_completion": tokenizer.decode(new_ids, **decode),
            "generated_token_ids": new_ids, "generated_tokens": len(new_ids),
            "stop_reason": "eos" if stopped_eos else "length",
            "hit_token_limit": len(new_ids) == max_new_tokens, "runtime_error": None}


class QwenBackend:
    """可选依赖延迟导入；CPU单元测试不用安装Transformers或读取真实模型。"""

    def __init__(self, protocol):
        offline_environment()
        if sys.prefix != protocol["environment_prefix"]:
            raise ValueError("use the dedicated M5 interpreter")
        if platform.python_version() != protocol["python_version"]:
            raise ValueError("Python version differs from protocol")
        packages = installed_packages()
        if packages != protocol["expected_packages"]:
            raise ValueError("M5 package versions differ from protocol")
        import torch
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig

        self.torch, self.protocol = torch, protocol
        self.model_dir = PROJECT / protocol["model_directory"]
        config = AutoConfig.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False)
        self.context_limit = config.max_position_embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False)
        template_sha = hashlib.sha256(self.tokenizer.chat_template.encode()).hexdigest()
        if template_sha != protocol["chat_template_sha256"]:
            raise ValueError("chat template SHA256 differs")
        self.generation_config = GenerationConfig(**protocol["generation"])
        self.runtime = {"packages": packages, "model_loaded": False}

    def encode(self, messages):
        return self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)

    def load(self):
        from transformers import AutoModelForCausalLM

        torch = self.torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(
                including_emulation=False):
            raise RuntimeError("CUDA and native BF16 are required; no CPU/quantization fallback")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation="sdpa", low_cpu_mem_usage=True,
        ).eval().requires_grad_(False)
        torch.cuda.synchronize()
        self.runtime.update({
            "model_loaded": True, "model_class": type(self.model).__name__,
            "model_dtype": str(next(self.model.parameters()).dtype),
            "model_device": str(next(self.model.parameters()).device),
            "unique_parameters": sum(p.numel() for p in self.model.parameters()),
            "gpu_name": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
            "load_seconds": time.perf_counter() - start,
            "generation_config": self.generation_config.to_dict(),
            "attention_implementation": self.model.config._attn_implementation,
            "tf32": False, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        })

    def generate(self, ids):
        torch, settings = self.torch, self.protocol["generation"]
        torch.manual_seed(self.protocol["seed_per_sample"])
        torch.cuda.manual_seed_all(self.protocol["seed_per_sample"])
        # input_ids/attention_mask: [batch=1, input_tokens]；整条输入无padding。
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
        mask = torch.ones_like(input_ids)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, attention_mask=mask,
                generation_config=self.generation_config, logits_to_keep=1,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        row = completion_record(self.tokenizer, ids, output[0].tolist(),
                                settings["eos_token_id"], settings["max_new_tokens"])
        row["seconds"] = seconds
        self.runtime.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        return row


def load_protocol(protocol_path, mode):
    raw = Path(protocol_path).read_bytes()
    protocol = parse_json(raw)
    if protocol.get("protocol_version") != "m5-qwen-calibration-v1.1":
        raise ValueError("unsupported Qwen protocol")
    if protocol.get("rule_version") != "m5-structure-v1.1":
        raise ValueError("unsupported rule version")
    actual = {}
    for name, expected in protocol["artifacts"].items():
        path = (PROJECT / name).resolve()
        if not path.is_relative_to(PROJECT):
            raise ValueError("artifact must be inside project")
        actual[name] = sha256_file(path)
        if actual[name] != expected:
            raise ValueError(f"artifact SHA256 differs: {name}")
    data = protocol["data_files"][mode]
    if data["path"] not in actual or protocol["system_prompt"] not in actual:
        raise ValueError("data and prompt must be SHA256-bound")
    model = PROJECT / protocol["model_directory"]
    for name in protocol["model_files"]:
        if (model / name).relative_to(PROJECT).as_posix() not in actual:
            raise ValueError("every model file must be SHA256-bound")
    if {p.name for p in model.iterdir() if p.is_file()} != set(protocol["model_files"]):
        raise ValueError("unexpected/missing model directory files")
    records = [parse_json(line) for line in (PROJECT / data["path"]).read_text(
        encoding="utf-8").splitlines()]
    if not records or len({r["review_id"] for r in records}) != len(records):
        raise ValueError("empty/duplicate review IDs")
    if len(records) != data["samples"] or sum(len(r["input"]["blocks"]) for r in records) != data["blocks"]:
        raise ValueError("sample/block counts differ")
    for record in records:
        validate_input(record["input"])
        gold = record["proposed_target"]
        if record["rule_version"] != protocol["rule_version"] or not valid_prediction(gold):
            raise ValueError("invalid reference or rule version")
        if [a["block_id"] for a in gold["annotations"]] != [b["block_id"] for b in record["input"]["blocks"]]:
            raise ValueError("reference coverage differs")
    return protocol, raw, actual, records


def report_text(metrics, predictions, mode):
    counts = metrics["counts"]
    lines = [f"# Qwen3-0.6B {mode} 固定协议运行", "",
             "仅校准/链路诊断，非独立人工金标或未见泛化评测；没有训练LoRA。", "",
             "| 指标 | 通过/总数 |", "| --- | ---: |"]
    for label, key, denominator in (
        ("JSON", "json_valid", "samples"), ("schema", "schema_valid", "samples"),
        ("编号覆盖", "coverage_valid", "samples"), ("整片段", "sample_exact", "samples"),
        ("类别", "kind_correct", "blocks"), ("块全部字段", "block_exact", "blocks"),
        ("标题含null", "title_correct", "blocks"),
        ("非空标题", "nonnull_title_correct", "nonnull_gold_titles"),
        ("reason含null", "reason_correct", "blocks"),
        ("非空reason", "nonnull_reason_correct", "nonnull_gold_reasons"),
    ):
        lines.append(f"| {label} | {counts[key]}/{counts[denominator]} |")
    lines += ["", f"固定八类macro-F1={metrics['macro_f1']:.6f}；micro-F1={metrics['micro_f1']:.6f}。", "",
              "| 类别 | 支持数 | TP | FP | FN | F1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for kind in KINDS:
        item = metrics["per_class"][kind]
        lines.append(f"| {kind} | {item['support']} | {item['tp']} | {item['fp']} | {item['fn']} | {item['f1']:.6f} |")
    lines += ["", f"输出触及token上限：{sum(p['hit_token_limit'] for p in predictions)}条；"
              f"运行异常：{sum(p['runtime_error'] is not None for p in predictions)}条。", "",
              "完整响应/token保存在predictions.jsonl；全部格式/字段错误及参考/输入保存在errors.jsonl。",
              "只去除生成序列最后一个终止EOS，不修JSON、不删围栏/思考、不补块或重排；"
              "坏整条回复沿用严格评分，所有失败都保留分母。",
              "这是一次预先固定种子的采样运行，不代表不同种子、驱动或硬件结果完全一致。", ""]
    return "\n".join(lines)


def run(protocol_path, output_dir, mode="calibration", backend_factory=QwenBackend):
    protocol_path, output_dir = Path(protocol_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError("output exists; choose a new directory")
    if not output_dir.is_relative_to(PROJECT / "logs"):
        raise ValueError("output directory must be under project logs")
    protocol, raw, actual, records = load_protocol(protocol_path, mode)
    prompt = (PROJECT / protocol["system_prompt"]).read_text(encoding="utf-8")
    backend = backend_factory(protocol)
    encoded = [backend.encode(messages_for(prompt, r["input"])) for r in records]
    limit, new_limit = protocol["max_input_tokens"], protocol["generation"]["max_new_tokens"]
    if limit <= 0 or new_limit <= 0 or limit + new_limit > backend.context_limit:
        raise ValueError("input plus output budget exceeds model context")
    if any(not ids or len(ids) > limit for ids in encoded):
        raise ValueError("input token budget exceeded; no truncation or generation performed")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "protocol.json", protocol)
    write_rows(output_dir / "input_tokens.jsonl", [
        {"review_id": r["review_id"], "token_ids": ids, "input_tokens": len(ids)}
        for r, ids in zip(records, encoded)])
    started = datetime.now(timezone.utc).isoformat()
    predictions = []
    try:
        backend.load()
        with (output_dir / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for record, ids in zip(records, encoded):
                try:
                    result = backend.generate(list(ids))
                except Exception as exc:
                    # 不重试/挑成功回复；生成异常用空响应计入全部分母。
                    result = {"response": "", "raw_completion": "", "generated_token_ids": [],
                              "generated_tokens": 0, "stop_reason": "error", "hit_token_limit": False,
                              "runtime_error": f"{type(exc).__name__}: {exc}", "seconds": None}
                row = {"review_id": record["review_id"], "input_tokens": len(ids), **result}
                predictions.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{mode} {len(predictions)}/{len(records)} {record['review_id']} "
                      f"tokens={row['generated_tokens']} stop={row['stop_reason']}", flush=True)
        if protocol_path.read_bytes() != raw or any(
            sha256_file(PROJECT / name) != digest for name, digest in actual.items()
        ):
            raise ValueError("bound protocol/artifact changed during run; results not accepted")
        metrics = evaluate((r["input"], r["proposed_target"], p["response"])
                           for r, p in zip(records, predictions))
        diagnostics = metrics.pop("diagnostics")
        write_rows(output_dir / "errors.jsonl", [
            {"review_id": r["review_id"], "input": r["input"], "raw_response": p["response"], **d}
            for r, p, d in zip(records, predictions, diagnostics) if d["problems"] or d["errors"]])
        write_json(output_dir / "metrics.json", metrics)
        (output_dir / "report.md").write_text(report_text(metrics, predictions, mode), encoding="utf-8")
        write_json(output_dir / "run_manifest.json", {
            "status": "completed", "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
            "protocol_sha256": hashlib.sha256(raw).hexdigest(), "artifacts": actual,
            "runtime": backend.runtime, "model_run": True, "lora_run": False,
            "runtime_errors": sum(p["runtime_error"] is not None for p in predictions),
            "outputs": {p.name: sha256_file(p) for p in sorted(output_dir.iterdir())},
        })
    except BaseException as exc:
        write_json(output_dir / "failure.json", {"error": f"{type(exc).__name__}: {exc}",
                                                "completed_samples": len(predictions)})
        raise
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT / "configs/m5_qwen_baseline_v1_1.json")
    parser.add_argument("--mode", choices=("smoke", "calibration"), default="calibration")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    metrics = run(args.protocol, args.output_dir, args.mode)
    print(json.dumps({"output_dir": str(args.output_dir), "counts": metrics["counts"],
                      "macro_f1": metrics["macro_f1"], "micro_f1": metrics["micro_f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

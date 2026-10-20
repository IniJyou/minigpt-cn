"""greedy调用边界修正版；复用v1只读诊断流程，显式禁止模型默认值回填。"""

import argparse
import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from minigpt_cn import m5_greedy_run as original
from minigpt_cn.m5_lora_run import inside
from minigpt_cn.m5_qwen_run import completion_record, sha256_file
from minigpt_cn.m5_rule_run import PROJECT
from minigpt_cn.m5_structure import parse_json

SOURCES = ("src/minigpt_cn/m5_greedy_run_v2.py", "tests/test_m5_greedy_run_v2.py",
           "scripts/m5_greedy_v2_wsl.sh", "docs/m5-greedy-call-fix.md")


class DecodeIntegrityError(BaseException):
    """配置完整性错误必须全局停止，不能被v1逐条Exception处理当成空回复。"""


def effective_config(model, configuration, expected):
    prepared, _ = model._prepare_generation_config(
        configuration, use_model_defaults=False, do_sample=False)
    if (prepared.get_generation_mode().value != "greedy_search"
            or any(getattr(prepared, key) != value for key, value in expected.items())):
        raise DecodeIntegrityError("effective generate configuration differs from frozen greedy settings")
    return prepared


def record_attempt(folder, condition):
    path = folder / "call_attempts.jsonl"
    rows = [parse_json(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    number = len(rows) + 1
    if number > 30 or condition != ("base" if number <= 15 else "adapter_step_000045"):
        raise DecodeIntegrityError("durable call budget/order differs")
    entry = {"attempt": number, "condition": condition, "created_utc": datetime.now(timezone.utc).isoformat(),
             "effective_mode": "greedy_search", "do_sample": False, "use_model_defaults": False}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class FixedGreedyBackend(original.GreedyBackend):
    def __init__(self, plan, api):
        super().__init__(plan, api)
        from transformers.generation.utils import GenerationMixin

        if plan["call_boundary_api_check"] not in plan["artifacts"]:
            raise DecodeIntegrityError("call-boundary regression evidence is not bound")
        fix = parse_json(inside(PROJECT, plan["call_boundary_api_check"]).read_bytes())
        if (not fix["passed"] or sha256_file(inspect.getsourcefile(GenerationMixin)) != fix["generation_utils_sha256"]):
            raise DecodeIntegrityError("actual generate implementation differs from CPU regression")
        self.plan = plan
        self.condition = "base"
        self.runtime["generate_overrides"] = {"do_sample": False, "use_model_defaults": False}
        self.runtime["generation_utils_sha256"] = fix["generation_utils_sha256"]

    def check_effective(self):
        configuration = effective_config(self.model, self.generation_config, self.plan["generation"])
        self.runtime.update({"effective_generation_config": configuration.to_dict(),
                             "effective_generation_mode": "greedy_search"})
        return configuration

    def load_base(self):
        super().load_base()
        self.check_effective()

    def load_adapter(self, path):
        super().load_adapter(path)
        self.condition = "adapter_step_000045"
        self.check_effective()

    def generate(self, ids):
        configuration = self.check_effective()
        record_attempt(inside(PROJECT, self.plan["experiment_directory"]), self.condition)
        torch, settings = self.torch, self.plan["generation"]
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # input_ids/attention_mask: [batch=1, input_tokens]，与冻结v1相同。
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda:0")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                generation_config=configuration, do_sample=False, use_model_defaults=False, logits_to_keep=1)
        torch.cuda.synchronize()
        result = completion_record(self.tokenizer, ids, output[0].tolist(), settings["eos_token_id"], 1024)
        result["seconds"] = time.perf_counter() - started
        self.runtime.update({"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        return result


def run(execution_path, phase="inspect", root=PROJECT, backend_factory=FixedGreedyBackend):
    if phase not in ("inspect", "run"):
        raise ValueError("only inspect/run; no train/test")
    execution = parse_json(Path(execution_path).read_bytes())
    if execution.get("revision") != "call-boundary-fix-v2" or not set(SOURCES) <= set(execution["artifacts"]):
        raise ValueError("v2 execution must bind the call-boundary fix and regression tests")
    return original.run(execution_path, phase, root, backend_factory)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=PROJECT / "configs/m5_greedy_diagnostic_v2/runtime.json")
    parser.add_argument("--phase", choices=("inspect", "run"), default="inspect")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.execution, args.phase), ensure_ascii=False))


if __name__ == "__main__":
    main()

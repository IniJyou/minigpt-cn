"""冻结的greedy valid诊断：默认只检查，执行最多30次推理，无训练/test入口。"""

import argparse
import gc
import inspect
import json
from pathlib import Path

from minigpt_cn.m5_lora_run import (
    evaluate_output,
    file_tree,
    inside,
    load_records,
    preflight_permission,
    quality_gates,
    seal,
    verify_files,
    verify_receipt,
)
from minigpt_cn.m5_qwen_run import QwenBackend, messages_for, sha256_file
from minigpt_cn.m5_rule_run import PROJECT, write_json, write_rows
from minigpt_cn.m5_structure import evaluate, parse_json

SOURCES = ("src/minigpt_cn/m5_greedy_run.py", "tests/test_m5_greedy_run.py",
           "scripts/m5_greedy_wsl.sh", "docs/m5-greedy-runner.md")


def load_execution(path, root=PROJECT):
    execution = parse_json(Path(path).read_bytes())
    if execution.get("version") != "m5-greedy-runtime-v1":
        raise ValueError("unsupported greedy runtime")
    artifacts = execution["artifacts"]
    if not set(SOURCES) | {execution["plan"], execution["cpu_checks"]} <= set(artifacts):
        raise ValueError("runtime must bind runner, tests, launcher, documentation and CPU checks")
    verify_files(root, artifacts)
    plan_path = inside(root, execution["plan"])
    if sha256_file(plan_path) != execution["plan_sha256"]:
        raise ValueError("greedy plan SHA256 differs")
    plan = parse_json(plan_path.read_bytes())
    if plan.get("version") != "m5-greedy-diagnostic-v1":
        raise ValueError("unsupported greedy plan")
    parent_runtime_path = inside(root, plan["parent_runtime"])
    parent_path = inside(root, plan["parent_plan"])
    if (sha256_file(parent_runtime_path) != plan["parent_runtime_sha256"]
            or sha256_file(parent_path) != plan["parent_plan_sha256"]):
        raise ValueError("parent protocol SHA256 differs")
    parent_execution = parse_json(parent_runtime_path.read_bytes())
    parent = parse_json(parent_path.read_bytes())
    dataset = parse_json(inside(root, parent["dataset_protocol"]).read_bytes())
    base = parse_json(inside(root, parent["base_protocol"]).read_bytes())
    # 所有父级绑定也必须进入新执行清单；只做摘要读取，不解析train/test记录。
    for bindings in (plan["artifacts"], parent_execution["artifacts"], parent["artifacts"],
                     dataset["artifacts"], base["artifacts"]):
        if any(artifacts.get(name) != digest for name, digest in bindings.items()):
            raise ValueError("parent artifact is missing from greedy execution bindings")
    model = inside(root, parent["model"]["model_directory"])
    if {p.name for p in model.iterdir() if p.is_file()} != set(base["model_files"]):
        raise ValueError("unexpected/missing model sidecar")
    folder = inside(root, parent_execution["experiment_directory"])
    parent_sha = sha256_file(parent_runtime_path)
    preflight_permission(folder, parent_sha)
    verify_receipt(folder / "train-valid", parent_sha)
    training = parse_json((folder / "train-valid/training_result.json").read_bytes())
    selection = parse_json((folder / "train-valid/selection.json").read_bytes())
    if (not training["completed"] or training["optimizer_steps"] != 180
            or training["supervised_exposures"] != parent["training"]["supervised_token_exposures_total"]):
        raise ValueError("parent training is incomplete")
    if selection["accepted"] or selection["selected_step"] is not None:
        raise ValueError("parent selection differs from the frozen diagnostic basis")
    adapter = folder / "train-valid/checkpoints/step_000045"
    expected = next(c["checkpoint_files"] for c in selection["candidates"] if c["step"] == 45)
    if file_tree(adapter) != expected or plan["adapter"]["files"] != expected or plan["adapter"]["step"] != 45:
        raise ValueError("fixed diagnostic adapter differs")
    conditions = [{"name": "base", "adapter_directory": None, "samples": 15},
                  {"name": "adapter_step_000045", "adapter_directory": adapter.relative_to(root).as_posix(), "samples": 15}]
    if plan["conditions"] != conditions or plan["limits"]["max_model_generation_calls"] != 30:
        raise ValueError("only the frozen base15 and step45 fifteen calls are permitted")
    if (plan["limits"]["training_steps"] != 0 or plan["limits"]["test_model_calls"] != 0
            or plan["limits"]["retry"] or plan["limits"]["other_checkpoints_or_seeds"]):
        raise ValueError("training, test and search are forbidden")
    api_path = inside(root, execution["generation_api_check"])
    if execution["generation_api_check"] not in artifacts:
        raise ValueError("generation API check is not bound")
    api = parse_json(api_path.read_bytes())
    if (plan["generation"] != api["greedy_generation"] or plan["generation"]["do_sample"] is not False
            or plan["generation"]["num_beams"] != 1
            or any(plan["generation"][key] is not None for key in ("temperature", "top_p", "top_k", "min_p"))):
        raise ValueError("only the checked greedy configuration is permitted")
    if (plan["model"] != parent["model"] or plan["system_prompt"] != parent["system_prompt"]
            or plan["valid_records"] != parent["records"]["valid"] or plan["valid_samples"] != 15
            or plan["valid_blocks"] != 62 or plan["max_input_tokens"] != 4096 or plan["max_new_tokens"] != 1024
            or plan["scoring"]["minimum_gates"] != parent["validation"]["minimum_gates"]):
        raise ValueError("task, data, budget or scoring differs from the parent")
    cpu = parse_json(inside(root, execution["cpu_checks"]).read_bytes())
    if not cpu["passed"]:
        raise ValueError("CPU boundary checks have not passed")
    output = inside(root, plan["experiment_directory"])
    if not output.is_relative_to((root / "logs").resolve()):
        raise ValueError("diagnostic output must be in project logs")
    return execution, plan, parent, api


class GreedyBackend(QwenBackend):
    """复用冻结Qwen推理接口，adapter以只读状态加载，不暴露训练方法。"""

    def __init__(self, plan, api):
        environment = plan["environment"]
        super().__init__({"environment_prefix": environment["prefix"],
                          "python_version": environment["python_version"],
                          "expected_packages": environment["expected_packages"],
                          "model_directory": plan["model"]["model_directory"],
                          "chat_template_sha256": plan["model"]["chat_template_sha256"],
                          "generation": plan["generation"], "seed_per_sample": plan["seed_per_sample"]})
        source = Path(inspect.getsourcefile(type(self.generation_config)))
        if (sha256_file(source) != api["configuration_source_sha256"]
                or self.generation_config.get_generation_mode().value != "greedy_search"):
            raise ValueError("actual generation API source/mode differs")
        self.runtime.update({"generation_mode": "greedy_search", "generation_api_source_sha256": sha256_file(source),
                             "training_steps": 0, "inference_only": True})

    def assert_frozen(self):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("inference model must be eval and fully frozen")

    def load_base(self):
        self.torch.use_deterministic_algorithms(False)
        super().load()
        self.assert_frozen()

    def load_adapter(self, path):
        from peft import PeftModel

        self.model = PeftModel.from_pretrained(self.model, str(path), is_trainable=False,
                                               local_files_only=True).eval().requires_grad_(False)
        self.assert_frozen()
        self.runtime.update({"adapter_directory": str(path), "adapter_files": file_tree(path)})

    def close(self):
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if self.torch.cuda.is_initialized():
            self.torch.cuda.empty_cache()


def encode_valid(backend, prompt, records, plan):
    encoded = [backend.encode(messages_for(prompt, row["input"])) for row in records]
    if any(not ids or len(ids) > plan["max_input_tokens"] for ids in encoded):
        raise ValueError("valid input budget exceeded; no truncation")
    if plan["max_input_tokens"] + plan["max_new_tokens"] > backend.context_limit:
        raise ValueError("generation budget exceeds context")
    return encoded


def paired_changes(records, historical, current):
    ids = [r["sample_id"] for r in records]
    if [p["sample_id"] for p in historical] != ids or [p["sample_id"] for p in current] != ids:
        raise ValueError("paired sample IDs differ")
    changes = []
    for row, before, after in zip(records, historical, current):
        scores = [evaluate([(row["input"], row["target"], p["response"])]) for p in (before, after)]
        detail = {"sample_id": row["sample_id"], "response_changed": before["response"] != after["response"]}
        for name, score in zip(("sampled", "greedy"), scores):
            detail[name] = {key: score["counts"][key] for key in (
                "json_valid", "schema_valid", "coverage_valid", "block_exact", "sample_exact")}
        detail["block_exact_delta"] = detail["greedy"]["block_exact"] - detail["sampled"]["block_exact"]
        changes.append(detail)
    return {"samples": changes, "more_exact_blocks": sum(c["block_exact_delta"] > 0 for c in changes),
            "fewer_exact_blocks": sum(c["block_exact_delta"] < 0 for c in changes),
            "same_exact_blocks": sum(c["block_exact_delta"] == 0 for c in changes)}


def run(execution_path, phase="inspect", root=PROJECT, backend_factory=GreedyBackend):
    if phase not in ("inspect", "run"):
        raise ValueError("only inspect or run is supported; no train/test")
    execution_path = Path(execution_path).resolve()
    execution_sha = sha256_file(execution_path)
    execution, plan, parent, api = load_execution(execution_path, root)
    output = inside(root, plan["experiment_directory"])
    if phase == "run" and output.exists():
        raise FileExistsError("diagnostic already claimed; no overwrite or retry")
    records = load_records(parent, "valid", root)
    if len(records) != 15 or sum(len(r["input"]["blocks"]) for r in records) != 62:
        raise ValueError("valid sample/block counts differ")
    prompt = inside(root, plan["system_prompt"]).read_text(encoding="utf-8")
    backend = backend_factory(plan, api)
    claimed, calls = False, 0
    try:
        encoded = encode_valid(backend, prompt, records, plan)
        preflight = {"samples": 15, "blocks": 62, "input_token_counts": [len(ids) for ids in encoded],
                     "input_token_range": [min(map(len, encoded)), max(map(len, encoded))], "model_loaded": False}
        if phase == "inspect":
            return {"phase": "inspect", **preflight, "generation_calls": 0}
        output.mkdir(parents=True, exist_ok=False)
        claimed = True
        write_json(output / "execution.json", execution)
        write_json(output / "protocol.json", plan)
        write_json(output / "started.json", {"execution_sha256": execution_sha, "phase": "run"})
        write_json(output / "input_preflight.json", preflight)
        metrics, pairs = {}, {}
        for index, condition in enumerate(plan["conditions"]):
            if index:
                backend = backend_factory(plan, api)
                if encode_valid(backend, prompt, records, plan) != encoded:
                    raise ValueError("condition tokenizer/input IDs differ")
            backend.load_base()
            if condition["adapter_directory"]:
                backend.load_adapter(inside(root, condition["adapter_directory"]))
            folder = output / condition["name"]
            folder.mkdir()
            predictions = []
            with (folder / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for row, ids in zip(records, encoded):
                    if calls >= 30:
                        raise ValueError("model call budget exhausted")
                    calls += 1
                    fatal = None
                    try:
                        prediction = backend.generate(list(ids))
                    except Exception as exc:
                        prediction = {"response": "", "raw_completion": "", "generated_token_ids": [],
                                      "generated_tokens": 0, "stop_reason": "error", "hit_token_limit": False,
                                      "runtime_error": f"{type(exc).__name__}: {exc}", "seconds": None}
                        if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError":
                            fatal = exc
                    prediction = {"sample_id": row["sample_id"], "input_tokens": len(ids), **prediction}
                    predictions.append(prediction)
                    stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    stream.flush()
                    if fatal is not None:
                        raise fatal
                    print(f"{condition['name']} {len(predictions)}/15 tokens={prediction['generated_tokens']}", flush=True)
            metrics[condition["name"]] = evaluate_output(records, predictions, folder)
            write_json(folder / "runtime.json", backend.runtime)
            historical_folder = inside(root, plan["historical_sampled_references"][condition["name"]])
            historical = [parse_json(line) for line in (historical_folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
            pairs[condition["name"]] = paired_changes(records, historical, predictions)
            backend.close()
        result = {"diagnostic_only": True, "model_generation_calls": calls, "training_steps": 0, "test_calls": 0,
                  "gates": quality_gates(metrics["adapter_step_000045"], metrics["base"], plan["scoring"]["minimum_gates"]),
                  "automatic_test_allowed": False, "original_selected_step": None}
        write_json(output / "paired_changes.json", pairs)
        write_json(output / "result.json", result)
        write_rows(output / "conditions.jsonl", [{"condition": name, "metrics": value} for name, value in metrics.items()])
        if calls != 30 or sha256_file(execution_path) != execution_sha:
            raise ValueError("call count or execution integrity differs")
        load_execution(execution_path, root)
        seal(output, execution_sha, {"status": "completed", "phase": "greedy-valid", "model_generation_calls": calls})
        return result
    except BaseException as exc:
        if claimed:
            write_json(output / "failure.json", {"error": f"{type(exc).__name__}: {exc}", "model_generation_calls": calls})
        raise
    finally:
        backend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=PROJECT / "configs/m5_greedy_diagnostic_v1/runtime.json")
    parser.add_argument("--phase", choices=("inspect", "run"), default="inspect")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.execution, args.phase), ensure_ascii=False))


if __name__ == "__main__":
    main()

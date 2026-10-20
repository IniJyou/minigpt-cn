"""M5本地体验：默认检查，--run对用户给定1..8块执行一次真实推理。"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from minigpt_cn.m5_memory_observe import PROJECT, inside, read, write
from minigpt_cn.m5_structure import parse_json, validate_input

CONFIG = "configs/m5_demo_v1.json"


def make_input(text=None, input_json=None):
    value = parse_json(Path(input_json).read_text(encoding="utf-8")) if input_json else {
        "blocks": [{"block_id": "b0001", "text": text if text is not None else "Definition: A group is a set with an associative operation, identity and inverses."}]}
    validate_input(value)
    return value


def run(value, config_path=PROJECT / CONFIG, execute=False, output=None, root=PROJECT, backend_factory=None):
    validate_input(value)
    config = read(config_path)
    adapter = inside(root, config["adapter_directory"])
    if not adapter.is_dir():
        raise FileNotFoundError("M5 adapter not available; open the closeout report for the recorded outcome")
    from minigpt_cn import m5_lora_run as lora

    if lora.file_tree(adapter) != config["adapter_files"]:
        raise ValueError("demo adapter identity differs")
    if not execute:
        return {"phase": "inspect", "adapter_directory": config["adapter_directory"],
                "quality_status": config["quality_status"], "input": value,
                "generation_calls": 0, "instruction": "Add --run to perform one actual local GPU inference"}
    from minigpt_cn import m5_train_fit_run as fit

    _, definition, plan, api, call_api, _ = fit.load_execution(root / "configs/m5_train_fit_v1/runtime.json", root)
    backend, folder = None, None
    try:
        backend = (backend_factory or fit.FitBackend)(plan, api, call_api)
        prompt = (root / definition["system_prompt"]).read_text(encoding="utf-8")
        ids = backend.encode(lora.messages_for(prompt, value))
        if len(ids) > 4096:
            raise ValueError("input exceeds4096 tokens; no truncation")
        if output is None:
            folder = Path(tempfile.mkdtemp(prefix="m5_manual_demo_", dir=root / "logs"))
        else:
            folder = inside(root / "logs", Path(output).resolve())
            folder.mkdir(parents=True, exist_ok=False)
        write(folder / "input.json", value)
        write(folder / "model.json", config)
        backend.load_base()
        backend.load_adapter(adapter)
        ledger = fit.AttemptLedger(folder / "call_attempts.jsonl", [{"attempt": 1, "kind": "manual_demo"}])
        def before():
            ledger.record({"attempt": 1, "kind": "manual_demo"})
        prediction = backend.generate(ids, before)
        if len(ledger.entries()) != 1:
            raise ValueError("demo must perform exactly one call")
        write(folder / "prediction.json", prediction)
        return {"phase": "manual_demo", "quality_status": config["quality_status"],
                "response": prediction["response"], "output_directory": str(folder), "generation_calls": 1}
    except BaseException as exc:
        if folder:
            write(folder / "failure.json", {"error": f"{type(exc).__name__}: {exc}", "retry": False})
        raise
    finally:
        if backend:
            primary = sys.exc_info()[1]
            try:
                backend.close()
            except BaseException:
                if primary is None:
                    raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", help="将文本原样作为一个块；不自动切段")
    group.add_argument("--input-json", type=Path, help='{"blocks":[{"block_id":"b0001","text":"..."}]}')
    parser.add_argument("--run", action="store_true", help="显式执行一次GPU推理")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    result = run(make_input(args.text, args.input_json), execute=args.run, output=args.output_dir)
    if args.run:
        print("实验模型：" + result["quality_status"], file=sys.stderr)
        print(result["response"])
        print("完整记录：" + result["output_directory"], file=sys.stderr)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

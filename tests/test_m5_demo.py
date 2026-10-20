"""手动体验入口的输入约束与默认零推理，不读取真实模型。"""

import json

import pytest

from minigpt_cn import m5_demo as demo


@pytest.mark.parametrize("text", ["Definition: A metric is a function.", "证明：由定义可得。", "<script>user text</script>"])
def test_text_preserved_as_one_block(text):
    assert demo.make_input(text) == {"blocks": [{"block_id": "b0001", "text": text}]}


@pytest.mark.parametrize("text", ["", " ", "a" * 2001])
def test_blank_or_oversize_rejected(text):
    with pytest.raises(ValueError):
        demo.make_input(text)


def test_default_does_not_construct_backend(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = tmp_path / "config.json"
    demo.write(config, {"adapter_directory": "adapter", "adapter_files": {}, "quality_status": "unaccepted"})
    result = demo.run(demo.make_input("Theorem: Test."), config, root=tmp_path,
                      backend_factory=lambda *args: pytest.fail("no model in default inspect"))
    assert result["generation_calls"] == 0 and result["phase"] == "inspect"


@pytest.mark.parametrize("mode", ["normal", "double_callback", "oversize"])
def test_explicit_demo_one_call_raw_output_and_no_retry(tmp_path, monkeypatch, mode):
    from minigpt_cn import m5_train_fit_run as fit
    from minigpt_cn.m5_greedy_run_v2 import DecodeIntegrityError

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    config = tmp_path / "demo.json"
    demo.write(config, {"adapter_directory": "adapter", "adapter_files": {}, "quality_status": "unaccepted"})
    (tmp_path / "prompt.txt").write_text("fixed prompt", encoding="utf-8")
    monkeypatch.setattr(fit, "load_execution", lambda *args: (None, {"system_prompt": "prompt.txt"}, {}, {}, {}, {}))
    state = {"loads": 0, "forwards": 0, "closed": 0}
    response = '{"annotations":[{"block_id":"b0001","kind":"body","title":null,"reason":null}]}'
    class Backend:
        def __init__(self, *args):
            pass
        def encode(self, messages):
            assert messages[0]["content"] == "fixed prompt"
            assert "target" not in messages[-1]["content"]
            return [1] * (4097 if mode == "oversize" else 3)
        def load_base(self):
            state["loads"] += 1
        def load_adapter(self, path):
            assert path == adapter
        def generate(self, ids, before):
            before()
            state["forwards"] += 1
            if mode == "double_callback":
                before()
                state["forwards"] += 1
            return {"response": response}
        def close(self):
            state["closed"] += 1
    output = tmp_path / "logs/manual"
    if mode == "normal":
        result = demo.run(demo.make_input("Definition: Test."), config, True, output, tmp_path, Backend)
        assert result["generation_calls"] == 1 and result["response"] == response
        assert json.loads((output / "prediction.json").read_text())["response"] == response
        with pytest.raises(FileExistsError):
            demo.run(demo.make_input("Definition: Test."), config, True, output, tmp_path, Backend)
    elif mode == "oversize":
        with pytest.raises(ValueError, match="no truncation"):
            demo.run(demo.make_input("Definition: Test."), config, True, output, tmp_path, Backend)
        assert state["loads"] == state["forwards"] == 0 and not output.exists()
    else:
        with pytest.raises(DecodeIntegrityError):
            demo.run(demo.make_input("Definition: Test."), config, True, output, tmp_path, Backend)
        assert state["forwards"] == 1 and (output / "failure.json").exists()
    assert state["closed"] >= 1

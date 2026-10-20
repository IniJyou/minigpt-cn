"""M5收尾的完整训练门槛、日志钩子和单次启动；CPU执行，不调用CUDA。"""

import json
from types import SimpleNamespace

import pytest
import torch

from minigpt_cn import m5_finish_run as finish


@pytest.mark.parametrize("damage", [None, "last_step", "additional_optimizer_steps", "training_microbatches",
                                   "supervised_exposures", "missing_update", "extra_update"])
def test_incomplete_training_cannot_generate(damage):
    result = {"last_step": 105, "additional_optimizer_steps": 60, "training_microbatches": 240,
              "supervised_exposures": 23010}
    updates = [{"absolute_step": step} for step in range(46, 106)]
    if damage == "missing_update":
        updates.pop()
    elif damage == "extra_update":
        updates.append({"absolute_step": 106})
    elif damage:
        result[damage] -= 1
    if damage:
        with pytest.raises(ValueError):
            finish.validate_training(result, updates)
    else:
        finish.validate_training(result, updates)


def test_local_optimizer_hook_records_without_changing_cpu_update(tmp_path, monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    class Base:
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(total_memory=3 * 1024**3)))
        def load_continuation(self):
            return optimizer, object(), {}
        def guarded_memory(self):
            return {"peak_allocated_bytes": 10, "peak_reserved_bytes": 20, "device_total_bytes": 3 * 1024**3}
    monkeypatch.setattr(finish.allocator, "allocator_evidence", lambda *args: {"verified": True})
    backend = finish.backend_class(Base, tmp_path)()
    restored, _, _ = backend.load_continuation()
    parameter.square().sum().backward()
    rng = torch.get_rng_state().clone()
    restored.step()
    assert backend.finish_updates == 1 and not torch.equal(parameter.detach(), torch.tensor([1.0]))
    assert torch.equal(rng, torch.get_rng_state())
    assert json.loads((tmp_path / "optimizer_returns.jsonl").read_text()) == {"absolute_step": 46}
    assert backend.guarded_memory()["peak_reserved_bytes"] == 20
    assert json.loads((tmp_path / "memory_samples.jsonl").read_text())["optimizer_updates_returned"] == 1


@pytest.mark.parametrize("phase", ["retry", "resume", "test", "train"])
def test_no_other_phase(tmp_path, phase):
    with pytest.raises(ValueError):
        finish.run(tmp_path / "absent", phase, tmp_path)


def test_single_launch_and_child_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "inspect_runtime", lambda *args: {"runtime_sha256": "fixed"})
    for key in finish.allocator.ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"] == finish.allocator.CANDIDATE
        return SimpleNamespace(returncode=1)
    assert finish.run(tmp_path / "runtime", "finish", tmp_path, execute)["child_returncode"] == 1
    with pytest.raises(FileExistsError):
        finish.run(tmp_path / "runtime", "finish", tmp_path, execute)
    assert len(calls) == 1

"""调用边界修正版的CPU测试，无真实权重或CUDA前向。"""

import copy
from types import SimpleNamespace

import pytest

from minigpt_cn import m5_greedy_run_v2 as fix


class Configuration(SimpleNamespace):
    def get_generation_mode(self):
        return SimpleNamespace(value="sample" if self.do_sample else "greedy_search")


def test_effective_config_passes_explicit_overrides_and_rejects_override_failure():
    expected = {"do_sample": False, "num_beams": 1}
    configuration = Configuration(**expected)
    seen = []

    def prepare(value, **kwargs):
        seen.append(kwargs)
        return copy.deepcopy(value), {}

    prepared = fix.effective_config(SimpleNamespace(_prepare_generation_config=prepare), configuration, expected)
    assert prepared.do_sample is False and seen == [{"use_model_defaults": False, "do_sample": False}]
    with pytest.raises(fix.DecodeIntegrityError):
        fix.effective_config(SimpleNamespace(_prepare_generation_config=lambda *a, **k: (Configuration(do_sample=True), {})),
                             configuration, expected)
    assert not issubclass(fix.DecodeIntegrityError, Exception)


def test_durable_attempt_log_caps_30_and_locks_condition_order(tmp_path):
    for i in range(30):
        fix.record_attempt(tmp_path, "base" if i < 15 else "adapter_step_000045")
    assert len((tmp_path / "call_attempts.jsonl").read_text().splitlines()) == 30
    with pytest.raises(fix.DecodeIntegrityError):
        fix.record_attempt(tmp_path, "adapter_step_000045")
    with pytest.raises(fix.DecodeIntegrityError):
        fix.record_attempt(tmp_path, "base")


def test_adapter_cannot_generate_before_base_condition(tmp_path):
    with pytest.raises(fix.DecodeIntegrityError):
        fix.record_attempt(tmp_path, "adapter_step_000045")
    assert not (tmp_path / "call_attempts.jsonl").exists()


@pytest.mark.parametrize("phase", ["train", "test", "train-valid"])
def test_no_other_entrypoints_before_file_read(tmp_path, phase):
    with pytest.raises(ValueError, match="no train/test"):
        fix.run(tmp_path / "no.json", phase, tmp_path)


def test_actual_transformers_default_merge_regression_when_available():
    transformers = pytest.importorskip("transformers")
    from transformers.generation.utils import GenerationMixin

    model = SimpleNamespace(generation_config=transformers.GenerationConfig(do_sample=True))
    model._prepare_generation_config = lambda config, **kwargs: GenerationMixin._prepare_generation_config(model, config, **kwargs)
    desired = transformers.GenerationConfig(do_sample=False, temperature=None, top_p=None, top_k=None, min_p=None)
    wrong, _ = model._prepare_generation_config(desired)
    assert wrong.do_sample is True
    effective = fix.effective_config(model, desired, {"do_sample": False, "num_beams": 1})
    assert effective.get_generation_mode().value == "greedy_search"

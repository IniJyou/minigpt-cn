"""公开源码可运行CPU测试；依赖未公开冻结产物的历史集成项单独注明。"""

from pathlib import Path

import pytest

# 只约束确实读取这些文件的fixture；普通合成测试不依赖本地产物。
M5_LOCAL_ARTIFACTS = {
    ("test_m5_allocator_run.py", "cpu_worker"): (
        "logs/m5_memory_observe_runner_20261019_v1/default_inspect_windows.json",
    ),
    ("test_m5_memory_observe.py", "saved45"): (
        "data/processed/m5_lora_revision_candidate_v1/token_checks.json",
        "logs/m5_train_fit_definition_20261019_v1/planned_schedule.json",
        "logs/m5_memory_observability_definition_20261019_v1/review.json",
    ),
    ("test_m5_train_fit_run.py", "workflow"): (
        "logs/m5_train_fit_definition_20261019_v1/evaluation_manifest.json",
        "logs/m5_train_replay_definition_20261019_v1/selected_samples.json",
        "data/processed/m5_lora_revision_candidate_v1/augmented.train.candidate.jsonl",
    ),
    ("test_m5_train_replay_run.py", "material"): (
        "logs/m5_train_replay_definition_20261019_v1/selected_samples.json",
        "logs/m5_train_replay_definition_20261019_v1/planned_call_order.json",
        "data/processed/m5_lora_revision_candidate_v1/augmented.train.candidate.jsonl",
        "data/processed/m5_lora_revision_candidate_v1/token_checks.json",
    ),
}


def missing_m5_artifacts(filename, fixture_names, root):
    return sorted({relative for (module, fixture), paths in M5_LOCAL_ARTIFACTS.items()
                   if filename == module and fixture in fixture_names
                   for relative in paths if not (Path(root) / relative).is_file()})


def pytest_sessionstart(session):
    # 原有合成测试在项目logs下创建临时目录；新克隆也需有这个空父目录。
    (Path(session.config.rootpath) / "logs").mkdir(exist_ok=True)


def pytest_collection_modifyitems(config, items):
    for item in items:
        missing = missing_m5_artifacts(item.path.name, item.fixturenames, config.rootpath)
        if missing:
            item.add_marker(pytest.mark.skip(reason="requires unpublished local M5 artifacts: " + ", ".join(missing)))

"""缺少产物仅跳过对应集成fixture；文件存在但损坏时仍交原测试校验。"""

from types import SimpleNamespace

import pytest
from conftest import M5_LOCAL_ARTIFACTS, missing_m5_artifacts, pytest_collection_modifyitems


@pytest.mark.parametrize("key", M5_LOCAL_ARTIFACTS)
def test_local_fixture_prerequisites_and_corrupt_files_are_not_silently_skipped(tmp_path, key):
    filename, fixture = key
    paths = M5_LOCAL_ARTIFACTS[key]
    assert set(missing_m5_artifacts(filename, [fixture], tmp_path)) == set(paths)
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("invalid JSON must reach the original integrity checks", encoding="utf-8")
    assert missing_m5_artifacts(filename, [fixture], tmp_path) == []


def test_synthetic_tests_are_never_skipped_for_missing_local_artifacts(tmp_path):
    assert missing_m5_artifacts("test_m5_train_fit_run.py", ["prefix_state"], tmp_path) == []
    assert missing_m5_artifacts("test_m5_demo.py", ["tmp_path"], tmp_path) == []
    assert missing_m5_artifacts("test_m5_finish_run.py", ["tmp_path"], tmp_path) == []


def test_collection_marks_only_the_integration_item(tmp_path):
    marked = []
    items = [SimpleNamespace(path=tmp_path / filename, fixturenames=fixtures,
                             add_marker=lambda mark, filename=filename: marked.append((filename, mark)))
             for filename, fixtures in (("test_m5_allocator_run.py", ["cpu_worker"]),
                                        ("test_m5_allocator_run.py", ["tmp_path"]),
                                        ("test_m5_demo.py", ["tmp_path"]))]
    pytest_collection_modifyitems(SimpleNamespace(rootpath=tmp_path), items)
    assert len(marked) == 1 and marked[0][0] == "test_m5_allocator_run.py"
    assert marked[0][1].name == "skip" and "unpublished local M5 artifacts" in marked[0][1].kwargs["reason"]

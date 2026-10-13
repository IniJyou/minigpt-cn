"""正文级去重与划分测试；只用内存样本，不依赖真实语料或作业仓库。"""

import hashlib
import os
import subprocess
import sys

import pytest

from minigpt_cn.data import choose_split, clean_document, deduplicate_texts


@pytest.mark.parametrize(
    "text, bucket, expected",
    [
        ("边界文档91", 0, "train"),
        ("边界文档150", 97, "train"),
        ("边界文档5", 98, "valid"),
        ("边界文档28", 99, "test"),
    ],
)
def test_fixed_pilot_hash_bucket_boundaries(text, bucket, expected):
    # 固定向量锁定 seed=42、NUL 分隔、UTF-8 与取模规则，不能换成比例切片或别的哈希。
    digest = hashlib.sha256(f"42\0{text}".encode("utf-8")).hexdigest()
    assert int(digest, 16) % 100 == bucket
    assert choose_split(text) == expected


def test_seed_is_part_of_the_hash_key():
    assert choose_split("边界文档5", seed=42) == "valid"
    assert choose_split("边界文档5", seed=43) != "valid"


def test_deduplication_preserves_first_occurrence_and_exact_text():
    texts = ["文档乙", "中 文", "文档乙", "中文", "中 文", "文档甲"]
    assert list(deduplicate_texts(texts)) == ["文档乙", "中 文", "中文", "文档甲"]
    assert list(deduplicate_texts([])) == []


def test_deduplication_consumes_a_generator_lazily_once():
    visited = []

    def documents():
        for text in ["甲", "甲", "乙"]:
            visited.append(text)
            yield text

    unique = deduplicate_texts(documents())
    assert visited == []
    assert next(unique) == "甲"
    assert visited == ["甲"]
    assert list(unique) == ["乙"]
    assert visited == ["甲", "甲", "乙"]


def test_clean_before_deduplication_combines_equivalent_documents():
    raw = ["  臺灣\r\n", "台湾", " \t\n", "cafe\u0301", "café"]
    cleaned = (clean_document(text) for text in raw)
    assert list(deduplicate_texts(text for text in cleaned if text is not None)) == [
        "台湾",
        "café",
    ]


def test_reordering_appending_and_duplicates_do_not_change_split_membership():
    texts = [f"中文文档{i}：😀" for i in range(200)]
    texts += ["边界文档91", "边界文档5", "边界文档28"]
    expected = {text: choose_split(text) for text in texts}
    reordered = list(reversed(texts)) + texts[:10] + ["追加新文档"]
    actual = {text: choose_split(text) for text in deduplicate_texts(reordered)}
    assert {text: actual[text] for text in texts} == expected
    groups = {name: set() for name in ("train", "valid", "test")}
    for text, split in actual.items():
        groups[split].add(text)
    assert all(groups.values())  # 三个固定边界样本保证这份测试输入覆盖所有集合。
    assert groups["train"].isdisjoint(groups["valid"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["valid"].isdisjoint(groups["test"])
    assert set.union(*groups.values()) == set(reordered)


def test_split_is_independent_of_python_hash_seed_across_processes():
    code = (
        "from minigpt_cn.data import choose_split; "
        "print([choose_split(text) for text in ('边界文档91', '边界文档5', '边界文档28')])"
    )
    outputs = []
    for seed in ("0", "123"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONUTF8="1")
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        outputs.append(result.stdout.strip())
    assert outputs == ["['train', 'valid', 'test']"] * 2


@pytest.mark.parametrize("text", [None, 42, b"text", "", " \t\r\n"])
def test_invalid_documents_are_rejected(text):
    error = TypeError if not isinstance(text, str) else ValueError
    with pytest.raises(error, match="text"):
        choose_split(text)
    with pytest.raises(error, match="text"):
        list(deduplicate_texts(["正文", text]))


@pytest.mark.parametrize("texts", ["单个正文", b"text"])
def test_bare_string_is_not_treated_as_a_document_collection(texts):
    with pytest.raises(TypeError, match="texts"):
        list(deduplicate_texts(texts))


@pytest.mark.parametrize("seed", [True, 1.5, "42", None])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(TypeError, match="seed"):
        choose_split("正文", seed=seed)

"""增量优化只改变计算方式：逐 ID 词表与有序 merges 必须等于原全扫描实现。"""

import heapq
import importlib
import random
from collections import Counter

import pytest

from minigpt_cn.tokenizer import BPETokenizer, train_bpe
from minigpt_cn.tokenizer.train_bpe_reference import train_bpe_reference

bpe = importlib.import_module("minigpt_cn.tokenizer.train_bpe")
EOS = "<|endoftext|>"


@pytest.mark.parametrize("seed", range(20))
def test_randomized_merge_prefixes_exactly_match_reference(tmp_path, seed):
    rng = random.Random(seed)
    specials = [[], [EOS], ["<e>", "<e><next>"], ["[.*]"]][seed % 4]
    chunks = ["aaaaaaa", "abababa", "aaaz", "zz", "数学语言", "😀🦉", "\t", "12345"]
    chunks += ["".join(rng.choices("abcd数学😀", k=rng.randrange(1, 30))) for _ in range(20)]
    chunks += specials
    text = "\n".join(rng.choices(chunks, k=60))
    path = tmp_path / "train.txt"
    path.write_text(text, encoding="utf-8")
    base = 256 + len(specials)
    # 不只检查最后一轮：词表预算不同，对比多个合并前缀，包含零合并和候选耗尽。
    for extra in (0, 1, 2, 16, 64, 256):
        expected = train_bpe_reference(path, base + extra, specials)
        actual = train_bpe(path, base + extra, specials)
        assert actual == expected, (seed, extra)
        tokenizer = BPETokenizer(*actual, specials)
        assert tokenizer.decode(tokenizer.encode(text)) == text
    assert path.read_text(encoding="utf-8") == text


def test_counts_and_inverted_index_are_correct_after_every_merge(tmp_path, monkeypatch):
    path = tmp_path / "train.txt"
    path.write_text("aaaaaaa\n" * 3 + "abababa\n" * 5 + f"数学数学😀{EOS}数学数学😀", encoding="utf-8")
    _, freq = bpe._initial_state(path, 600, [EOS])
    weights = list(freq.values())
    original = bpe._pop_best_pair
    checked = []

    def audit(heap, counts, index, words):
        expected_counts, expected_index = Counter(), {}
        for word_id, (ids, weight) in enumerate(zip(words, weights)):
            for pair, count in Counter(zip(ids, ids[1:])).items():
                expected_counts[pair] += weight * count
                expected_index.setdefault(pair, set()).add(word_id)
        assert counts == expected_counts
        assert all(count > 0 for count in counts.values())
        assert index == expected_index
        checked.append(True)
        return original(heap, counts, index, words)

    monkeypatch.setattr(bpe, "_pop_best_pair", audit)
    assert train_bpe(path, 600, [EOS]) == train_bpe_reference(path, 600, [EOS])
    assert len(checked) > 5


def test_merge_is_non_overlapping_even_when_pair_repeats():
    merged, changes = bpe._merge_pair((1, 1, 1, 1, 1), (1, 1), 256)
    assert merged == (256, 256, 1)
    assert changes[(1, 1)] == -4  # 不是只减掉两次替换：四个旧相邻边全部消失。


@pytest.mark.parametrize(
    "ids,pair",
    [
        ((1, 1, 1, 1, 1), (1, 1)),
        ((1, 2, 1, 2, 1, 2), (1, 2)),
        ((3, 1, 2, 4), (1, 2)),
        ((1, 2), (1, 2)),
        ((1, 2, 2, 1, 2), (1, 2)),
        ((1, 2, 3), (4, 5)),
    ],
)
def test_neighbor_changes_equal_full_recount(ids, pair):
    merged, changes = bpe._merge_pair(ids, pair, 256)
    expected = Counter(zip(merged, merged[1:]))
    expected.subtract(Counter(zip(ids, ids[1:])))
    # 不使用 Counter 减法：它会丢掉负数，而负数正是需要删去的旧边。
    assert {p: c for p, c in changes.items() if c} == {p: c for p, c in expected.items() if c}


def test_heap_discards_stale_and_removed_pairs():
    vocab = {0: b"a", 1: b"b", 2: b"z"}
    entries = [bpe._PairCandidate((0, 1), 99, vocab), bpe._PairCandidate((1, 2), 88, vocab)]
    entries += [bpe._PairCandidate((0, 1), 3, vocab)] * 2
    heapq.heapify(entries)
    counts, index, words = {(0, 1): 3}, {(0, 1): {0}}, [(0, 1)]
    assert bpe._pop_best_pair(entries, counts, index, words) == (0, 1)
    counts.clear()
    assert bpe._pop_best_pair(entries, counts, index, words) is None


def test_heap_priority_uses_frequency_then_bytes_not_ids():
    vocab = {0: b"z", 1: b"a", 256: b"aa"}
    counts = {(256, 1): 2, (0, 0): 2, (1, 0): 3}
    heap = [bpe._PairCandidate(pair, count, vocab) for pair, count in counts.items()]
    heapq.heapify(heap)
    assert bpe._pop_best_pair(heap, counts, {}, []) == (1, 0)  # 频次高先出。
    del counts[(1, 0)]
    assert bpe._pop_best_pair(heap, counts, {}, []) == (0, 0)  # 同频时 z 大于 aa。


@pytest.mark.parametrize("words", [[(2, 3), (0, 1)], [(2, 3, 0, 1)]])
def test_identical_byte_keys_preserve_first_occurrence(words):
    vocab = {0: b"a", 1: b"b", 2: b"a", 3: b"b"}
    counts = {(0, 1): 1, (2, 3): 1}
    index = {(0, 1): {len(words) - 1}, (2, 3): {0}}
    heap = [bpe._PairCandidate(pair, count, vocab) for pair, count in counts.items()]
    heapq.heapify(heap)
    assert bpe._pop_best_pair(heap, counts, index, words) == (2, 3)
    del counts[(2, 3)]
    assert bpe._pop_best_pair(heap, counts, index, words) == (0, 1)


def test_heap_compaction_releases_old_entries():
    vocab = {0: b"a", 1: b"b"}
    heap = [bpe._PairCandidate((0, 1), count, vocab) for count in range(1, 1100)]
    counts = {(0, 1): 2}
    compacted = bpe._compact_heap(heap, counts, vocab)
    assert len(compacted) == 1 and compacted[0].key == (2, b"a", b"b")


def test_forced_compaction_does_not_change_training(tmp_path, monkeypatch):
    monkeypatch.setattr(bpe, "_HEAP_MIN_COMPACT_SIZE", 0)
    monkeypatch.setattr(bpe, "_HEAP_COMPACT_FACTOR", 1)
    path = tmp_path / "train.txt"
    path.write_text("ababaaa\naaaaz\nzz\n数学是一门语言😀\n" * 5, encoding="utf-8")
    assert train_bpe(path, 500, [EOS]) == train_bpe_reference(path, 500, [EOS])

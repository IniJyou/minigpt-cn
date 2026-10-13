"""增量 byte-level BPE：只更新受影响词块，用优先队列选择最佳 pair。

公开接口与原实现相同，预分词、特殊 token、字节平局规则均不变。
仍整读 UTF-8 文本；倒排索引以额外内存换取速度，不宣称支持无限大语料。
原全扫描算法保留在 train_bpe_reference.py，仅用于小规模一致性对照。
"""

import heapq
from collections import Counter
from pathlib import Path

import regex

from minigpt_cn.tokenizer.bpe import PAT

_HEAP_COMPACT_FACTOR = 4
_HEAP_MIN_COMPACT_SIZE = 1024


class _PairCandidate:
    """Python 的 heapq 是最小堆，反向比较得到最大频次、最大字节序。"""

    __slots__ = ("pair", "key")

    def __init__(self, pair, count, vocab):
        self.pair = pair
        # 不能拿 token ID 比较平局：新 ID 较大，其 bytes 却未必较大。
        self.key = (count, vocab[pair[0]], vocab[pair[1]])

    def __lt__(self, other):
        return self.key > other.key


def _initial_state(input_path, vocab_size, special_tokens):
    # 与参考版相同的配置检查；bool 是 int 子类，必须显式排除。
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size 必须为整数")
    if not isinstance(special_tokens, list) or not all(isinstance(t, str) for t in special_tokens):
        raise TypeError("special_tokens 必须为字符串列表")
    if any(not t for t in special_tokens) or len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens 中的字符串必须非空且不能重复")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size 至少为256加特殊token数量")

    text = Path(input_path).read_text(encoding="utf-8")
    vocab = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    # 特殊 token 必须先隔离，既不参与统计，也不能跨它合并；最长匹配优先。
    if special_tokens:
        pattern = "|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        parts = regex.split(f"({pattern})", text)
    else:
        parts = [text]
    specials = set(special_tokens)
    freq = Counter()
    for part in parts:
        if part not in specials:
            for match in regex.finditer(PAT, part):
                freq[tuple(match.group().encode("utf-8"))] += 1
    # 返回后不再保留全文/parts；仍有整读阶段的峰值内存，尚不是流式预分词。
    return vocab, freq


def _merge_pair(ids, pair, new_id):
    # 词块是变长一维 ID 元组，不是张量。aaaa 的相邻 aa 计数为 3，替换只有 2 次。
    # 从左到右跳过两个 ID，保证同一 token 不被重复用于两次合并。
    left, right = pair
    merged, i = [], 0
    changes = Counter()
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == left and ids[i + 1] == right:
            # 当前序列 = 已处理的 merged 前缀 + 尚未处理的 ids[i:]。
            # x,a,b,y -> x,new,y：删 (x,a)/(a,b)/(b,y)，加 (x,new)/(new,y)。
            # 只改最多三个旧边、两个新边；开头/结尾没有邻居时不凭空加边。
            changes[pair] -= 1
            if merged:
                previous = merged[-1]
                changes[(previous, left)] -= 1
                changes[(previous, new_id)] += 1
            if i + 2 < len(ids):
                following = ids[i + 2]
                changes[(right, following)] -= 1
                changes[(new_id, following)] += 1
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    # Counter 可包含净变化为零的边，调用者跳过即可。返回的变化未乘词块权重。
    return tuple(merged), changes


def _pop_best_pair(heap, pair_counts, pair_words, words):
    while heap:
        candidate = heapq.heappop(heap)
        # 懒删除：堆里允许留着旧频次；当前 Counter 才是事实来源。
        # 同一 pair 的旧节点若恰好恢复到相同频次，也有相同优先级，可以使用。
        if pair_counts.get(candidate.pair, 0) != candidate.key[0]:
            continue
        ties = {candidate.pair: candidate}
        while heap and heap[0].key == candidate.key:
            other = heapq.heappop(heap)
            if pair_counts.get(other.pair, 0) == other.key[0]:
                ties[other.pair] = other
        best = candidate.pair
        if len(ties) > 1:
            # 防守相同 bytes 对应不同 ID 的完全平局：参考版 max 保留首次出现的 pair。
            # 固定 word_id 沿初始词块顺序；若词块后来相同，分开按原权重统计仍等价。
            def first_occurrence(pair):
                word_id = min(pair_words[pair])
                ids = words[word_id]
                position = next(i for i in range(len(ids) - 1) if ids[i : i + 2] == pair)
                return word_id, position

            best = min(ties, key=first_occurrence)
        for pair, entry in ties.items():
            if pair != best:
                heapq.heappush(heap, entry)
        return best
    return None


def _compact_heap(heap, pair_counts, vocab):
    # 懒删除不能让历史节点无限积累。超过当前候选数的常数倍就重建，释放旧节点。
    limit = max(_HEAP_MIN_COMPACT_SIZE, _HEAP_COMPACT_FACTOR * len(pair_counts))
    if len(heap) > limit:
        heap = [_PairCandidate(pair, count, vocab) for pair, count in pair_counts.items()]
        heapq.heapify(heap)
    return heap


def train_bpe(
    input_path: str | Path, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """返回 (vocab, 有序 bytes merges)，不修改输入、不写文件。

    vocab_size 包含 256 个初始字节和特殊 token；候选耗尽时提前结束，不填充。
    调用者仍负责只传 train 文本。此函数不改变 tokenizer 的文件格式或编码接口。
    """
    vocab, freq = _initial_state(input_path, vocab_size, special_tokens)
    if len(vocab) == vocab_size:
        return vocab, []

    # 固定词块编号和权重：不同词块合并后即使相同，也可分别计数，不必重建全局 freq。
    words, weights = list(freq), list(freq.values())
    del freq
    pair_counts = Counter()
    pair_words = {}  # 倒排索引：pair -> 含它的 word_id 集合；每个词块只登记一次。
    word_pair_counts = []  # 缓存每个词块的局部计数，之后只改邻边，不再全文重数它。
    for word_id, ids in enumerate(words):
        local_counts = Counter(zip(ids, ids[1:]))
        word_pair_counts.append(local_counts)
        for pair, occurrences in local_counts.items():
            # C(pair) = Σ词块 权重 × 词块内相邻出现次数，统计仍允许重叠。
            pair_counts[pair] += weights[word_id] * occurrences
            pair_words.setdefault(pair, set()).add(word_id)
    heap = [_PairCandidate(pair, count, vocab) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    merges = []
    while len(vocab) < vocab_size:
        best = _pop_best_pair(heap, pair_counts, pair_words, words)
        if best is None:
            break
        new_id = len(vocab)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        merges.append((vocab[best[0]], vocab[best[1]]))

        changes = Counter()
        # 必须复制集合再遍历，因为更新索引会删除/添加成员；其他词块完全不用扫描。
        for word_id in tuple(pair_words[best]):
            ids = words[word_id]
            merged, local_changes = _merge_pair(ids, best, new_id)
            local_counts = word_pair_counts[word_id]
            weight = weights[word_id]
            # ΔC(pair) = Σ受影响词块 权重 × (合并后出现次数 - 合并前出现次数)。
            # 索引只记录“是否存在”：局部计数从正数变零才移除，从零变正数才加入。
            for pair, change in local_changes.items():
                if not change:
                    continue
                old_count = local_counts[pair]
                count = old_count + change
                if count:
                    local_counts[pair] = count
                    if not old_count:
                        pair_words.setdefault(pair, set()).add(word_id)
                else:
                    del local_counts[pair]
                    members = pair_words[pair]
                    members.remove(word_id)
                    if not members:
                        del pair_words[pair]
                changes[pair] += weight * change
            words[word_id] = merged

        # 一轮所有词块更新完才入堆；中间状态不参与下一次选择，净变化为零时不入堆。
        for pair, change in changes.items():
            if not change:
                continue
            count = pair_counts[pair] + change
            if count:
                pair_counts[pair] = count
                heapq.heappush(heap, _PairCandidate(pair, count, vocab))
            else:
                del pair_counts[pair]
        heap = _compact_heap(heap, pair_counts, vocab)
    return vocab, merges

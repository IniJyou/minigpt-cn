"""保留迁移后的逐轮全扫描 BPE，作为优化版的词表/合并顺序参考。

仍整读 UTF-8 文本、每轮重新统计 pair，尚未验证正式16K训练的速度与内存。
"""

from collections import Counter
from pathlib import Path

import regex

from minigpt_cn.tokenizer.bpe import PAT


def _contains_pair(ids, pair):
    # 原作业优化：不含本轮最佳 pair 的词块直接保留，不必重新构造 ID 序列。
    return any((ids[i], ids[i + 1]) == pair for i in range(len(ids) - 1))


def train_bpe_reference(
    input_path: str | Path, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """返回 (vocab, merges)，格式与现有 BPETokenizer 和作业实现一致。

    vocab_size 包含256个初始字节与特殊token，是上限；没有可合并pair时提前停止。
    调用者负责只传训练集文本，不要混入验证集/测试集。此函数不修改输入文件。
    """
    # 项目新增：先检查配置，再读文件；bool是int子类，需要显式排除。
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError("vocab_size 必须为整数")
    if not isinstance(special_tokens, list) or not all(isinstance(t, str) for t in special_tokens):
        raise TypeError("special_tokens 必须为字符串列表")
    if any(not t for t in special_tokens) or len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special_tokens 中的字符串必须非空且不能重复")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size 至少为256加特殊token数量")

    text = Path(input_path).read_text(encoding="utf-8")  # 原作业：读文件全文。
    # 原作业：0..255分别对应一个字节；即使某个字符没见过，也能拆成UTF-8字节编码。
    vocab = {i: bytes([i]) for i in range(256)}
    special_ids = {}
    for token in special_tokens:
        special_ids[token] = len(vocab)
        vocab[len(vocab)] = token.encode("utf-8")  # value是bytes，不是token ID。

    # 原作业：先保护特殊token，再预分词；它们不能参加pair统计或跨边界合并。
    if special_tokens:
        pattern = "|".join(regex.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        parts = regex.split(f"({pattern})", text)
    else:
        parts = [text]

    freq = Counter()
    for part in parts:
        if part in special_ids:
            continue
        # 项目适配：复用编码器PAT，直接累加，省掉原chunks中间列表；预分词规则不变。
        # 这里没有batch/张量：每个词块是变长一维ID元组，长度为该词块的UTF-8字节数。
        for match in regex.finditer(PAT, part):
            ids = tuple(match.group().encode("utf-8"))
            freq[ids] += 1

    merges = []
    while len(vocab) < vocab_size:
        # 原作业：C(a,b)=sum_chunk freq(chunk)*相邻pair在chunk中出现的次数。
        # 统计允许重叠，例如aaaa中(a,a)出现3次；实际替换仍必须非重叠。
        pair_counts = Counter()
        for ids, count in freq.items():
            for i in range(len(ids) - 1):
                pair_counts[(ids[i], ids[i + 1])] += count
        if not pair_counts:
            break  # 语料没有候选时停止，不靠填充token凑满目标词表。

        # 原作业：先最大频次，再最大字节字典序；不能用ID比较平局。
        # 新token的ID虽然更大，但其字节值未必更大。
        best = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))
        new_id = len(vocab)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        # merges按学习顺序保存bytes pair，不能保存best的整数ID。
        merges.append((vocab[best[0]], vocab[best[1]]))

        new_freq = Counter()
        for ids, count in freq.items():
            if not _contains_pair(ids, best):
                new_freq[ids] += count
                continue
            new_ids, i = [], 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == best:
                    new_ids.append(new_id)
                    i += 2  # 原作业关键：跳过两个，避免重复使用已合并的token。
                else:
                    new_ids.append(ids[i])
                    i += 1
            # 不同旧词块若合并成同一序列，频次要累加，不能覆盖。
            new_freq[tuple(new_ids)] += count
        freq = new_freq
    return vocab, merges

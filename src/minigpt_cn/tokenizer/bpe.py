"""迁移已训练的 byte-level BPE 编解码及 pilot JSON 格式，不重新训练词表。"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self.byte_to_id = {token: index for index, token in self.vocab.items()}
        self.merge_to_rank = {
            (self.byte_to_id[a], self.byte_to_id[b]): (rank, self.byte_to_id[a + b])
            for rank, (a, b) in enumerate(self.merges)
        }
        special_alt = "|".join(
            regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True)
        )
        self._split_pat = f"({special_alt})" if special_alt else None
        self._combined_pat = f"{special_alt}|{PAT}" if special_alt else PAT
        self._max_special_len = max(map(len, self.special_tokens), default=0)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        result = []
        # 先保护特殊 token，再按原正则预分词；不能跨预分词边界合并。
        parts = regex.split(self._split_pat, text) if self._split_pat else [text]
        for part in parts:
            if part in self.special_tokens:
                result.append(self.byte_to_id[part.encode("utf-8")])
                continue
            for chunk in regex.findall(PAT, part):
                ids = [self.byte_to_id[bytes([byte])] for byte in chunk.encode("utf-8")]
                result.extend(self._apply_merges(ids))
        return result

    def _apply_merges(self, ids):
        while len(ids) >= 2:
            best_rank, best_pair, best_id = None, None, None
            for pair in zip(ids, ids[1:]):
                if pair in self.merge_to_rank:
                    rank, merged_id = self.merge_to_rank[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank, best_pair, best_id = rank, pair, merged_id
            if best_pair is None:
                break
            merged, index = [], 0
            while index < len(ids):
                if index + 1 < len(ids) and (ids[index], ids[index + 1]) == best_pair:
                    merged.append(best_id)
                    index += 2  # 非重叠替换，不能重复使用同一个字节。
                else:
                    merged.append(ids[index])
                    index += 1
            ids = merged
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        # 必须先拼接字节再解码；单个 token 可能只包含半个 UTF-8 字符。
        # 保留 pilot 的 errors='replace'：非法组合仍显示替换字符，不声称已修复。
        return b"".join(self.vocab[index] for index in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, chunks: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for chunk in chunks:
            buffer += chunk
            cut = self._safe_cut(buffer)
            if cut:
                yield from self.encode(buffer[:cut])
                buffer = buffer[cut:]
        if buffer:
            yield from self.encode(buffer)

    def _safe_cut(self, buffer):
        # 与原实现一致：保留尚未确定的末尾预分词和可能跨块的特殊 token。
        limit = len(buffer)
        for length in range(1, min(self._max_special_len, limit) + 1):
            tail = buffer[-length:]
            if any(
                len(tail) < len(token) and token.startswith(tail) for token in self.special_tokens
            ):
                limit -= length
                break
        special_spans = (
            []
            if not self._split_pat
            else [
                match.span()
                for match in regex.finditer(self._split_pat, buffer[:limit])
                if match.group() in self.special_tokens
            ]
        )
        previous_end, last_end = None, None
        for match in regex.finditer(self._combined_pat, buffer[:limit]):
            if any(
                match.start() < end and match.end() > start and match.span() != (start, end)
                for start, end in special_spans
            ):
                continue
            previous_end, last_end = last_end, match.end()
        cut = previous_end or 0
        if cut and buffer[cut - 1].isspace():
            if cut == len(buffer):
                return 0
            if not buffer[cut].isspace() and not any(
                buffer[cut:].startswith(token) for token in self.special_tokens
            ):
                return 0
        return cut

    def save(self, path: str | Path):
        # bytes 以 hex 保存，避免 UTF-8 解码损坏词表；兼容原 tokenizer_2048.json。
        payload = {
            "special_tokens": self.special_tokens,
            "vocab": {str(index): token.hex() for index, token in self.vocab.items()},
            "merges": [[a.hex(), b.hex()] for a, b in self.merges],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = {int(index): bytes.fromhex(token) for index, token in payload["vocab"].items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in payload["merges"]]
        return cls(vocab, merges, payload["special_tokens"])

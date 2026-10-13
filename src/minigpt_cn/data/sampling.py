"""BPE 的确定性完整文档采样；只保留预算附近的候选，不缓存全量正文。"""

import hashlib
import heapq
from dataclasses import dataclass

HASH_RULE = "SHA256(UTF8('bpe-sample-v1' + NUL + seed + NUL + text)); tie=UTF8(text)"


def _sampling_key(text: str, seed: int) -> tuple[bytes, bytes]:
    # 与 train/valid/test 划分分开使用哈希域；NUL 是真正的 \0 字符，不是两个字符 \\0。
    body = text.encode("utf-8")
    digest = hashlib.sha256(f"bpe-sample-v1\0{seed}\0".encode("utf-8") + body).digest()
    # 哈希偶然相同时用正文字节打破平局；标题、行号和输入顺序不参与排序。
    return digest, body


@dataclass
class _Candidate:
    key: tuple[bytes, bytes]
    characters: int
    source: str | None
    bytes: int

    def __lt__(self, other: "_Candidate") -> bool:
        # heapq 原本是最小堆；反转比较后，根节点是哈希顺序最靠后的候选。
        return self.key > other.key


class _HashPrefixSampler:
    """保存哈希排序后，累计字节首次达到预算的最短前缀。

    最后一篇可能越界，导出时不写它；预算恰好填满则保留恰好填满的前缀。
    保留字节数 < 预算 + 末篇字节数（总量不足预算时保留全部）。还有堆对象开销，
    不是硬 RSS 上限；单篇巨型文档仍须能解析。参数由 export_train_text 统一校验。
    """

    def __init__(self, max_bytes: int, seed: int, boundary_bytes: int):
        self.max_bytes = max_bytes
        self.seed = seed
        self.boundary_bytes = boundary_bytes
        self._heap: list[_Candidate] = []
        self._bytes = 0

    def add(self, text: str, source: str | None) -> None:
        key = _sampling_key(text, self.seed)
        candidate = _Candidate(key, len(text), source, len(key[1]) + self.boundary_bytes)
        heapq.heappush(self._heap, candidate)
        self._bytes += candidate.bytes
        # 只要去掉顺序最靠后的整篇后仍覆盖预算，它就不可能进入最终输出。
        # 后续新增更小哈希只会让预算更早用完，因此已删候选永远不必找回来。
        while self._bytes - self._heap[0].bytes >= self.max_bytes:
            self._bytes -= heapq.heappop(self._heap).bytes

    def ordered(self) -> list[_Candidate]:
        # 候选集合不是源文件前缀；最终写出严格遵循正常（从小到大）的哈希顺序。
        return sorted(self._heap, key=lambda candidate: candidate.key)

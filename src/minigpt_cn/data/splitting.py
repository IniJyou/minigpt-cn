"""清洗后正文的精确去重与固定 hash 划分；不读写文件。"""

import hashlib
from collections.abc import Iterable, Iterator
from typing import Literal


def _validate_text(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text 必须是清洗后的正文字符串")
    if not text.strip():
        raise ValueError("text 不能为空或全是空白，请先调用 clean_document 过滤")


def deduplicate_texts(texts: Iterable[str]) -> Iterator[str]:
    """按完整正文精确去重，按输入顺序逐篇返回首次出现的正文。

    调用前应清洗并过滤 None；这里不隐式清洗，也不比较标题、来源或 token。
    输入只遍历一次，但 seen 保存全部唯一正文，不是固定内存的全量语料方案。
    """
    if isinstance(texts, (str, bytes)):
        raise TypeError("texts 必须是文档集合，不能直接传入单个字符串或 bytes")
    seen: set[str] = set()
    for text in texts:
        _validate_text(text)
        # 直接比较正文，不把“哈希相同”当成“正文相同”；保留原 pilot 的精确语义。
        if text not in seen:
            seen.add(text)
            yield text


def choose_split(text: str, seed: int = 42) -> Literal["train", "valid", "test"]:
    """复用原 pilot 的 98/1/1 规则，同一正文在同一种子下永远属于同一集合。

    比例是 hash 桶的比例，不保证小样本的文档数或 token 数恰好为 98/1/1。
    正文和种子必须在整个实验中固定；追加或重排输入不会重分已有文档。
    """
    _validate_text(text)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed 必须是整数")
    # 与原作业完全相同：十进制种子 + NUL 分隔符 + 正文，整体 UTF-8 编码。
    # SHA-256 固定跨进程结果；不能用受 PYTHONHASHSEED 影响的 Python hash(text)。
    key = f"{seed}\0{text}".encode("utf-8")
    bucket = int(hashlib.sha256(key).hexdigest(), 16) % 100
    # 100 个桶中：0..97 为 train，98 为 valid，99 为 test（valid 与项目读取命名一致）。
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "valid"
    return "test"

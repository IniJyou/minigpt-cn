"""单篇文档清洗；不读写文件，不负责去重、划分或 token 编码。"""

import unicodedata
from functools import lru_cache

from opencc import OpenCC


@lru_cache(maxsize=1)
def _simplifier() -> OpenCC:
    # 多篇文档复用同一字典；只有启用繁转简时才构建转换器。
    return OpenCC("t2s.json")


def normalize_text(text: str, *, to_simplified: bool = True) -> str:
    """保留原 pilot 的规范化顺序；可显式关闭繁转简。"""
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")
    if not isinstance(to_simplified, bool):
        raise TypeError("to_simplified 必须是布尔值")

    # 1. NFC 合并等价的组合字符，例如 e + 重音符 → é。
    # 不用 NFKC：不主动改变全角字符等兼容形式。
    text = unicodedata.normalize("NFC", text)
    # 2. 与原 pilot 相同的繁转简；关闭后保留繁体正文。
    if to_simplified:
        text = _simplifier().convert(text)
    # 3. Windows CRLF、旧式 CR 都统一为 LF，先替换 CRLF 防止出现双换行。
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 4. 仅去掉逐行首尾空白；保留行内空格、tab 和段落间的空行。
    lines = [line.strip() for line in text.split("\n")]
    # 5. 整篇去掉首尾空白，不删除正文标点、英文或 emoji。
    return "\n".join(lines).strip()


def clean_document(
    text: str,
    *,
    min_chars: int = 1,
    max_chars: int | None = None,
    to_simplified: bool = True,
) -> str | None:
    """返回规范化正文；空文档或长度不合要求时返回 None（整篇过滤，不截断）。

    默认只过滤空文档，与原 pilot 一致。长度为规范化后的 len(text)：
    统计 Unicode 码点（包含保留的空格/换行），不是 UTF-8 字节数或 token 数。
    """
    # bool 是 int 的子类，不能把 True 当作长度阈值。
    if not isinstance(min_chars, int) or isinstance(min_chars, bool):
        raise TypeError("min_chars 必须是整数")
    if min_chars < 1:
        raise ValueError("min_chars 必须至少为 1")
    if max_chars is not None:
        if not isinstance(max_chars, int) or isinstance(max_chars, bool):
            raise TypeError("max_chars 必须是整数或 None")
        if max_chars < min_chars:
            raise ValueError("max_chars 不能小于 min_chars")

    cleaned = normalize_text(text, to_simplified=to_simplified)
    # 必须先规范化再测长度；两端空白或组合字符会改变长度。
    if len(cleaned) < min_chars or (max_chars is not None and len(cleaned) > max_chars):
        return None
    return cleaned

"""数据处理子包。

已有清洗、正文精确去重、hash 划分与 token 读取；全量处理和编码管线后续逐步接入。
"""

from .cleaning import clean_document, normalize_text
from .splitting import choose_split, deduplicate_texts

__all__ = ["choose_split", "clean_document", "deduplicate_texts", "normalize_text"]

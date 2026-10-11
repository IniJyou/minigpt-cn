"""数据处理子包。

已有清洗、去重、划分、JSONL/训练文本入口与 token 读取；dump 解析和编码管线后续接入。
"""

from .cleaning import clean_document, normalize_text
from .splitting import choose_split, deduplicate_texts

__all__ = ["choose_split", "clean_document", "deduplicate_texts", "normalize_text"]

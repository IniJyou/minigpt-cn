"""数据处理子包。

已有单篇文本清洗与 token 读取；去重、数据划分和编码管线后续逐步接入。
"""

from .cleaning import clean_document, normalize_text

__all__ = ["clean_document", "normalize_text"]

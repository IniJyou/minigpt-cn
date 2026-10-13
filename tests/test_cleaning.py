"""单篇文档测试；不下载语料，不依赖作业仓库或真实数据文件。"""

import unicodedata

import pytest

from minigpt_cn.data import clean_document, normalize_text


def test_matches_pilot_normalization_order():
    # NFC、繁转简、CRLF/CR、逐行 strip、整体 strip 同时出现。
    raw = " \r\n\t臺灣的數學\t \r\n  cafe\u0301  \r  😀 English  \r\n "
    assert normalize_text(raw) == "台湾的数学\ncafé\n😀 English"


def test_can_preserve_traditional_chinese():
    assert normalize_text("  臺灣的數學\r\n", to_simplified=False) == "臺灣的數學"


def test_nfc_not_nfkc_and_preserves_body_format():
    raw = "  ＡＢＣ  １２３\t英文，😀。 \r\n \t\r\n   下一段  "
    assert normalize_text(raw) == "ＡＢＣ  １２３\t英文，😀。\n\n下一段"


@pytest.mark.parametrize("text", ["", " \t\r\n\u3000", "e\u0301", "繁體數學", "a\n\n b 😀"])
def test_normalization_is_idempotent(text):
    cleaned = normalize_text(text)
    assert unicodedata.is_normalized("NFC", cleaned)
    assert normalize_text(cleaned) == cleaned


@pytest.mark.parametrize("text", ["", " \t\n\r\u3000"])
def test_empty_documents_are_filtered(text):
    assert clean_document(text) is None


def test_default_retains_short_nonempty_document():
    assert clean_document("  中  ") == "中"


def test_inclusive_length_bounds_apply_after_normalization():
    assert clean_document("  e\u0301  ", min_chars=1, max_chars=1) == "é"
    assert clean_document(" 臺灣😀 ", min_chars=3, max_chars=3) == "台湾😀"
    assert clean_document("中", min_chars=2) is None
    assert clean_document("数学百科", max_chars=3) is None  # 不截断成“数学百”。
    assert clean_document("繁體", to_simplified=False) == "繁體"


def test_length_counts_retained_whitespace_not_bytes_or_tokens():
    assert clean_document("中\n😀", min_chars=3, max_chars=3) == "中\n😀"
    assert clean_document("中\n😀", max_chars=2) is None


@pytest.mark.parametrize("text", [None, 42, b"text"])
def test_non_string_input_is_rejected(text):
    with pytest.raises(TypeError, match="text"):
        normalize_text(text)
    with pytest.raises(TypeError, match="text"):
        clean_document(text)


@pytest.mark.parametrize("flag", [1, "true", None])
def test_non_boolean_conversion_flag_is_rejected(flag):
    with pytest.raises(TypeError, match="to_simplified"):
        clean_document("正文", to_simplified=flag)


@pytest.mark.parametrize(
    "kwargs, error, field",
    [
        ({"min_chars": True}, TypeError, "min_chars"),
        ({"min_chars": 1.5}, TypeError, "min_chars"),
        ({"min_chars": 0}, ValueError, "min_chars"),
        ({"min_chars": -1}, ValueError, "min_chars"),
        ({"max_chars": True}, TypeError, "max_chars"),
        ({"max_chars": 2.5}, TypeError, "max_chars"),
        ({"min_chars": 3, "max_chars": 2}, ValueError, "max_chars"),
    ],
)
def test_invalid_length_bounds_are_rejected(kwargs, error, field):
    with pytest.raises(error, match=field):
        clean_document("正文", **kwargs)

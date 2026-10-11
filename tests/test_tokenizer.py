"""微型内存词表测试；不依赖真实 tokenizer 文件。"""

import pytest

from minigpt_cn.tokenizer import BPETokenizer


@pytest.fixture
def tokenizer():
    vocab = {index: bytes([index]) for index in range(256)}
    vocab.update({256: b"<|endoftext|>", 257: b"aa", 258: b"aaa"})
    return BPETokenizer(vocab, [(b"a", b"a"), (b"aa", b"a")], ["<|endoftext|>"])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "中文百科：数学与计算机。",
        "emoji 😀🌏",
        "空格  与\n换行\t",
        "aaa aaaa",
        "中文<|endoftext|>English",
        "<|endoftext|><|endoftext|>",
    ],
)
def test_round_trip_and_streaming(tokenizer, text):
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    for chunk_size in (1, 2, 5):
        chunks = [text[start : start + chunk_size] for start in range(0, len(text), chunk_size)]
        assert list(tokenizer.encode_iterable(chunks)) == ids


def test_merge_order_special_id_and_save_reload(tokenizer, tmp_path):
    assert tokenizer.encode("aaa") == [258]
    assert tokenizer.encode("aaaa") == [257, 257]
    assert tokenizer.encode("<|endoftext|>") == [256]
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.vocab == tokenizer.vocab
    assert restored.merges == tokenizer.merges
    assert restored.special_tokens == tokenizer.special_tokens
    assert restored.vocab_size == 259
    assert restored.encode("aaa中文😀") == tokenizer.encode("aaa中文😀")


def test_decode_joins_utf8_bytes_before_decoding(tokenizer):
    assert tokenizer.decode(list("中".encode("utf-8"))) == "中"
    assert tokenizer.decode([0xFF]) == "\ufffd"  # 保留已知的非法字节替换行为。

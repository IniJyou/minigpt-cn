"""M2第一小步：小语料训练、merge规则及现有编解码接口；不依赖作业仓库。"""

import pytest

from minigpt_cn.tokenizer import BPETokenizer, train_bpe

EOS = "<|endoftext|>"


def corpus(tmp_path, text):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_pair_counts_weight_chunk_frequency_and_do_not_change_input(tmp_path):
    text = "ab\nab\nab\ncd"
    path = corpus(tmp_path, text)
    vocab, merges = train_bpe(path, 257, [])
    assert merges == [(b"a", b"b")]  # ab出现3次，cd出现1次，而非各算一个独特词块。
    assert vocab[256] == b"ab" and path.read_text(encoding="utf-8") == text


def test_tie_uses_bytes_not_new_token_ids_and_is_deterministic(tmp_path):
    path = corpus(tmp_path, "aaaz\nzz")
    vocab, merges = train_bpe(path, 259, [])
    # 第二轮aa的ID=256虽然大于z的ID=122，但字节b'z'大于b'aa'，应先合并zz。
    assert merges == [(b"a", b"a"), (b"z", b"z"), (b"aa", b"a")]
    assert [vocab[i] for i in (256, 257, 258)] == [b"aa", b"zz", b"aaa"]
    assert train_bpe(path, 259, []) == (vocab, merges)
    reversed_path = tmp_path / "reversed.txt"
    reversed_path.write_text("zz\naaaz", encoding="utf-8")
    assert train_bpe(reversed_path, 259, []) == (vocab, merges)


def test_pair_replacement_is_non_overlapping(tmp_path):
    vocab, merges = train_bpe(corpus(tmp_path, "aaaaa"), 258, [])
    assert merges == [(b"a", b"a"), (b"aa", b"aa")]
    tokenizer = BPETokenizer(vocab, merges)
    assert tokenizer.encode("aaaaa") == [257, ord("a")]


def test_merge_order_preserves_word_frequency(tmp_path):
    vocab, merges = train_bpe(corpus(tmp_path, "abab\nabab\nab"), 258, [])
    assert merges == [(b"a", b"b"), (b"ab", b"ab")]
    assert vocab[257] == b"abab"


def test_never_merges_across_pretokenization_boundaries(tmp_path):
    vocab, merges = train_bpe(corpus(tmp_path, "a1b2"), 300, [])
    assert len(vocab) == 256 and merges == []  # 字母/数字被PAT分为四个单字节词块。


def test_special_token_is_atomic_and_blocks_cross_boundary_merges(tmp_path):
    text = f"aa{EOS}bb"
    vocab, merges = train_bpe(corpus(tmp_path, text), 300, [EOS])
    assert vocab[256] == EOS.encode() and len(vocab) == 259
    assert merges == [(b"b", b"b"), (b"a", b"a")]
    assert BPETokenizer(vocab, merges, [EOS]).encode(text) == [258, 256, 257]


def test_overlapping_special_tokens_match_longest_but_keep_declared_id_order(tmp_path):
    specials = ["<eos>", "<eos><next>"]
    text = "aa<eos><next>bb<eos>cc"
    vocab, merges = train_bpe(corpus(tmp_path, text), 300, specials)
    assert [vocab[256], vocab[257]] == [t.encode() for t in specials]
    assert set(vocab[i] for i in range(258, len(vocab))) == {b"aa", b"bb", b"cc"}
    tokenizer = BPETokenizer(vocab, merges, specials)
    assert (
        tokenizer.encode(specials[1]) == [257] and tokenizer.decode(tokenizer.encode(text)) == text
    )


@pytest.mark.parametrize("text", ["", EOS + EOS])
def test_no_candidates_returns_actual_vocabulary_without_padding(tmp_path, text):
    vocab, merges = train_bpe(corpus(tmp_path, text), 300, [EOS])
    assert len(vocab) == 257 and merges == []
    assert set(vocab) == set(range(257))


def test_budget_includes_initial_bytes_and_special_tokens(tmp_path):
    vocab, merges = train_bpe(corpus(tmp_path, "aaaa"), 257, [EOS])
    assert len(vocab) == 257 and merges == []
    assert all(vocab[i] == bytes([i]) for i in range(256))


def test_trained_tokenizer_chinese_emoji_round_trip_and_save_reload(tmp_path):
    text = (f"数学是一门语言。😀\n数学与计算机。{EOS}\n" * 3) + "English 123"
    vocab, merges = train_bpe(corpus(tmp_path, text), 300, [EOS])
    assert set(vocab) == set(range(len(vocab)))
    assert 257 < len(vocab) <= 300 and len(merges) == len(vocab) - 257
    assert all(vocab[i] == bytes([i]) for i in range(256))
    tokenizer = BPETokenizer(vocab, merges, [EOS])
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert (restored.vocab, restored.merges, restored.special_tokens) == (vocab, merges, [EOS])
    for sample in (text, "没在训练文本中的字：🦉\t\n", EOS):
        ids = tokenizer.encode(sample)
        assert all(0 <= i < len(vocab) for i in ids)
        assert restored.encode(sample) == ids and restored.decode(ids) == sample
    assert restored.encode(EOS) == [256]


@pytest.mark.parametrize(
    "size,specials,error",
    [
        (True, [], TypeError),
        (257.5, [], TypeError),
        ("300", [], TypeError),
        (255, [], ValueError),
        (256, [EOS], ValueError),
        (300, EOS, TypeError),
        (300, None, TypeError),
        (300, [1], TypeError),
        (300, [""], ValueError),
        (300, [EOS, EOS], ValueError),
    ],
)
def test_invalid_training_configuration_has_clear_exception(tmp_path, size, specials, error):
    with pytest.raises(error, match="vocab_size|special_tokens"):
        train_bpe(corpus(tmp_path, "aaaa"), size, specials)


def test_missing_input_file_has_clear_exception(tmp_path):
    with pytest.raises(FileNotFoundError):
        train_bpe(tmp_path / "missing.txt", 300, [])

"""从原始维基文本修复显示字段；缺公式/未知复杂转换不得冒充完整正文。"""

import json

import pytest
from test_parse_xml import document, make_download, page
from test_wikitext import clean, read_records

from minigpt_cn.data import extract as cli
from minigpt_cn.data import wikitext as markup


@pytest.mark.parametrize(
    "raw,visible",
    [
        ("{{langx|el|μαθηματικά}}", "μαθηματικά"),
        ("{{langx|code=ja|text=数学}}", "数学"),
        (
            "{{langx|el|μαθηματικά|translit=mathēmatiká|lit=学习}}",
            "μαθηματικά（mathēmatiká）（学习）",
        ),
        ("{{tsl|en|Japanese Aero Engine Corporation|日本航空發動機公司}}", "日本航空發動機公司"),
        ("{{tsl|en|RJ500|引擎|RJ500引擎}}", "RJ500引擎"),
        ("{{tsl|en|RJ500||发动机}}", "发动机"),
        ("{{Translink|en|John Gilmore}}", "John Gilmore"),
        ("{{link-ja|日本航空發動機公司|日本航空機エンジン協会}}", "日本航空發動機公司"),
        ("{{link-en|公司|Company|公司显示名称}}", "公司显示名称"),
        ("{{le|公司|Company|d=公司简称}}", "公司简称"),
        ("{{link-ru|名字||{{nowrap|显示名字}}}}", "显示名字"),
        ("{{Zy|酢|cù|ㄘㄨˋ}}", "酢"),
        ("{{Zy|亳|bó|ㄅㄛˋ|bok3}}", "亳"),
        ("-{吒}-", "吒"),
        ("-{R|吒}-", "吒"),
        ("-{zh-cn:软件;zh-tw:軟體;}-", "软件"),
        ("-{zh-tw:軟體;zh-hans:软件;}-", "软件"),
        ("[[名-{吒}-|顯-{示}-]]", "顯示"),
        ("{{tsl|en|Oxalis|-{酢}-浆草}}", "酢浆草"),
    ],
    ids=[f"display-{i}" for i in range(19)],
)
def test_known_display_fields_restore_content(raw, visible):
    result = clean("中文正文：" + raw + "。")
    assert result["text"] == "中文正文：" + visible + "。"
    assert not result["reject_reasons"] and not result["issues"]
    assert not any(s in result["text"] for s in ("{{", "[[", "-{", "ㄘㄨˋ"))


@pytest.mark.parametrize("raw", ["{{tsl|en}}", "{{langx|en|}}", "{{link-ja}}", "{{Zy}}"])
def test_missing_known_display_argument_rejected(raw):
    result = clean("中文正文，" + raw + "负责工作。")
    assert "missing_display_content" in result["reject_reasons"]


@pytest.mark.parametrize(
    "raw",
    [
        "-{H|zh-cn:软件;zh-tw:軟體;}-",
        "-{A|zh-cn:软件;zh-tw:軟體;}-",
        "-{zh-cn:软件;zh-cn:軟體;}-",
        "-{zh-unknown:软件;zh-tw:軟體;}-",
        "-{外-{内}-文}-",
        "-{R|zh-cn:软件;zh-tw:軟體;}-",
        "-{缺少结束",
    ],
    ids=[f"conversion-{i}" for i in range(7)],
)
def test_unsupported_conversion_not_silently_removed(raw):
    assert "unsupported_language_conversion" in clean("中文正文。" + raw)["reject_reasons"]


def test_visible_math_rejected_but_math_in_ignored_content_does_not_poison_body():
    r = clean("在<math>r</math>处密度正比于<math>r^2</math>。")
    assert "unsupported_visible_math" in r["reject_reasons"]
    assert r["metrics"]["unsupported_math_tags"] == 2
    r = clean("真实中文正文。<ref><math>x</math></ref>{{infobox|公式=<math>x</math>}}")
    assert r["text"] == "真实中文正文。" and not r["reject_reasons"]


def test_empty_sections_removed_and_parent_with_child_prose_kept():
    r = clean(
        "序言中文正文。\n==成绩==\n{|\n|数据\n|}\n===2004年===\n"
        "==历史==\n===简介===\n{{navbox}}\n===发展===\n这里有完整中文叙述。\n"
        "==空结尾==\n<ref>只有引用</ref>"
    )
    assert r["text"] == "序言中文正文。\n\n历史\n\n发展\n\n这里有完整中文叙述。"
    assert r["metrics"]["removed_empty_sections"] == 4
    assert r["metrics"]["omitted_tables"] == 1
    assert not r["reject_reasons"]


@pytest.mark.parametrize("gap", ["（ ， ）", "( )", "等， ，", "（ ）"])
def test_broken_inline_structure_is_not_repaired_by_guessing(gap):
    assert "broken_inline_structure" in clean("中文正文" + gap + "仍有叙述。")["reject_reasons"]


def test_unknown_template_still_warns_and_nowiki_not_reinterpreted():
    r = clean("中文{{未知|不应猜的参数}}正文。")
    assert "unknown_template" in r["issues"] and "不应猜" not in r["text"]
    assert "unsupported_inline_template" in r["reject_reasons"]
    r = clean("中文<nowiki>-{吒}-</nowiki>正文。")
    assert "-{吒}-" in r["text"] and "unsupported_language_conversion" in r["reject_reasons"]


def test_block_unknown_is_pending_but_inline_date_or_name_hole_is_rejected():
    r = clean("{{未知导航}}\n这是一段独立完整的中文叙述。")
    assert r["issues"] == ["unknown_template"] and not r["reject_reasons"]
    for raw in (
        "克利斯登（Kristen，{{bd|1926年|8月27日}}），正文。",
        "全职工作于{{未知名字|机构}}，进行计算。",
    ):
        assert "unsupported_inline_template" in clean(raw)["reject_reasons"]
    assert "unsupported_inline_template" in clean("{{未知}}[[组织]]。中文正文。")["reject_reasons"]


def test_present_display_argument_must_not_fall_back_when_rendered_empty():
    r = clean("中文{{tsl|en|English|有内容的参数|<ref>不可作为正文</ref>}}正文。")
    assert "missing_display_content" in r["reject_reasons"]
    assert "有内容的参数" not in r["text"]
    assert "missing_display_content" in clean("中文{{nowrap|}}正文。")["reject_reasons"]


def test_rendering_after_conversion_still_has_node_budget(monkeypatch):
    monkeypatch.setattr(markup, "MAX_NODES", 1)
    assert clean("中文-{词}-正文。")["reject_reasons"] == ["markup_node_limit"]


def test_extraction_pipeline_never_exports_rejected_math_or_gap_as_text(tmp_path):
    pages = [
        page(
            title="好正文",
            text="{{Zy|酢|cù}}浆草是一种植物，{{tsl|en|Oxalis|酢浆草}}是它的名字。这里保留足够长的中文叙述。",
        ),
        page(title="公式", text="在<math>r</math>处，密度正比于<math>r^2</math>。").replace(
            "<id>1</id>", "<id>2</id>", 1
        ),
        page(
            title="缺词",
            text="中文正文含有（ ， ）的缺失部分，仍然不应当导出为合格正文。",
        ).replace("<id>1</id>", "<id>3</id>", 1),
    ]
    path, _ = make_download(tmp_path, document(pages))
    out = tmp_path / "body"
    m = cli.extract_bodies([path], out)
    assert m["processing"]["algorithm"] == "local_wikitext_body_v2"
    assert m["counts"]["quarantine"] == 1 and m["counts"]["rejected"] == 2
    good = read_records(out, "quarantine.jsonl")[0]
    assert good["text"].startswith("酢浆草") and good["status"] == "quarantine"
    assert all("text" not in r for r in read_records(out, "rejected.jsonl"))
    assert not m["ready_for_training"] and not read_records(out, "approved.jsonl")
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == m

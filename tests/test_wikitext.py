"""小型离线标记样例；不联网，重点检查正文不混入模板/页头/编辑指令。"""

import hashlib
import json

import pytest
from test_parse_xml import document, make_download, page

from minigpt_cn.data import extract as cli
from minigpt_cn.data import wikitext as markup
from minigpt_cn.data.wikitext import extract_wikitext


def clean(text, source="zhwiki", title="測試", **kwargs):
    return extract_wikitext(text, source=source, title=title, min_chars=1, **kwargs)


def read_records(directory, name):
    return [
        json.loads(line) for line in (directory / name).read_text(encoding="utf-8").splitlines()
    ]


def test_nested_templates_links_entities_categories_and_references():
    result = clean(
        "'''數學'''是[[學科|一門學科]]。{{lang|en|{{nowrap|English}}}} &amp; 😀\n"
        "<ref>引用{{cite web|url=https://example.org|title=不可混入}}</ref>"
        "[[File:a.jpg|thumb|圖片]][[分类:学科]][[en:Mathematics]]"
        "[https://example.org 可見標籤]<!--隐藏-->"
    )
    assert result["text"] == "數學是一門學科。English & 😀\n可見標籤"
    assert result["reject_reasons"] == []
    assert result["issues"] == []
    assert result["metrics"]["metadata_links"] == 3


def test_tail_sections_remove_descendants_but_keep_next_sibling():
    result = clean(
        "== 正文 ==\n這是正文。\n== 参考文献 ==\n引用\n=== 子節 ===\n引用二\n== 歷史 ==\n這是歷史。"
    )
    assert "正文" in result["text"] and "歷史" in result["text"]
    assert "引用" not in result["text"] and "子節" not in result["text"]


def test_poem_onlyinclude_header_variants_and_work_group():
    result = clean(
        "{{header|title=[[詩集]]|author=作者|section=第一章|next=另一作品}}不屬於正文\n"
        "<onlyinclude><poem>天{{另|地|空}}玄黃。\n宇宙{{ul|洪荒}}。</poem></onlyinclude>"
        "{{PD-old}}[[Category:古籍]]",
        source="zhwikisource",
        title="詩集/第一章",
    )
    assert result["text"] == "天地玄黃。\n宇宙洪荒。"
    assert result["metrics"]["onlyinclude_selected"]
    assert result["metrics"]["first_variant_choices"] == 1
    assert result["license_hints"] == ["pd-old"]
    assert result["headers"][0]["author"] == "作者"
    assert result["work_group"]["status"] == "needs_review"
    assert result["work_group"]["chapter"] == "第一章"
    other = clean("其他中文正文。", source="zhwikisource", title="詩集/第二章")
    assert other["work_group"]["suggested_id"] == result["work_group"]["suggested_id"]
    assert "不屬於" not in result["text"]


def test_onlyinclude_in_ref_or_template_is_not_selected():
    result = clean(
        "真實正文。<ref><onlyinclude>假的正文</onlyinclude></ref>"
        "{{cite web|title=<onlyinclude>另一假正文</onlyinclude>}}",
        source="zhwikisource",
    )
    assert result["text"] == "真實正文。"
    assert not result["metrics"]["onlyinclude_selected"]


@pytest.mark.parametrize(
    "markup,reason",
    [
        ("{{versions}}[[甲版本]][[乙版本]]", "versions_or_disambiguation"),
        ("{{檢索|論語}}[[/學而第一]][[/為政第二]]", "directory"),
        ("{{transclude pages|書.pdf|1|20}}", "requires_transclusion"),
        ('<pages index="書.pdf" from="1" to="20"/>', "requires_transclusion"),
        ("{{:其他頁面}}", "requires_transclusion"),
    ],
)
def test_nonbody_pages_are_explicitly_rejected(markup, reason):
    assert reason in clean(markup, source="zhwikisource")["reject_reasons"]


def test_unknown_templates_not_guessed_and_issues_not_silenced():
    result = clean("中文正文{{未知|錯誤參數}}。{{#invoke:模块|私有參數}}")
    assert "錯誤參數" not in result["text"]
    assert "私有參數" not in result["text"]
    assert result["issues"] == [
        "unexpanded_parser_function",
        "unknown_template",
        "unsupported_inline_template",
    ]
    assert "unsupported_inline_template" in result["reject_reasons"]
    assert result["unknown_templates"] == ["未知"]


def test_table_math_and_styles_do_not_pollute_body():
    result = clean(
        '中文正文。<div style="color:red">保留文字<br/>第二行。</div>\n'
        '{| class="wikitable"\n| 不保留表格\n|}\n<math>x+y</math>'
    )
    assert result["text"] == "中文正文。\n保留文字\n第二行。"
    assert "style" not in result["text"]
    assert "unsupported_visible_math" in result["reject_reasons"]
    assert result["metrics"]["omitted_tables"] == 1


def test_nowiki_keeps_literals_and_flags_residual_markup():
    result = clean("中文<nowiki>[[literal]]</nowiki>")
    assert result["text"] == "中文[[literal]]"
    assert "residual_markup" in result["issues"]
    assert "residual_markup" in result["reject_reasons"]


def test_list_and_section_markers_keep_text_without_false_unknown_tags():
    result = clean(
        "中文正文。\n* 第一項。\n* 第二項。\n"
        '<section begin="詩"/><small>小字</small><section end="詩"/>'
    )
    assert "第一項" in result["text"] and "第二項" in result["text"]
    assert "小字" in result["text"] and not result["issues"]


def test_transclusion_inside_removed_reference_does_not_reject_body():
    result = clean("中文正文。<ref>{{:不屬於正文}}</ref>")
    assert result["text"] == "中文正文。" and not result["reject_reasons"]


@pytest.mark.parametrize("literal", ["<|endoftext|>", "\x00"])
def test_reserved_or_control_text_rejected(literal):
    assert "reserved_or_control_text" in clean("中文正文。" + literal)["reject_reasons"]


def test_normalization_keeps_traditional_unicode_and_paragraphs():
    result = clean(" 繁體 e\u0301 😀\r\n\r\n段落二。 ")
    assert result["text"] == "繁體 é 😀\n\n段落二。"
    assert result["metrics"]["raw_chars"] > result["metrics"]["text_chars"]


def test_limits_and_low_han_ratio(monkeypatch):
    assert clean("中文" * 10, max_raw_chars=5)["reject_reasons"] == ["raw_chars_limit"]
    assert "low_han_ratio" in clean("hello world")["reject_reasons"]
    assert (
        "short_or_empty_body"
        in extract_wikitext("短", source="zhwiki", title="短")["reject_reasons"]
    )
    # parser自身会把极深标记退回字面文本；降低我们的上限独立验证渲染深度保护。
    monkeypatch.setattr(markup, "MAX_DEPTH", 2)
    deep = "{{nowrap|" * 4 + "中文" + "}}" * 4
    assert clean(deep)["reject_reasons"] == ["markup_depth_limit"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source": "other"},
        {"min_chars": True},
        {"min_chars": 0},
        {"max_raw_chars": 0},
    ],
)
def test_bad_options(kwargs):
    options = {"source": "zhwiki", "title": "测试", **kwargs}
    with pytest.raises(ValueError):
        extract_wikitext("中文正文", **options)


def fixture_dump(tmp_path, source="zhwikisource"):
    return make_download(
        tmp_path,
        document(
            [
                page(
                    title="詩集/第一章",
                    text="{{header|title=[[詩集]]}}<poem>中文正文第一行。\n"
                    "中文正文第二行，這是測試文檔。</poem>{{PD-old}}",
                ),
                page(title="目录", text="{{檢索|詩集}}[[/一]][[/二]]").replace(
                    "<id>1</id>", "<id>2</id>", 1
                ),
                page(title="模板页", namespace=10).replace("<id>1</id>", "<id>3</id>", 1),
            ],
            source,
        ),
        source,
    )


def test_pipeline_fail_closed_metadata_hashes_counts_and_no_input_writes(tmp_path):
    path, _ = fixture_dump(tmp_path)
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    output = tmp_path / "body"
    report = cli.extract_bodies([path], output, max_pages=None)
    assert report["scope"] == "full_dump" and not report["ready_for_training"]
    assert report["sources"][0]["xml"]["xml_document_complete"]
    assert report["counts"] == {"approved": 0, "quarantine": 1, "rejected": 2, "candidates": 1}
    record = read_records(output, "quarantine.jsonl")[0]
    assert "wikitext" not in record
    assert record["license"]["review_status"] == "per_document_review_required"
    assert record["license_hints"] == ["pd-old"]
    assert record["revision_url"].endswith("oldid=11")
    assert record["history_url"].endswith("curid=1&action=history")
    assert record["text_sha256"] == hashlib.sha256(record["text"].encode()).hexdigest()
    assert record["work_group"]["status"] == "needs_review"
    assert all("text" not in r for r in read_records(output, "rejected.jsonl"))
    for info in report["files"].values():
        assert info["sha256"] == cli._sha256(output / info["path"])
        assert info["bytes"] == (output / info["path"]).stat().st_size
    assert {p.name: p.read_bytes() for p in path.parent.iterdir()} == before
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        cli.extract_bodies([path], output)


def approve(record):
    return {
        **record,
        "decision": "approved",
        "quality_checked": True,
        "reviewer": "test-only",
        "reviewed_at": "2026-10-12",
        "notes": "离线虚构样例，不代表真实版权结论",
        "license_identifier": "TEST-LICENSE",
        "license_evidence": ["https://example.org/legal"],
        "work_id": "test:诗集",
        "work_evidence": ["https://example.org/work"],
    }


def test_exact_revision_review_and_work_group_reload(tmp_path):
    path, _ = fixture_dump(tmp_path)
    cli.extract_bodies([path], tmp_path / "first")
    record = approve(read_records(tmp_path / "first", "review_queue.jsonl")[0])
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(json.dumps(record), encoding="utf-8")
    report = cli.extract_bodies([path], tmp_path / "reviewed", reviews_path=reviews)
    assert report["counts"]["approved"] == 1 and report["counts"]["quarantine"] == 0
    accepted = read_records(tmp_path / "reviewed", "approved.jsonl")[0]
    assert accepted["work_group"]["id"] == "test:诗集"
    assert accepted["work_group"]["status"] == "reviewed"
    assert accepted["review"]["license_identifier"] == "TEST-LICENSE"
    record["text_sha256"] = "0" * 64
    reviews.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="摘要过期"):
        cli.extract_bodies([path], tmp_path / "bad", reviews_path=reviews)
    assert not (tmp_path / "bad" / "manifest.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"decision": "pending"},
        {"quality_checked": False},
        {"license_evidence": []},
        {"work_id": ""},
        {"reviewer": ""},
        {"work_evidence": ["http://example.org"]},
    ],
)
def test_invalid_reviews_fail_before_output(tmp_path, change):
    path, _ = fixture_dump(tmp_path)
    cli.extract_bodies([path], tmp_path / "first")
    record = {**approve(read_records(tmp_path / "first", "review_queue.jsonl")[0]), **change}
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        cli.extract_bodies([path], tmp_path / "bad", reviews_path=reviews)
    assert not (tmp_path / "bad").exists()


def test_cli_scope_and_bad_budget(tmp_path, capsys):
    path, _ = fixture_dump(tmp_path)
    report = cli.main(["--download-manifest", str(path), "--output-dir", str(tmp_path / "cli")])
    assert report["scope"] == "prefix_sample"
    assert '"ready_for_training": false' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(
            ["--download-manifest", str(path), "--output-dir", str(tmp_path / "full"), "--full"]
        )
    with pytest.raises(ValueError, match="字节预算"):
        cli.extract_bodies([path], tmp_path / "limited", max_xml_bytes=100)
    assert not (tmp_path / "limited" / "manifest.json").exists()


def test_input_change_and_duplicate_sources_never_write_success(tmp_path, monkeypatch):
    path, _ = fixture_dump(tmp_path)
    with pytest.raises(ValueError, match="重复"):
        cli.extract_bodies([path, path], tmp_path / "duplicate")
    assert not (tmp_path / "duplicate").exists()
    original, calls = cli._verified_input, []

    def changing(manifest):
        source, dump, info = original(manifest)
        calls.append(1)
        if len(calls) > 1:
            info = {**info, "sha256": "0" * 64}
        return source, dump, info

    monkeypatch.setattr(cli, "_verified_input", changing)
    with pytest.raises(RuntimeError, match="输入发生变化"):
        cli.extract_bodies([path], tmp_path / "changed")
    assert not (tmp_path / "changed" / "manifest.json").exists()


def test_review_cannot_approve_structural_rejection_or_missing_revision(tmp_path):
    path, _ = fixture_dump(tmp_path)
    cli.extract_bodies([path], tmp_path / "first")
    rejected = read_records(tmp_path / "first", "rejected.jsonl")[0]
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(json.dumps(approve(rejected)), encoding="utf-8")
    with pytest.raises(ValueError, match="结构性拒绝"):
        cli.extract_bodies([path], tmp_path / "bad", reviews_path=reviews)
    record = approve(read_records(tmp_path / "first", "review_queue.jsonl")[0])
    record["revision_id"] = 9999
    reviews.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="未遇到"):
        cli.extract_bodies([path], tmp_path / "missing", reviews_path=reviews)
    assert not (tmp_path / "missing" / "manifest.json").exists()


def test_full_stream_does_not_stop_at_old_10000_sample_limit(tmp_path):
    path, _ = make_download(tmp_path, document([page(text="中文正文。" * 4)] * 10001))
    report = cli.extract_bodies([path], tmp_path / "full", max_pages=None)
    assert report["sources"][0]["xml"]["pages_scanned"] == 10001
    assert report["sources"][0]["xml"]["xml_document_complete"]
    assert report["sources"][0]["xml"]["peak_root_children"] < 100
    assert report["counts"]["quarantine"] == 10001

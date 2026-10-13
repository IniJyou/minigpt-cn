"""保守的本地正文提取：AST解析语法，白名单决定内容；不执行模板或下载转录页。"""

import hashlib
import re
from collections import Counter

import mwparserfromhell as mw
import regex
from mwparserfromhell.nodes import (
    Argument,
    Comment,
    ExternalLink,
    Heading,
    HTMLEntity,
    Tag,
    Template,
    Text,
    Wikilink,
)

from minigpt_cn.data.cleaning import normalize_text

ALGORITHM = "local_wikitext_body_v2"
MAX_DEPTH = 48
MAX_NODES = 100000
DROP_TAGS = {
    "ref",
    "references",
    "gallery",
    "table",
    "timeline",
    "score",
    "svg",
    "script",
    "style",
    "code",
    "source",
    "syntaxhighlight",
    "pre",
    "noinclude",
    "imagemap",
    "templatestyles",
}
INLINE_TAGS = {
    "b",
    "i",
    "em",
    "strong",
    "span",
    "font",
    "u",
    "s",
    "sub",
    "sup",
    "small",
    "big",
    "cite",
    "tt",
    "abbr",
}
BLOCK_TAGS = {
    "div",
    "p",
    "poem",
    "center",
    "blockquote",
    "onlyinclude",
    "includeonly",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
}
# 明确知道显示语义的模板才取参数；不能对所有模板一律取“最后一个参数”。
FIRST_ARGUMENT = {
    "nowrap",
    "nobr",
    "small",
    "big",
    "center",
    "left",
    "right",
    "ul",
    "propernoun",
    "wavybookmark",
    "*",
    "另",
    "另2",
    "校",
    "僻字",
}
METADATA = {
    "header",
    "header2",
    "footer",
    "textquality",
    "reflist",
    "references",
    "notelist",
    "col-begin",
    "col-end",
    "col-break",
    "clear",
    "toc",
    "目錄",
    "目录",
    "檢索",
    "检索",
    "versions",
    "disambig",
    "disambiguation",
    "消歧义",
    "消歧義",
    "navbox",
    "main",
    "see also",
    "seealso",
    "further",
    "authority control",
    "defaultsort",
}
TAIL_SECTIONS = {
    "参考资料",
    "參考資料",
    "参考文献",
    "參考文獻",
    "注释",
    "註釋",
    "外部链接",
    "外部連結",
    "延伸阅读",
    "延伸閱讀",
    "参见",
    "參見",
    "references",
    "external links",
    "notes",
}
NON_BODY_NAMESPACES = {
    "file",
    "image",
    "category",
    "文件",
    "档案",
    "檔案",
    "图像",
    "圖像",
    "分类",
    "分類",
}
HAN = regex.compile(r"\p{Script=Han}")
CONVERSION = re.compile(r"-\{(.*?)\}-", re.DOTALL)
VARIANTS = ("zh-hans", "zh-cn", "zh-sg", "zh-my", "zh", "zh-hant", "zh-tw", "zh-hk", "zh-mo")
LINK_TEMPLATES = {"le", "link-en", "link-ja", "link-fr", "link-ru", "link-ko"}
EMPTY_PARENTHESES = re.compile(r"\([\s,，、;；:：/／]*\)|（[\s,，、;；:：/／]*）")
REPEATED_COMMA = re.compile(r"[,，]\s*[,，]")


def _name(template):
    return str(template.name).strip().replace("_", " ").casefold()


def _parameter(template, name):
    return template.get(name).value if template.has(name) else mw.parse("")


class _Renderer:
    def __init__(self):
        self.counts = Counter()
        self.issues = set()
        self.unknown = set()
        self.visited_nodes = 0

    def render(self, code, depth=0):
        if depth > MAX_DEPTH:
            raise ValueError("markup_depth_limit")
        parts, pending, excluded_level = [], [], None
        for index, node in enumerate(code.nodes):
            self.visited_nodes += 1
            if self.visited_nodes > MAX_NODES:
                raise ValueError("markup_node_limit")
            if isinstance(node, Heading):
                if excluded_level is not None and node.level > excluded_level:
                    continue
                excluded_level = None
                title = self.render(node.title, depth + 1).strip()
                # 暂存标题，遇到实际正文才写出。空表格/导航/引用不让标题“变成正文”。
                # 新的同级/上级标题结束旧空节；父标题只有子节正文时仍会保留。
                while pending and pending[-1][0] >= node.level:
                    pending.pop()
                    self.counts["removed_empty_sections"] += 1
                if title.casefold() in TAIL_SECTIONS:
                    excluded_level = node.level
                    self.counts["removed_sections"] += 1
                else:
                    pending.append((node.level, title))
            elif excluded_level is None:
                body = self.node(node, depth)
                if body is None:
                    # 未知模板单独一行可能是导航，继续保留待审警告；嵌在句子里则可能
                    # 是姓名/日期/术语。不能删成“全职工作于 ，”再作为候选输出。
                    for neighbor_index, side in ((index - 1, -1), (index + 1, 0)):
                        if 0 <= neighbor_index < len(code.nodes):
                            neighbor = code.nodes[neighbor_index]
                            if isinstance(neighbor, (Wikilink, ExternalLink)):
                                self.issues.add("unsupported_inline_template")
                            if (
                                isinstance(neighbor, Text)
                                and str(neighbor).split("\n")[side].strip()
                            ):
                                self.issues.add("unsupported_inline_template")
                    body = " "
                if body.strip():
                    parts.extend("\n" + title + "\n" for _, title in pending)
                    pending.clear()
                parts.append(body)
        self.counts["removed_empty_sections"] += len(pending)
        return "".join(parts)

    def conversion(self, text):
        """只选择局部明文/地区字词，不执行H/A等影响后文的全局转换规则。"""

        def choose(match):
            payload = match[1].strip()
            if payload.startswith("R|"):
                payload = payload[2:]
                if re.match(r"zh(?:-[a-z]+)?:", payload):
                    return match[0]  # 本轮不解释R|内部的地区规则。
            elif "|" in payload:
                return match[0]  # 有未知标志或内部标记，保留残余并在最终检查中拒绝。
            if "|" in payload:
                return match[0]
            if any(c in payload for c in "{}[]<>") or "=>" in payload:
                return match[0]
            if re.match(r"zh(?:-[a-z]+)?:", payload):
                values = {}
                for entry in payload.rstrip(";").split(";"):
                    key, sep, value = entry.strip().partition(":")
                    if not sep or key not in VARIANTS or key in values or not value.strip():
                        return match[0]
                    values[key] = value.strip()
                payload = next(values[key] for key in VARIANTS if key in values)
            self.counts["local_conversion_choices"] += 1
            return payload

        return CONVERSION.sub(choose, text)

    def display(self, template, fields, depth):
        """仅按已知模板的显示参数优先级取值，不取语言代码/URL/CSS或随机末参数。"""
        for field in fields:
            value = _parameter(template, field)
            if str(value).strip():
                self.counts["display_templates"] += 1
                body = self.render(value, depth + 1)
                if body.strip():
                    return body
                break  # 已指定的显示内容解析失败，不能退回另一参数假装成功。
        self.issues.add("missing_display_content")
        return " "

    def node(self, node, depth):
        if isinstance(node, Text):
            raw = str(node)
            converted = self.conversion(raw)
            # 库可能把含转换标记的链接整段退回Text；去掉局部语法后再按AST解析链接。
            return self.render(mw.parse(converted), depth + 1) if converted != raw else raw
        if isinstance(node, HTMLEntity):
            try:
                return node.normalize()
            except (ValueError, OverflowError):
                self.issues.add("invalid_entity")
                return ""
        if isinstance(node, Comment):
            return ""
        if isinstance(node, Wikilink):
            title = str(node.title).strip()
            prefix, separator, _ = title.partition(":")
            if separator and not title.startswith(":"):
                if prefix.casefold() in NON_BODY_NAMESPACES or re.fullmatch(
                    r"[a-z-]{2,12}", prefix
                ):
                    self.counts["metadata_links"] += 1
                    return ""
            self.counts["visible_links"] += 1
            return (
                self.render(node.text, depth + 1)
                if node.text is not None
                else self.render(mw.parse(title.lstrip(":").split("#", 1)[0]), depth + 1)
            )
        if isinstance(node, ExternalLink):
            self.counts["external_links"] += 1
            return self.render(node.title, depth + 1) if node.title is not None else ""
        if isinstance(node, Template):
            name = _name(node)
            self.counts["templates"] += 1
            if name in {"lang", "langx"}:
                body = self.display(node, ("text", "2"), depth)
                if name == "langx":
                    # 保留可选音译/直译，不生成依赖外部语言表的标签或HTML样式。
                    for fields in (("translit", "3"), ("translation", "lit", "4")):
                        if any(str(_parameter(node, f)).strip() for f in fields):
                            body += "（" + self.display(node, fields, depth) + "）"
                return body
            if name in {"tsl", "translink"}:
                return self.display(node, ("4", "3", "2"), depth)
            if name in LINK_TEMPLATES:
                return self.display(node, ("d", "3", "1"), depth)
            if name == "zy":
                # 本地样例Zy|亳|bó|ㄅㄛˋ|bok3：正文必须保留第一个“字”参数。
                # 只保留基字，不把拼音/注音/粤音当成字，也不模拟ruby排版。
                return self.display(node, ("1",), depth)
            if name in FIRST_ARGUMENT:
                if name in {"另", "另2", "校", "僻字"}:
                    self.counts["first_variant_choices"] += 1
                return self.display(node, ("1",), depth)
            if name == "gap":
                return " "
            if name in METADATA or name.startswith(("cite ", "citation", "infobox", "pd-")):
                self.counts["metadata_templates"] += 1
                return " "
            if name.startswith(":") or name in {"transclude pages", "pages"}:
                self.issues.add("requires_transclusion")
            elif name.startswith("#"):
                self.issues.add("unexpanded_parser_function")
            else:
                self.issues.add("unknown_template")
                # 每篇最多存16种名称；全量报告不积累数百万模板名。
                if len(self.unknown) < 16:
                    self.unknown.add(name[:100])
            return None  # 调用者检查行内上下文；不把未知模板参数猜成正文。
        if isinstance(node, Tag):
            name = str(node.tag).strip().casefold()
            if name == "math":
                # 不静默删掉行内公式。尚无可靠纯文本数学渲染，本轮拒绝整个正文。
                # 被忽略的ref/infobox内部不进入这里，不影响独立叙述正文。
                self.issues.add("unsupported_visible_math")
                self.counts["unsupported_math_tags"] += 1
                return " "
            if name == "pages":
                self.issues.add("requires_transclusion")
                return ""
            if name in DROP_TAGS:
                self.counts["removed_tags"] += 1
                if name == "table":
                    self.counts["omitted_tables"] += 1
                return "\n" if name == "table" else ""
            if name in {"br", "hr"}:
                return "\n"
            if name == "section" and node.self_closing:
                return ""  # Wikisource转录片段的begin/end定位标记，不是作品正文。
            if name == "nowiki":
                return str(node.contents or "")  # 不再解析，字面量中的尖括号应保留。
            if name in INLINE_TAGS | BLOCK_TAGS:
                body = self.render(node.contents, depth + 1) if node.contents is not None else ""
                return "\n" + body + "\n" if name in BLOCK_TAGS else body
            self.issues.add("unknown_tag")
            return " "
        if isinstance(node, Argument):
            self.issues.add("unexpanded_argument")
            return " "
        self.issues.add("unknown_node")
        return " "


def _onlyinclude_blocks(code):
    """只选择可见正文里的外层onlyinclude；不收集ref/模板参数里的伪正文。"""
    for node in code.nodes:
        if isinstance(node, Tag):
            name = str(node.tag).strip().casefold()
            if name == "onlyinclude" and node.contents is not None:
                yield node.contents
            elif name in INLINE_TAGS | BLOCK_TAGS and node.contents is not None:
                yield from _onlyinclude_blocks(node.contents)


def _has_transclusion(code, depth=0):
    if depth > MAX_DEPTH:
        raise ValueError("markup_depth_limit")
    for node in code.nodes:
        if isinstance(node, Template):
            if _name(node).startswith(":") or _name(node) in {"transclude pages", "pages"}:
                return True
        elif isinstance(node, Tag):
            name = str(node.tag).strip().casefold()
            if name == "pages":
                return True
            if name in INLINE_TAGS | BLOCK_TAGS and node.contents is not None:
                if _has_transclusion(node.contents, depth + 1):
                    return True
    return False  # 引用/被丢弃的元数据里的转录不决定正文完整性。


def _metadata(code, source, title):
    headers, license_hints, names = [], set(), set()
    for template in code.filter_templates(recursive=True):
        name = _name(template)
        names.add(name)
        if name.startswith("pd-") and len(license_hints) < 32:
            license_hints.add(name)
        if name in {"header", "header2"} and len(headers) < 4:
            headers.append(
                {
                    key: str(_parameter(template, key))[:512].strip()
                    for key in (
                        "title",
                        "section",
                        "author",
                        "translator",
                        "year",
                        "previous",
                        "next",
                    )
                    if template.has(key)
                }
            )
    root, _, chapter = title.partition("/")
    evidence = ["page_title_prefix" if chapter else "page_title"]
    if source == "zhwikisource" and headers:
        links = mw.parse(headers[0].get("title", "")).filter_wikilinks()
        if links:
            root = str(links[0].title).lstrip(":").split("#", 1)[0].split("/", 1)[0]
            evidence.append("header_title_link")
    # 分组键只为建议：繁简合并有助于人工发现同一作品，不能自动证明作品/版本相同。
    key = normalize_text(root, to_simplified=True) if source == "zhwikisource" else title
    group = {
        "status": "needs_review" if source == "zhwikisource" else "page_identity",
        "suggested_id": source + ":" + hashlib.sha256(key.encode()).hexdigest(),
        "suggested_title": root if source == "zhwikisource" else title,
        "chapter": chapter if source == "zhwikisource" else None,
        "evidence": evidence if source == "zhwikisource" else ["page_identity"],
    }
    # 版本选择页与检索目录不能作为作品正文；不按作者/previous/next自动合并作品。
    kind = "body"
    if names & {"versions", "disambig", "disambiguation", "消歧义", "消歧義"}:
        kind = "versions_or_disambiguation"
    elif source == "zhwikisource" and names & {"檢索", "检索"}:
        kind = "directory"
    return {
        "headers": headers,
        "license_hints": sorted(license_hints),
        "work_group": group,
        "page_kind": kind,
    }


def extract_wikitext(wikitext, *, source, title, min_chars=20, max_raw_chars=1000000):
    """返回正文、质量指标与拒绝原因；无许可自动批准、繁转简正文或跨页展开。

    这里的长度是Unicode码点，不是UTF-8字节或模型token；每篇单独解析，语料不整体入RAM。
    """
    if not isinstance(wikitext, str) or not isinstance(title, str):
        raise TypeError("wikitext/title必须是字符串")
    if source not in {"zhwiki", "zhwikisource"}:
        raise ValueError("source必须是zhwiki或zhwikisource")
    for name, value in (("min_chars", min_chars), ("max_raw_chars", max_raw_chars)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name}必须是正整数")
    result = {
        "text": "",
        "issues": [],
        "reject_reasons": [],
        "metrics": {
            "raw_chars": len(wikitext),
            "text_chars": 0,
            "han_ratio": 0.0,
        },
    }
    if len(wikitext) > max_raw_chars:
        result["reject_reasons"] = ["raw_chars_limit"]
        return result
    try:
        code = mw.parse(wikitext)
        if len(code.filter(recursive=True)) > MAX_NODES:
            raise ValueError("markup_node_limit")
        metadata = _metadata(code, source, title)
        renderer = _Renderer()
        selected = list(_onlyinclude_blocks(code)) if source == "zhwikisource" else []
        # 用完整AST收集跨页依赖，不能因为选择onlyinclude而误报“已完整展开”。
        if _has_transclusion(code):
            renderer.issues.add("requires_transclusion")
        body = (
            "\n".join(renderer.render(block) for block in selected)
            if selected
            else (renderer.render(code))
        )
        body = re.sub(r"\n{3,}", "\n\n", normalize_text(body, to_simplified=False))
        # 行首列表标记/魔术字只在AST去除模板等结构后处理，不用正则模拟嵌套语法。
        body = re.sub(r"(?m)^[*#;:]+\s*", "", body)
        body = re.sub(r"__(?:[A-Z][A-Z_]+)__", "", body).strip()
        nonspace = sum(not char.isspace() for char in body)
        han_ratio = len(HAN.findall(body)) / max(nonspace, 1)
        if any(mark in body for mark in ("{{", "}}", "[[", "]]", "{|", "|}")):
            renderer.issues.add("residual_markup")
        if re.search(r"</?[A-Za-z][^>\n]{0,500}>", body):
            renderer.issues.add("residual_markup")
        if "-{" in body or "}-" in body:
            renderer.issues.add("unsupported_language_conversion")
        if EMPTY_PARENTHESES.search(body) or REPEATED_COMMA.search(body):
            renderer.issues.add("broken_inline_structure")
        # 有些parser会静默移除NUL；从原文检查控制字符，避免“清掉后假装正常”。
        if "<|endoftext|>" in body or any(
            (ord(c) < 32 or ord(c) == 127) and c not in "\r\n\t" for c in wikitext
        ):
            renderer.issues.add("reserved_or_control_text")
        reasons = []
        if metadata["page_kind"] != "body":
            reasons.append(metadata["page_kind"])
        if "requires_transclusion" in renderer.issues:
            reasons.append("requires_transclusion")  # 即使有部分直写正文，也不能冒充完整作品。
        for issue in (
            "residual_markup",
            "reserved_or_control_text",
            "unsupported_visible_math",
            "unsupported_inline_template",
            "unsupported_language_conversion",
            "missing_display_content",
            "broken_inline_structure",
        ):
            if issue in renderer.issues:
                reasons.append(issue)  # 不把CSS/残余语法或EOS注入当作合格纯正文。
        if nonspace < min_chars:
            reasons.append("short_or_empty_body")
        if han_ratio < 0.1:
            reasons.append("low_han_ratio")
        result.update(
            metadata,
            text=body,
            issues=sorted(renderer.issues),
            reject_reasons=reasons,
            unknown_templates=sorted(renderer.unknown),
            metrics={
                "raw_chars": len(wikitext),
                "text_chars": len(body),
                "nonspace_chars": nonspace,
                "han_ratio": han_ratio,
                "retained_ratio": len(body) / max(len(wikitext), 1),
                "onlyinclude_selected": bool(selected),
                **renderer.counts,
            },
        )
    except (ValueError, RecursionError) as error:
        # 保守拒绝整个页面，绝不把截断/失败输出标成正常正文。
        reason = (
            str(error)
            if str(error) in {"markup_depth_limit", "markup_node_limit"}
            else ("markup_parse_failure")
        )
        result["reject_reasons"] = [reason]
    return result

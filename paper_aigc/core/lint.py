"""论文规范自检：图表编号、参考文献格式、中英文标点、重复词、术语、超长句、段落末尾。

只读 core/types.Document 契约，不修改文档、不联网、无第三方依赖。
lint_document() 设计为永不抛异常：单项检查内部出错时跳过该项，空文档返回空报告。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .textnorm import (
    count_cjk,
    is_furniture_like,
    is_heading_like,
    is_reference_like,
    normalize_text,
    split_sentences,
)
from .types import Document, Paragraph

ISSUE_CODES: Tuple[str, ...] = (
    "table_seq",
    "table_missing",
    "ref_format",
    "mixed_punct",
    "dup_word",
    "term_inconsistent",
    "long_sentence",
    "no_period",
)

LEVEL_ERROR = "error"
LEVEL_WARN = "warn"
LEVEL_INFO = "info"
_LEVELS: Tuple[str, ...] = (LEVEL_ERROR, LEVEL_WARN, LEVEL_INFO)
_CODE_ORDER: Dict[str, int] = {code: i for i, code in enumerate(ISSUE_CODES)}

SNIPPET_LIMIT = 60
LONG_SENTENCE_LIMIT = 80

_CAPTION_RE = re.compile(r"^(附表|附图|表|图)\s*([0-9]+|[一二三四五六七八九十百零〇两]+)")
_MENTION_RE = re.compile(r"(附表|附图|表|图)\s*([0-9]+|[一二三四五六七八九十百零〇两]+)")
_REF_HEADING_RE = re.compile(
    r"^(参考文献|参考资料|引用文献|文献列表|references?|bibliography|works\s+cited)"
    r"\s*[:：]?\s*$",
    re.IGNORECASE,
)
_REF_NUM_RE = re.compile(r"^\s*\[(\d+)\]")
_REF_TYPE_RE = re.compile(
    r"\[(?:J|M|R|D|C|N|P|S|G|A|Z|EB/OL|DB/OL|CP/OL|J/OL|M/OL|R/OL|D/OL)\]",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")

_MIXED_PUNCT_RE = re.compile(r"[\u4e00-\u9fff][,\.?!()]|[,\.?!()][\u4e00-\u9fff]")
_ASCII_PUNCT_RE = re.compile(r"[,\.?!()]")

_DUP_SINGLE_RE = re.compile(r"([\u4e00-\u9fff])\1")
_DUP_MULTI_RE = re.compile(r"([\u4e00-\u9fff]{2,4})\1")

# 汉语里合法的叠字，不能当「重复用词」报。
_LEGIT_REDUP_SINGLE: Set[str] = set(
    "一人常渐慢纷仅刚看试谢妈爸哥姐弟妹天年处种层步件家字声片阵双些点滴团块张条次回遍番面方个每各"
)

# 合法的「一 + 量词」叠用，例如「一个一个」「一步一步」。
_LEGIT_REDUP_MULTI: Set[str] = {
    "一个一个",
    "一天一天",
    "一年一年",
    "一次一次",
    "一条一条",
    "一件一件",
    "一步一步",
    "一点一点",
    "一批一批",
    "一代一代",
    "一行一行",
    "一字一字",
    "一遍一遍",
    "一层一层",
    "一家一家",
    "一本一本",
    "一篇一篇",
    "一段一段",
    "一章一章",
    "一句一句",
}

# 中文数字引用的上下文守卫：避免把「发表一次」「代表一定」当成「表一」。
_CN_MENTION_PREFIX: Set[str] = set("如见由据在和与及或上下附")
_CN_MENTION_SUFFIX: Tuple[str, ...] = (
    "所示",
    "中",
    "为",
    "所",
    "给",
    "列",
    "显",
    "和",
    "与",
    "、",
    "，",
    "。",
    "及",
    "的",
    "：",
    "；",
    "）",
    ")",
    "”",
)

# 内置中英术语词表：两种写法同时出现即视为术语混用。
_TERM_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("净资产收益率", "ROE"),
    ("国内生产总值", "GDP"),
    ("人工智能", "AI"),
    ("居民消费价格指数", "CPI"),
    ("机器学习", "ML"),
    ("自然语言处理", "NLP"),
)

_CN_DIGITS: Dict[str, int] = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}

_TRAILING_CLOSERS = "”’」』）)】》\"'"
_END_PUNCT = "。！？!?…；;：:.．"


def _snippet(text: str, limit: int = SNIPPET_LIMIT) -> str:
    """压成单行并截断到 limit 个字符（超长补省略号）。"""
    value = normalize_text(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)] + "…"


def _around(text: str, position: int, radius: int = 24) -> str:
    """取命中位置附近的上下文片段，长度受 SNIPPET_LIMIT 约束。"""
    try:
        position = int(position)
    except Exception:
        position = 0
    start = max(0, position - radius)
    end = min(len(text or ""), position + radius)
    return _snippet((text or "")[start:end])


def _parse_number(token: str) -> Optional[int]:
    """解析阿拉伯数字或简单中文数字（一…九十九）；失败返回 None。"""
    value = (token or "").strip()
    if not value:
        return None
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" in value:
        left, _, right = value.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    digits = [_CN_DIGITS.get(ch) for ch in value]
    if any(digit is None for digit in digits):
        return None
    result = 0
    for digit in digits:
        result = result * 10 + int(digit)
    return result


def _mention_allowed(text: str, match) -> bool:
    """阿拉伯数字引用直接放行；中文数字引用要求上下文像「如表一所示」。"""
    token = match.group(2)
    if token.isdigit():
        return True
    start, end = match.span()
    previous = text[start - 1] if start > 0 else ""
    if previous and previous not in _CN_MENTION_PREFIX and previous not in "，。；：、（(":
        return False
    tail = text[end:]
    if not tail:
        return True
    return any(tail.startswith(suffix) for suffix in _CN_MENTION_SUFFIX)


def _paragraph_text(paragraph: Paragraph) -> str:
    try:
        return normalize_text(getattr(paragraph, "text", "") or "")
    except Exception:
        return ""


def _paragraph_index(paragraph: Paragraph) -> int:
    try:
        return int(getattr(paragraph, "index", -1))
    except Exception:
        return -1


def _is_heading(paragraph: Paragraph, text: str) -> bool:
    try:
        if getattr(paragraph, "is_heading", False):
            return True
        return bool(is_heading_like(text, getattr(paragraph, "style", "") or ""))
    except Exception:
        return False


def _is_reference(paragraph: Paragraph, text: str) -> bool:
    try:
        if getattr(paragraph, "is_reference", False):
            return True
        return bool(is_reference_like(text))
    except Exception:
        return False


def _is_ref_heading(text: str) -> bool:
    return bool(_REF_HEADING_RE.match((text or "").strip()))


def _looks_like_caption(text: str) -> bool:
    """短行、无句末标点、以「表N/图N」开头，才算图表题。"""
    value = (text or "").strip()
    if not _CAPTION_RE.match(value):
        return False
    return len(value) <= 60 and not any(ch in value for ch in "。！？!?")


def _caption_kind(prefix: str) -> str:
    return "表" if prefix in ("表", "附表") else "图"


def _is_table_row(text: str) -> bool:
    value = (text or "").strip()
    return value.startswith("|") or value.count("|") >= 2


def _content_paragraphs(paragraphs: Sequence[Paragraph]):
    """只产出正文段：跳过标题、参考文献、图表题。"""
    for paragraph in paragraphs:
        text = _paragraph_text(paragraph)
        if not text:
            continue
        if _is_heading(paragraph, text):
            continue
        if _is_reference(paragraph, text):
            continue
        if _looks_like_caption(text):
            continue
        yield paragraph, text


@dataclass
class LintIssue:
    """单条规范问题。paragraph_index 无对应段落时为 -1，snippet 不超过 60 字。"""

    code: str
    level: str
    message: str
    paragraph_index: int = -1
    snippet: str = ""

    def __post_init__(self) -> None:
        self.code = str(self.code or "")
        self.level = self.level if self.level in _LEVELS else LEVEL_INFO
        self.message = str(self.message or "")
        self.snippet = _snippet(str(self.snippet or ""))
        try:
            self.paragraph_index = int(self.paragraph_index)
        except Exception:
            self.paragraph_index = -1

    def to_dict(self) -> dict:
        return _issue_to_dict(self)


@dataclass
class LintReport:
    """自检结果。counts 固定包含 error / warn / info 三个键。"""

    issues: List[LintIssue] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = _count_levels(self.issues)


def _count_levels(issues: Sequence[LintIssue]) -> Dict[str, int]:
    counts: Dict[str, int] = {level: 0 for level in _LEVELS}
    for issue in issues or []:
        level = getattr(issue, "level", LEVEL_INFO)
        if level not in counts:
            counts[level] = 0
        counts[level] += 1
    return counts


def _issue_to_dict(issue: LintIssue) -> dict:
    try:
        return {
            "code": str(getattr(issue, "code", "")),
            "level": str(getattr(issue, "level", LEVEL_INFO)),
            "message": str(getattr(issue, "message", "")),
            "paragraph_index": int(getattr(issue, "paragraph_index", -1)),
            "snippet": _snippet(str(getattr(issue, "snippet", "") or "")),
        }
    except Exception:
        return {
            "code": "unknown",
            "level": LEVEL_INFO,
            "message": "",
            "paragraph_index": -1,
            "snippet": "",
        }


def _dedupe(issues: Sequence[LintIssue]) -> List[LintIssue]:
    result: List[LintIssue] = []
    seen: Set[Tuple[str, int, str]] = set()
    for issue in issues:
        key = (issue.code, issue.paragraph_index, issue.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result


def _sorted(issues: Sequence[LintIssue]) -> List[LintIssue]:
    def sort_key(issue: LintIssue):
        index = issue.paragraph_index
        return (
            0 if index < 0 else 1,
            index if index >= 0 else 0,
            _CODE_ORDER.get(issue.code, len(ISSUE_CODES)),
            issue.message,
        )

    return sorted(issues, key=sort_key)


def _check_tables(paragraphs: Sequence[Paragraph]) -> List[LintIssue]:
    """图表编号连续性 / 重复 / 正文引用与图表题是否对应。"""
    issues: List[LintIssue] = []
    captions: List[Tuple[str, int, Paragraph, str, int]] = []
    caption_ids: Set[int] = set()
    for paragraph in paragraphs:
        text = _paragraph_text(paragraph)
        if not text or _is_heading(paragraph, text):
            continue
        match = _CAPTION_RE.match(text)
        if not match or not _looks_like_caption(text):
            continue
        number = _parse_number(match.group(2))
        if number is None:
            continue
        kind = _caption_kind(match.group(1))
        captions.append((kind, number, paragraph, text, match.start(1)))
        caption_ids.add(id(paragraph))

    mentions: List[Tuple[str, int, Paragraph, str, int]] = []
    for paragraph in paragraphs:
        text = _paragraph_text(paragraph)
        if not text or _is_heading(paragraph, text):
            continue
        if id(paragraph) in caption_ids:
            continue
        if _is_reference(paragraph, text):
            continue
        for match in _MENTION_RE.finditer(text):
            number = _parse_number(match.group(2))
            if number is None:
                continue
            if not _mention_allowed(text, match):
                continue
            mentions.append(
                (_caption_kind(match.group(1)), number, paragraph, text, match.start(1))
            )

    caption_numbers: Dict[str, Set[int]] = {"表": set(), "图": set()}
    for kind, number, _paragraph, _text, _position in captions:
        caption_numbers[kind].add(number)

    for kind in ("表", "图"):
        sequence = [item for item in captions if item[0] == kind]
        if not sequence:
            continue
        numbers = [number for _kind, number, _p, _t, _pos in sequence]
        seen: Dict[int, Paragraph] = {}
        for number, paragraph, text, _position in [
            (item[1], item[2], item[3], item[4]) for item in sequence
        ]:
            if number in seen:
                issues.append(
                    LintIssue(
                        "table_seq",
                        LEVEL_ERROR,
                        "%s编号重复：%s%d 出现了不止一次。" % (kind, kind, number),
                        _paragraph_index(paragraph),
                        text,
                    )
                )
            else:
                seen[number] = paragraph
        missing = [
            number
            for number in range(min(numbers), max(numbers) + 1)
            if number not in seen
        ]
        for number in missing:
            issues.append(
                LintIssue(
                    "table_seq",
                    LEVEL_WARN,
                    "%s编号不连续：缺少 %s%d。" % (kind, kind, number),
                    _paragraph_index(sequence[0][2]),
                    sequence[0][3],
                )
            )
        if numbers and numbers[0] != 1:
            issues.append(
                LintIssue(
                    "table_seq",
                    LEVEL_WARN,
                    "%s编号未从 1 开始（首个为 %s%d）。" % (kind, kind, numbers[0]),
                    _paragraph_index(sequence[0][2]),
                    sequence[0][3],
                )
            )
        if numbers != sorted(numbers):
            issues.append(
                LintIssue(
                    "table_seq",
                    LEVEL_INFO,
                    "%s编号顺序与文档出现顺序不一致，建议按正文顺序排列。" % kind,
                    _paragraph_index(sequence[0][2]),
                    sequence[0][3],
                )
            )

    mention_numbers: Dict[str, Set[int]] = {"表": set(), "图": set()}
    reported: Set[Tuple[str, int, int]] = set()
    for kind, number, paragraph, text, position in mentions:
        mention_numbers[kind].add(number)
        if number in caption_numbers[kind]:
            continue
        index = _paragraph_index(paragraph)
        key = (kind, number, index)
        if key in reported:
            continue
        reported.add(key)
        issues.append(
            LintIssue(
                "table_missing",
                LEVEL_ERROR,
                "正文引用了「%s%d」，但文档中未找到对应的%s题（如「%s%d ...」）。"
                % (kind, number, kind, kind, number),
                index,
                _around(text, position),
            )
        )

    for kind, number, paragraph, text, _position in captions:
        if number not in mention_numbers[kind]:
            issues.append(
                LintIssue(
                    "table_seq",
                    LEVEL_INFO,
                    "%s题「%s%d」在正文中没有被引用。" % (kind, kind, number),
                    _paragraph_index(paragraph),
                    text,
                )
            )
    return issues


def _reference_paragraphs(
    paragraphs: Sequence[Paragraph],
) -> List[Tuple[Paragraph, str]]:
    """收集参考文献段：显式 is_reference / 参考文献小节内 / 形如文献条目的段落。"""
    result: List[Tuple[Paragraph, str]] = []
    in_reference_section = False
    for paragraph in paragraphs:
        text = _paragraph_text(paragraph)
        if not text:
            continue
        if _is_heading(paragraph, text) or _is_ref_heading(text):
            in_reference_section = _is_ref_heading(text)
            continue
        if in_reference_section or _is_reference(paragraph, text):
            result.append((paragraph, text))
    return result


def _check_references(paragraphs: Sequence[Paragraph]) -> List[LintIssue]:
    """参考文献：缺序号 / 缺年份 / 缺文献类型标识 / 序号不连续。"""
    issues: List[LintIssue] = []
    expected = 1
    for paragraph, text in _reference_paragraphs(paragraphs):
        index = _paragraph_index(paragraph)
        number_match = _REF_NUM_RE.match(text)
        if number_match:
            number = int(number_match.group(1))
            if number != expected:
                issues.append(
                    LintIssue(
                        "ref_format",
                        LEVEL_WARN,
                        "参考文献序号不连续：上一条应为 [%d]，本条为 [%d]。"
                        % (expected, number),
                        index,
                        text,
                    )
                )
            expected = number + 1
        else:
            issues.append(
                LintIssue(
                    "ref_format",
                    LEVEL_ERROR,
                    "参考文献缺少序号（应以 [1] 这样的编号开头）。",
                    index,
                    text,
                )
            )
        if not _YEAR_RE.search(text):
            issues.append(
                LintIssue(
                    "ref_format",
                    LEVEL_ERROR,
                    "参考文献缺少年份（未找到 19xx / 20xx）。",
                    index,
                    text,
                )
            )
        if not _REF_TYPE_RE.search(text):
            issues.append(
                LintIssue(
                    "ref_format",
                    LEVEL_WARN,
                    "参考文献缺少文献类型标识（如 [J] / [M] / [R] / [D]）。",
                    index,
                    text,
                )
            )
    return issues


def _check_mixed_punct(paragraphs: Sequence[Paragraph]) -> List[LintIssue]:
    """中文段落里夹英文标点（紧邻汉字）。"""
    issues: List[LintIssue] = []
    for paragraph, text in _content_paragraphs(paragraphs):
        if count_cjk(text) < 3:
            continue
        matches = list(_MIXED_PUNCT_RE.finditer(text))
        if not matches:
            continue
        found: List[str] = []
        for match in matches:
            for char in _ASCII_PUNCT_RE.findall(match.group(0)):
                if char not in found:
                    found.append(char)
        issues.append(
            LintIssue(
                "mixed_punct",
                LEVEL_WARN,
                "中文段落中混用英文标点「%s」，建议改用中文标点。"
                % "、".join(found),
                _paragraph_index(paragraph),
                _around(text, matches[0].start()),
            )
        )
    return issues


def _check_dup_words(paragraphs: Sequence[Paragraph]) -> List[LintIssue]:
    """连续重复字（的的、了了）或 2-4 字词连发（研究研究）。"""
    issues: List[LintIssue] = []
    for paragraph, text in _content_paragraphs(paragraphs):
        found: List[str] = []
        for match in _DUP_SINGLE_RE.finditer(text):
            char = match.group(1)
            if char in _LEGIT_REDUP_SINGLE:
                continue
            token = char * 2
            if token not in found:
                found.append(token)
        for match in _DUP_MULTI_RE.finditer(text):
            word = match.group(1)
            if len(set(word)) == 1:
                continue
            token = word * 2
            if token in _LEGIT_REDUP_MULTI:
                continue
            if token not in found:
                found.append(token)
            if len(found) >= 4:
                break
        for token in found[:4]:
            issues.append(
                LintIssue(
                    "dup_word",
                    LEVEL_WARN,
                    "连续重复用词「%s」，建议删除重复部分。" % token,
                    _paragraph_index(paragraph),
                    _around(text, text.find(token)),
                )
            )
    return issues


def _check_terms(paragraphs: Sequence[Paragraph]) -> List[LintIssue]:
    """中英术语混用（内置小词表，如 净资产收益率 / ROE）。"""
    issues: List[LintIssue] = []
    items: List[Tuple[Paragraph, str]] = []
    for paragraph in paragraphs:
        text = _paragraph_text(paragraph)
        if not text:
            continue
        if _is_reference(paragraph, text):
            continue
        items.append((paragraph, text))

    for chinese, english in _TERM_PAIRS:
        english_re = re.compile(
            r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(english)
        )
        first_chinese: Optional[Tuple[Paragraph, str, int]] = None
        first_english: Optional[Tuple[Paragraph, str, int]] = None
        for paragraph, text in items:
            if first_chinese is None:
                position = text.find(chinese)
                if position >= 0:
                    first_chinese = (paragraph, text, position)
            if first_english is None:
                match = english_re.search(text)
                if match:
                    first_english = (paragraph, text, match.start())
            if first_chinese is not None and first_english is not None:
                break
        if first_chinese is None or first_english is None:
            continue
        paragraph, text, position = first_english
        issues.append(
            LintIssue(
                "term_inconsistent",
                LEVEL_WARN,
                "术语中英混用：文中同时出现「%s」与「%s」，建议统一为其中一种表达。"
                % (chinese, english),
                _paragraph_index(paragraph),
                _around(text, position),
            )
        )
    return issues


def _check_long_sentences(paragraphs: Sequence[Paragraph]) -> List[LintIssue]:
    """单句汉字数超过 80 的段落，每段最多报 3 条。"""
    issues: List[LintIssue] = []
    for paragraph, text in _content_paragraphs(paragraphs):
        reported = 0
        for sentence in split_sentences(text):
            cjk_count = count_cjk(sentence)
            if cjk_count <= LONG_SENTENCE_LIMIT:
                continue
            issues.append(
                LintIssue(
                    "long_sentence",
                    LEVEL_INFO,
                    "单句汉字数 %d 超过 %d，建议拆分为多句。"
                    % (cjk_count, LONG_SENTENCE_LIMIT),
                    _paragraph_index(paragraph),
                    sentence,
                )
            )
            reported += 1
            if reported >= 3:
                break
    return issues


def _check_no_period(paragraphs: Sequence[Paragraph]) -> List[LintIssue]:
    """非标题、非表格行的段落末尾缺少句末标点。"""
    issues: List[LintIssue] = []
    for paragraph, text in _content_paragraphs(paragraphs):
        if _is_table_row(text):
            continue
        try:
            if is_furniture_like(text):
                continue
        except Exception:
            pass
        stripped = text.rstrip(_TRAILING_CLOSERS + " \t")
        if not stripped:
            continue
        if stripped[-1] in _END_PUNCT:
            continue
        issues.append(
            LintIssue(
                "no_period",
                LEVEL_INFO,
                "段落末尾缺少句末标点（建议以。！？等结尾）。",
                _paragraph_index(paragraph),
                text,
            )
        )
    return issues


def lint_document(doc: Document) -> LintReport:
    """对整篇文档做规范自检；空文档返回空报告，任何情况下都不抛异常。"""
    issues: List[LintIssue] = []
    try:
        paragraphs = list(getattr(doc, "paragraphs", None) or [])
    except Exception:
        paragraphs = []

    checkers = (
        _check_tables,
        _check_references,
        _check_mixed_punct,
        _check_dup_words,
        _check_terms,
        _check_long_sentences,
        _check_no_period,
    )
    for checker in checkers:
        try:
            issues.extend(checker(paragraphs))
        except Exception:
            # 单项检查失败不影响其他检查；自检必须给出结果而不是让整篇失败。
            continue

    try:
        issues = _sorted(_dedupe(issues))
    except Exception:
        issues = []
    return LintReport(issues=issues)


def to_meta(report: LintReport) -> dict:
    """转成可直接塞进报告 meta 的 dict：{"issues": [...], "counts": {...}}。"""
    issues: List[dict] = []
    counts: Dict[str, int] = {level: 0 for level in _LEVELS}
    try:
        source = list(getattr(report, "issues", None) or [])
    except Exception:
        source = []
    for issue in source:
        item = _issue_to_dict(issue)
        issues.append(item)
        level = item.get("level", LEVEL_INFO)
        if level not in counts:
            counts[level] = 0
        counts[level] += 1
    return {"issues": issues, "counts": counts}


__all__ = [
    "ISSUE_CODES",
    "LEVEL_ERROR",
    "LEVEL_INFO",
    "LEVEL_WARN",
    "LONG_SENTENCE_LIMIT",
    "LintIssue",
    "LintReport",
    "SNIPPET_LIMIT",
    "lint_document",
    "to_meta",
]

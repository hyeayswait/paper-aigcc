"""中文文本规范化与切分工具。无第三方依赖（环境无网络，不能安装 jieba）。"""
from __future__ import annotations

import re
from typing import List

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']*")
_DIGIT_RE = re.compile(r"\d+(?:\.\d+)?%?")

_SENT_END = "。！？!?；;…"
_CLAUSE_SEP = "，,、：:；;"
_CLOSERS = "”’」』）)】》\"'"

_SECTION_RE = re.compile(
    r"^(第[一二三四五六七八九十百零〇\d]+[章节篇部分回]"
    r"|[一二三四五六七八九十]+、"
    r"|\d+(\.\d+)*[ 、\.]"
    r"|[（(][一二三四五六七八九十\d]+[）)])"
)

_REF_START_PATTERNS = [
    r"^\[\d+\]\s*",
    r"^\d{1,3}[\.、]\s*[\u4e00-\u9fff]{2,4}\s*[,，\.]",
    r"^[\u4e00-\u9fff]{2,4}\s*(等)?\s*[\.．,，]\s*\d{4}",
    r"^\d{1,3}\s*[\u4e00-\u9fff]{2,4}\s*[\.．,，]\s*\d{4}",
]
_REF_MARKERS = re.compile(r"\[[JMDNRCSPG]\]|\[EB/OL\]|\[Z\]|\[A\]|doi\s*:|DOI\s*:")


def normalize_text(text: str) -> str:
    """统一空白、全角空格、零宽字符；去掉首尾空白。"""
    if not text:
        return ""
    t = text.replace("\u3000", " ").replace("\xa0", " ").replace("\t", " ")
    t = t.replace("\u200b", "").replace("\ufeff", "")
    t = re.sub(r"[ \r\n]+", " ", t)
    return t.strip()


def is_cjk(ch: str) -> bool:
    return bool(ch) and bool(_CJK_RE.match(ch))


def count_cjk(text: str) -> int:
    return len(_CJK_RE.findall(text or ""))


def count_words(text: str) -> int:
    """中文字数口径：汉字数 + 英文单词数 + 数字串数。"""
    if not text:
        return 0
    return count_cjk(text) + len(_EN_WORD_RE.findall(text)) + len(_DIGIT_RE.findall(text))


def tokenize_words(text: str) -> List[str]:
    """无分词器版 token 化：汉字单字 + 小写英文词 + 数字串。

    用于 TTR / 熵 / 重复率等统计，不追求语言学正确性。
    """
    if not text:
        return []
    tokens: List[str] = []
    for m in re.finditer(r"[\u4e00-\u9fff]|[A-Za-z][A-Za-z0-9\-']*|\d+(?:\.\d+)?", text):
        tokens.append(m.group(0).lower())
    return tokens


def split_sentences(text: str) -> List[str]:
    """按句末标点切句，自动吞掉连续的句末标点和后引号/括号。"""
    text = normalize_text(text)
    if not text:
        return []
    out: List[str] = []
    buf: List[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        buf.append(ch)
        if ch in _SENT_END:
            while i + 1 < n and (text[i + 1] in _SENT_END or text[i + 1] in _CLOSERS):
                i += 1
                buf.append(text[i])
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def split_clauses(text: str) -> List[str]:
    """按逗号/顿号/分号/冒号/句末标点切分句。"""
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"[，,、：:；;。！？!?…]+", text)
    return [p.strip() for p in parts if p and p.strip()]


def char_ngrams(text: str, n: int = 2) -> List[str]:
    """只取汉字序列做 n-gram。"""
    seq = "".join(_CJK_RE.findall(text or ""))
    if n <= 0 or len(seq) < n:
        return []
    return [seq[i : i + n] for i in range(len(seq) - n + 1)]


def is_heading_like(text: str, style: str = "") -> bool:
    s = (style or "").lower()
    if s.startswith("heading") or "标题" in s or s in ("title", "subtitle"):
        return True
    t = (text or "").strip()
    if not t or len(t) > 40:
        return False
    if any(c in t for c in "。！？!?"):
        return False
    return bool(_SECTION_RE.match(t))


_CAPTION_RE = re.compile(r"^(表|图|附表|附图|Table|Figure)\s*[0-9一二三四五六七八九十]+")
_KEYWORD_RE = re.compile(r"^(关键词|关键字|Key\s*words|Keywords)\s*[：:]", re.IGNORECASE)
_NOTE_RE = re.compile(r"^(注|资料来源|数据来源|单位|说明)\s*[：:]")
_AUTHOR_RE = re.compile(r"(指导教师|指导老师|作者简介|学号|班级|院系|专业名称)")


def is_furniture_like(text: str) -> bool:
    """识别图表题、关键词行、署名/班级、文档短标题等非正文行。

    这些行不该参与评分，否则会被当成很短的"段落"污染分数分布。
    """
    t = normalize_text(text)
    if not t:
        return True
    # 图表题/注释必须是「短行且没有句末标点」；否则像「表4里，营收从……」这类
    # 以表号开头的正文会被误判成图表题而漏掉评分。
    if len(t) <= 45 and not any(c in t for c in "。！？!?"):
        if _CAPTION_RE.match(t):
            return True
        if _NOTE_RE.match(t):
            return True
    if _KEYWORD_RE.match(t):
        return True
    if len(t) <= 60 and _AUTHOR_RE.search(t):
        return True
    if len(t) <= 25 and not any(c in t for c in "。！？!?；;，,：:"):
        return True
    return False


def is_reference_like(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _REF_MARKERS.search(t):
        return True
    for pat in _REF_START_PATTERNS:
        if re.search(pat, t):
            return True
    return False

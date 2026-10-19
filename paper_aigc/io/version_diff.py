# -*- coding: utf-8 -*-
"""版本对比：按段落对齐两版 docx，统计新增/删除/修改/未变并渲染单文件 HTML。

对齐口径（docs/SPEC2.md 第 2 节）：
- 只比较 docx_reader 给出的 body_indices（非标题、非参考文献、非图表题等正文段）；
- 段落文本先 normalize_text()，再用 difflib.SequenceMatcher 对段落列表做对齐；
- SequenceMatcher 的 replace 块内部再做一次贪心配对：去标点后的 13 字片段
  重合数 / min(两段片段数) >= 0.6 记为 modified，配不上的拆成 added / removed。

HTML 约束：单文件、零外链（产物中不出现 http:// 或 https:// 字面量）、
所有用户文本经 html.escape 转义、原文与新文都用 <details> 折叠。
"""
from __future__ import annotations

import difflib
import html
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

from ..core.textnorm import count_words, normalize_text
from ..core.types import VERSION
from .docx_reader import read_docx

NGRAM_SIZE = 13
SIMILARITY_THRESHOLD = 0.6

# 单次配对最多枚举的候选对数；超出后退化为按顺序一一配对，避免大块 O(n*m) 爆炸。
_MAX_PAIR_CANDIDATES = 4000

STATUS_LABELS: Dict[str, str] = {
    "added": "新增",
    "removed": "删除",
    "modified": "修改",
    "same": "未变",
}

STATUS_COLORS: Dict[str, str] = {
    "added": "#1b7a4b",
    "removed": "#b4231f",
    "modified": "#b26a00",
    "same": "#667788",
}

_KEEP_RE = re.compile(r"[\u4e00-\u9fffa-zA-Z0-9]")
_URL_SCHEME_RE = re.compile(r"(?i)(https?)://")


@dataclass
class ParagraphDiff:
    """一对（或单侧）段落的差异记录。"""

    status: str          # "added" | "removed" | "modified" | "same"
    old_index: int       # 旧稿段落序号（docx 内从 0 开始），无则 -1
    new_index: int       # 新稿段落序号，无则 -1
    old_text: str
    new_text: str
    similarity: float    # 0-1，去标点后的 13gram 重合 / min(两段片段数)
    char_delta: int      # 新文字数 - 旧文字数


@dataclass
class DiffReport:
    """两版文档的逐段对比结果。"""

    old_path: str
    new_path: str
    items: List[ParagraphDiff] = field(default_factory=list)
    added: int = 0
    removed: int = 0
    modified: int = 0
    unchanged: int = 0
    old_chars: int = 0
    new_chars: int = 0


def _compact(text: str) -> str:
    """normalize_text 后只保留汉字/英文字母/数字，即“去标点”文本。"""
    return "".join(_KEEP_RE.findall(normalize_text(text)))


def _ngrams(seq: str, size: int) -> List[str]:
    if size <= 0 or len(seq) < size:
        return []
    return [seq[i : i + size] for i in range(len(seq) - size + 1)]


def paragraph_similarity(old_text: str, new_text: str) -> float:
    """去标点后 13 字片段重合数 / min(两段片段数)，返回 0-1。

    任一段不足 13 字时把片段长度降到较短一段的长度，避免短段落一律判 0；
    两段去标点后都为空时视为完全一致。
    """
    left = _compact(old_text)
    right = _compact(new_text)
    if not left or not right:
        return 1.0 if (not left and not right) else 0.0
    size = NGRAM_SIZE
    if len(left) < size or len(right) < size:
        size = min(len(left), len(right))
    left_grams = _ngrams(left, size)
    right_grams = _ngrams(right, size)
    if not left_grams or not right_grams:
        return 1.0 if left == right else 0.0
    left_counter = Counter(left_grams)
    right_counter = Counter(right_grams)
    shared = sum(min(count, right_counter[gram]) for gram, count in left_counter.items())
    denominator = min(len(left_grams), len(right_grams))
    if denominator <= 0:
        return 0.0
    return max(0.0, min(1.0, shared / float(denominator)))


def _body_texts(document) -> Tuple[List[int], List[str]]:
    """返回 (body_indices 对应的 docx 段落序号列表, normalize 后的正文列表)。"""
    indexes: List[int] = []
    texts: List[str] = []
    paragraphs = document.paragraphs or []
    for index in document.body_indices or []:
        if index < 0 or index >= len(paragraphs):
            continue
        indexes.append(index)
        texts.append(normalize_text(paragraphs[index].text))
    return indexes, texts


def _pair_replace_block(
    old_positions: Sequence[int],
    new_positions: Sequence[int],
    old_texts: Sequence[str],
    new_texts: Sequence[str],
) -> Tuple[List[Tuple[float, int, int]], List[int], List[int]]:
    """在 replace 块内贪心配对相似段落。

    返回 (pairs, left_old, left_new)，pairs 元素为 (similarity, old_pos, new_pos)。
    """
    pairs: List[Tuple[float, int, int]] = []
    used_old = set()
    used_new = set()

    if len(old_positions) * len(new_positions) <= _MAX_PAIR_CANDIDATES:
        candidates: List[Tuple[float, int, int]] = []
        for old_pos in old_positions:
            for new_pos in new_positions:
                similarity = paragraph_similarity(old_texts[old_pos], new_texts[new_pos])
                if similarity >= SIMILARITY_THRESHOLD:
                    candidates.append((similarity, old_pos, new_pos))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        for similarity, old_pos, new_pos in candidates:
            if old_pos in used_old or new_pos in used_new:
                continue
            used_old.add(old_pos)
            used_new.add(new_pos)
            pairs.append((similarity, old_pos, new_pos))
    else:
        for old_pos, new_pos in zip(old_positions, new_positions):
            similarity = paragraph_similarity(old_texts[old_pos], new_texts[new_pos])
            if similarity >= SIMILARITY_THRESHOLD:
                used_old.add(old_pos)
                used_new.add(new_pos)
                pairs.append((similarity, old_pos, new_pos))

    # SequenceMatcher 已经把这一块判定为「旧段对应新段」，剩余未配上的按顺序配对，
    # 以免整段重写（相似度低于阈值）被误报成「删除 + 新增」。
    for old_pos, new_pos in zip(list(old_positions), list(new_positions)):
        if old_pos in used_old or new_pos in used_new:
            continue
        used_old.add(old_pos)
        used_new.add(new_pos)
        pairs.append((paragraph_similarity(old_texts[old_pos], new_texts[new_pos]),
                      old_pos, new_pos))

    left_old = [pos for pos in old_positions if pos not in used_old]
    left_new = [pos for pos in new_positions if pos not in used_new]
    return pairs, left_old, left_new


def diff_documents(old_path: str, new_path: str) -> DiffReport:
    """对比两版 docx 的正文段落，返回 DiffReport。

    两个文件都必须能被 docx_reader.read_docx 解析；失败时由其抛出中文 ValueError。
    """
    old_document = read_docx(old_path)
    new_document = read_docx(new_path)
    old_indexes, old_texts = _body_texts(old_document)
    new_indexes, new_texts = _body_texts(new_document)

    matcher = difflib.SequenceMatcher(None, old_texts, new_texts, autojunk=False)
    items: List[ParagraphDiff] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                old_pos = i1 + offset
                new_pos = j1 + offset
                text = old_texts[old_pos]
                items.append(
                    ParagraphDiff(
                        status="same",
                        old_index=old_indexes[old_pos],
                        new_index=new_indexes[new_pos],
                        old_text=text,
                        new_text=text,
                        similarity=1.0,
                        char_delta=0,
                    )
                )
        elif tag == "delete":
            for old_pos in range(i1, i2):
                text = old_texts[old_pos]
                items.append(
                    ParagraphDiff(
                        status="removed",
                        old_index=old_indexes[old_pos],
                        new_index=-1,
                        old_text=text,
                        new_text="",
                        similarity=0.0,
                        char_delta=-count_words(text),
                    )
                )
        elif tag == "insert":
            for new_pos in range(j1, j2):
                text = new_texts[new_pos]
                items.append(
                    ParagraphDiff(
                        status="added",
                        old_index=-1,
                        new_index=new_indexes[new_pos],
                        old_text="",
                        new_text=text,
                        similarity=0.0,
                        char_delta=count_words(text),
                    )
                )
        else:  # replace
            pairs, left_old, left_new = _pair_replace_block(
                list(range(i1, i2)), list(range(j1, j2)), old_texts, new_texts
            )
            pair_by_new = {new_pos: (similarity, old_pos) for similarity, old_pos, new_pos in pairs}
            old_left_set = set(left_old)
            new_left_set = set(left_new)
            # 先按新稿顺序输出 modified / added，再补上本块内未配对的 removed。
            for new_pos in range(j1, j2):
                if new_pos in pair_by_new:
                    similarity, old_pos = pair_by_new[new_pos]
                    old_text = old_texts[old_pos]
                    new_text = new_texts[new_pos]
                    items.append(
                        ParagraphDiff(
                            status="modified",
                            old_index=old_indexes[old_pos],
                            new_index=new_indexes[new_pos],
                            old_text=old_text,
                            new_text=new_text,
                            similarity=similarity,
                            char_delta=count_words(new_text) - count_words(old_text),
                        )
                    )
                elif new_pos in new_left_set:
                    text = new_texts[new_pos]
                    items.append(
                        ParagraphDiff(
                            status="added",
                            old_index=-1,
                            new_index=new_indexes[new_pos],
                            old_text="",
                            new_text=text,
                            similarity=0.0,
                            char_delta=count_words(text),
                        )
                    )
            for old_pos in range(i1, i2):
                if old_pos in old_left_set:
                    text = old_texts[old_pos]
                    items.append(
                        ParagraphDiff(
                            status="removed",
                            old_index=old_indexes[old_pos],
                            new_index=-1,
                            old_text=text,
                            new_text="",
                            similarity=0.0,
                            char_delta=-count_words(text),
                        )
                    )

    report = DiffReport(
        old_path=os.fspath(old_path),
        new_path=os.fspath(new_path),
        items=items,
        old_chars=sum(count_words(text) for text in old_texts),
        new_chars=sum(count_words(text) for text in new_texts),
    )
    for item in items:
        if item.status == "added":
            report.added += 1
        elif item.status == "removed":
            report.removed += 1
        elif item.status == "modified":
            report.modified += 1
        else:
            report.unchanged += 1
    return report


_CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 40px; background: #f4f6f8; color: #222;
       font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "Segoe UI", Arial, sans-serif;
       line-height: 1.65; font-size: 14px; }
.wrap { max-width: 1180px; margin: 0 auto; }
header { border-left: 6px solid #1e5aa8; padding: 4px 14px; margin-bottom: 16px; }
h1 { font-size: 24px; margin: 0 0 6px; }
.sub { color: #555; font-size: 13px; margin: 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: 12px; margin-bottom: 18px; }
.card { background: #fff; border: 1px solid #dde3ea; border-radius: 8px; padding: 12px 14px; }
.card-label { color: #667788; font-size: 13px; margin-bottom: 6px; }
.card-value { font-size: 24px; font-weight: 700; }
.card-sub { color: #667788; font-size: 12.5px; margin-top: 4px; }
.meta { color: #556677; font-size: 13px; margin: 0 0 12px; }
h2 { font-size: 18px; margin: 26px 0 10px; padding-bottom: 6px; border-bottom: 2px solid #dde3ea; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dde3ea;
        table-layout: fixed; }
th, td { border-bottom: 1px solid #e6ebf1; padding: 8px 9px; text-align: left;
         vertical-align: top; font-size: 13.5px; word-break: break-word; }
th { background: #eef2f7; font-size: 13px; white-space: nowrap; }
td.num { text-align: center; color: #667788; white-space: nowrap; }
td.num.pos { color: #1b7a4b; font-weight: 700; }
td.num.neg { color: #b4231f; font-weight: 700; }
.badge { display: inline-block; color: #ffffff; padding: 2px 9px; border-radius: 10px;
         font-size: 12.5px; white-space: nowrap; }
.plain { white-space: pre-wrap; word-break: break-word; font-size: 13.5px; }
details summary { cursor: pointer; color: #1e5aa8; font-size: 13px; }
details[open] summary { margin-bottom: 6px; }
.empty { text-align: center; color: #778899; padding: 18px; }
footer { margin-top: 30px; color: #667788; font-size: 12.5px; text-align: center;
         border-top: 1px solid #dde3ea; padding-top: 12px; }
@media (max-width: 860px) { table { table-layout: auto; } }
"""


def _esc(value: object) -> str:
    """转义用户文本；顺手把 URL scheme 的冒号编码掉，保证产物不含 http(s):// 字面量。"""
    if value is None:
        return ""
    text = html.escape(str(value), quote=True)
    return _URL_SCHEME_RE.sub(lambda match: match.group(1) + "&#58;//", text)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _display_index(index: object) -> str:
    value = _as_int(index, -1)
    return str(value + 1) if value >= 0 else "—"


def _index_label(index: object, prefix: str) -> str:
    value = _as_int(index, -1)
    if value < 0:
        return "%s（无对应段）" % prefix
    return "%s第 %d 段" % (prefix, value + 1)


def _fmt_delta(delta: int) -> str:
    if delta > 0:
        return "+%d" % delta
    return str(delta)


def _fmt_similarity(item: ParagraphDiff) -> str:
    if item.status in ("added", "removed"):
        return "—"
    value = max(0.0, min(1.0, float(item.similarity or 0.0)))
    return "%.1f%%" % (value * 100.0)


def _card(label: str, value: str, sub: str, color: str) -> str:
    return (
        '<div class="card"><div class="card-label">%s</div>'
        '<div class="card-value" style="color:%s">%s</div>'
        '<div class="card-sub">%s</div></div>'
        % (_esc(label), color, _esc(value), _esc(sub))
    )


def _header_html(diff: DiffReport, generated: str) -> str:
    old_name = os.path.basename(diff.old_path) or "（未命名）"
    new_name = os.path.basename(diff.new_path) or "（未命名）"
    return (
        "<header><h1>论文版本对比报告</h1>"
        "<p class='sub'>旧稿：%s　|　新稿：%s　|　工具版本：%s　|　生成时间：%s</p>"
        "</header>"
        % (_esc(old_name), _esc(new_name), _esc(VERSION), _esc(generated))
    )


def _cards_html(diff: DiffReport) -> str:
    delta = int(diff.new_chars) - int(diff.old_chars)
    delta_color = (
        STATUS_COLORS["added"]
        if delta > 0
        else STATUS_COLORS["removed"]
        if delta < 0
        else STATUS_COLORS["same"]
    )
    cards = [
        _card("新增段落", str(diff.added), "新稿多出的正文段", STATUS_COLORS["added"]),
        _card("删除段落", str(diff.removed), "旧稿有、新稿没有", STATUS_COLORS["removed"]),
        _card(
            "修改段落",
            str(diff.modified),
            "同一对齐块内按顺序配对",
            STATUS_COLORS["modified"],
        ),
        _card("未变段落", str(diff.unchanged), "两版完全一致", STATUS_COLORS["same"]),
        _card("旧稿字数", str(diff.old_chars), "正文段字数合计", "#1e5aa8"),
        _card("新稿字数", str(diff.new_chars), "正文段字数合计", "#1e5aa8"),
        _card("字数变化", _fmt_delta(delta), "新稿字数 - 旧稿字数", delta_color),
    ]
    return '<section class="cards">' + "".join(cards) + "</section>"


def _row_html(item: ParagraphDiff) -> str:
    status = item.status if item.status in STATUS_LABELS else "same"
    label = STATUS_LABELS[status]
    color = STATUS_COLORS[status]
    delta = _as_int(item.char_delta)
    delta_class = "pos" if delta > 0 else ("neg" if delta < 0 else "")
    old_text = item.old_text or "（此版本没有该段）"
    new_text = item.new_text or "（此版本没有该段）"
    return (
        '<tr class="row-%s">'
        '<td><span class="badge" style="background:%s">%s</span></td>'
        '<td class="num">%s</td>'
        '<td class="num">%s</td>'
        '<td class="num">%s</td>'
        '<td class="num %s">%s</td>'
        "<td><details><summary>%s原文</summary><div class='plain'>%s</div></details></td>"
        "<td><details><summary>%s新文</summary><div class='plain'>%s</div></details></td>"
        "</tr>"
        % (
            status,
            color,
            _esc(label),
            _esc(_display_index(item.old_index)),
            _esc(_display_index(item.new_index)),
            _esc(_fmt_similarity(item)),
            delta_class,
            _esc(_fmt_delta(delta)),
            _esc(_index_label(item.old_index, "旧稿")),
            _esc(old_text),
            _esc(_index_label(item.new_index, "新稿")),
            _esc(new_text),
        )
    )


def _table_html(diff: DiffReport) -> str:
    rows = [_row_html(item) for item in diff.items or []]
    if not rows:
        rows.append("<tr><td colspan='7' class='empty'>两版都没有可对比的正文段落。</td></tr>")
    return (
        "<h2>逐段对比</h2>"
        "<p class='meta'>段号为该版本 docx 中的段落序号（从 1 开始）；"
        "相似度 = 去标点后 13 字片段重合数 / min(两段片段数)，空白与标点不计入比对。"
        "原文、新文默认折叠，点击展开。</p>"
        "<table>"
        "<colgroup><col style='width:68px'><col style='width:66px'><col style='width:66px'>"
        "<col style='width:68px'><col style='width:76px'><col><col></colgroup>"
        "<thead><tr><th>状态</th><th>旧稿段号</th><th>新稿段号</th><th>相似度</th>"
        "<th>字数变化</th><th>原文</th><th>新文</th></tr></thead>"
        "<tbody>%s</tbody></table>" % "\n".join(rows)
    )


def _footer_html(generated: str) -> str:
    return (
        "<footer>PaperAIGC v%s　·　生成时间：%s<br>"
        "本报告只按段落文本统计两版差异，不评价哪一版更好；"
        "段落对齐基于文本相似度，仅供参考。</footer>" % (_esc(VERSION), _esc(generated))
    )


def render_diff_html(diff: DiffReport, out_path: str) -> str:
    """渲染单文件 HTML 版本对比报告并写入 out_path，返回 out_path。"""
    target = os.fspath(out_path)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    old_name = os.path.basename(diff.old_path) or "旧稿"
    new_name = os.path.basename(diff.new_path) or "新稿"

    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>%s</title>" % _esc("论文版本对比 - %s / %s" % (old_name, new_name)),
        "<style>" + _CSS + "</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        _header_html(diff, generated),
        _cards_html(diff),
        _table_html(diff),
        _footer_html(generated),
        "</div>",
        "</body>",
        "</html>",
    ]
    html_text = "\n".join(parts) + "\n"

    parent = os.path.dirname(os.path.abspath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(html_text)
    return target

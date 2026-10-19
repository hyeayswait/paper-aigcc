"""PDF 解析：把 PDF 论文转换成共享契约中的 Document。

实现要点（对照 docs/SPEC2.md 第 1 节）：
- 依赖 vendor/ 里的 pymupdf（PyMuPDF），导入前先 ensure_vendor()；
- 逐页取 get_text("blocks") 的文本块，按空行与块边界切段；
- 去掉各页重复出现的页眉 / 页脚行，以及纯页码行；
- 段落分类复用 core.textnorm：heading / furniture / reference；
- body_indices 规则与 docx_reader 完全一致：
  非标题、非参考文献、非图表题且字数 >= BODY_MIN_CHARS。
"""
from __future__ import annotations

import os
import re
from typing import Iterator, List, Set

from .._vendor import ensure_vendor

ensure_vendor()

try:  # vendor/pymupdf 缺失时降级，不阻断 paper_aigc.io 其他模块
    import fitz  # type: ignore
    _FITZ_IMPORT_ERROR = ""
except Exception as exc:  # noqa: BLE001
    fitz = None  # type: ignore
    _FITZ_IMPORT_ERROR = str(exc)

from ..core.textnorm import (
    count_words,
    is_furniture_like,
    is_heading_like,
    is_reference_like,
    normalize_text,
)
from ..core.types import Document, Paragraph

BODY_MIN_CHARS = 10

# 目录行（标题后跟页码）不当作文档标题，与 docx_reader 一致。
_TOC_TAIL_RE = re.compile(r"[\s\.·…]+\d{1,3}$")

# 纯页码行：2 / - 2 - / 第 2 页 / 2/10 / Page 2 / 罗马数字页码。
_PAGE_NUMBER_RE = re.compile(
    r"^(?:"
    r"[-—–_·.\s]*\d{1,4}[-—–_·.\s]*"
    r"|第\s*\d{1,4}\s*页(?:\s*[/／]\s*\d{1,4}\s*页?)?"
    r"|\d{1,4}\s*[/／]\s*\d{1,4}"
    r"|page\s*\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?"
    r"|[-—–_·.\s]*[ivxlcdm]{1,7}[-—–_·.\s]*"
    r")$",
    re.IGNORECASE,
)

# 判定页眉页脚时，每页只看最前 / 最后这么多个非空行。
_EDGE_LINE_LIMIT = 2


def is_pdf_supported() -> bool:
    """pymupdf 是否可用（vendor/ 已正确安装）。"""
    return fitz is not None


def _validate_pdf_path(path: str) -> str:
    """校验路径存在且扩展名为 .pdf；不合法统一抛中文 ValueError。"""
    file_path = os.fspath(path)
    if not os.path.isfile(file_path):
        raise ValueError("文件不存在或不是文件：%s" % file_path)
    if os.path.splitext(file_path)[1].lower() != ".pdf":
        raise ValueError("仅支持 .pdf 格式的 PDF 文档：%s" % os.path.basename(file_path))
    return file_path


def _extract_pages(pdf_document) -> List[List[List[str]]]:
    """抽取逐页文本块。

    返回 pages[page][block] = 行列表；块内空行保留为 "" 以便后续切段。
    只收文本块（block_type == 0），图片块丢弃。
    """
    pages: List[List[List[str]]] = []
    for page in pdf_document:
        try:
            blocks = list(page.get_text("blocks") or [])
        except Exception:
            blocks = []
        # 按阅读顺序（先纵坐标、再横坐标）排序，保证页眉页脚在首尾。
        try:
            blocks.sort(key=lambda block: (round(float(block[1]), 1), float(block[0])))
        except Exception:
            pass
        page_blocks: List[List[str]] = []
        for block in blocks:
            # block = (x0, y0, x1, y1, text, block_no, block_type)
            if len(block) >= 7 and block[6] != 0:
                continue
            raw = block[4] if len(block) > 4 else ""
            if not isinstance(raw, str) or not raw.strip():
                continue
            lines = [normalize_text(line) for line in raw.splitlines()]
            while lines and not lines[0]:
                lines.pop(0)
            while lines and not lines[-1]:
                lines.pop()
            if lines:
                page_blocks.append(lines)
        if page_blocks:
            pages.append(page_blocks)
    return pages


def _collect_edge_lines(pages: List[List[List[str]]]) -> Set[str]:
    """找出在多页首 / 尾重复出现的行（页眉、页脚）。

    单页文档无法判断重复，直接返回空集合；纯页码行由 _PAGE_NUMBER_RE 单独剔除。
    """
    if len(pages) < 2:
        return set()
    top_counts = {}
    bottom_counts = {}
    for blocks in pages:
        lines = [line for block in blocks for line in block if line]
        if not lines:
            continue
        for line in dict.fromkeys(lines[:_EDGE_LINE_LIMIT]):
            top_counts[line] = top_counts.get(line, 0) + 1
        for line in dict.fromkeys(lines[-_EDGE_LINE_LIMIT:]):
            bottom_counts[line] = bottom_counts.get(line, 0) + 1
    repeated: Set[str] = set()
    for line, count in top_counts.items():
        if count >= 2:
            repeated.add(line)
    for line, count in bottom_counts.items():
        if count >= 2:
            repeated.add(line)
    return repeated


def _is_ascii_word_char(char: str) -> bool:
    return bool(char) and char.isascii() and char.isalnum()


def _join_lines(lines: List[str]) -> str:
    """把 PDF 块内硬折行的多行合并成一段。

    中文换行直接拼接，英文单词之间补空格，行尾连字符断词做去连字符处理。
    """
    merged = ""
    for line in lines:
        line = (line or "").strip()
        if not line:
            continue
        if not merged:
            merged = line
            continue
        if merged.endswith("-") and line[:1].isascii() and line[:1].isalpha():
            merged = merged[:-1] + line
        elif _is_ascii_word_char(merged[-1]) and _is_ascii_word_char(line[0]):
            merged = merged + " " + line
        else:
            merged = merged + line
    return normalize_text(merged)


def _is_dropped_line(line: str, repeated_lines: Set[str]) -> bool:
    if not line:
        return False
    if line in repeated_lines:
        return True
    return bool(_PAGE_NUMBER_RE.match(line))


def _iter_paragraph_texts(
    pages: List[List[List[str]]], repeated_lines: Set[str]
) -> Iterator[str]:
    """按块边界与块内空行切段，跳过页眉 / 页脚 / 页码行。"""
    for blocks in pages:
        for block in blocks:
            chunk: List[str] = []
            for line in block:
                if not line:  # 块内空行 = 段落边界
                    text = _join_lines(chunk)
                    if text:
                        yield text
                    chunk = []
                    continue
                if _is_dropped_line(line, repeated_lines):
                    continue
                chunk.append(line)
            text = _join_lines(chunk)
            if text:
                yield text


def read_pdf(path: str) -> Document:
    """解析 PDF 并返回 Document。

    文件不存在 / 非 .pdf / pymupdf 不可用 / 已加密或损坏时抛中文 ValueError。
    """
    file_path = _validate_pdf_path(path)
    file_name = os.path.basename(file_path)

    if fitz is None:
        detail = ("；导入错误：%s" % _FITZ_IMPORT_ERROR) if _FITZ_IMPORT_ERROR else ""
        raise ValueError(
            "未找到 pymupdf 依赖（vendor/ 目录不可用），无法解析 PDF：%s%s"
            % (file_name, detail)
        )

    try:
        pdf_document = fitz.open(file_path)
    except Exception as exc:
        raise ValueError(
            "无法解析该 .pdf 文件（可能已损坏或不是 PDF 文档）：%s" % file_name
        ) from exc

    try:
        if getattr(pdf_document, "needs_pass", False):
            raise ValueError("该 PDF 已加密，需要密码才能解析：%s" % file_name)
        try:
            pages = _extract_pages(pdf_document)
        except Exception as exc:
            raise ValueError(
                "无法解析该 .pdf 文件的文本内容（可能已损坏或为扫描件）：%s" % file_name
            ) from exc
    finally:
        try:
            pdf_document.close()
        except Exception:
            pass

    repeated_lines = _collect_edge_lines(pages)

    paragraphs: List[Paragraph] = []
    body_indices: List[int] = []
    title = ""

    for text in _iter_paragraph_texts(pages, repeated_lines):
        index = len(paragraphs)
        heading = is_heading_like(text, "")
        reference = is_reference_like(text)
        furniture = is_furniture_like(text)
        char_count = count_words(text)
        paragraphs.append(
            Paragraph(
                index=index,
                text=text,
                style="",
                is_heading=heading,
                is_reference=reference,
                char_count=char_count,
            )
        )
        if heading and not title and not _TOC_TAIL_RE.search(text):
            title = text
        if (not heading) and (not reference) and (not furniture) and char_count >= BODY_MIN_CHARS:
            body_indices.append(index)

    if not title:
        title = os.path.splitext(file_name)[0]

    total_words = sum(paragraphs[i].char_count for i in body_indices)
    return Document(
        path=file_path,
        title=title,
        paragraphs=paragraphs,
        body_indices=body_indices,
        total_words=total_words,
    )

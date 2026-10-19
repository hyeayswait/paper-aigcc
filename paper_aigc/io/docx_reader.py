"""docx 解析：把 Word 文档转换成共享契约中的 Document。

收录规则（与 docx_writer 完全一致，保证段落 index 一一对应）：
- 只用 document.paragraphs，表格 / 文本框 / 页眉页脚不处理；
- 只收录 normalize_text() 后非空的段落，index 按收录顺序从 0 递增；
- body_indices = 非标题、非参考文献且字数 >= BODY_MIN_CHARS 的段落 index。
"""
from __future__ import annotations

import os
import re
from typing import List, Tuple

import docx

from ..core.textnorm import (
    count_words,
    is_furniture_like,
    is_heading_like,
    is_reference_like,
    normalize_text,
)
from ..core.types import Document, Paragraph

BODY_MIN_CHARS = 10

# 目录行（标题后跟页码）不当作文档标题
_TOC_TAIL_RE = re.compile(r"[\s\.·…]+\d{1,3}$")


def _style_name(paragraph) -> str:
    try:
        return paragraph.style.name or ""
    except Exception:
        return ""


def collect_paragraphs(docx_document) -> List[Tuple[object, str, str]]:
    """按收录规则返回 [(paragraph, 规范化文本, style_name)]。

    docx_reader 与 docx_writer 共用该函数，避免两处规则漂移。
    """
    items: List[Tuple[object, str, str]] = []
    for paragraph in docx_document.paragraphs:
        text = normalize_text(paragraph.text)
        if not text:
            continue
        items.append((paragraph, text, _style_name(paragraph)))
    return items


def _open_docx(path: str):
    """校验路径并打开 docx；任何不合法情况统一抛中文 ValueError。"""
    file_path = os.fspath(path)
    if not os.path.isfile(file_path):
        raise ValueError("文件不存在或不是文件：%s" % file_path)
    if os.path.splitext(file_path)[1].lower() != ".docx":
        raise ValueError("仅支持 .docx 格式的 Word 文档：%s" % os.path.basename(file_path))
    try:
        return docx.Document(file_path)
    except Exception as exc:
        raise ValueError(
            "无法解析该 .docx 文件（可能已损坏或不是 Word 文档）：%s"
            % os.path.basename(file_path)
        ) from exc


def read_docx(path: str) -> Document:
    """解析 docx 并返回 Document；文件不存在或非 docx 抛中文 ValueError。"""
    docx_document = _open_docx(path)
    items = collect_paragraphs(docx_document)

    paragraphs: List[Paragraph] = []
    body_indices: List[int] = []
    title = ""

    for index, (_paragraph, text, style) in enumerate(items):
        heading = is_heading_like(text, style)
        reference = is_reference_like(text)
        furniture = is_furniture_like(text)
        char_count = count_words(text)
        paragraph = Paragraph(
            index=index,
            text=text,
            style=style,
            is_heading=heading,
            is_reference=reference,
            char_count=char_count,
        )
        paragraphs.append(paragraph)
        if heading and not title and not _TOC_TAIL_RE.search(text):
            title = text
        if (not heading) and (not reference) and (not furniture) and char_count >= BODY_MIN_CHARS:
            body_indices.append(index)

    if not title:
        for paragraph in paragraphs:
            if paragraph.is_heading or paragraph.is_reference:
                continue
            body = paragraph.text
            if len(body) <= 40 and not any(c in body for c in "。！？!?；;，,：:"):
                title = body
                break
    if not title:
        title = os.path.splitext(os.path.basename(os.fspath(path)))[0]

    total_words = sum(paragraphs[i].char_count for i in body_indices)
    return Document(
        path=os.fspath(path),
        title=title,
        paragraphs=paragraphs,
        body_indices=body_indices,
        total_words=total_words,
    )

"""统一文档加载入口：按扩展名把 .docx / .pdf / .txt / .md 解析成 Document。

- .docx -> docx_reader.read_docx（保留段落 style，判定规则与 docx 完全一致）
- .pdf  -> pdf_reader.read_pdf（pymupdf；不可用时由 read_pdf 抛中文 ValueError）
- .txt / .md -> read_text（整篇按空行切段，复用 textnorm 的 heading/furniture/reference 规则）

所有解析错误统一抛中文 ValueError，方便上层直接展示给用户。
"""
from __future__ import annotations

import os
import re
from typing import List

from ..core.textnorm import (
    count_words,
    is_furniture_like,
    is_heading_like,
    is_reference_like,
    normalize_text,
)
from ..core.types import Document, Paragraph
from .docx_reader import BODY_MIN_CHARS, read_docx
from .pdf_reader import read_pdf

SUPPORTED_EXTENSIONS: List[str] = [".docx", ".pdf", ".txt", ".md"]

# 纯文本优先按 UTF-8（含 BOM）解码，失败再按 GB18030（兼容 GBK/GB2312）。
_TEXT_ENCODINGS = ("utf-8-sig", "gb18030")

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_BLANK_LINE_RE = re.compile(r"(?:\r?\n)[ \t\u3000]*(?:\r?\n)+")


def supported_extensions() -> list:
    """返回支持的文件扩展名（小写、带点）的新副本。"""
    return list(SUPPORTED_EXTENSIONS)


def _validate_path(path: str) -> str:
    file_path = os.fspath(path)
    if not os.path.isfile(file_path):
        raise ValueError("文件不存在或不是文件：%s" % file_path)
    return file_path


def _read_text_file(file_path: str) -> str:
    """读取文本文件，自动尝试 UTF-8 / GB18030；失败抛中文 ValueError。"""
    try:
        with open(file_path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise ValueError("无法读取文本文件：%s" % os.path.basename(file_path)) from exc
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        encodings = ("utf-16",)
    else:
        encodings = _TEXT_ENCODINGS
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(
        "无法解析该文本文件（编码不受支持，请另存为 UTF-8）：%s"
        % os.path.basename(file_path)
    )


def _split_text_chunks(text: str) -> List[str]:
    """整篇按空行切段（保留段内单换行，交给 _join_lines 处理硬折行）。"""
    if not text:
        return []
    normalized = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    return [chunk for chunk in _BLANK_LINE_RE.split(normalized) if chunk.strip()]


def _is_ascii_word_char(char: str) -> bool:
    return bool(char) and char.isascii() and char.isalnum()


def _join_lines(lines: List[str]) -> str:
    """合并段内硬折行：中文直接拼接，英文单词间补空格，行尾连字符断词还原。"""
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


def read_text(path: str) -> Document:
    """解析 .txt / .md：整篇按空行切段，并复用 textnorm 的收录规则。

    Markdown 的 ``# 标题`` 会去掉井号并按标题处理；标题缺失时回退到文件名。
    """
    file_path = _validate_path(path)
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in (".txt", ".md"):
        raise ValueError("仅支持 .txt / .md 格式的文本文件：%s" % file_name)

    text = _read_text_file(file_path)
    is_markdown = ext == ".md"

    paragraphs: List[Paragraph] = []
    body_indices: List[int] = []
    title = ""

    for chunk in _split_text_chunks(text):
        lines = chunk.split("\n")
        style = ""
        if is_markdown and lines:
            match = _MD_HEADING_RE.match(lines[0].strip())
            if match:
                style = "heading%d" % len(match.group(1))
                lines = [match.group(2)] + lines[1:]
        body = _join_lines(lines)
        if not body:
            continue
        index = len(paragraphs)
        heading = is_heading_like(body, style)
        reference = is_reference_like(body)
        furniture = is_furniture_like(body)
        char_count = count_words(body)
        paragraphs.append(
            Paragraph(
                index=index,
                text=body,
                style=style,
                is_heading=heading,
                is_reference=reference,
                char_count=char_count,
            )
        )
        if heading and not title:
            title = body
        if (not heading) and (not reference) and (not furniture) and char_count >= BODY_MIN_CHARS:
            body_indices.append(index)

    if not title:
        # 与 docx_reader 一致：没有显式标题时，取首个短且无句末标点的非参考文献行。
        for paragraph in paragraphs:
            if paragraph.is_heading or paragraph.is_reference:
                continue
            candidate = paragraph.text
            if len(candidate) <= 40 and not any(c in candidate for c in "。！？!?；;，,：:"):
                title = candidate
                break
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


def read_document(path: str) -> Document:
    """按扩展名分派解析器并返回 Document；不支持的格式抛中文 ValueError。"""
    file_path = _validate_path(path)
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".docx":
        return read_docx(file_path)
    if ext == ".pdf":
        return read_pdf(file_path)
    if ext in (".txt", ".md"):
        return read_text(file_path)
    raise ValueError(
        "不支持的文档格式：%s（支持：%s）"
        % (ext or "无扩展名", "、".join(SUPPORTED_EXTENSIONS))
    )

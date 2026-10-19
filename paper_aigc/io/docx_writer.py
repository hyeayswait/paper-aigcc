"""docx 回写：把改写后的段落文本写回 Word，尽量保留原有格式。

段落定位规则与 docx_reader 完全一致（共用 collect_paragraphs）。
命中段落：run[0].text = 新文本，其余 run 置空；没有 run 时 add_run。
未命中段落一字不动；原始文件不被修改。
"""
from __future__ import annotations

import os
from typing import Dict

import docx

from .docx_reader import _open_docx, collect_paragraphs


def write_docx(source_path: str, replacements: Dict[int, str], out_path: str) -> str:
    """按段落 index 替换文本并另存为 out_path，返回 out_path。"""
    src = os.fspath(source_path)
    dst = os.fspath(out_path)

    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("输出路径不能与源文件相同，原始文件不会被覆盖。")

    docx_document = _open_docx(src)
    items = collect_paragraphs(docx_document)

    for index, (paragraph, _text, _style) in enumerate(items):
        if index not in replacements:
            continue
        new_text = replacements[index]
        if new_text is None:
            continue
        new_text = str(new_text)
        runs = paragraph.runs
        if runs:
            runs[0].text = new_text
            for run in runs[1:]:
                run.text = ""
        else:
            paragraph.add_run(new_text)

    parent = os.path.dirname(os.path.abspath(dst))
    if parent:
        os.makedirs(parent, exist_ok=True)
    docx_document.save(dst)
    return dst

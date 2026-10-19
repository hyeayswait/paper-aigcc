# -*- coding: utf-8 -*-
"""命令行：对比两版论文 docx，打印摘要并输出逐段差异 HTML。

用法：
    python tools/version_diff.py --old a.docx --new b.docx --out diff.html
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paper_aigc.io.version_diff import diff_documents, render_diff_html


def _basename(path: str) -> str:
    return os.path.basename(os.fspath(path)) or os.fspath(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="对比两版论文 docx，输出逐段差异 HTML 报告（单文件、零外链）"
    )
    parser.add_argument("--old", required=True, help="旧版 .docx 路径")
    parser.add_argument("--new", required=True, help="新版 .docx 路径")
    parser.add_argument("--out", required=True, help="输出 HTML 文件路径")
    args = parser.parse_args(argv)

    try:
        diff = diff_documents(args.old, args.new)
    except ValueError as exc:
        print("错误：%s" % exc, file=sys.stderr)
        return 2

    try:
        out_path = render_diff_html(diff, args.out)
    except OSError as exc:
        print("错误：无法写入报告文件（%s）：%s" % (args.out, exc), file=sys.stderr)
        return 2

    delta = diff.new_chars - diff.old_chars
    print("版本对比完成")
    print("旧稿：%s（正文 %d 字）" % (_basename(diff.old_path), diff.old_chars))
    print("新稿：%s（正文 %d 字）" % (_basename(diff.new_path), diff.new_chars))
    print(
        "新增 %d 段 | 删除 %d 段 | 修改 %d 段 | 未变 %d 段"
        % (diff.added, diff.removed, diff.modified, diff.unchanged)
    )
    print("字数变化：%+d" % delta)
    print("报告：%s" % os.path.abspath(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

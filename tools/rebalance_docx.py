# -*- coding: utf-8 -*-
"""对 docx 正文做「节奏再平衡」：只调整句子边界标点，不改任何字词。

    python tools/rebalance_docx.py --src 送检版.docx --dst 送检版_v3.docx --long 2 --short 1
"""
from __future__ import annotations

import argparse
import io
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from paper_aigc.io.docx_reader import read_docx
from paper_aigc.io.docx_writer import write_docx
from paper_aigc.rewrite.optimize import document_quality, rebalance_rhythm


def main() -> int:
    parser = argparse.ArgumentParser(description="调整 docx 正文的句子节奏（只改标点）")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--long", type=int, default=2, help="每段至少几个长句")
    parser.add_argument("--short", type=int, default=1, help="每段至少保留几个短句")
    parser.add_argument("--max-author-voice", type=int, default=3,
                        help="全篇最多保留几次「笔者认为」（多出的删除）")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    doc = read_docx(args.src)
    before = [doc.paragraphs[i].text for i in doc.body_indices]
    replacements = {}
    budget = int(args.max_author_voice)
    for index in doc.body_indices:
        original = doc.paragraphs[index].text
        tuned = rebalance_rhythm(original, long_target=args.long, keep_short=args.short)
        for phrase in ("笔者认为，", "笔者认为", "本文认为，", "本文认为", "我判断，", "我判断"):
            while phrase in tuned:
                if budget > 0:
                    budget -= 1
                    break
                tuned = tuned.replace(phrase, "", 1)
        tuned = tuned.replace("。。", "。").strip()
        if tuned != original:
            replacements[index] = tuned

    after = [replacements.get(i, doc.paragraphs[i].text) for i in doc.body_indices]
    print("调整段落数：%d / %d" % (len(replacements), len(doc.body_indices)))
    print("调整前：", document_quality(before))
    print("调整后：", document_quality(after))

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    write_docx(args.src, replacements, args.dst)
    print("已写出：%s" % args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())

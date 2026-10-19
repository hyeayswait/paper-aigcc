# -*- coding: utf-8 -*-
"""诊断工具：打印某段文本的全部原始特征、子分与最终分数。

    python tools/feature_diag.py --text "要诊断的段落……"
    python tools/feature_diag.py --docx samples\样例论文_AI味混排.docx --index 5
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paper_aigc.core import scoring
from paper_aigc.core.features import FEATURE_META, extract_features_dict
from paper_aigc.io.docx_reader import read_docx


def _show(text: str) -> None:
    feats = extract_features_dict(text)
    subs = scoring._subscores(feats)
    ps = scoring.score_paragraph(text)
    print("=" * 96)
    print("字符数 %d   分数 %.1f   等级 %s" % (len(text), ps.score, ps.level))
    print("=" * 96)
    print("%-24s %-14s %9s %8s %7s %7s" % ("特征", "中文名", "原始值", "子分", "权重", "贡献"))
    for key, meta in FEATURE_META.items():
        weight = float(meta["weight"])
        contribution = weight * subs[key] / scoring._TOTAL_WEIGHT * 100.0
        print("%-24s %-14s %9.4f %8.2f %7.2f %7.1f" % (
            key, str(meta["name"]), feats.get(key, 0.0), subs[key], weight, contribution))
    print("-" * 96)
    print("Top 信号：", "；".join("%s %.1f" % (n, c) for _k, n, c in ps.top_signals) or "无")
    print("改写建议：", ps.suggestion or "—")


def main() -> int:
    parser = argparse.ArgumentParser(description="查看一段文本的特征明细")
    parser.add_argument("--text", default="", help="直接给一段文本")
    parser.add_argument("--docx", default="", help="从 docx 里取一段")
    parser.add_argument("--index", type=int, default=-1, help="段落 index（0 起）")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    if args.text:
        _show(args.text)
        return 0
    if args.docx:
        doc = read_docx(args.docx)
        if args.index < 0:
            for idx in doc.body_indices:
                para = next(p for p in doc.paragraphs if p.index == idx)
                print("#%d  %s" % (idx, para.text[:60]))
            return 0
        para = next((p for p in doc.paragraphs if p.index == args.index), None)
        if para is None:
            print("找不到 index=%d 的段落" % args.index, file=sys.stderr)
            return 2
        _show(para.text)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

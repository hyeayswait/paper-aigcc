# -*- coding: utf-8 -*-
"""在公开中文 AIGC 检测数据上评估当前启发式指数的判别力。

    python tools/eval_detector.py --csv data\aigcd_test.csv --limit 3000
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from paper_aigc.core import scoring


def load_aigcd(path: str, limit: int = 0):
    texts, labels, sources = [], [], []
    with io.open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if not text or label not in ("0", "1"):
                continue
            texts.append(text)
            labels.append(int(label))
            sources.append(row.get("source") or "")
            if limit and len(texts) >= limit:
                break
    return texts, labels, sources


def auc(scores, labels) -> float:
    """Mann-Whitney U 形式计算 AUC，正类 = 1（AI）。"""
    pairs = sorted(zip(scores, labels))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = sum(r for r, (_, label) in zip(ranks, pairs) if label == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def percentile(values, q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def summarize(scores, labels, name: str) -> dict:
    human = [s for s, y in zip(scores, labels) if y == 0]
    ai = [s for s, y in zip(scores, labels) if y == 1]
    value = auc(scores, labels)
    # 以「人类文本 95 分位」为阈值：人类误报率约 5%
    thr = percentile(human, 0.95)
    tp = sum(1 for s, y in zip(scores, labels) if y == 1 and s >= thr)
    fp = sum(1 for s, y in zip(scores, labels) if y == 0 and s >= thr)
    recall = tp / float(len(ai)) if ai else 0.0
    fpr = fp / float(len(human)) if human else 0.0
    result = {
        "name": name,
        "n": len(scores),
        "n_human": len(human),
        "n_ai": len(ai),
        "auc": round(value, 4),
        "human_mean": round(sum(human) / len(human), 1) if human else 0,
        "ai_mean": round(sum(ai) / len(ai), 1) if ai else 0,
        "human_p50": round(percentile(human, 0.50), 1),
        "human_p95": round(thr, 1),
        "ai_p50": round(percentile(ai, 0.50), 1),
        "threshold_human_p95": round(thr, 1),
        "recall_at_threshold": round(recall, 3),
        "fpr_at_threshold": round(fpr, 3),
    }
    print("\n[%s] n=%d（人类 %d / AI %d）" % (name, len(scores), len(human), len(ai)))
    print("  AUC = %.4f" % value)
    print("  人类分数：均值 %.1f  中位数 %.1f  P95 %.1f" % (
        result["human_mean"], result["human_p50"], result["human_p95"]))
    print("  AI   分数：均值 %.1f  中位数 %.1f" % (result["ai_mean"], result["ai_p50"]))
    print("  以人类 P95=%.1f 为阈值：召回 %.1f%%  人类误报 %.1f%%" % (
        thr, recall * 100, fpr * 100))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="评估启发式指数的判别力")
    parser.add_argument("--csv", default=os.path.join(_ROOT, "data", "aigcd_test.csv"))
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

    started = time.time()
    texts, labels, sources = load_aigcd(args.csv, args.limit)
    print("载入 %d 条样本（%.1fs）" % (len(texts), time.time() - started))
    scores = [scoring.score_text(text) for text in texts]
    print("评分完成（%.1fs）" % (time.time() - started))

    results = {"dataset": os.path.basename(args.csv), "overall": summarize(scores, labels, "全部")}
    results["by_source"] = {}
    for source in sorted({s for s in sources if s}):
        sub_scores = [sc for sc, src in zip(scores, sources) if src == source]
        sub_labels = [lab for lab, src in zip(labels, sources) if src == source]
        if len(set(sub_labels)) < 2:
            continue
        results["by_source"][source] = summarize(sub_scores, sub_labels, source)

    if args.out:
        with io.open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n结果已写入 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

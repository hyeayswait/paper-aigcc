# -*- coding: utf-8 -*-
"""用公开中文 AIGC 检测数据重新拟合评分的特征权重。

    python tools/fit_calibration.py --train data\aigcd_train.csv --test data\aigcd_test.csv

产物：
    paper_aigc/calibration/model_v1.json   （标准化参数 + 逻辑回归系数 + 评估指标）
    out/fit_report.json                    （训练/测试评估明细）
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from paper_aigc.core.features import FEATURE_META, extract_features_dict
from paper_aigc.core.textnorm import count_words

MIN_WORDS = 60          # 太短的样本丢弃：与论文段落尺度更接近
FEATURE_KEYS = list(FEATURE_META.keys())


def load_csv(path: str, limit: int = 0):
    rows = []
    with io.open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if label not in ("0", "1") or not text:
                continue
            if count_words(text) < MIN_WORDS:
                continue
            rows.append((text, int(label), row.get("source") or ""))
            if limit and len(rows) >= limit:
                break
    return rows


def featurize(rows):
    X = np.zeros((len(rows), len(FEATURE_KEYS)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    for i, (text, label, _src) in enumerate(rows):
        feats = extract_features_dict(text)
        for j, key in enumerate(FEATURE_KEYS):
            value = float(feats.get(key, 0.0))
            X[i, j] = value if np.isfinite(value) else 0.0
        y[i] = label
    return X, y


def auc(scores, labels) -> float:
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = float(labels.sum())
    neg = float(len(labels) - pos)
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = ranks[sorted_labels == 1].sum()
    return float((rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def fit_logreg(X, y, l2: float = 1.0, iters: int = 60):
    n, d = X.shape
    design = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    for step_i in range(iters):
        z = design @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = design.T @ (p - y)
        grad[1:] += l2 * w[1:]
        weights = p * (1.0 - p)
        hess = (design * weights[:, None]).T @ design
        reg = np.eye(d + 1) * l2
        reg[0, 0] = 1e-6
        hess = hess + reg
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(hess, grad, rcond=None)[0]
        w -= delta
        if np.max(np.abs(delta)) < 1e-7:
            break
    return w


def predict(X, mean, std, coef, intercept):
    Z = (X - mean) / std
    z = Z @ coef + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.path.join(_ROOT, "data", "aigcd_train.csv"))
    parser.add_argument("--test", default=os.path.join(_ROOT, "data", "aigcd_test.csv"))
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1.0)
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    t0 = time.time()
    train_rows = load_csv(args.train, args.limit_train)
    test_rows = load_csv(args.test, args.limit_test)
    print("样本：训练 %d（AI %d）/ 测试 %d（AI %d），过滤阈值 >=%d 字" % (
        len(train_rows), sum(r[1] for r in train_rows),
        len(test_rows), sum(r[1] for r in test_rows), MIN_WORDS))

    print("提取训练特征……" )
    Xtr, ytr = featurize(train_rows)
    print("  %.1fs" % (time.time() - t0))
    print("提取测试特征……")
    Xte, yte = featurize(test_rows)
    print("  %.1fs" % (time.time() - t0))

    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std < 1e-9] = 1.0
    Ztr = (Xtr - mean) / std
    w = fit_logreg(Ztr, ytr, l2=args.l2)

    p_tr = predict(Xtr, mean, std, w[1:], w[0])
    p_te = predict(Xte, mean, std, w[1:], w[0])
    print("\n训练集 AUC = %.4f" % auc(p_tr, ytr))
    test_auc = auc(p_te, yte)
    print("测试集 AUC = %.4f" % test_auc)

    human = p_te[yte == 0]
    ai = p_te[yte == 1]
    thr = float(np.percentile(human, 95))
    recall = float((ai >= thr).mean()) if len(ai) else 0.0
    fpr = float((human >= thr).mean()) if len(human) else 0.0
    print("阈值=人类P95=%.4f -> 召回 %.1f%% / 误报 %.1f%%" % (thr, recall * 100, fpr * 100))

    coef_map = {key: round(float(w[i + 1]), 4) for i, key in enumerate(FEATURE_KEYS)}
    order = sorted(coef_map.items(), key=lambda kv: -abs(kv[1]))
    print("\n特征权重（标准化后，正=更像 AI）：")
    for key, value in order:
        print("  %-22s %+.4f  %s" % (key, value, FEATURE_META[key]["name"]))

    model = {
        "version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "AnxForever/chinese-ai-detection-dataset (HC3 + THUCNews + GPT-4.1/Gemini 平行语料)",
        "feature_keys": FEATURE_KEYS,
        "standardize": {"mean": [round(float(v), 6) for v in mean],
                        "std": [round(float(v), 6) for v in std]},
        "coef": [round(float(v), 6) for v in w[1:]],
        "intercept": round(float(w[0]), 6),
        "l2": args.l2,
        "min_words": MIN_WORDS,
        "metrics": {
            "train_auc": round(float(auc(p_tr, ytr)), 4),
            "test_auc": round(test_auc, 4),
            "human_p95": round(thr, 4),
            "recall_at_p95": round(recall, 3),
            "fpr_at_p95": round(fpr, 3),
            "n_train": len(train_rows),
            "n_test": len(test_rows),
        },
    }

    model_dir = os.path.join(_ROOT, "paper_aigc", "calibration")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model_v1.json")
    with io.open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)
    print("\n模型已保存：%s" % model_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

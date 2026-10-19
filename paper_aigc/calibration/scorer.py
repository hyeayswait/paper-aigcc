# -*- coding: utf-8 -*-
"""加载训练好的校准模型，给出「这段文本是 AI 生成」的概率。

模型来自 tools/fit_calibration.py，训练数据是公开中文 AIGC 检测集
（HC3 中文 + THUCNews + GPT-4.1/Gemini 平行语料）。
注意：这不是知网模型，概率量纲也不是知网 AIGC 率，只是比手工锚点更靠谱的相对刻度。
"""
from __future__ import annotations

import io
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np

from ..core.features import extract_features_dict
from ..core.textnorm import count_words

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_v1.json")

_model_cache: List[Optional["CalibratedModel"]] = []


class CalibratedModel:
    """逻辑回归校准器：特征 -> P(该段为 AI 生成)。"""

    def __init__(self, path: str = MODEL_PATH) -> None:
        with io.open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.path = path
        self.feature_keys: List[str] = list(data["feature_keys"])
        self.mean = np.array(data["standardize"]["mean"], dtype=np.float64)
        self.std = np.array(data["standardize"]["std"], dtype=np.float64)
        self.coef = np.array(data["coef"], dtype=np.float64)
        self.intercept = float(data["intercept"])
        self.min_words = int(data.get("min_words", 0))
        self.metrics: Dict[str, object] = dict(data.get("metrics") or {})
        self.dataset = str(data.get("dataset") or "")
        self.created = str(data.get("created") or "")

    @property
    def test_auc(self) -> float:
        try:
            return float(self.metrics.get("test_auc", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @property
    def human_p95(self) -> float:
        """人类文本概率的 95 分位：超过它的段落按「偏高」处理。"""
        try:
            return float(self.metrics.get("human_p95", 0.5))
        except (TypeError, ValueError):
            return 0.5

    def probability(self, text: str) -> float:
        feats = extract_features_dict(text or "")
        vector = np.array([float(feats.get(key, 0.0)) for key in self.feature_keys], dtype=np.float64)
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
        z = float(((vector - self.mean) / self.std) @ self.coef + self.intercept)
        if not math.isfinite(z):
            return 0.0
        return 1.0 / (1.0 + math.exp(max(-30.0, min(30.0, -z))))

    def score(self, text: str) -> float:
        """把概率映射成 0-100 的指数（= 概率 ×100）。"""
        if count_words(text or "") < self.min_words:
            # 过短文本特征不可靠，退回中性偏低
            return 0.0
        return round(self.probability(text) * 100.0, 1)

    def describe(self) -> str:
        return "公开数据校准 v1（测试集 AUC %.3f，人类 P95=%.2f，样本 %s）" % (
            self.test_auc, self.human_p95, self.metrics.get("n_train", "?"))


def is_available(path: str = MODEL_PATH) -> bool:
    return os.path.isfile(path)


def get_model(path: str = MODEL_PATH) -> Optional[CalibratedModel]:
    if not _model_cache:
        try:
            _model_cache.append(CalibratedModel(path) if is_available(path) else None)
        except Exception:
            _model_cache.append(None)
    return _model_cache[0]


def model_metrics(path: str = MODEL_PATH) -> Dict[str, object]:
    model = get_model(path)
    return dict(model.metrics) if model is not None else {}

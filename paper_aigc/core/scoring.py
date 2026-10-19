"""评分引擎：把 17 个文体特征合成 0-100 的启发式「AI 痕迹指数」。

重要说明：该指数没有与知网做过任何校准，绝对值不能当作知网 AIGC 率引用。
可信的用法只有两种：同一版本内改写前后对比，以及段落之间的相对排序。
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .features import FEATURE_META, MIN_CJK_FOR_FEATURES, extract_features_dict
from .textnorm import count_cjk, count_words
from .types import (
    LEVELS,
    VERSION,
    Document,
    Paragraph,
    ParagraphScore,
    ScoreReport,
)

# 分数分段阈值：<25 低，<45 中，<60 偏高，>=60 高
LEVEL_THRESHOLDS: Tuple[float, float, float] = (25.0, 45.0, 60.0)

# 每个特征对应一句中文改写建议，最多拼两句
SUGGESTION_TEMPLATES: Dict[str, str] = {
    "sentence_len_cv": "拆开连续的同长度句式，让长短句交替出现",
    "avg_sentence_len": "把过长的句子拆成两层意思，单句尽量控制在二三十字",
    "short_sentence_ratio": "补一些短句作为停顿，避免通篇长句铺陈",
    "long_sentence_ratio": "压缩超长句，一个句子只承载一个核心判断",
    "clause_len_cv": "调整分句节奏，不要让所有分句都一样长",
    "connective_density": "删掉可有可无的过渡词，靠语义自然衔接",
    "boilerplate_density": "替换学术套话，直接陈述具体结论",
    "parallelism_density": "打散排比和对仗结构，少用固定的关联框架",
    "hedge_density": "减少模糊限制语，把判断的边界说清楚",
    "idiom_density": "减少四字格堆叠，换成更自然的说法",
    "num_ratio": "适当补充具体数据或量化描述",
    "first_person_density": "加入作者的判断语气，例如「笔者认为」",
    "passive_density": "把被动句改成主动句，明确动作的发出者",
    "char_ttr": "丰富用词，避免反复使用同一批词",
    "repetition_ratio": "改写反复出现的短语和句式",
    "char_entropy": "扩展词汇面，减少单一用字的反复出现",
    "punct_diversity": "适度混用问句、破折号、括号等不同标点",
}

# 空文本/超短文本/无可识别特征时的兜底建议
_DEFAULT_SUGGESTION = "该段文体特征接近人工写作常见区间，暂不建议改写。"
_FALLBACK_SUGGESTION = "该段存在若干 AI 化特征，建议调整句式与用词。"

_TOTAL_WEIGHT: float = sum(float(meta["weight"]) for meta in FEATURE_META.values())

# ---------------------------------------------------------------- 评分引擎
ENGINE_HEURISTIC = "heuristic"
ENGINE_CALIBRATED = "calibrated"
_ENGINE_STATE = {"mode": ENGINE_HEURISTIC}

# 校准模式下分数 = P(AI)×100，等级按人类文本概率分位切：
# <40 低、<60 中、<80 偏高、>=80 高（80 ≈ 公开数据里人类文本的 P95）
CALIBRATED_LEVEL_THRESHOLDS: Tuple[float, float, float] = (40.0, 60.0, 80.0)


def set_engine(mode: str) -> str:
    """切换评分引擎：heuristic（手工锚点）/ calibrated（公开数据逻辑回归）。"""
    value = str(mode or "").strip().lower()
    if value not in (ENGINE_HEURISTIC, ENGINE_CALIBRATED):
        raise ValueError("未知评分引擎：%s（可用 heuristic / calibrated）" % mode)
    _ENGINE_STATE["mode"] = value
    return value


def get_engine() -> str:
    return _ENGINE_STATE["mode"]


def engine_thresholds() -> Tuple[float, float, float]:
    if _ENGINE_STATE["mode"] == ENGINE_CALIBRATED:
        return CALIBRATED_LEVEL_THRESHOLDS
    return LEVEL_THRESHOLDS


def _calibrated_model():
    """校准模式下返回模型；不可用则返回 None（自动回退到手工锚点）。"""
    if _ENGINE_STATE["mode"] != ENGINE_CALIBRATED:
        return None
    try:
        from ..calibration.scorer import get_model

        return get_model()
    except Exception:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def _apply_anchors(raw_value: float, anchors: object) -> float:
    """用分段线性锚点把原始值映射到 0-1 子分；超端点取端点值。"""
    if not anchors:
        return 0.0
    points: List[Tuple[float, float]] = []
    for pair in anchors:  # type: ignore[union-attr]
        points.append((float(pair[0]), float(pair[1])))
    if not points:
        return 0.0
    x = float(raw_value)
    if not math.isfinite(x):
        return 0.0
    if x <= points[0][0]:
        return _clamp(points[0][1], 0.0, 1.0)
    if x >= points[-1][0]:
        return _clamp(points[-1][1], 0.0, 1.0)
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= x <= x1:
            if x1 == x0:
                return _clamp(y1, 0.0, 1.0)
            ratio = (x - x0) / (x1 - x0)
            return _clamp(y0 + ratio * (y1 - y0), 0.0, 1.0)
    return _clamp(points[-1][1], 0.0, 1.0)


def _subscores(feature_values: Dict[str, float]) -> Dict[str, float]:
    """把原始特征值逐项映射成 0-1 子分。"""
    result: Dict[str, float] = {}
    for key, meta in FEATURE_META.items():
        anchors = meta["anchors"]
        raw_value = float(feature_values.get(key, 0.0))
        result[key] = _apply_anchors(raw_value, anchors)
    return result


def _raw_score(subs: Dict[str, float]) -> float:
    """加权平均后放大到 0-100。"""
    if _TOTAL_WEIGHT <= 0:
        return 0.0
    total = 0.0
    for key, meta in FEATURE_META.items():
        total += float(meta["weight"]) * subs.get(key, 0.0)
    return total / _TOTAL_WEIGHT * 100.0


def _top_signals(subs: Dict[str, float]) -> List[Tuple[str, str, float]]:
    """按 (子分-0.5)*权重 降序，只保留正向贡献，最多 4 条。"""
    candidates: List[Tuple[float, str, str, float]] = []
    for key, meta in FEATURE_META.items():
        weight = float(meta["weight"])
        sub = subs.get(key, 0.0)
        strength = (sub - 0.5) * weight
        if strength <= 0:
            continue
        contribution = round(weight * sub / _TOTAL_WEIGHT * 100.0, 1)
        candidates.append((strength, key, str(meta["name"]), contribution))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [(key, name, contribution) for _, key, name, contribution in candidates[:4]]


def _build_suggestion(top_signals: List[Tuple[str, str, float]]) -> str:
    """按高贡献特征拼一到两句中文改写建议。"""
    if not top_signals:
        return _DEFAULT_SUGGESTION
    parts: List[str] = []
    for key, _name, _contribution in top_signals[:2]:
        template = SUGGESTION_TEMPLATES.get(key, "")
        if template:
            parts.append(template)
    if not parts:
        return _FALLBACK_SUGGESTION
    return "；".join(parts) + "。"


def level_of(score: float, thresholds: Optional[Tuple[float, float, float]] = None) -> str:
    """把 0-100 分数映射为等级；默认阈值随当前引擎切换。"""
    value = float(score)
    if not math.isfinite(value):
        return LEVELS[0]
    low, mid, high = thresholds or engine_thresholds()
    if value < low:
        return LEVELS[0]
    if value < mid:
        return LEVELS[1]
    if value < high:
        return LEVELS[2]
    return LEVELS[3]


def _empty_feature_dict() -> Dict[str, float]:
    return {key: 0.0 for key in FEATURE_META}


def _empty_paragraph_score(raw: str, index: int) -> ParagraphScore:
    return ParagraphScore(
        index=index,
        text=raw,
        score=0.0,
        level=LEVELS[0],
        features=_empty_feature_dict(),
        top_signals=[],
        suggestion="",
    )


def _score_paragraph_heuristic(raw: str, index: int) -> ParagraphScore:
    features = extract_features_dict(raw)
    subs = _subscores(features)
    score = _clamp(round(_raw_score(subs), 1), 0.0, 100.0)
    signals = _top_signals(subs)
    return ParagraphScore(
        index=index,
        text=raw,
        score=score,
        level=level_of(score),
        features=features,
        top_signals=signals,
        suggestion=_build_suggestion(signals),
    )


def _calibrated_contributions(model, features: Dict[str, float]):
    """每个特征对 logit 的贡献；正=把该段推向「AI 生成」。"""
    rows: List[Tuple[str, str, float, float]] = []
    total_abs = abs(float(model.intercept)) or 1e-9
    for position, key in enumerate(model.feature_keys):
        try:
            std = float(model.std[position]) or 1e-9
            z = (float(features.get(key, 0.0)) - float(model.mean[position])) / std
            value = float(model.coef[position]) * z
        except (TypeError, ValueError, IndexError):
            value = 0.0
        name = str(FEATURE_META.get(key, {}).get("name", key))
        rows.append((key, name, value, 0.0))
        total_abs += abs(value)
    scale = 100.0 / total_abs
    return [(key, name, value, value * scale) for key, name, value, _zero in rows]


def _score_paragraph_calibrated(model, raw: str, index: int) -> ParagraphScore:
    features = extract_features_dict(raw)
    rows = _calibrated_contributions(model, features)
    logit = sum(row[2] for row in rows) + float(model.intercept)
    logit = max(-30.0, min(30.0, logit))
    probability = 1.0 / (1.0 + math.exp(-logit))
    score = _clamp(round(probability * 100.0, 1), 0.0, 100.0)
    positive = [row for row in rows if row[2] > 0]
    positive.sort(key=lambda item: -item[2])
    signals = [(key, name, round(points, 1)) for key, name, _value, points in positive[:4]]
    return ParagraphScore(
        index=index,
        text=raw,
        score=score,
        level=level_of(score),
        features=features,
        top_signals=signals,
        suggestion=_build_suggestion(signals),
    )


def score_paragraph(text: str, index: int = 0) -> ParagraphScore:
    """给单段文本评分。短文本与空文本直接返回 0 分。"""
    raw = text or ""
    if count_cjk(raw) < MIN_CJK_FOR_FEATURES:
        return _empty_paragraph_score(raw, index)
    model = _calibrated_model()
    if model is not None and count_words(raw) >= int(getattr(model, "min_words", 0)):
        return _score_paragraph_calibrated(model, raw, index)
    return _score_paragraph_heuristic(raw, index)


def score_text(text: str) -> float:
    """只返回 0-100 的启发式指数。"""
    return score_paragraph(text, 0).score


def _profile_of(report: ScoreReport) -> Dict[str, object]:
    """全文画像：风格离散度 + 每段都触顶的共性特征。

    当所有段落分数挤在一起时，段落排序没有区分价值；这时应该告诉用户
    「问题在全文共性」而不是让他去挑几段。
    """
    scores = [float(item.score) for item in report.paragraphs]
    count = len(scores)
    if count == 0:
        return {
            "paragraphs": 0,
            "dispersion": 0.0,
            "homogeneous": False,
            "global_signals": [],
            "note": "没有可参与评分的正文段落。",
        }

    mean = sum(scores) / float(count)
    dispersion = math.sqrt(sum((value - mean) ** 2 for value in scores) / float(count))

    model = _calibrated_model()
    contribution_rows: List[Dict[str, float]] = []
    for item in report.paragraphs:
        if model is not None:
            contribution_rows.append({
                key: points for key, _name, _value, points in _calibrated_contributions(model, item.features)
            })
        else:
            subs = _subscores(item.features)
            contribution_rows.append({
                key: float(FEATURE_META[key]["weight"]) * subs.get(key, 0.0) / _TOTAL_WEIGHT * 100.0
                for key in FEATURE_META
            })
    baseline = 4.0 if model is not None else 3.0
    global_signals: List[Dict[str, object]] = []
    for key, meta in FEATURE_META.items():
        values = [row.get(key, 0.0) for row in contribution_rows]
        elevated_ratio = sum(1 for value in values if value >= baseline) / float(count)
        if elevated_ratio < 0.6:
            continue
        global_signals.append({
            "key": key,
            "name": str(meta["name"]),
            "elevated_ratio": round(elevated_ratio, 2),
            "mean_contribution": round(sum(values) / float(count), 1),
        })
    global_signals.sort(key=lambda item: (-float(item["mean_contribution"]), str(item["key"])))

    homogeneous = count >= 5 and dispersion < 6.0
    names = "、".join(str(item["name"]) for item in global_signals[:4]) or "未发现明显共性特征"
    if homogeneous:
        note = (
            "全文 %d 段文风高度一致（分数标准差仅 %.1f 分）。这说明问题不在某几段，"
            "而是全文共性的：%s。这种情况下段落排序的参考价值有限，建议整篇改写，"
            "而不是只挑排名靠前的几段。"
            % (count, dispersion, names)
        )
    else:
        note = (
            "全文 %d 段文风差异较明显（分数标准差 %.1f 分），可以优先处理排名靠前的段落。"
            "全文共性问题：%s。"
            % (count, dispersion, names)
        )
    return {
        "paragraphs": count,
        "dispersion": round(dispersion, 1),
        "homogeneous": bool(homogeneous),
        "global_signals": global_signals,
        "note": note,
    }


def score_document(doc: Document) -> ScoreReport:
    """给整篇文档评分：只统计 body_indices 指向的正文段落。"""
    report = ScoreReport(document=doc)

    by_index: Dict[int, Paragraph] = {}
    for paragraph in doc.paragraphs:
        if paragraph.index not in by_index:
            by_index[paragraph.index] = paragraph

    scored: List[Tuple[Paragraph, ParagraphScore]] = []
    for index in doc.body_indices:
        paragraph = by_index.get(index)
        if paragraph is None:
            continue
        scored.append((paragraph, score_paragraph(paragraph.text, paragraph.index)))

    report.paragraphs = [item[1] for item in scored]

    level_counts: Dict[str, int] = {level: 0 for level in LEVELS}
    for paragraph_score in report.paragraphs:
        level_counts[paragraph_score.level] = level_counts.get(paragraph_score.level, 0) + 1
    report.level_counts = level_counts

    if not scored:
        report.overall = 0.0
        report.level = LEVELS[0]
    else:
        weights = [max(0, int(paragraph.char_count)) for paragraph, _ in scored]
        total_weight = sum(weights)
        if total_weight > 0:
            weighted = 0.0
            for (_paragraph, paragraph_score), weight in zip(scored, weights):
                weighted += paragraph_score.score * weight
            overall = weighted / total_weight
        else:
            overall = sum(item[1].score for item in scored) / len(scored)
        report.overall = _clamp(round(overall, 1), 0.0, 100.0)
        report.level = level_of(report.overall)

    model = _calibrated_model()
    if model is not None:
        report.calibration = "public-data-v1:AUC=%.3f" % float(model.test_auc)
        engine_name = "calibrated-logreg"
        calibration_meta = {
            "engine": engine_name,
            "model": "public-data-v1",
            "test_auc": float(model.test_auc),
            "human_p95": float(model.human_p95),
            "dataset": model.dataset,
        }
    else:
        report.calibration = "uncalibrated"
        engine_name = "stylometric-heuristic"
        calibration_meta = {"engine": engine_name}
    report.meta = {
        "version": VERSION,
        "feature_count": len(FEATURE_META),
        "engine": engine_name,
        "calibration": calibration_meta,
        "level_thresholds": list(engine_thresholds()),
        "profile": _profile_of(report),
    }
    return report

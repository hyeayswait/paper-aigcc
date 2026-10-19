"""文体特征提取 —— 17 个固定特征键与锚点表。

定位说明：本工具没有接触过知网，也没有知网标注数据。这里输出的是可解释的
文体统计量，只用于同一版本内的前后对比与段落间相对排序，不是知网 AIGC 率，
也不能与知网的检测分数直接比较。

实现约束：
- 纯标准库，不依赖网络，也不依赖第三方分词器；
- 不使用随机数与当前时间，同一文本多次调用结果完全一致；
- 空文本、纯英文、仅标点的输入返回全 0，绝不抛异常。
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

from .textnorm import (
    char_ngrams,
    count_cjk,
    split_clauses,
    split_sentences,
    tokenize_words,
)
from .types import FeatureSet

# 汉字数少于该阈值的文本不做文体统计：样本太小，统计量没有解释力。
MIN_CJK_FOR_FEATURES = 5

# token 数超过该阈值才算「长文本」，改用 MATTR（窗宽 50）抵消长度效应；
# 段落通常不足 200 token，此时普通 TTR 对「同一批词反复出现」更敏感。
MATTR_MIN_TOKENS = 200

# 锚点表格式：[(原始值 x, 0-1 子分 y), ...]，x 升序。
# y 越大表示该特征越像 AI 生成；超端点时取端点值，中间由 scoring.py 线性插值。
# 锚点表放在 FEATURE_META[key]["anchors"]，由 features.py 提供、scoring.py 消费。
FEATURE_META: Dict[str, Dict[str, object]] = {
    "sentence_len_cv": {
        "name": "句长变异系数",
        "weight": 0.7,
        "desc": "句长标准差与均值之比；越低说明句子长度越整齐、越像机器生成",
        "anchors": [(0.0, 1.0), (0.20, 0.80), (0.40, 0.50), (0.65, 0.20), (0.90, 0.0)],
    },
    "avg_sentence_len": {
        "name": "平均句长",
        "weight": 1.0,
        "desc": "每句汉字数的均值；AI 文本常偏长且结构完整",
        "anchors": [(4.0, 0.10), (10.0, 0.25), (20.0, 0.45), (32.0, 0.68), (45.0, 0.84), (65.0, 0.93), (100.0, 1.0)],
    },
    "short_sentence_ratio": {
        "name": "短句占比",
        "weight": 0.5,
        "desc": "汉字数小于 12 的句子占比；人工写作通常短句更多",
        "anchors": [(0.0, 1.0), (0.05, 0.82), (0.15, 0.58), (0.30, 0.32), (0.50, 0.12), (0.75, 0.0)],
    },
    "long_sentence_ratio": {
        "name": "长句占比",
        "weight": 1.1,
        "desc": "汉字数大于 45 的句子占比；AI 倾向写长复句",
        "anchors": [(0.0, 0.02), (0.05, 0.22), (0.12, 0.48), (0.22, 0.70), (0.35, 0.86), (0.55, 0.94), (0.80, 0.99), (1.0, 1.0)],
    },
    "clause_len_cv": {
        "name": "分句长变异系数",
        "weight": 0.35,
        "desc": "分句长度标准差与均值之比；越低说明分句节奏越单一（弱信号：本样例方向不稳，降权）",
        "anchors": [(0.10, 1.0), (0.30, 0.82), (0.45, 0.62), (0.60, 0.42), (0.75, 0.22), (0.95, 0.05), (1.20, 0.0)],
    },
    "connective_density": {
        "name": "连接词密度",
        "weight": 1.1,
        "desc": "每百字过渡/连接词个数；AI 常靠显式连接词推进论述",
        "anchors": [(0.0, 0.05), (1.0, 0.30), (2.0, 0.52), (3.5, 0.78), (5.0, 0.95), (8.0, 1.0)],
    },
    "boilerplate_density": {
        "name": "套话密度",
        "weight": 1.3,
        "desc": "每百字学术套话命中数；模板化综述的典型标志",
        "anchors": [(0.0, 0.03), (0.8, 0.25), (1.6, 0.50), (3.0, 0.78), (4.5, 0.95), (7.0, 1.0)],
    },
    "parallelism_density": {
        "name": "对仗排比密度",
        "weight": 1.0,
        "desc": "每百字「不仅…而且」「一方面…另一方面」类结构数",
        "anchors": [(0.0, 0.05), (0.5, 0.30), (1.2, 0.58), (2.2, 0.82), (3.5, 1.0)],
    },
    "hedge_density": {
        "name": "模糊限制语密度",
        "weight": 0.7,
        "desc": "每百字「可能/在一定程度上」等模糊限制语个数",
        "anchors": [(0.0, 0.08), (0.8, 0.30), (1.8, 0.55), (3.2, 0.82), (5.0, 1.0)],
    },
    "idiom_density": {
        "name": "四字格密度",
        "weight": 0.5,
        "desc": "常见四字格命中数占汉字总数的比例；四字格堆叠常见于机写文本",
        "anchors": [(0.0, 0.15), (0.01, 0.35), (0.02, 0.55), (0.04, 0.78), (0.07, 1.0)],
    },
    "num_ratio": {
        "name": "数字密度",
        "weight": 0.7,
        "desc": "每百字数字/百分数个数；具体数据越多越像人工实证写作",
        "anchors": [(0.0, 0.95), (0.8, 0.78), (2.0, 0.55), (4.0, 0.30), (7.0, 0.10), (10.0, 0.0)],
    },
    "first_person_density": {
        "name": "第一人称密度",
        "weight": 0.6,
        "desc": "每百字我/我们/笔者/本人出现次数；作者介入是人工写作的迹象",
        "anchors": [(0.0, 0.80), (0.5, 0.62), (1.5, 0.38), (3.0, 0.16), (5.0, 0.0)],
    },
    "passive_density": {
        "name": "被动标记密度",
        "weight": 0.6,
        "desc": "每百字被/由/受到/得以/加以等被动标记个数",
        "anchors": [(0.0, 0.08), (0.6, 0.32), (1.5, 0.58), (2.8, 0.82), (4.0, 1.0)],
    },
    "char_ttr": {
        "name": "词面丰富度",
        "weight": 0.8,
        "desc": "字符类 TTR（长文本用 MATTR，窗宽 50）；越低说明用词越单一",
        "anchors": [(0.55, 1.0), (0.68, 0.88), (0.76, 0.66), (0.83, 0.42), (0.88, 0.24), (0.93, 0.10), (1.0, 0.0)],
    },
    "repetition_ratio": {
        "name": "三元组重复率",
        "weight": 0.9,
        "desc": "重复出现的汉字 3-gram 占全部 3-gram 的比例",
        "anchors": [(0.0, 0.10), (0.02, 0.30), (0.05, 0.55), (0.10, 0.80), (0.20, 1.0)],
    },
    "char_entropy": {
        "name": "汉字一元熵",
        "weight": 0.2,
        "desc": "汉字 unigram 香农熵（bits）；偏低说明用字集中（弱信号：对段落长度敏感，仅作参考）",
        "anchors": [(3.0, 1.0), (4.5, 0.85), (5.5, 0.70), (6.4, 0.55), (7.2, 0.38), (8.0, 0.22), (8.8, 0.08), (9.6, 0.0)],
    },
    "punct_diversity": {
        "name": "标点多样性",
        "weight": 0.4,
        "desc": "标点种类的香农熵（bits）；标点单一常见于机写文本",
        "anchors": [(0.0, 1.0), (0.6, 0.82), (1.2, 0.60), (1.8, 0.38), (2.4, 0.18), (3.3, 0.05)],
    },
}

FEATURE_KEYS: List[str] = list(FEATURE_META.keys())

# 过渡/连接词（逐词条统计出现次数，词条之间允许语义重叠）
_CONNECTIVES: Tuple[str, ...] = (
    "因此", "所以", "然而", "但是", "此外", "另外", "首先", "其次", "再次",
    "最后", "总之", "总的来说", "与此同时", "进一步", "进而", "从而", "由于",
    "尽管", "虽然", "不仅", "而且", "并且", "换言之", "具体而言", "一方面",
    "另一方面", "与之相比", "在这个意义上", "在此基础上", "在此背景下",
    "从整体上看", "需要指出的是", "需要注意的是", "从这个意义上讲", "综上",
)

# 学术套话：模板化综述的高频固定表达
# 学术套话：模板化综述/机写文本的高频固定表达。
# 说明：这里刻意收录「首先/其次/此外/综上所述/角度」这类结构标记，因为它们是
# 模板化写作最稳定的表面证据；词条之间可能语义重叠（如既算过渡又算套话），
# 这会让模板化文本被多计几次，属于预期的放大效应。
_BOILERPLATE: Tuple[str, ...] = (
    # 篇章推进的固定结构标记
    "首先", "其次", "再次", "此外", "另外", "同时",
    "综上所述", "总而言之", "总的来说", "总之",
    "值得注意的是", "需要指出的是", "需要注意的是",
    "在此背景下", "在此基础上", "本文旨在", "研究结果表明",
    "由此可见", "换言之", "也就是说",
    # 常见四字/短语套话
    "具有重要意义", "具有重要的理论意义", "具有重要的现实意义",
    "具有广阔的应用前景", "提供了新的视角", "提供了有益参考",
    "不可忽视", "不可或缺", "日益", "起到重要作用", "发挥重要作用",
    "在当今社会", "随着社会的发展", "随着科技的发展", "众所周知",
    "不言而喻", "深入探讨", "系统梳理", "全面分析", "进一步提升",
    "有效促进", "有力推动", "进一步", "在很大程度上", "角度来看",
    "一方面", "另一方面", "不仅", "而且",
    # 方法与结果部分的模板化表达
    "本研究采用", "混合研究方法", "综合运用", "全面系统", "力求全面",
    "严格遵循", "知情同意", "匿名化处理", "多层次", "深入剖析",
    "通过上述", "较为有效", "提供参考", "相互关联", "共同作用",
    "单一指标", "有效识别", "应用场景", "制度设计", "多种手段",
    "数据收集阶段", "数据分析阶段", "研究设计", "后续研究",
    "难以从根本上", "定量分析与定性分析",
    # 发展与趋势类固定搭配
    "的快速发展", "的迅猛发展", "的广泛应用", "有效整合", "有效提升",
    "进一步拓展", "长效机制", "重要方向", "重要举措", "关键环节",
    "核心要素", "有力支撑", "重要保障", "内在要求", "必然选择",
    "现实需要", "统筹推进", "协同推进", "全面推进", "持续优化",
    "深度融合", "新的机遇", "新的挑战", "面临挑战", "重要议题",
)

# 套话正则：覆盖「随着……的发展」这类中间夹修饰语的固定框架
_BOILERPLATE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"随着[^。！？!?]{0,15}?发展"),
    re.compile(r"从[^。！？!?]{0,10}?角度来看"),
)

# 模糊限制语：让判断显得缓和、留有余地的表达
_HEDGES: Tuple[str, ...] = (
    "可能", "也许", "或许", "大概", "在一定程度上", "某种程度上",
    "一般来说", "通常", "往往", "似乎", "大致", "相对而言", "不排除",
    "有待", "尚需", "有所", "较为", "可以说是", "基本上", "或多或少",
    "从某种意义上", "大体上", "倾向于", "普遍认为", "一般认为",
)

# 排比/对仗结构：用正则匹配最常见的关联框架
_PARALLEL_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"不仅[^。！？!?]{0,30}?而且"),
    re.compile(r"不但[^。！？!?]{0,30}?还"),
    re.compile(r"既[^。！？!?]{0,20}?又"),
    re.compile(r"一方面[^。！？!?]{0,40}?另一方面"),
    re.compile(r"无论是[^。！？!?]{0,30}?还是"),
    re.compile(r"既[^。！？!?]{0,15}?也"),
)

# 第一人称标记：长词写在前面，避免「我们」被重复计成「我」+「们」
_FIRST_PERSON_RE = re.compile(r"我们|我|笔者|本人")

# 被动标记：「自由/理由」里的「由」不算被动，「由于」也不算
_PASSIVE_RE = re.compile(r"被|受到|得以|加以|(?<![自理])由(?!于)")

# 数字/百分数
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")

# 标点：非字母数字、非空白即视为标点（含全角标点）
_PUNCT_RE = re.compile(r"[^\w\s]")

# 常见四字格词表：只认真正的四字词/成语，避免把任意四个汉字都当成四字格
_IDIOMS: Tuple[str, ...] = tuple(sorted({
    "举足轻重", "相辅相成", "密不可分", "潜移默化", "循序渐进", "因地制宜",
    "实事求是", "与时俱进", "开拓创新", "精益求精", "一丝不苟", "持之以恒",
    "融会贯通", "深入浅出", "层层递进", "环环相扣", "有条不紊", "卓有成效",
    "行之有效", "切实可行", "立竿见影", "不可或缺", "息息相关", "大有可为",
    "任重道远", "刻不容缓", "势在必行", "一脉相承", "源远流长", "博大精深",
    "独树一帜", "别具一格", "以人为本", "统筹兼顾", "扬长避短", "取长补短",
    "推陈出新", "继往开来", "承前启后", "脚踏实地", "高瞻远瞩", "未雨绸缪",
    "防患未然", "见微知著",
}))


def _count_terms(text: str, terms: Tuple[str, ...]) -> int:
    """统计一组固定词条在文本中的总出现次数。"""
    total = 0
    for term in terms:
        total += text.count(term)
    return total


def _safe_div(numerator: float, denominator: float) -> float:
    """安全除法：分母非正时返回 0.0，保证结果有限。"""
    if denominator <= 0:
        return 0.0
    value = numerator / denominator
    if not math.isfinite(value):
        return 0.0
    return value


def _cv(values: List[float]) -> float:
    """变异系数：总体标准差 / 均值。样本不足或均值为 0 时返回 0。"""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) / mean


def _mattr(tokens: List[str], window: int = 50) -> float:
    """移动平均类符/形符比（MATTR）。文本不超过一个窗口时退化为普通 TTR。"""
    n = len(tokens)
    if n <= 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    total = 0.0
    windows = 0
    for start in range(0, n - window + 1):
        chunk = tokens[start:start + window]
        total += len(set(chunk)) / window
        windows += 1
    return _safe_div(total, float(windows))


def _shannon_entropy(chars: List[str]) -> float:
    """对字符列表做香农熵（bits）。空列表返回 0。"""
    n = len(chars)
    if n <= 0:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in chars:
        counts[ch] = counts.get(ch, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log(p, 2)
    return entropy


def _cjk_chars(text: str) -> List[str]:
    """取出全部汉字字符（保持原顺序）。"""
    return [ch for ch in (text or "") if "\u4e00" <= ch <= "\u9fff"]


def extract_features(text: str) -> FeatureSet:
    """提取 17 个文体特征。空文本/超短文本返回全 0，不抛异常。"""
    raw = text or ""
    cjk_count = count_cjk(raw)
    if cjk_count < MIN_CJK_FOR_FEATURES:
        return FeatureSet({key: 0.0 for key in FEATURE_KEYS})

    # 句级统计
    sentence_lens: List[float] = []
    for sentence in split_sentences(raw):
        length = count_cjk(sentence)
        if length > 0:
            sentence_lens.append(float(length))
    n_sentence = len(sentence_lens)
    avg_sentence_len = _safe_div(sum(sentence_lens), float(n_sentence))
    sentence_len_cv = _cv(sentence_lens)
    short_sentence_ratio = _safe_div(
        float(sum(1 for x in sentence_lens if x < 12)), float(n_sentence)
    )
    long_sentence_ratio = _safe_div(
        float(sum(1 for x in sentence_lens if x > 45)), float(n_sentence)
    )

    # 分句级统计
    clause_lens: List[float] = []
    for clause in split_clauses(raw):
        length = count_cjk(clause)
        if length > 0:
            clause_lens.append(float(length))
    clause_len_cv = _cv(clause_lens)

    # 词表级统计
    tokens = tokenize_words(raw)
    if len(tokens) > MATTR_MIN_TOKENS:
        char_ttr = _mattr(tokens, 50)
    else:
        char_ttr = _safe_div(float(len(set(tokens))), float(len(tokens)))

    trigrams = char_ngrams(raw, 3)
    if trigrams:
        repetition_ratio = _safe_div(
            float(len(trigrams) - len(set(trigrams))), float(len(trigrams))
        )
    else:
        repetition_ratio = 0.0

    # 密度类：统一以「每百字」为量纲，分母用汉字数
    per_hundred = 100.0 / cjk_count
    connective_density = _count_terms(raw, _CONNECTIVES) * per_hundred
    boilerplate_hits = _count_terms(raw, _BOILERPLATE)
    boilerplate_hits += sum(len(pattern.findall(raw)) for pattern in _BOILERPLATE_PATTERNS)
    boilerplate_density = boilerplate_hits * per_hundred
    parallelism_density = sum(len(p.findall(raw)) for p in _PARALLEL_PATTERNS) * per_hundred
    hedge_density = _count_terms(raw, _HEDGES) * per_hundred
    idiom_density = _safe_div(float(_count_terms(raw, _IDIOMS)), float(cjk_count))
    num_ratio = len(_NUMBER_RE.findall(raw)) * per_hundred
    first_person_density = len(_FIRST_PERSON_RE.findall(raw)) * per_hundred
    passive_density = len(_PASSIVE_RE.findall(raw)) * per_hundred

    # 熵类
    char_entropy = _shannon_entropy(_cjk_chars(raw))
    punct_chars = _PUNCT_RE.findall(raw)
    punct_diversity = _shannon_entropy(punct_chars)

    raw_values: Dict[str, float] = {
        "sentence_len_cv": sentence_len_cv,
        "avg_sentence_len": avg_sentence_len,
        "short_sentence_ratio": short_sentence_ratio,
        "long_sentence_ratio": long_sentence_ratio,
        "clause_len_cv": clause_len_cv,
        "connective_density": connective_density,
        "boilerplate_density": boilerplate_density,
        "parallelism_density": parallelism_density,
        "hedge_density": hedge_density,
        "idiom_density": idiom_density,
        "num_ratio": num_ratio,
        "first_person_density": first_person_density,
        "passive_density": passive_density,
        "char_ttr": char_ttr,
        "repetition_ratio": repetition_ratio,
        "char_entropy": char_entropy,
        "punct_diversity": punct_diversity,
    }

    # 按 FEATURE_META 的固定顺序输出，并过滤 NaN/inf/负值
    values: Dict[str, float] = {}
    for key in FEATURE_KEYS:
        value = float(raw_values.get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            value = 0.0
        values[key] = value
    return FeatureSet(values)


def extract_features_dict(text: str) -> Dict[str, float]:
    """返回特征字典的浅拷贝，便于直接序列化或比较。"""
    return dict(extract_features(text).values)

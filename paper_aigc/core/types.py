"""共享数据类型 —— 所有模块的唯一契约。

注意：本工具**未与知网校准**。ScoreReport 中的分数是一个 0-100 的启发式
"AI 痕迹指数"，绝对值不可与知网 AIGC 率直接比较；同一版本内的前后对比、
段落间排序是有效的。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

VERSION = "0.1.0"

DISCLAIMER_ZH = (
    "本分数是本工具基于文体特征计算的启发式「AI 痕迹指数」，没有使用知网的检测模型，"
    "也没有用知网结果做过校准。它不能当作知网 AIGC 率来引用，两者绝对值不可直接比较。"
    "可用的结论只有两类：同一版本内的前后对比（改写是否让文本更自然），以及段落之间的相对排序。"
)

LEVELS: List[str] = ["低", "中", "偏高", "高"]


@dataclass
class Paragraph:
    """一个非空正文段落。"""

    index: int
    text: str
    style: str = ""
    is_heading: bool = False
    is_reference: bool = False
    char_count: int = 0

    def __post_init__(self) -> None:
        if not self.char_count and self.text:
            from .textnorm import count_words

            self.char_count = count_words(self.text)


@dataclass
class Document:
    """一份待检测文档。仅收录非空段落。"""

    path: str
    title: str = ""
    paragraphs: List[Paragraph] = field(default_factory=list)
    body_indices: List[int] = field(default_factory=list)
    total_words: int = 0

    @property
    def file_name(self) -> str:
        import os

        return os.path.basename(self.path)


@dataclass
class FeatureSet:
    """一段文本的特征向量。key 必须来自 features.FEATURE_META。"""

    values: Dict[str, float] = field(default_factory=dict)


@dataclass
class ParagraphScore:
    """单段评分结果。score 为 0-100，越高越像 AI 生成。"""

    index: int
    text: str
    score: float = 0.0
    level: str = "低"
    features: Dict[str, float] = field(default_factory=dict)
    top_signals: List[Tuple[str, str, float]] = field(default_factory=list)
    suggestion: str = ""


@dataclass
class ScoreReport:
    """整篇评分结果。"""

    document: Document
    paragraphs: List[ParagraphScore] = field(default_factory=list)
    overall: float = 0.0
    level: str = "低"
    level_counts: Dict[str, int] = field(default_factory=dict)
    calibration: str = "uncalibrated"
    disclaimer: str = DISCLAIMER_ZH
    meta: Dict[str, object] = field(default_factory=dict)

    def sorted_by_risk(self) -> List[ParagraphScore]:
        return sorted(self.paragraphs, key=lambda p: p.score, reverse=True)


@dataclass
class RewriteResult:
    """单段改写结果。"""

    index: int
    original: str
    revised: str = ""
    score_before: float = 0.0
    score_after: float = 0.0
    attempts: int = 0
    ok: bool = False
    error: str = ""
    notes: List[str] = field(default_factory=list)

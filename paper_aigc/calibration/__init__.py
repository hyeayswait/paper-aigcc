"""公开数据校准层：把手工启发式指数替换为在真实中文 AIGC 语料上拟合的概率。"""
from __future__ import annotations

from .scorer import (
    CalibratedModel,
    is_available,
    model_metrics,
    get_model,
)

__all__ = ["CalibratedModel", "is_available", "model_metrics", "get_model"]

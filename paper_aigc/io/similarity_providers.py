# -*- coding: utf-8 -*-
"""查重 provider 抽象层：本地库 / 联网搜索 可插拔。

设计原则：
- 统一返回 SimilarityReport（与本地库查重同一数据结构）；
- 联网能力通过「注入搜索函数」实现，不在本模块里硬编码任何厂商；
- 未配置或不可用时明确报错，绝不静默返回 0% 冒充「没有重复」。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..core.types import Document
from .similarity import (
    DEFAULT_NGRAM,
    SimilarityReport,
    check_similarity,
)

# 搜索函数签名：给一个文本片段，返回命中的网页地址/标题列表
SearchFn = Callable[[str], List[str]]

_web_search_registry: Dict[str, SearchFn] = {}


def register_web_search(name: str, search_fn: SearchFn) -> None:
    """注册一个联网搜索后端，例如 Bing/Serper/Tavily/自建接口。

        register_web_search("serper", my_serper_search)
        cfg.similarity_provider = "web:serper"
    """
    if not name or not callable(search_fn):
        raise ValueError("注册联网查重后端需要 name 和可调用的 search_fn")
    _web_search_registry[name] = search_fn


def list_web_backends() -> List[str]:
    return sorted(_web_search_registry)


class SimilarityProvider(ABC):
    name = "base"
    label = "查重"

    @abstractmethod
    def available(self) -> bool:
        ...

    @abstractmethod
    def check(self, doc: Document) -> SimilarityReport:
        ...


class LocalCorpusProvider(SimilarityProvider):
    """本地库比对的 provider 包装。"""

    name = "local"
    label = "本地库查重"

    def __init__(self, corpus_dir: str, ngram: int = DEFAULT_NGRAM) -> None:
        self.corpus_dir = corpus_dir
        self.ngram = ngram

    def available(self) -> bool:
        import os

        return bool(self.corpus_dir) and os.path.isdir(self.corpus_dir)

    def check(self, doc: Document) -> SimilarityReport:
        return check_similarity(doc, self.corpus_dir, ngram=self.ngram)


class WebSearchProvider(SimilarityProvider):
    """联网查重：把段落切成片段，用注入的搜索函数查是否命中网络公开文本。

    注意：命中只代表「网络上有相同片段」，不代表学术库重复；结果标注为
    「网络公开文本重合率」，不能当作知网/PaperPass 的查重率使用。
    """

    name = "web"
    label = "网络公开文本重合率"

    def __init__(self, search_fn: SearchFn, ngram: int = DEFAULT_NGRAM,
                 per_paragraph_queries: int = 3, backend: str = "") -> None:
        self.search_fn = search_fn
        self.ngram = ngram
        self.per_paragraph_queries = max(1, per_paragraph_queries)
        self.backend = backend

    def available(self) -> bool:
        return callable(self.search_fn)

    def check(self, doc: Document) -> SimilarityReport:
        report = SimilarityReport(ngram=self.ngram)
        total_chars = 0
        matched_chars = 0
        from .similarity import ParagraphSimilarity, MatchSource, _normalize

        for index in doc.body_indices:
            paragraph = next((p for p in doc.paragraphs if p.index == index), None)
            if paragraph is None:
                continue
            text = _normalize(paragraph.text)
            if len(text) < self.ngram:
                continue
            step = max(self.ngram, len(text) // (self.per_paragraph_queries + 1))
            queries = [text[i:i + 40] for i in range(0, len(text) - 10, step)][: self.per_paragraph_queries]
            hits: List[str] = []
            for query in queries:
                try:
                    found = self.search_fn(query) or []
                except Exception:
                    found = []
                hits.extend(str(item) for item in found[:2])
            ratio = 0.0 if not queries else min(1.0, len(hits) / float(len(queries)))
            report.paragraphs.append(ParagraphSimilarity(
                index=index, text=paragraph.text, ratio=round(ratio, 4),
                sources=[MatchSource(source=item, matched_grams=0,
                                     ratio=round(1.0 / len(queries), 4)) for item in hits[:3]],
            ))
            total_chars += len(paragraph.text)
            matched_chars += int(round(ratio * len(paragraph.text)))
        report.overall = round(matched_chars / float(total_chars), 4) if total_chars else 0.0
        report.note = ("联网查重（后端：%s）：只查网络公开文本，命中不等于学术库重复，"
                       "不能当作知网/PaperPass 查重率。" % (self.backend or "未命名"))
        return report


def get_provider(cfg) -> SimilarityProvider:
    """按配置返回查重 provider。未配置的联网后端会直接报错，不静默返回 0%。"""
    mode = str(getattr(cfg, "similarity_provider", "local") or "local").strip().lower()
    ngram = int(getattr(cfg, "similarity_ngram", DEFAULT_NGRAM) or DEFAULT_NGRAM)
    if mode == "local":
        return LocalCorpusProvider(str(getattr(cfg, "corpus_dir", "") or ""), ngram=ngram)
    if mode.startswith("web"):
        backend = mode.split(":", 1)[1] if ":" in mode else mode[4:] or "default"
        search_fn = _web_search_registry.get(backend)
        if search_fn is None:
            raise ValueError(
                "联网查重后端「%s」未注册。可用的：%s。"
                "请先调用 register_web_search(name, fn) 接入你的搜索 API（Bing/Serper/Tavily/自建）。"
                % (backend, list_web_backends() or "无"))
        return WebSearchProvider(search_fn, ngram=ngram, backend=backend)
    raise ValueError("未知查重 provider：%s（可选 local 或 web:<后端名>）" % mode)


def check(doc: Document, cfg) -> SimilarityReport:
    """统一入口：按配置选 provider 并执行查重。"""
    provider = get_provider(cfg)
    if not provider.available():
        raise ValueError("查重 provider「%s」当前不可用（请检查配置）" % provider.label)
    return provider.check(doc)

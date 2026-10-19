# -*- coding: utf-8 -*-
"""查重 provider 层测试：本地 / 联网（注入假搜索函数）/ 错误处理。"""
from __future__ import annotations

import os
import tempfile
import unittest

from paper_aigc.config import AppConfig
from paper_aigc.core.types import Document, Paragraph
from paper_aigc.io import similarity_providers as providers


def make_doc(texts):
    doc = Document(path="x.docx", title="t")
    for index, text in enumerate(texts):
        paragraph = Paragraph(index=index, text=text)
        doc.paragraphs.append(paragraph)
        doc.body_indices.append(index)
        doc.total_words += paragraph.char_count
    return doc


BODY = "家电行业已经进入存量竞争阶段，企业之间的竞争由规模扩张转向经营质量比拼，价格压力与渠道竞争都会压缩盈利空间。"


class ProviderTest(unittest.TestCase):
    def test_default_is_local(self) -> None:
        provider = providers.get_provider(AppConfig())
        self.assertIsInstance(provider, providers.LocalCorpusProvider)

    def test_unknown_provider_raises(self) -> None:
        cfg = AppConfig()
        cfg.similarity_provider = "nope"
        with self.assertRaises(ValueError):
            providers.get_provider(cfg)

    def test_web_provider_requires_registration(self) -> None:
        cfg = AppConfig()
        cfg.similarity_provider = "web:notregister"
        with self.assertRaises(ValueError):
            providers.get_provider(cfg)

    def test_web_provider_with_fake_search(self) -> None:
        calls = []

        def fake_search(query: str):
            calls.append(query)
            return ["https://example.com/a"] if "存量竞争" in query else []

        providers.register_web_search("fake", fake_search)
        cfg = AppConfig()
        cfg.similarity_provider = "web:fake"
        report = providers.check(make_doc([BODY]), cfg)
        self.assertTrue(calls)
        self.assertGreater(report.overall_percent, 0.0)
        self.assertIn("网络公开文本", report.note)

    def test_unavailable_local_provider_raises(self) -> None:
        cfg = AppConfig()
        cfg.corpus_dir = os.path.join(tempfile.mkdtemp(), "missing")
        with self.assertRaises(ValueError):
            providers.check(make_doc([BODY]), cfg)


if __name__ == "__main__":
    unittest.main()

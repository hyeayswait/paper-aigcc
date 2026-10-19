# -*- coding: utf-8 -*-
"""降重专项测试：重复率下降、无重复跳过、异常降级、取消与重试（全离线）。"""
from __future__ import annotations

import os
import tempfile
import unittest

from paper_aigc.config import AppConfig
from paper_aigc.core.types import Document, Paragraph
from paper_aigc.rewrite.dedup import (
    build_dedup_prompt,
    build_dedup_retry_prompt,
    dedup_document,
)

CORPUS_TEXT = (
    "家电行业已经进入存量竞争阶段，企业之间的竞争由规模扩张转向经营质量比拼，"
    "价格压力与渠道竞争都会压缩盈利空间。"
)
REWRITTEN_TEXT = (
    "家电企业如今面对的是存量市场，靠铺规模取胜的路子已经走不通。"
    "比拼的重点回到经营质量，价格战与渠道竞争会挤压利润，盈利空间因此收窄。"
)
UNRELATED_TEXT = (
    "我统计了十七位教师的访谈记录，其中只有三位能说清检测系统的名称，"
    "其余人只记得学校开过一次会。"
)
NUMBER_TEXT = (
    "表4显示营业总收入从 1234.56 亿元增长至 2345.67 亿元，增速达到 15.20%[3]，"
    "但加权净资产收益率从 18.50% 回落至 16.20%。"
)
NUMBER_REWRITE = (
    "营业总收入从 1234.56 亿元升至 2345.67 亿元，这是表4 给出的数字；"
    "增速 15.20%[3]，不低。但加权净资产收益率从 18.50% 回落到 16.20%，"
    "资本使用效率在下滑。"
)


class FakeClient:
    def __init__(self, reply: str = REWRITTEN_TEXT, error: Exception = None,
                 replies=None) -> None:
        self.reply = reply
        self.error = error
        self.replies = list(replies or [])
        self.calls = []

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        if self.replies:
            return self.replies.pop(0)
        return self.reply


def make_doc(texts) -> Document:
    doc = Document(path="fake.docx", title="测试文档")
    for index, text in enumerate(texts):
        paragraph = Paragraph(index=index, text=text)
        doc.paragraphs.append(paragraph)
        doc.body_indices.append(index)
        doc.total_words += paragraph.char_count
    return doc


def make_cfg(**kwargs) -> AppConfig:
    base = dict(api_key="sk-test", max_rewrite_attempts=3, temperature=1.1, max_tokens=1024)
    base.update(kwargs)
    return AppConfig(**base)


class DedupDocumentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.mkdtemp()
        self.corpus = os.path.join(self.temp, "corpus")
        os.makedirs(self.corpus)
        with open(os.path.join(self.corpus, "ref.txt"), "w", encoding="utf-8") as handle:
            handle.write(CORPUS_TEXT)

    def test_identical_paragraph_is_rewritten(self) -> None:
        doc = make_doc([CORPUS_TEXT])
        client = FakeClient()
        results = dedup_document(doc, client, make_cfg(), self.corpus)
        self.assertEqual(set(results.keys()), {0})
        outcome = results[0]
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.error, "")
        self.assertEqual(outcome.revised, REWRITTEN_TEXT)
        self.assertGreaterEqual(outcome.score_before, 90.0)
        self.assertLessEqual(outcome.score_before, 100.0)
        self.assertLess(outcome.score_after, outcome.score_before)
        self.assertGreaterEqual(outcome.attempts, 1)
        self.assertGreaterEqual(len(client.calls), 1)

    def test_unrelated_paragraph_is_skipped(self) -> None:
        doc = make_doc([UNRELATED_TEXT])
        client = FakeClient()
        results = dedup_document(doc, client, make_cfg(), self.corpus)
        self.assertEqual(results, {})
        self.assertEqual(client.calls, [])

    def test_client_error_is_captured_not_raised(self) -> None:
        doc = make_doc([CORPUS_TEXT])
        client = FakeClient(error=RuntimeError("API Key 无效"))
        results = dedup_document(doc, client, make_cfg(), self.corpus)
        self.assertEqual(len(results), 1)
        outcome = results[0]
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.error)
        self.assertIn("API Key 无效", outcome.error)
        self.assertEqual(outcome.score_after, outcome.score_before)

    def test_missing_corpus_returns_empty(self) -> None:
        doc = make_doc([CORPUS_TEXT])
        client = FakeClient()
        results = dedup_document(doc, client, make_cfg(),
                                 os.path.join(self.temp, "missing"))
        self.assertEqual(results, {})
        self.assertEqual(client.calls, [])

    def test_cancel_before_start_skips_all(self) -> None:
        doc = make_doc([CORPUS_TEXT, CORPUS_TEXT])
        client = FakeClient()
        results = dedup_document(doc, client, make_cfg(), self.corpus,
                                 cancel=lambda: True)
        self.assertEqual(results, {})
        self.assertEqual(client.calls, [])

    def test_cancel_midway_keeps_finished_results(self) -> None:
        doc = make_doc([CORPUS_TEXT, CORPUS_TEXT])
        client = FakeClient()
        results = dedup_document(doc, client, make_cfg(), self.corpus,
                                 cancel=lambda: len(client.calls) >= 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(client.calls), 1)

    def test_retries_when_rewrite_loses_facts(self) -> None:
        corpus = os.path.join(self.temp, "num_corpus")
        os.makedirs(corpus)
        with open(os.path.join(corpus, "num.txt"), "w", encoding="utf-8") as handle:
            handle.write(NUMBER_TEXT)
        bad = "收入涨了很多。利润也在涨。"
        client = FakeClient(replies=[bad, NUMBER_REWRITE])
        results = dedup_document(make_doc([NUMBER_TEXT]), client, make_cfg(), corpus)
        outcome = results[0]
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.attempts, 2)
        self.assertEqual(len(client.calls), 2)
        self.assertLess(outcome.score_after, outcome.score_before)

    def test_attempts_capped_at_three(self) -> None:
        client = FakeClient(replies=["", "", "", ""])
        results = dedup_document(make_doc([CORPUS_TEXT]), client,
                                 make_cfg(max_rewrite_attempts=9), self.corpus)
        outcome = results[0]
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.error)
        self.assertEqual(outcome.attempts, 3)
        self.assertEqual(len(client.calls), 3)

    def test_best_version_is_kept(self) -> None:
        """第一版通过校验但未降重，第二版更低，应保留第二版。"""
        client = FakeClient(replies=[CORPUS_TEXT, REWRITTEN_TEXT])
        results = dedup_document(make_doc([CORPUS_TEXT]), client, make_cfg(), self.corpus)
        outcome = results[0]
        self.assertTrue(outcome.ok, outcome.error)
        self.assertEqual(outcome.revised, REWRITTEN_TEXT)
        self.assertLess(outcome.score_after, outcome.score_before)
        self.assertEqual(len(client.calls), 2)

    def test_progress_messages_reported(self) -> None:
        messages = []
        dedup_document(make_doc([CORPUS_TEXT]), FakeClient(), make_cfg(), self.corpus,
                       progress=messages.append)
        self.assertTrue(messages)
        self.assertTrue(any("降重" in message for message in messages))

    def test_progress_exception_does_not_break(self) -> None:
        def broken(_message):
            raise RuntimeError("progress 回调异常")
        results = dedup_document(make_doc([CORPUS_TEXT]), FakeClient(), make_cfg(),
                                 self.corpus, progress=broken)
        self.assertTrue(results[0].ok)


class PromptTest(unittest.TestCase):
    def test_prompt_contains_constraints_and_original(self) -> None:
        prompt = build_dedup_prompt(CORPUS_TEXT, 0.9, 0.15)
        self.assertIn(CORPUS_TEXT, prompt)
        for token in ("同义", "语序", "数字", "百分比", "引用标记 [n]", "表4/图2",
                      "不得新增事实", "只输出正文"):
            self.assertIn(token, prompt)
        self.assertIn("90.0%", prompt)
        self.assertIn("15.0%", prompt)

    def test_retry_prompt_contains_previous_and_problems(self) -> None:
        prompt = build_dedup_retry_prompt(CORPUS_TEXT, REWRITTEN_TEXT, 50.0, 0.15,
                                          ["漏掉数字：15.20"])
        self.assertIn(REWRITTEN_TEXT, prompt)
        self.assertIn("漏掉数字：15.20", prompt)
        self.assertIn(CORPUS_TEXT, prompt)
        self.assertIn("50.0%", prompt)


if __name__ == "__main__":
    unittest.main()

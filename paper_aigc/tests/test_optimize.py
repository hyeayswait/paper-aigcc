# -*- coding: utf-8 -*-
"""全文优化模块测试：事实护栏 + 迭代收敛 + 结构保持 + 异常降级（全离线）。"""
from __future__ import annotations

import unittest

from paper_aigc.config import AppConfig
from paper_aigc.core import scoring
from paper_aigc.core.types import Document, Paragraph
from paper_aigc.rewrite import optimize
from paper_aigc.rewrite.optimize import (
    apply_replacements,
    build_optimize_prompt,
    build_optimize_retry_prompt,
    optimize_document,
    validate_rewrite,
)

HIGH_TEXT = (
    "随着企业规模持续扩张，表4显示营业总收入从 1234.56 亿元增长至 2345.67 亿元，"
    "增速达到 15.20%[3]，但加权净资产收益率从 18.50% 回落至 16.20%，"
    "这表明企业的资本使用效率存在一定程度下降，值得进一步关注与分析。"
)
LOW_TEXT = (
    "表4 的数字我看了。收入 1234.56 亿元到 2345.67 亿元。增速 15.20%[3]。不慢。"
    "但 ROE 从 18.50% 掉到 16.20%。钱投出去了，回报没跟上。这才是问题。"
)


class FakeClient:
    def __init__(self, reply: str = LOW_TEXT, error: Exception = None) -> None:
        self.reply = reply
        self.error = error
        self.calls = []

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
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
    base = dict(api_key="sk-test", optimize_target=18.0, optimize_rounds=2,
                optimize_max_calls=50, concurrency=2, max_rewrite_attempts=2, temperature=1.1)
    base.update(kwargs)
    return AppConfig(**base)


class ValidateRewriteTest(unittest.TestCase):
    def test_accepts_faithful_rewrite(self) -> None:
        passed, problems = validate_rewrite(HIGH_TEXT, LOW_TEXT)
        self.assertTrue(passed, problems)
        self.assertEqual(problems, [])

    def test_rejects_missing_number(self) -> None:
        passed, problems = validate_rewrite(HIGH_TEXT, "收入涨了不少。利润也在涨。")
        self.assertFalse(passed)
        self.assertTrue(any("漏掉数字" in item for item in problems), problems)

    def test_rejects_invented_number(self) -> None:
        text = LOW_TEXT + "新增 999 亿元。"
        passed, problems = validate_rewrite(HIGH_TEXT, text)
        self.assertFalse(passed)
        self.assertTrue(any("凭空新增数字" in item for item in problems), problems)

    def test_rejects_lost_citation(self) -> None:
        text = LOW_TEXT.replace("[3]", "")
        passed, problems = validate_rewrite(HIGH_TEXT, text)
        self.assertFalse(passed)
        self.assertTrue(any("引用标记" in item for item in problems), problems)

    def test_rejects_lost_table_reference(self) -> None:
        text = LOW_TEXT.replace("表4", "该表")
        passed, problems = validate_rewrite(HIGH_TEXT, text)
        self.assertFalse(passed)
        self.assertTrue(any("表号" in item for item in problems), problems)

    def test_rejects_length_collapse(self) -> None:
        passed, problems = validate_rewrite(HIGH_TEXT, "收入 1234.56 2345.67 15.20 18.50 16.20 表4[3]")
        self.assertFalse(passed)
        self.assertTrue(any("塌缩" in item or "字数" in item for item in problems), problems)

    def test_rejects_empty(self) -> None:
        passed, problems = validate_rewrite(HIGH_TEXT, "   ")
        self.assertFalse(passed)
        self.assertIn("改写结果为空", problems)

    def test_clean_output_strips_fence_and_prefix(self) -> None:
        raw = "```text\n改写后：表4 的数字我看了。\n```"
        self.assertEqual(optimize._clean_output(raw), "表4 的数字我看了。")


class PromptTest(unittest.TestCase):
    def test_prompt_contains_rules_and_target(self) -> None:
        prompt = build_optimize_prompt(HIGH_TEXT, [("boilerplate_density", "套话密度", 6.1)], 18.0)
        self.assertIn(HIGH_TEXT, prompt)
        self.assertIn("18", prompt)
        self.assertIn("22-32", prompt)
        self.assertIn("套话密度", prompt)

    def test_retry_prompt_includes_previous_and_problems(self) -> None:
        prompt = build_optimize_retry_prompt(HIGH_TEXT, LOW_TEXT, 30.0, 18.0, [], ["漏掉数字：15.20"])
        self.assertIn("漏掉数字：15.20", prompt)
        self.assertIn("30.0", prompt)
        self.assertIn(HIGH_TEXT, prompt)


class OptimizeDocumentTest(unittest.TestCase):
    def test_reaches_target(self) -> None:
        doc = make_doc([HIGH_TEXT, HIGH_TEXT, HIGH_TEXT])
        report = scoring.score_document(doc)
        self.assertGreater(report.overall, 18.0)
        client = FakeClient()
        results = optimize_document(doc, report, client, make_cfg(), target=18.0)
        self.assertTrue(results)
        accepted = {i: r.revised for i, r in results.items() if r.ok and r.revised}
        self.assertEqual(len(accepted), 3)
        after = scoring.score_document(apply_replacements(doc, accepted))
        self.assertLess(after.overall, 18.0)
        self.assertGreater(len(client.calls), 0)

    def test_skips_paragraphs_already_below_target(self) -> None:
        doc = make_doc([LOW_TEXT, LOW_TEXT])
        report = scoring.score_document(doc)
        self.assertLess(report.overall, 18.0)
        client = FakeClient()
        results = optimize_document(doc, report, client, make_cfg(), target=18.0)
        self.assertEqual(results, {})
        self.assertEqual(client.calls, [])

    def test_client_error_is_captured_not_raised(self) -> None:
        doc = make_doc([HIGH_TEXT])
        report = scoring.score_document(doc)
        client = FakeClient(error=RuntimeError("API Key 无效"))
        results = optimize_document(doc, report, client, make_cfg(optimize_rounds=1), target=18.0)
        self.assertEqual(len(results), 1)
        outcome = list(results.values())[0]
        self.assertFalse(outcome.ok)
        self.assertTrue(outcome.error)

    def test_cancel_stops_before_calling(self) -> None:
        doc = make_doc([HIGH_TEXT])
        report = scoring.score_document(doc)
        client = FakeClient()
        results = optimize_document(doc, report, client, make_cfg(optimize_rounds=1),
                                    target=18.0, cancel=lambda: True)
        self.assertEqual(results, {})
        self.assertEqual(client.calls, [])

    def test_budget_limits_total_calls(self) -> None:
        doc = make_doc([HIGH_TEXT] * 4)
        report = scoring.score_document(doc)
        client = FakeClient()
        optimize_document(doc, report, client, make_cfg(optimize_max_calls=1,
                                                        concurrency=1, optimize_rounds=1), target=18.0)
        self.assertLessEqual(len(client.calls), 1)

    def test_rejects_rewrite_that_drops_numbers(self) -> None:
        """模型把数字改丢时，本次改写不得被接受。"""
        doc = make_doc([HIGH_TEXT])
        report = scoring.score_document(doc)
        client = FakeClient(reply="收入涨了很多。利润也在涨。但回报没跟上。")
        results = optimize_document(doc, report, client, make_cfg(optimize_rounds=1), target=18.0)
        outcome = list(results.values())[0]
        self.assertFalse(outcome.ok)
        self.assertIn("漏掉数字", outcome.error)


class ApplyReplacementsTest(unittest.TestCase):
    def test_preserves_paragraph_count_and_headings(self) -> None:
        heading = Paragraph(index=0, text="一、引言", is_heading=True)
        body = Paragraph(index=1, text=HIGH_TEXT)
        tail = Paragraph(index=2, text="[1] 张三. 某文献[J]. 2024.")
        doc = Document(path="x.docx", title="t", paragraphs=[heading, body, tail],
                       body_indices=[1], total_words=body.char_count)
        new_doc = apply_replacements(doc, {1: LOW_TEXT})
        self.assertEqual(len(new_doc.paragraphs), 3)
        self.assertEqual(new_doc.paragraphs[0].text, "一、引言")
        self.assertTrue(new_doc.paragraphs[0].is_heading)
        self.assertEqual(new_doc.paragraphs[1].text, LOW_TEXT)
        self.assertEqual(new_doc.paragraphs[2].text, tail.text)
        self.assertEqual(new_doc.body_indices, [1])
        self.assertEqual(new_doc.total_words, Paragraph(index=1, text=LOW_TEXT).char_count)


class StrayEnglishTest(unittest.TestCase):
    def test_rejects_english_not_in_original(self) -> None:
        text = LOW_TEXT.replace("钱投出去了", "钱 also 投出去了")
        passed, problems = validate_rewrite(HIGH_TEXT, text)
        self.assertFalse(passed)
        self.assertTrue(any("混入" in item and "英文" in item for item in problems), problems)

    def test_allows_english_present_in_original(self) -> None:
        passed, problems = validate_rewrite(HIGH_TEXT, LOW_TEXT)
        self.assertTrue(passed, problems)


class DocumentQualityTest(unittest.TestCase):
    def test_reports_rhythm_stats(self) -> None:
        stats = optimize.document_quality([LOW_TEXT, LOW_TEXT])
        self.assertGreater(stats["sentences"], 0)
        self.assertGreater(stats["avg_len"], 0)
        self.assertGreaterEqual(stats["author_voice"], 0)

    def test_counts_author_voice(self) -> None:
        text = "笔者认为这可行。笔者认为这不行。"
        stats = optimize.document_quality([text])
        self.assertEqual(stats["author_voice"], 2)


class TargetRateTest(unittest.TestCase):
    """target_rate 优先级高于 optimize_target。"""

    def test_high_target_rate_skips_rewrite(self) -> None:
        doc = make_doc([HIGH_TEXT])
        report = scoring.score_document(doc)
        client = FakeClient()
        cfg = make_cfg()
        cfg.target_rate = 95.0
        cfg.optimize_target = 0.0
        results = optimize_document(doc, report, client, cfg, target=None)
        self.assertEqual(results, {})

    def test_low_target_rate_triggers_rewrite(self) -> None:
        doc = make_doc([HIGH_TEXT])
        report = scoring.score_document(doc)
        client = FakeClient()
        cfg = make_cfg()
        cfg.target_rate = 10.0
        cfg.optimize_target = 95.0
        results = optimize_document(doc, report, client, cfg, target=None)
        self.assertTrue(results)


class ConfigDefaultTest(unittest.TestCase):
    def test_default_target_rate_is_50(self) -> None:
        from paper_aigc.config import AppConfig

        cfg = AppConfig()
        self.assertEqual(cfg.target_rate, 50.0)
        self.assertEqual(cfg.corpus_dir, "")


if __name__ == "__main__":
    unittest.main()

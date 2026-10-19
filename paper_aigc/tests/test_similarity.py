# -*- coding: utf-8 -*-
"""本地库查重模块测试（全离线）。"""
from __future__ import annotations

import os
import tempfile
import unittest

from paper_aigc.core.types import Document, Paragraph
from paper_aigc.io.similarity import check_similarity, to_meta

CORPUS_TEXT = (
    "家电行业已经进入存量竞争阶段，企业之间的竞争由规模扩张转向经营质量比拼，"
    "价格压力与渠道竞争都会压缩盈利空间，上游原材料价格波动也带来成本不确定性。"
)
BODY_TEXT = (
    "家电行业已经进入存量竞争阶段，企业之间的竞争由规模扩张转向经营质量比拼，"
    "价格压力与渠道竞争都会压缩盈利空间，上游原材料价格波动也带来成本不确定性。"
)


def make_doc(texts):
    doc = Document(path="x.docx", title="t")
    for index, text in enumerate(texts):
        paragraph = Paragraph(index=index, text=text)
        doc.paragraphs.append(paragraph)
        doc.body_indices.append(index)
        doc.total_words += paragraph.char_count
    return doc


class SimilarityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.mkdtemp()
        self.corpus = os.path.join(self.temp, "corpus")
        os.makedirs(self.corpus)
        with open(os.path.join(self.corpus, "ref.txt"), "w", encoding="utf-8") as handle:
            handle.write(CORPUS_TEXT)

    def test_identical_paragraph_is_flagged(self) -> None:
        report = check_similarity(make_doc([BODY_TEXT]), self.corpus)
        self.assertGreater(report.overall_percent, 90.0)
        self.assertEqual(report.corpus_files, 1)
        self.assertTrue(report.paragraphs[0].sources)

    def test_unrelated_paragraph_is_clean(self) -> None:
        text = "我统计了十七位教师的访谈记录，其中只有三位能说清检测系统的名称。"
        report = check_similarity(make_doc([text]), self.corpus)
        self.assertLess(report.overall_percent, 10.0)

    def test_punctuation_does_not_affect_match(self) -> None:
        punctuated = CORPUS_TEXT.replace("，", "；").replace("。", "！")
        report = check_similarity(make_doc([punctuated]), self.corpus)
        self.assertGreater(report.overall_percent, 90.0)

    def test_empty_corpus_returns_note(self) -> None:
        report = check_similarity(make_doc([BODY_TEXT]), os.path.join(self.temp, "missing"))
        self.assertEqual(report.overall_percent, 0.0)
        self.assertIn("未找到本地比对库", report.note)

    def test_internal_duplicates_detected(self) -> None:
        report = check_similarity(make_doc([BODY_TEXT, BODY_TEXT + "补充一句。"]), self.corpus)
        self.assertTrue(report.internal_duplicates)

    def test_to_meta_shape(self) -> None:
        report = check_similarity(make_doc([BODY_TEXT]), self.corpus)
        meta = to_meta(report)
        for key in ("overall_percent", "corpus_files", "corpus_chars", "ngram", "note",
                    "paragraphs", "internal_duplicates"):
            self.assertIn(key, meta)
        self.assertTrue(meta["paragraphs"])
        self.assertIn("source", meta["paragraphs"][0]["sources"][0])


class SelfCompareGuardTest(unittest.TestCase):
    def test_warns_when_paper_itself_in_corpus(self) -> None:
        temp = tempfile.mkdtemp()
        corpus = os.path.join(temp, "corpus")
        os.makedirs(corpus)
        body = "家电行业已经进入存量竞争阶段，企业之间的竞争由规模扩张转向经营质量比拼，价格压力与渠道竞争都会压缩盈利空间。"
        with open(os.path.join(corpus, "自己.docx.txt"), "w", encoding="utf-8") as handle:
            handle.write(body)
        report = check_similarity(make_doc([body]), corpus)
        self.assertTrue(report.self_compare)
        self.assertIn("自比", report.note)

    def test_no_warning_for_unrelated_corpus(self) -> None:
        temp = tempfile.mkdtemp()
        corpus = os.path.join(temp, "corpus")
        os.makedirs(corpus)
        with open(os.path.join(corpus, "ref.txt"), "w", encoding="utf-8") as handle:
            handle.write("完全无关的另外一段文字，用来验证不会误报自比警告。")
        report = check_similarity(make_doc([CORPUS_TEXT]), corpus)
        self.assertFalse(report.self_compare)
        self.assertNotIn("自比", report.note)


if __name__ == "__main__":
    unittest.main()

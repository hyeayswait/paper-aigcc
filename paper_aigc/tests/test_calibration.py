# -*- coding: utf-8 -*-
"""校准层与评分引擎切换的测试。"""
from __future__ import annotations

import os
import unittest

from paper_aigc.calibration import get_model, is_available, model_metrics
from paper_aigc.core import scoring

SAMPLE_AI = (
    "随着人工智能技术的快速发展，生成式语言模型在学术写作领域的应用日益广泛，"
    "其在提升写作效率的同时，也引发了关于学术诚信与内容原创性的广泛讨论。"
    "在此背景下，如何有效识别与规范人工智能生成内容，已经成为学术界与教育管理部门共同关注的重要议题。"
)
SAMPLE_HUMAN = (
    "问卷发了两轮。第一轮 312 份，有效 287 份；第二轮只补发给了未回复的 96 人，回来 41 份。"
    "有效样本合计 328。这个回收率不算高，所以我没做加权，直接在文中说明了局限。"
)


@unittest.skipUnless(is_available(), "校准模型文件不存在")
class CalibratedModelTest(unittest.TestCase):
    def test_model_metrics_are_sane(self) -> None:
        metrics = model_metrics()
        self.assertGreater(float(metrics["test_auc"]), 0.7)
        self.assertLessEqual(float(metrics["test_auc"]), 1.0)
        self.assertGreater(int(metrics["n_train"]), 1000)

    def test_probability_in_range(self) -> None:
        model = get_model()
        for text in (SAMPLE_AI, SAMPLE_HUMAN, "短句。测试。"):
            probability = model.probability(text)
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_short_text_scores_zero(self) -> None:
        model = get_model()
        self.assertEqual(model.score("太短。"), 0.0)

    def test_model_prefers_ai_text(self) -> None:
        model = get_model()
        self.assertGreater(model.probability(SAMPLE_AI), model.probability(SAMPLE_HUMAN))


class EngineSwitchTest(unittest.TestCase):
    def tearDown(self) -> None:
        scoring.set_engine("heuristic")

    def test_unknown_engine_rejected(self) -> None:
        with self.assertRaises(ValueError):
            scoring.set_engine("nope")

    def test_thresholds_follow_engine(self) -> None:
        scoring.set_engine("heuristic")
        self.assertEqual(scoring.engine_thresholds(), (25.0, 45.0, 60.0))
        scoring.set_engine("calibrated")
        self.assertEqual(scoring.engine_thresholds(), (40.0, 60.0, 80.0))

    def test_calibrated_scores_are_probability_scaled(self) -> None:
        scoring.set_engine("calibrated")
        result = scoring.score_paragraph(SAMPLE_AI)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 100.0)
        self.assertIn(result.level, ("低", "中", "偏高", "高"))

    @unittest.skipUnless(is_available(), "校准模型文件不存在")
    def test_calibrated_mode_differs_from_heuristic(self) -> None:
        scoring.set_engine("heuristic")
        heuristic = scoring.score_paragraph(SAMPLE_AI).score
        scoring.set_engine("calibrated")
        calibrated = scoring.score_paragraph(SAMPLE_AI).score
        self.assertNotAlmostEqual(heuristic, calibrated, places=1)

    def test_document_meta_records_engine(self) -> None:
        from paper_aigc.core.types import Document, Paragraph

        doc = Document(path="x.docx", title="t")
        for index, text in enumerate((SAMPLE_AI, SAMPLE_HUMAN)):
            paragraph = Paragraph(index=index, text=text)
            doc.paragraphs.append(paragraph)
            doc.body_indices.append(index)
            doc.total_words += paragraph.char_count
        scoring.set_engine("calibrated")
        report = scoring.score_document(doc)
        self.assertEqual(report.meta["engine"], "calibrated-logreg")
        self.assertIn("public-data", report.calibration)
        self.assertEqual(len(report.meta["level_thresholds"]), 3)


if __name__ == "__main__":
    unittest.main()

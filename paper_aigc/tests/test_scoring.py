"""core/scoring.py 的单元测试：等级映射、分数区间、确定性、文档聚合。"""
from __future__ import annotations

import math
import unittest

from paper_aigc.core import scoring
from paper_aigc.core.features import FEATURE_META
from paper_aigc.core.types import (
    DISCLAIMER_ZH,
    LEVELS,
    VERSION,
    Document,
    Paragraph,
)

AI_TEXT = (
    "在当今社会，人工智能技术不仅是推动产业升级的重要引擎，而且是重塑教育生态的关键力量。"
    "一方面，智能技术能够有效提升教学效率，另一方面，它也可能在一定程度上加剧教育不公平。"
    "值得注意的是，相关研究普遍认为，技术应用既需要制度保障，又需要教师参与。"
    "首先，政策层面应当统筹推进基础设施建设，进一步完善相关标准与规范。"
    "其次，学校层面应当有效整合优质资源，建立健全教师培训的长效机制。"
    "此外，教师层面应当持续优化教学设计，把技术工具真正融入课堂实践，进而提升学生的核心素养。"
    "然而，当前研究仍存在明显不足，相关结论有待进一步验证，实践路径尚需系统梳理。"
    "综上所述，人工智能与教育的深度融合具有重要的理论意义和广阔的应用前景，"
    "同时也面临着诸多现实挑战，需要在今后的研究中不断探索与完善。"
)

HUMAN_TEXT = (
    "我带的两个班这学期试了新的作业办法。第一次交上来的东西很糟，"
    "三十七个人里有十四个直接复制了网上的答案。后来我把题目改成必须结合课堂实验数据，"
    "情况才慢慢好转。说到底，学生不是不会写，是不知道写什么值得写。"
)

LONG_TEXT = AI_TEXT * 5


def _make_document() -> Document:
    """构造一份含标题、正文、参考文献的文档，只给 1、3 段打分。"""
    paragraphs = [
        Paragraph(index=0, text="第一章 绪论", style="Heading 1", is_heading=True),
        Paragraph(index=1, text=AI_TEXT, style="正文"),
        Paragraph(index=2, text="[1] 张某. 教育技术研究[J]. 2023(4): 12-18.", is_reference=True),
        Paragraph(index=3, text=HUMAN_TEXT, style="正文"),
    ]
    total = paragraphs[1].char_count + paragraphs[3].char_count
    return Document(
        path="test.docx",
        title="测试文档",
        paragraphs=paragraphs,
        body_indices=[1, 3],
        total_words=total,
    )


class LevelMappingTest(unittest.TestCase):
    """等级阈值必须严格按 SPEC 第 4 节。"""

    def test_thresholds_are_exact(self) -> None:
        self.assertEqual(scoring.LEVEL_THRESHOLDS, (25.0, 45.0, 60.0))
        self.assertEqual(LEVELS, ["低", "中", "偏高", "高"])

    def test_level_boundaries(self) -> None:
        cases = [
            (-10.0, "低"),
            (0.0, "低"),
            (24.9, "低"),
            (25.0, "中"),
            (44.9, "中"),
            (45.0, "偏高"),
            (59.9, "偏高"),
            (60.0, "高"),
            (100.0, "高"),
            (150.0, "高"),
        ]
        for score, expected in cases:
            self.assertEqual(scoring.level_of(score), expected, score)

    def test_non_finite_falls_back_to_low(self) -> None:
        self.assertEqual(scoring.level_of(float("nan")), "低")
        self.assertEqual(scoring.level_of(float("inf")), "低")


class ScoreTextTest(unittest.TestCase):
    """单文本评分的区间、精度与确定性。"""

    def test_score_in_range_and_rounded(self) -> None:
        for text in (AI_TEXT, HUMAN_TEXT, LONG_TEXT, "今天天气很好。我们去公园散步。"):
            score = scoring.score_text(text)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)
            self.assertIsInstance(score, float)
            self.assertEqual(score, round(score, 1))

    def test_deterministic(self) -> None:
        for text in (AI_TEXT, HUMAN_TEXT, LONG_TEXT):
            self.assertEqual(scoring.score_text(text), scoring.score_text(text))

    def test_empty_and_non_chinese_score_zero(self) -> None:
        for text in ("", "   ", "\n\t", "！！！……", "Hello world. No Chinese here.", "你好。"):
            self.assertEqual(scoring.score_text(text), 0.0, text)

    def test_ai_text_scores_higher_than_human_text(self) -> None:
        ai_score = scoring.score_text(AI_TEXT)
        human_score = scoring.score_text(HUMAN_TEXT)
        self.assertGreaterEqual(ai_score, 55.0)
        self.assertGreater(ai_score, human_score)
        self.assertLess(human_score, 50.0)

    def test_score_paragraph_keeps_index_and_text(self) -> None:
        paragraph = scoring.score_paragraph(AI_TEXT, index=7)
        self.assertEqual(paragraph.index, 7)
        self.assertEqual(paragraph.text, AI_TEXT)
        self.assertEqual(paragraph.level, scoring.level_of(paragraph.score))
        self.assertEqual(set(paragraph.features.keys()), set(FEATURE_META.keys()))
        self.assertTrue(all(math.isfinite(v) for v in paragraph.features.values()))

    def test_short_text_paragraph_is_zero(self) -> None:
        paragraph = scoring.score_paragraph("你好。", index=2)
        self.assertEqual(paragraph.score, 0.0)
        self.assertEqual(paragraph.level, "低")
        self.assertEqual(len(paragraph.features), 17)
        self.assertEqual(paragraph.top_signals, [])
        self.assertEqual(paragraph.suggestion, "")

    def test_top_signals_follow_contract(self) -> None:
        paragraph = scoring.score_paragraph(AI_TEXT, index=0)
        signals = paragraph.top_signals
        self.assertLessEqual(len(signals), 4)
        self.assertGreater(len(signals), 0)

        total_weight = sum(float(meta["weight"]) for meta in FEATURE_META.values())
        subs = {}
        for key, meta in FEATURE_META.items():
            subs[key] = scoring._apply_anchors(paragraph.features[key], meta["anchors"])

        strengths = []
        for signal in signals:
            self.assertEqual(len(signal), 3)
            key, name, contribution = signal
            self.assertIn(key, FEATURE_META)
            self.assertEqual(name, FEATURE_META[key]["name"])
            self.assertIsInstance(contribution, float)
            self.assertGreater(contribution, 0.0)
            strength = (subs[key] - 0.5) * float(FEATURE_META[key]["weight"])
            self.assertGreater(strength, 0.0)
            strengths.append(strength)
            expected = round(
                float(FEATURE_META[key]["weight"]) * subs[key] / total_weight * 100.0, 1
            )
            self.assertEqual(contribution, expected)

        self.assertEqual(strengths, sorted(strengths, reverse=True))
        positive = 0
        for key, meta in FEATURE_META.items():
            if (subs[key] - 0.5) * float(meta["weight"]) > 0:
                positive += 1
        self.assertEqual(len(signals), min(4, positive))

    def test_suggestion_is_chinese_and_ends_with_period(self) -> None:
        paragraph = scoring.score_paragraph(AI_TEXT, index=0)
        self.assertTrue(paragraph.suggestion)
        self.assertTrue(paragraph.suggestion.endswith("。"))
        self.assertTrue(any("\u4e00" <= ch <= "\u9fff" for ch in paragraph.suggestion))
        self.assertNotEqual(paragraph.suggestion, "")

    def test_weird_text_never_raises_or_nan(self) -> None:
        for text in ("\x00abc", "😀😀😀", "汉" * 40, "，。！？" * 20, "a" * 2000):
            score = scoring.score_text(text)
            self.assertTrue(math.isfinite(score))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 100.0)


class ScoreDocumentTest(unittest.TestCase):
    """整篇文档的聚合规则。"""

    def test_only_body_indices_are_scored(self) -> None:
        doc = _make_document()
        report = scoring.score_document(doc)
        self.assertEqual(len(report.paragraphs), 2)
        self.assertEqual([p.index for p in report.paragraphs], [1, 3])
        self.assertIs(report.document, doc)

    def test_report_fields_and_disclaimer(self) -> None:
        report = scoring.score_document(_make_document())
        self.assertGreaterEqual(report.overall, 0.0)
        self.assertLessEqual(report.overall, 100.0)
        self.assertEqual(report.overall, round(report.overall, 1))
        self.assertEqual(report.level, scoring.level_of(report.overall))
        self.assertEqual(report.calibration, "uncalibrated")
        self.assertEqual(report.disclaimer, DISCLAIMER_ZH)
        self.assertIn("知网", report.disclaimer)
        self.assertEqual(set(report.level_counts.keys()), set(LEVELS))
        self.assertEqual(sum(report.level_counts.values()), 2)
        for paragraph_score in report.paragraphs:
            self.assertEqual(paragraph_score.level, scoring.level_of(paragraph_score.score))

    def test_overall_is_char_count_weighted(self) -> None:
        doc = _make_document()
        report = scoring.score_document(doc)
        total_weight = 0
        weighted = 0.0
        for paragraph in doc.paragraphs:
            if paragraph.index not in doc.body_indices:
                continue
            score = scoring.score_paragraph(paragraph.text, paragraph.index).score
            total_weight += paragraph.char_count
            weighted += score * paragraph.char_count
        expected = round(weighted / total_weight, 1)
        self.assertAlmostEqual(report.overall, expected, places=6)

    def test_paragraph_scores_match_direct_scoring(self) -> None:
        doc = _make_document()
        report = scoring.score_document(doc)
        for paragraph_score in report.paragraphs:
            direct = scoring.score_paragraph(paragraph_score.text, paragraph_score.index)
            self.assertEqual(paragraph_score.score, direct.score)
            self.assertEqual(paragraph_score.level, direct.level)

    def test_meta_is_filled(self) -> None:
        report = scoring.score_document(_make_document())
        self.assertEqual(report.meta.get("version"), VERSION)
        self.assertEqual(report.meta.get("feature_count"), 17)
        self.assertEqual(report.meta.get("engine"), "stylometric-heuristic")
        # 全文画像由 Lead 追加：必须存在且字段完整
        profile = report.meta.get("profile")
        self.assertIsInstance(profile, dict)
        for key in ("paragraphs", "dispersion", "homogeneous", "global_signals", "note"):
            self.assertIn(key, profile)
        self.assertIsInstance(profile["homogeneous"], bool)
        self.assertIsInstance(profile["global_signals"], list)
        self.assertTrue(str(profile["note"]))

    def test_profile_flags_homogeneous_text(self) -> None:
        """同一段文字复制多份，风格必然高度同质，应标记 homogeneous。"""
        doc = Document(path="dup.docx", title="重复文档")
        text = (
            "本研究采用混合研究方法，综合运用文献分析、问卷调查与深度访谈等多种手段，"
            "力求全面系统地揭示研究问题。在数据收集阶段，严格遵循学术伦理规范，"
            "确保所有参与者均知情同意，并对个人信息进行匿名化处理。"
        )
        for index in range(6):
            paragraph = Paragraph(index=index, text=text)
            doc.paragraphs.append(paragraph)
            doc.body_indices.append(index)
            doc.total_words += paragraph.char_count
        profile = scoring.score_document(doc).meta["profile"]
        self.assertTrue(profile["homogeneous"])
        self.assertLess(profile["dispersion"], 6.0)

    def test_empty_document_returns_zero(self) -> None:
        report = scoring.score_document(Document(path="empty.docx", title="空文档"))
        self.assertEqual(report.overall, 0.0)
        self.assertEqual(report.level, "低")
        self.assertEqual(report.paragraphs, [])
        self.assertEqual(report.level_counts, {level: 0 for level in LEVELS})
        self.assertEqual(report.meta["feature_count"], 17)
        self.assertEqual(report.calibration, "uncalibrated")

    def test_missing_body_index_is_ignored(self) -> None:
        doc = Document(
            path="broken.docx",
            paragraphs=[Paragraph(index=0, text=AI_TEXT)],
            body_indices=[99],
        )
        report = scoring.score_document(doc)
        self.assertEqual(report.paragraphs, [])
        self.assertEqual(report.overall, 0.0)

    def test_empty_text_paragraph_does_not_raise(self) -> None:
        doc = Document(
            path="blank.docx",
            paragraphs=[Paragraph(index=0, text="")],
            body_indices=[0],
        )
        report = scoring.score_document(doc)
        self.assertEqual(len(report.paragraphs), 1)
        self.assertEqual(report.paragraphs[0].score, 0.0)
        self.assertEqual(report.overall, 0.0)
        self.assertEqual(report.level, "低")

    def test_document_is_not_mutated(self) -> None:
        doc = _make_document()
        before_indices = list(doc.body_indices)
        before_texts = [p.text for p in doc.paragraphs]
        scoring.score_document(doc)
        self.assertEqual(doc.body_indices, before_indices)
        self.assertEqual([p.text for p in doc.paragraphs], before_texts)

    def test_document_deterministic(self) -> None:
        first = scoring.score_document(_make_document())
        second = scoring.score_document(_make_document())
        self.assertEqual(first.overall, second.overall)
        self.assertEqual(
            [(p.index, p.score) for p in first.paragraphs],
            [(p.index, p.score) for p in second.paragraphs],
        )


if __name__ == "__main__":
    unittest.main()

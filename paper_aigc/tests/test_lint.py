"""T6 单元测试：论文规范自检（core/lint.py）。

每类检查各造一个最小 Document（不依赖 docx、不联网、不写仓库文件）；
另测空文档、to_meta 结构、paragraph_index / snippet 约束与「永不抛异常」。
"""
from __future__ import annotations

import json
import unittest

from paper_aigc.core.lint import (
    ISSUE_CODES,
    LEVEL_ERROR,
    LEVEL_INFO,
    LEVEL_WARN,
    LintIssue,
    LintReport,
    lint_document,
    to_meta,
)
from paper_aigc.core.types import Document, Paragraph

REQUIRED_CODES = {
    "table_seq",
    "table_missing",
    "ref_format",
    "mixed_punct",
    "dup_word",
    "term_inconsistent",
    "long_sentence",
    "no_period",
}


def _doc(items) -> Document:
    """items: [(text, is_heading?, is_reference?), ...]，按顺序生成段落。"""
    paragraphs = []
    for index, item in enumerate(items):
        text = item[0]
        heading = bool(item[1]) if len(item) > 1 else False
        reference = bool(item[2]) if len(item) > 2 else False
        paragraphs.append(
            Paragraph(
                index=index,
                text=text,
                is_heading=heading,
                is_reference=reference,
            )
        )
    body = [
        paragraph.index
        for paragraph in paragraphs
        if not paragraph.is_heading and not paragraph.is_reference
    ]
    return Document(
        path="测试论文.docx",
        title="测试论文",
        paragraphs=paragraphs,
        body_indices=body,
        total_words=sum(paragraph.char_count for paragraph in paragraphs),
    )


def _codes(report: LintReport):
    return [issue.code for issue in report.issues]


def _issues_of(report: LintReport, code: str):
    return [issue for issue in report.issues if issue.code == code]


class LintBasicsTest(unittest.TestCase):
    def test_required_codes_implemented(self) -> None:
        self.assertEqual(set(ISSUE_CODES), REQUIRED_CODES)
        self.assertEqual(len(ISSUE_CODES), len(REQUIRED_CODES))

    def test_empty_document_returns_empty_report(self) -> None:
        report = lint_document(Document(path="空.docx"))
        self.assertEqual(report.issues, [])
        self.assertEqual(report.counts, {"error": 0, "warn": 0, "info": 0})

    def test_document_with_blank_paragraphs_does_not_raise(self) -> None:
        document = Document(
            path="空白.docx",
            paragraphs=[
                Paragraph(index=0, text=""),
                Paragraph(index=1, text="   "),
                Paragraph(index=2, text="\u200b"),
            ],
        )
        report = lint_document(document)
        self.assertEqual(report.issues, [])
        self.assertEqual(to_meta(report)["counts"], {"error": 0, "warn": 0, "info": 0})

    def test_lint_document_does_not_raise_without_body_indices(self) -> None:
        document = Document(
            path="无body.docx",
            paragraphs=[Paragraph(index=0, text="本研究采用定量分析,结果显示差异显著。")],
        )
        report = lint_document(document)
        self.assertIn("mixed_punct", _codes(report))

    def test_lint_never_raises_on_invalid_input(self) -> None:
        # 契约上应该是 Document，但自检函数需要对坏输入也保持不抛异常。
        report = lint_document(None)  # type: ignore[arg-type]
        self.assertEqual(report.issues, [])
        self.assertEqual(report.counts, {"error": 0, "warn": 0, "info": 0})

    def test_lint_issue_defaults(self) -> None:
        issue = LintIssue(code="no_period", level=LEVEL_INFO, message="示例")
        self.assertEqual(issue.paragraph_index, -1)
        self.assertEqual(issue.snippet, "")

    def test_lint_report_autofills_counts(self) -> None:
        report = LintReport(
            issues=[
                LintIssue("mixed_punct", LEVEL_WARN, "w"),
                LintIssue("ref_format", LEVEL_ERROR, "e"),
            ]
        )
        self.assertEqual(report.counts, {"error": 1, "warn": 1, "info": 0})

    def test_to_meta_structure_and_json_serializable(self) -> None:
        report = lint_document(_doc([("本研究采用定量分析,结果显示差异显著。", False, False)]))
        meta = to_meta(report)

        self.assertEqual(set(meta.keys()), {"issues", "counts"})
        self.assertEqual(set(meta["counts"].keys()), {"error", "warn", "info"})
        self.assertEqual(sum(meta["counts"].values()), len(meta["issues"]))
        self.assertIsInstance(meta["issues"], list)
        for item in meta["issues"]:
            self.assertEqual(
                set(item.keys()),
                {"code", "level", "message", "paragraph_index", "snippet"},
            )
            self.assertIsInstance(item["paragraph_index"], int)
            self.assertLessEqual(len(item["snippet"]), 60)
        json.dumps(meta, ensure_ascii=False)


class TableCheckTest(unittest.TestCase):
    def test_table_number_gap_detected(self) -> None:
        document = _doc(
            [
                ("表1 各系统检测结果", False, False),
                ("如表1和表3所示，各系统结果总体稳定。", False, False),
                ("表3 补充对比结果", False, False),
            ]
        )
        report = lint_document(document)
        issues = _issues_of(report, "table_seq")
        self.assertTrue(issues)
        self.assertTrue(any("不连续" in issue.message and "表2" in issue.message for issue in issues))
        self.assertTrue(any(issue.level == LEVEL_WARN for issue in issues))

    def test_duplicate_table_caption_is_error(self) -> None:
        document = _doc(
            [
                ("表1 检测结果", False, False),
                ("表1 检测结果", False, False),
                ("如表1所示，结果可信。", False, False),
            ]
        )
        report = lint_document(document)
        issues = _issues_of(report, "table_seq")
        duplicates = [issue for issue in issues if "重复" in issue.message]
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(duplicates[0].level, LEVEL_ERROR)
        self.assertEqual(duplicates[0].paragraph_index, 1)

    def test_body_reference_without_caption_detected(self) -> None:
        document = _doc(
            [
                ("表1 检测结果", False, False),
                ("如表2所示，各系统差异明显。", False, False),
            ]
        )
        report = lint_document(document)
        issues = _issues_of(report, "table_missing")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].paragraph_index, 1)
        self.assertEqual(issues[0].level, LEVEL_ERROR)
        self.assertIn("表2", issues[0].message)
        self.assertIn("表2", issues[0].snippet)

    def test_figure_sequence_and_missing_checked(self) -> None:
        document = _doc(
            [
                ("图1 研究框架", False, False),
                ("如图1和图3所示，框架覆盖完整流程。", False, False),
                ("图3 流程示意", False, False),
                ("如图2所示，细节另有说明。", False, False),
            ]
        )
        report = lint_document(document)
        self.assertTrue(_issues_of(report, "table_seq"))
        missing = _issues_of(report, "table_missing")
        self.assertTrue(any("图2" in issue.message for issue in missing))
        self.assertTrue(any(issue.paragraph_index == 3 for issue in missing))

    def test_chinese_numeral_caption_matches_arabic_reference(self) -> None:
        document = _doc(
            [
                ("表一 指标说明", False, False),
                ("如表一所示，指标定义清晰。", False, False),
            ]
        )
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "table_missing"), [])

    def test_chinese_numeral_false_positive_guarded(self) -> None:
        # 「发表一次」「代表一定」里的「表一」不是图表引用。
        document = _doc(
            [("本研究发表一次阶段性成果，代表一定的研究水平。", False, False)]
        )
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "table_missing"), [])
        self.assertEqual(_issues_of(report, "table_seq"), [])

    def test_caption_without_body_reference_is_info(self) -> None:
        document = _doc([("表1 检测结果", False, False)])
        report = lint_document(document)
        issues = [
            issue
            for issue in _issues_of(report, "table_seq")
            if "没有被引用" in issue.message
        ]
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].level, LEVEL_INFO)


class ReferenceCheckTest(unittest.TestCase):
    def test_reference_missing_sequence_number(self) -> None:
        document = _doc(
            [("张三. 生成式人工智能与学术写作规范[J]. 现代教育技术, 2020.", False, True)]
        )
        report = lint_document(document)
        issues = _issues_of(report, "ref_format")
        self.assertTrue(any(issue.level == LEVEL_ERROR and "缺少序号" in issue.message for issue in issues))

    def test_reference_missing_year(self) -> None:
        document = _doc([("[1] 张三. 某文[J]. 某刊.", False, True)])
        report = lint_document(document)
        issues = _issues_of(report, "ref_format")
        self.assertTrue(any("年份" in issue.message for issue in issues))

    def test_reference_missing_type_marker(self) -> None:
        document = _doc([("[1] 张三. 某文. 某刊, 2020.", False, True)])
        report = lint_document(document)
        issues = _issues_of(report, "ref_format")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].level, LEVEL_WARN)
        self.assertIn("文献类型标识", issues[0].message)

    def test_reference_number_not_continuous(self) -> None:
        document = _doc(
            [
                ("[1] 张三. 某文[J]. 某刊, 2020.", False, True),
                ("[3] 李四. 某文[M]. 某社, 2021.", False, True),
            ]
        )
        report = lint_document(document)
        issues = _issues_of(report, "ref_format")
        self.assertTrue(any("序号不连续" in issue.message for issue in issues))
        self.assertTrue(any(issue.paragraph_index == 1 for issue in issues))

    def test_valid_references_produce_no_ref_format(self) -> None:
        document = _doc(
            [
                ("[1] 张三. 某文[J]. 某刊, 2020.", False, True),
                ("[2] 李四. 某文[M]. 某社, 2021.", False, True),
            ]
        )
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "ref_format"), [])

    def test_reference_paragraph_not_checked_as_body_text(self) -> None:
        document = _doc([("[1] 张三, 李四. 某文[J]. 某刊, 2020.", False, True)])
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "mixed_punct"), [])


class MixedPunctTest(unittest.TestCase):
    def test_ascii_comma_between_chinese_detected(self) -> None:
        document = _doc([("本研究采用定量分析,结果显示差异显著。", False, False)])
        report = lint_document(document)
        issues = _issues_of(report, "mixed_punct")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].paragraph_index, 0)
        self.assertEqual(issues[0].level, LEVEL_WARN)
        self.assertIn(",", issues[0].message)

    def test_ascii_parenthesis_between_chinese_detected(self) -> None:
        document = _doc([("具体结果另见附录(附表)中的说明。", False, False)])
        report = lint_document(document)
        issues = _issues_of(report, "mixed_punct")
        self.assertEqual(len(issues), 1)
        self.assertIn("(", issues[0].message)
        self.assertIn(")", issues[0].message)

    def test_ascii_punct_not_next_to_chinese_not_flagged(self) -> None:
        document = _doc([("模型 AUC 达到 0.92, 并在英文语料上验证。", False, False)])
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "mixed_punct"), [])


class DupWordTest(unittest.TestCase):
    def test_single_char_repeat_detected(self) -> None:
        document = _doc([("这个问题值得的的深入讨论，结论仍需验证。", False, False)])
        report = lint_document(document)
        issues = _issues_of(report, "dup_word")
        self.assertTrue(any("的的" in issue.message for issue in issues))
        self.assertEqual(issues[0].level, LEVEL_WARN)

    def test_two_char_word_repeat_detected(self) -> None:
        document = _doc([("我们需要研究研究这一问题，再给出结论。", False, False)])
        report = lint_document(document)
        issues = _issues_of(report, "dup_word")
        self.assertTrue(any("研究研究" in issue.message for issue in issues))

    def test_legitimate_reduplication_not_flagged(self) -> None:
        document = _doc([("他常常查阅一一对应的资料，并慢慢改进研究方法。", False, False)])
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "dup_word"), [])

    def test_legitimate_numeral_classifier_repeat_not_flagged(self) -> None:
        document = _doc([("他们一个一个地核对数据，确保结果可靠。", False, False)])
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "dup_word"), [])


class TermInconsistentTest(unittest.TestCase):
    def test_chinese_and_english_terms_flagged(self) -> None:
        document = _doc(
            [
                ("净资产收益率保持在较高水平。", False, False),
                ("ROE 连续三年上升，说明盈利能力改善。", False, False),
            ]
        )
        report = lint_document(document)
        issues = _issues_of(report, "term_inconsistent")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].paragraph_index, 1)
        self.assertIn("净资产收益率", issues[0].message)
        self.assertIn("ROE", issues[0].message)

    def test_gdp_pair_flagged(self) -> None:
        document = _doc(
            [
                ("国内生产总值增速保持在合理区间。", False, False),
                ("GDP 增速与就业数据相互印证。", False, False),
            ]
        )
        report = lint_document(document)
        self.assertTrue(_issues_of(report, "term_inconsistent"))

    def test_aigc_is_not_matched_as_ai(self) -> None:
        document = _doc(
            [
                ("人工智能生成内容受到广泛关注。", False, False),
                ("AIGC 检测工具在高校中逐步普及。", False, False),
            ]
        )
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "term_inconsistent"), [])


class LongSentenceTest(unittest.TestCase):
    def test_sentence_over_80_chinese_chars_detected(self) -> None:
        long_text = "甲乙丙丁戊己庚辛壬癸" * 9 + "。"
        document = _doc([(long_text, False, False)])
        report = lint_document(document)
        issues = _issues_of(report, "long_sentence")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].level, LEVEL_INFO)
        self.assertIn("90", issues[0].message)
        self.assertEqual(_issues_of(report, "dup_word"), [])

    def test_short_sentences_not_flagged(self) -> None:
        document = _doc(
            [("本研究采用定量分析的方法。数据来自两轮问卷调查。结论需要谨慎解读。", False, False)]
        )
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "long_sentence"), [])


class NoPeriodTest(unittest.TestCase):
    def test_missing_end_punct_detected(self) -> None:
        document = _doc(
            [("本研究在数据收集阶段遇到了一些困难，但最终未能给出明确解释", False, False)]
        )
        report = lint_document(document)
        issues = _issues_of(report, "no_period")
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].level, LEVEL_INFO)
        self.assertEqual(issues[0].paragraph_index, 0)

    def test_punctuated_paragraph_not_flagged(self) -> None:
        document = _doc([("本研究采用定量分析与定性分析相结合的方法。", False, False)])
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "no_period"), [])

    def test_closing_quote_after_period_not_flagged(self) -> None:
        document = _doc([("受访教师表示：「这样处理更稳妥。」", False, False)])
        report = lint_document(document)
        self.assertEqual(_issues_of(report, "no_period"), [])

    def test_heading_caption_and_table_row_skipped(self) -> None:
        document = _doc(
            [
                ("三、研究方法", True, False),
                ("表1 检测结果", False, False),
                ("| 检测系统名称 | 疑似AI率 | 检测时间 | 备注说明 |", False, False),
                ("本节说明研究方法的整体设计。", False, False),
            ]
        )
        report = lint_document(document)
        flagged = {issue.paragraph_index for issue in _issues_of(report, "no_period")}
        self.assertNotIn(0, flagged)
        self.assertNotIn(1, flagged)
        self.assertNotIn(2, flagged)
        self.assertNotIn(3, flagged)


class CleanDocumentTest(unittest.TestCase):
    def test_clean_document_has_no_issues(self) -> None:
        document = _doc(
            [
                ("一、引言", True, False),
                ("表1给出了各系统的检测结果，其中本工具表现稳定。", False, False),
                ("表1 各系统检测结果", False, False),
                ("参考文献", True, False),
                (
                    "[1] 张三. 生成式人工智能与学术写作规范[J]. 现代教育技术, 2024, 34(5): 12-19.",
                    False,
                    True,
                ),
                ("[2] 李四. 论文规范研究[M]. 北京: 某出版社, 2023.", False, True),
            ]
        )
        report = lint_document(document)
        self.assertEqual(report.issues, [])
        self.assertEqual(report.counts, {"error": 0, "warn": 0, "info": 0})

    def test_issue_fields_always_valid(self) -> None:
        document = _doc(
            [
                ("一、引言", True, False),
                ("本研究采用定量分析,结果显示差异显著。", False, False),
                ("这个问题值得的的深入讨论，并且需要研究研究。", False, False),
                ("净资产收益率上升。", False, False),
                ("ROE 持续改善。", False, False),
                ("甲乙丙丁戊己庚辛壬癸" * 9 + "。", False, False),
                ("最后一段没有句末标点，需要补上", False, False),
            ]
        )
        report = lint_document(document)
        self.assertTrue(report.issues)
        for issue in report.issues:
            self.assertIn(issue.code, ISSUE_CODES)
            self.assertIn(issue.level, {LEVEL_ERROR, LEVEL_WARN, LEVEL_INFO})
            self.assertIsInstance(issue.paragraph_index, int)
            self.assertGreaterEqual(issue.paragraph_index, -1)
            self.assertIsInstance(issue.snippet, str)
            self.assertLessEqual(len(issue.snippet), 60)
            self.assertTrue(issue.message)

    def test_snippet_truncated_to_60_chars(self) -> None:
        filler = "甲乙丙丁戊己庚辛壬癸" * 12
        document = _doc([(filler + ",结果稳定。", False, False)])
        report = lint_document(document)
        mixed = _issues_of(report, "mixed_punct")
        self.assertEqual(len(mixed), 1)
        self.assertLessEqual(len(mixed[0].snippet), 60)

    def test_issues_sorted_by_paragraph_index(self) -> None:
        document = _doc(
            [
                ("第三段没有句末标点，需要补上。", False, False),
                ("本研究采用定量分析,结果显示差异显著。", False, False),
            ]
        )
        report = lint_document(document)
        indices = [issue.paragraph_index for issue in report.issues]
        self.assertEqual(indices, sorted(indices))


if __name__ == "__main__":
    unittest.main()

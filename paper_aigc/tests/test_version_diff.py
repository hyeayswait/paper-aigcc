# -*- coding: utf-8 -*-
"""版本对比模块测试：现场生成临时 docx，全部离线运行。"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import docx

from paper_aigc.io.version_diff import (
    NGRAM_SIZE,
    diff_documents,
    paragraph_similarity,
    render_diff_html,
)

P1 = "家电行业已经进入存量竞争阶段，企业之间的竞争由规模扩张转向经营质量比拼，价格压力与渠道竞争都会压缩盈利空间。"
P2 = "某公司的营业收入在最近五年保持增长，净利润率稳定在百分之八左右，经营现金流整体表现良好。"
P3 = "本文采用杜邦分析法拆解净资产收益率，发现总资产周转率是主要驱动因素，杠杆水平相对稳定。"
P4 = "渠道库存与终端零售数据需要交叉验证，单一口径的销量统计容易高估真实需求。"
P5 = "本段用于验证新增与删除的识别，内容与其余段落无关，讨论的是上游铜材价格波动。"
P6 = "这一段用于验证脚本转义，<script>alert('x')</script> 必须原样转义后再展示。"


def write_docx(path: str, texts) -> str:
    """用 python-docx 现场生成一个只含正文段落的 docx。"""
    document = docx.Document()
    for text in texts:
        document.add_paragraph(text)
    document.save(path)
    return path


class DiffTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp, True)

    def make_pair(self, old_texts, new_texts):
        old_path = write_docx(os.path.join(self.temp, "old.docx"), old_texts)
        new_path = write_docx(os.path.join(self.temp, "new.docx"), new_texts)
        return old_path, new_path, diff_documents(old_path, new_path)


class SimilarityTest(unittest.TestCase):
    def test_punctuation_does_not_affect_similarity(self) -> None:
        punctuated = P2.replace("，", "；").replace("。", "！")
        self.assertGreaterEqual(paragraph_similarity(P2, punctuated), 0.99)

    def test_identical_text_scores_one(self) -> None:
        self.assertAlmostEqual(paragraph_similarity(P2, P2), 1.0, places=6)

    def test_unrelated_text_scores_low(self) -> None:
        self.assertLess(paragraph_similarity(P2, P5), 0.3)

    def test_short_paragraphs_do_not_crash(self) -> None:
        self.assertGreaterEqual(paragraph_similarity("短句。", "短句。"), 0.99)
        self.assertEqual(paragraph_similarity("", ""), 1.0)
        self.assertEqual(paragraph_similarity(P2, ""), 0.0)
        self.assertEqual(NGRAM_SIZE, 13)


class DiffCountsTest(DiffTestCase):
    def test_modified_paragraph_is_detected(self) -> None:
        new_p2 = P2 + "整体来看，这一趋势在未来两年仍将延续。"
        _old, _new, report = self.make_pair([P1, P2, P3], [P1, new_p2, P3])
        self.assertEqual(report.modified, 1)
        self.assertEqual(report.added, 0)
        self.assertEqual(report.removed, 0)
        self.assertEqual(report.unchanged, 2)
        modified = [item for item in report.items if item.status == "modified"]
        self.assertEqual(len(modified), 1)
        self.assertGreaterEqual(modified[0].similarity, 0.6)
        self.assertEqual(modified[0].old_index, 1)
        self.assertEqual(modified[0].new_index, 1)
        self.assertGreater(modified[0].char_delta, 0)
        self.assertIn("整体来看", modified[0].new_text)
        self.assertNotIn("整体来看", modified[0].old_text)

    def test_added_and_removed_paragraphs(self) -> None:
        _old, _new, report = self.make_pair([P1, P2, P3], [P1, P3, P5])
        self.assertEqual(report.added, 1)
        self.assertEqual(report.removed, 1)
        self.assertEqual(report.modified, 0)
        self.assertEqual(report.unchanged, 2)
        removed = [item for item in report.items if item.status == "removed"]
        added = [item for item in report.items if item.status == "added"]
        self.assertEqual(removed[0].old_text, P2)
        self.assertEqual(removed[0].new_index, -1)
        self.assertEqual(added[0].new_text, P5)
        self.assertEqual(added[0].old_index, -1)
        self.assertLess(removed[0].char_delta, 0)
        self.assertGreater(added[0].char_delta, 0)

    def test_identical_documents_are_all_unchanged(self) -> None:
        _old, _new, report = self.make_pair([P1, P2, P3], [P1, P2, P3])
        self.assertEqual(report.unchanged, 3)
        self.assertEqual(report.added, 0)
        self.assertEqual(report.removed, 0)
        self.assertEqual(report.modified, 0)
        self.assertEqual(report.old_chars, report.new_chars)
        self.assertTrue(all(item.status == "same" for item in report.items))
        self.assertTrue(all(item.similarity == 1.0 for item in report.items))

    def test_counts_and_chars_match_items(self) -> None:
        new_p2 = P2 + "补充一句总结性表述。"
        _old, _new, report = self.make_pair([P1, P2, P3], [P1, new_p2, P5])
        # 同一对齐块内「未配对」的段落按顺序算作 modified（整段重写不应报成增删）
        self.assertEqual(report.added, 0)
        self.assertEqual(report.removed, 0)
        self.assertEqual(report.modified, 2)
        self.assertEqual(report.unchanged, 1)
        self.assertEqual(len(report.items), report.added + report.removed + report.modified + report.unchanged)
        self.assertEqual(sum(item.char_delta for item in report.items), report.new_chars - report.old_chars)

    def test_missing_file_raises_chinese_value_error(self) -> None:
        old_path = write_docx(os.path.join(self.temp, "only_old.docx"), [P1])
        missing = os.path.join(self.temp, "not_exists.docx")
        with self.assertRaises(ValueError) as ctx:
            diff_documents(old_path, missing)
        self.assertIn("不存在", str(ctx.exception))


class RenderHtmlTest(DiffTestCase):
    def render(self):
        new_p2 = P2 + "整体来看，这一趋势在未来两年仍将延续。"
        _old, _new, report = self.make_pair([P1, P2, P3, P6], [P1, new_p2, P5, P6])
        out_path = os.path.join(self.temp, "diff.html")
        returned = render_diff_html(report, out_path)
        with open(out_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        return report, returned, out_path, content

    def test_returns_out_path_and_writes_file(self) -> None:
        _report, returned, out_path, content = self.render()
        self.assertTrue(os.path.isfile(out_path))
        self.assertEqual(os.fspath(returned), os.fspath(out_path))
        self.assertGreater(len(content), 1000)
        self.assertIn("<!DOCTYPE html>", content)

    def test_summary_labels_present(self) -> None:
        _report, _returned, _out_path, content = self.render()
        for label in ("新增段落", "删除段落", "修改段落", "未变段落", "字数变化"):
            self.assertIn(label, content)

    def test_rows_use_details_and_statuses(self) -> None:
        report, _returned, _out_path, content = self.render()
        self.assertGreaterEqual(content.count("<details>"), len(report.items) * 2)
        for label in ("修改", "未变"):
            self.assertIn(">%s<" % label, content)
        # 新增/删除在本场景没有对应行，只在摘要卡里出现
        for label in ("新增段落", "删除段落"):
            self.assertIn(label, content)
        self.assertIn("逐段对比", content)

    def test_no_external_resources(self) -> None:
        _report, _returned, _out_path, content = self.render()
        lowered = content.lower()
        self.assertNotIn("http://", lowered)
        self.assertNotIn("https://", lowered)
        self.assertNotIn("<link", lowered)
        self.assertNotIn("src=", lowered)

    def test_script_text_is_escaped(self) -> None:
        _report, _returned, _out_path, content = self.render()
        self.assertIn("&lt;script&gt;", content)
        self.assertNotIn("<script", content.lower())

    def test_url_like_text_is_neutralized(self) -> None:
        doc_text = "参考网址为 http://example.com/a 和 https://example.com/b，此处仅作测试。"
        _old, _new, report = self.make_pair([P1, doc_text], [P1, doc_text])
        out_path = os.path.join(self.temp, "url.html")
        render_diff_html(report, out_path)
        with open(out_path, "r", encoding="utf-8") as handle:
            content = handle.read()
        self.assertNotIn("http://", content.lower())
        self.assertNotIn("https://", content.lower())

    def test_empty_documents_render_without_crash(self) -> None:
        _old, _new, report = self.make_pair([], [])
        self.assertEqual(report.items, [])
        out_path = os.path.join(self.temp, "empty.html")
        render_diff_html(report, out_path)
        self.assertTrue(os.path.isfile(out_path))


if __name__ == "__main__":
    unittest.main()

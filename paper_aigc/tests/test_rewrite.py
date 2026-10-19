"""T3 改写模块单元测试。

全部离线：requests.post 与评分模块都被 mock，不读取真实 API Key、不联网。
"""
from __future__ import annotations

import unittest
from unittest import mock

import requests

import paper_aigc.rewrite.pipeline as pipeline
from paper_aigc.config import AppConfig
from paper_aigc.core.types import (
    Document,
    Paragraph,
    ParagraphScore,
    RewriteResult,
    ScoreReport,
)
from paper_aigc.rewrite import prompts
from paper_aigc.rewrite.client import DeepSeekClient, RewriteError


def make_cfg(**kwargs):
    values = dict(
        api_key="sk-test-key",
        base_url="https://api.deepseek.com",
        model="deepseek-chat",
        temperature=1.15,
        max_tokens=1024,
        timeout=5,
        retries=1,
        high_threshold=60.0,
        rewrite_target=35.0,
        max_rewrite_attempts=3,
        min_rewrite_chars=40,
        concurrency=2,
    )
    values.update(kwargs)
    return AppConfig(**values)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or ""
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def ok_response(content):
    return FakeResponse(200, {"choices": [{"message": {"content": content}}]})


class FakeClient:
    """替代 DeepSeekClient 的假客户端：按顺序吐出回复或抛异常。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, temperature=None, max_tokens=None):
        self.calls.append(messages)
        if not self.replies:
            raise RewriteError("余额不足，请前往 DeepSeek 平台充值后重试。")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def scoring_stub(values):
    """造一个假的 scoring 模块：score_paragraph 依次返回给定分数。"""
    remaining = list(values)
    module = mock.Mock()

    def _score_paragraph(text, index=0):
        value = remaining.pop(0) if remaining else 0.0
        return ParagraphScore(index=index, text=text, score=value)

    module.score_paragraph.side_effect = _score_paragraph
    return module


LONG_A = (
    "随着人工智能技术的快速发展，深度学习在图像识别领域取得了显著进展。"
    "本文认为该方法在标准数据集上的准确率达到 95.3%，但是仍然存在泛化能力不足的问题。"
    "因此，未来的研究应当重点关注跨域迁移与模型可解释性。"
)
LONG_B = (
    "近年来，多模态大模型在视觉问答任务上表现突出，相关研究不断涌现。"
    "本文通过对比实验发现，引入跨模态对齐后，准确率提升了 4.7 个百分点，"
    "但推理成本也明显上升。"
)
REVISED_A = "图像识别的准确率到了 95.3%，这是深度学习带来的进步。笔者更关心泛化：换一批数据，效果会不会掉？这决定方法能不能真的落地。"
REVISED_B = "准确率 95.3% 这个数字很好看，可它来自标准数据集。笔者担心换一批数据之后还能不能撑住，所以后面得看跨域迁移与可解释性。"
REVISED_C = "标准数据集上 95.3% 的准确率确实说明深度学习的进展。但泛化能力呢？笔者认为这才是接下来要解决的问题，尤其是跨域迁移。"


class DeepSeekClientTestCase(unittest.TestCase):
    def test_missing_api_key_raises_chinese_error(self):
        with self.assertRaises(RewriteError) as ctx:
            DeepSeekClient(make_cfg(api_key="   "))
        self.assertIn("未配置 DeepSeek API Key", str(ctx.exception))

    def test_empty_messages_rejected_before_network(self):
        client = DeepSeekClient(make_cfg())
        with self.assertRaises(RewriteError) as ctx:
            client.chat([])
        self.assertIn("消息为空", str(ctx.exception))

    @mock.patch("paper_aigc.rewrite.client.requests.post")
    def test_chat_sends_openai_payload_and_bearer_header(self, post):
        post.return_value = ok_response("改写后的文本")
        client = DeepSeekClient(make_cfg())
        out = client.chat(
            [{"role": "user", "content": "你好"}], temperature=0.3, max_tokens=64
        )
        self.assertEqual(out, "改写后的文本")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.deepseek.com/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-test-key")
        self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")
        self.assertEqual(kwargs["json"]["model"], "deepseek-chat")
        self.assertEqual(kwargs["json"]["temperature"], 0.3)
        self.assertEqual(kwargs["json"]["max_tokens"], 64)
        self.assertFalse(kwargs["json"]["stream"])
        self.assertEqual(kwargs["json"]["messages"][0]["content"], "你好")
        self.assertEqual(kwargs["timeout"], 5)

    @mock.patch("paper_aigc.rewrite.client.time.sleep")
    @mock.patch("paper_aigc.rewrite.client.requests.post")
    def test_retries_server_error_then_succeeds(self, post, sleep):
        post.side_effect = [FakeResponse(500, text="boom"), ok_response("ok")]
        client = DeepSeekClient(make_cfg(retries=2))
        self.assertEqual(client.chat([{"role": "user", "content": "hi"}]), "ok")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called()

    @mock.patch("paper_aigc.rewrite.client.requests.post")
    def test_auth_error_not_retried_and_message_friendly(self, post):
        post.return_value = FakeResponse(401, {"error": {"message": "invalid key"}})
        client = DeepSeekClient(make_cfg(retries=3))
        with self.assertRaises(RewriteError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("API Key 无效或无权限", str(ctx.exception))
        self.assertEqual(post.call_count, 1)

    @mock.patch("paper_aigc.rewrite.client.requests.post")
    def test_insufficient_balance_message(self, post):
        post.return_value = FakeResponse(402, {"error": {"message": "Insufficient Balance"}})
        with self.assertRaises(RewriteError) as ctx:
            DeepSeekClient(make_cfg()).chat([{"role": "user", "content": "hi"}])
        self.assertIn("余额不足", str(ctx.exception))

    @mock.patch("paper_aigc.rewrite.client.time.sleep")
    @mock.patch("paper_aigc.rewrite.client.requests.post")
    def test_rate_limit_exhausts_retries_with_chinese_message(self, post, sleep):
        post.return_value = FakeResponse(429, {"error": {"message": "rate limit"}})
        client = DeepSeekClient(make_cfg(retries=2))
        with self.assertRaises(RewriteError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("触发限流", str(ctx.exception))
        self.assertEqual(post.call_count, 3)

    @mock.patch("paper_aigc.rewrite.client.requests.post")
    def test_network_error_is_chinese_and_no_retry_when_disabled(self, post):
        post.side_effect = requests.exceptions.ConnectionError("boom")
        client = DeepSeekClient(make_cfg(retries=0))
        with self.assertRaises(RewriteError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("网络不可达", str(ctx.exception))
        self.assertEqual(post.call_count, 1)

    @mock.patch("paper_aigc.rewrite.client.requests.post")
    def test_empty_choices_raises(self, post):
        post.return_value = FakeResponse(200, {"choices": []})
        with self.assertRaises(RewriteError) as ctx:
            DeepSeekClient(make_cfg()).chat([{"role": "user", "content": "hi"}])
        self.assertIn("空内容", str(ctx.exception))

    def test_check_connection_success_and_failure_do_not_raise(self):
        client = DeepSeekClient(make_cfg())
        with mock.patch.object(client, "chat", return_value="正常"):
            ok, message = client.check_connection()
        self.assertTrue(ok)
        self.assertIn("连接正常", message)
        with mock.patch.object(
            client, "chat", side_effect=RewriteError("API Key 无效或无权限，请在设置中检查。")
        ):
            ok, message = client.check_connection()
        self.assertFalse(ok)
        self.assertIn("API Key 无效", message)


class PromptsTestCase(unittest.TestCase):
    def test_system_prompt_is_about_naturalizing_not_evasion(self):
        self.assertIn("自然化改写", prompts.SYSTEM_PROMPT)
        self.assertIn("保留", prompts.SYSTEM_PROMPT)
        for banned in ("规避检测", "绕过检测", "对抗检测", "降低 AIGC 率", "躲避检测"):
            self.assertNotIn(banned, prompts.SYSTEM_PROMPT)

    def test_rewrite_prompt_keeps_original_and_signal_names(self):
        prompt = prompts.build_rewrite_prompt(
            LONG_A,
            [("sentence_len_cv", "句长变异系数", 8.4), ("connective_density", "连接词密度", 5.2)],
        )
        self.assertIn(LONG_A, prompt)
        self.assertIn("句长变异系数", prompt)
        self.assertIn("连接词密度", prompt)
        self.assertIn("不得新增", prompt)
        self.assertIn("只输出改写后的", prompt)

    def test_context_block_only_when_context_given(self):
        without = prompts.build_rewrite_prompt(LONG_A, [])
        self.assertNotIn("【上下文】", without)
        with_ctx = prompts.build_rewrite_prompt(LONG_A, [], context="论文标题：测试")
        self.assertIn("【上下文】", with_ctx)
        self.assertIn("论文标题：测试", with_ctx)

    def test_retry_prompt_mentions_previous_result_and_target(self):
        prompt = prompts.build_retry_prompt(
            LONG_A, REVISED_A, 58.0, 35.0, [("hedge_density", "模糊限制语密度", 3.0)]
        )
        self.assertIn(REVISED_A, prompt)
        self.assertIn("58.0", prompt)
        self.assertIn("35.0", prompt)
        self.assertIn("模糊限制语密度", prompt)
        self.assertIn("换一种句式", prompt)


class RewriteParagraphTestCase(unittest.TestCase):
    def test_success_on_first_attempt(self):
        paragraph = Paragraph(index=0, text=LONG_A)
        score = ParagraphScore(
            index=0,
            text=LONG_A,
            score=72.0,
            top_signals=[("connective_density", "连接词密度", 6.2)],
        )
        client = FakeClient([REVISED_A])
        progress = []
        with mock.patch.object(pipeline, "scoring", scoring_stub([21.5])):
            result = pipeline.rewrite_paragraph(
                paragraph, score, client, make_cfg(), progress=progress.append
            )
        self.assertTrue(result.ok)
        self.assertEqual(result.error, "")
        self.assertEqual(result.attempts, 1)
        self.assertAlmostEqual(result.score_before, 72.0)
        self.assertAlmostEqual(result.score_after, 21.5)
        self.assertEqual(result.revised, REVISED_A)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(any("完成" in message for message in progress))

    def test_retries_with_retry_prompt_until_target(self):
        paragraph = Paragraph(index=3, text=LONG_A)
        score = ParagraphScore(
            index=3,
            text=LONG_A,
            score=80.0,
            top_signals=[("boilerplate_density", "套话密度", 5.0)],
        )
        client = FakeClient([REVISED_A, REVISED_B, REVISED_C])
        with mock.patch.object(pipeline, "scoring", scoring_stub([58.0, 46.0, 24.0])):
            result = pipeline.rewrite_paragraph(paragraph, score, client, make_cfg())
        self.assertTrue(result.ok)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(len(client.calls), 3)
        self.assertAlmostEqual(result.score_after, 24.0)
        self.assertEqual(result.revised, REVISED_C)
        second_prompt = client.calls[1][1]["content"]
        self.assertIn(REVISED_A, second_prompt)
        self.assertIn("58.0", second_prompt)
        self.assertIn("35.0", second_prompt)
        self.assertIn("system", client.calls[0][0]["role"])
        self.assertIn("自然化改写", client.calls[0][0]["content"])

    def test_api_error_is_captured_in_result(self):
        paragraph = Paragraph(index=1, text=LONG_A)
        score = ParagraphScore(index=1, text=LONG_A, score=75.0)
        client = FakeClient(
            [RewriteError("API Key 无效或无权限，请在设置中检查 DeepSeek API Key。")]
        )
        with mock.patch.object(pipeline, "scoring", scoring_stub([10.0])):
            result = pipeline.rewrite_paragraph(paragraph, score, client, make_cfg())
        self.assertFalse(result.ok)
        self.assertIn("API Key", result.error)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.revised, "")

    def test_unchanged_output_is_flagged(self):
        paragraph = Paragraph(index=2, text=LONG_A)
        score = ParagraphScore(index=2, text=LONG_A, score=70.0)
        client = FakeClient([LONG_A])
        with mock.patch.object(pipeline, "scoring", scoring_stub([30.0])):
            result = pipeline.rewrite_paragraph(paragraph, score, client, make_cfg())
        self.assertFalse(result.ok)
        self.assertEqual(result.error, "改写无变化")

    def test_empty_paragraph_is_skipped(self):
        paragraph = Paragraph(index=5, text="")
        score = ParagraphScore(index=5, text="", score=80.0)
        client = FakeClient([REVISED_A])
        result = pipeline.rewrite_paragraph(paragraph, score, client, make_cfg())
        self.assertFalse(result.ok)
        self.assertIn("为空", result.error)
        self.assertEqual(client.calls, [])

    def test_scoring_exception_is_captured(self):
        paragraph = Paragraph(index=4, text=LONG_A)
        score = ParagraphScore(index=4, text=LONG_A, score=71.0)
        module = mock.Mock()
        module.score_paragraph.side_effect = RuntimeError("评分器炸了")
        client = FakeClient([REVISED_A])
        with mock.patch.object(pipeline, "scoring", module):
            result = pipeline.rewrite_paragraph(paragraph, score, client, make_cfg())
        self.assertFalse(result.ok)
        self.assertIn("异常", result.error)
        self.assertTrue(result.error)


def make_document():
    para0 = Paragraph(index=0, text=LONG_A)
    para1 = Paragraph(index=1, text=LONG_B)
    doc = Document(
        path="demo.docx",
        title="测试论文",
        paragraphs=[para0, para1],
        body_indices=[0, 1],
    )
    report = ScoreReport(
        document=doc,
        paragraphs=[
            ParagraphScore(index=0, text=LONG_A, score=78.0),
            ParagraphScore(index=1, text=LONG_B, score=66.0),
        ],
    )
    return doc, report


class RewriteDocumentTestCase(unittest.TestCase):
    def test_short_paragraph_is_skipped(self):
        para = Paragraph(index=0, text="方法有效。")
        doc = Document(path="short.docx", title="短文", paragraphs=[para], body_indices=[0])
        report = ScoreReport(
            document=doc,
            paragraphs=[ParagraphScore(index=0, text=para.text, score=90.0)],
        )
        client = FakeClient([REVISED_A])
        progress = []
        with mock.patch.object(pipeline, "scoring", scoring_stub([10.0])):
            results = pipeline.rewrite_document(
                doc, report, client, make_cfg(), progress=progress.append
            )
        self.assertEqual(results, {})
        self.assertEqual(client.calls, [])
        self.assertTrue(any("没有达到" in message for message in progress))

    def test_low_score_paragraph_is_skipped(self):
        doc, report = make_document()
        report.paragraphs[0].score = 20.0
        report.paragraphs[1].score = 12.0
        client = FakeClient([REVISED_A, REVISED_B])
        with mock.patch.object(pipeline, "scoring", scoring_stub([10.0])):
            results = pipeline.rewrite_document(doc, report, client, make_cfg())
        self.assertEqual(results, {})
        self.assertEqual(client.calls, [])

    def test_high_risk_paragraphs_are_rewritten_concurrently(self):
        doc, report = make_document()
        client = FakeClient([REVISED_A, REVISED_B])
        progress = []
        with mock.patch.object(pipeline, "scoring", scoring_stub([20.0, 25.0])):
            results = pipeline.rewrite_document(
                doc, report, client, make_cfg(concurrency=2), progress=progress.append
            )
        self.assertEqual(sorted(results.keys()), [0, 1])
        self.assertTrue(all(result.ok for result in results.values()))
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all(result.score_after <= 35.0 for result in results.values()))
        self.assertTrue(any("共有 2 段" in message for message in progress))
        self.assertTrue(any("进度" in message for message in progress))

    def test_client_failure_does_not_raise_and_records_errors(self):
        doc, report = make_document()
        client = FakeClient([])
        with mock.patch.object(pipeline, "scoring", scoring_stub([10.0])):
            results = pipeline.rewrite_document(doc, report, client, make_cfg())
        self.assertEqual(sorted(results.keys()), [0, 1])
        for result in results.values():
            self.assertFalse(result.ok)
            self.assertTrue(result.error)
            self.assertIn("余额不足", result.error)

    def test_cancel_stops_before_any_request(self):
        doc, report = make_document()
        client = FakeClient([REVISED_A, REVISED_B])
        progress = []
        with mock.patch.object(pipeline, "scoring", scoring_stub([20.0])):
            results = pipeline.rewrite_document(
                doc,
                report,
                client,
                make_cfg(),
                progress=progress.append,
                cancel=lambda: True,
            )
        self.assertEqual(results, {})
        self.assertEqual(client.calls, [])
        self.assertTrue(any("取消" in message for message in progress))

    def test_result_indexes_match_document(self):
        doc, report = make_document()
        client = FakeClient([REVISED_A, REVISED_B])
        with mock.patch.object(pipeline, "scoring", scoring_stub([20.0, 20.0])):
            results = pipeline.rewrite_document(doc, report, client, make_cfg())
        for index, result in results.items():
            self.assertIsInstance(result, RewriteResult)
            self.assertEqual(result.index, index)
            self.assertTrue(result.original)


if __name__ == "__main__":
    unittest.main()

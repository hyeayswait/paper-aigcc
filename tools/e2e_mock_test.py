# -*- coding: utf-8 -*-
"""端到端自测：本地 mock DeepSeek 服务，验证 client -> pipeline -> 重评 -> docx -> 报告 全链路。"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.stdout.reconfigure(encoding="utf-8")

from paper_aigc.config import AppConfig
from paper_aigc.core import scoring
from paper_aigc.io.docx_reader import read_docx
from paper_aigc.io.docx_writer import write_docx
from paper_aigc.io.report import render_html
from paper_aigc.rewrite.client import DeepSeekClient
from paper_aigc.rewrite.pipeline import rewrite_document

SAMPLE = os.path.join(_ROOT, "samples", "样例论文_AI味混排.docx")
OUT = os.path.join(_ROOT, "out")
os.makedirs(OUT, exist_ok=True)

# 一段明显"人写"的替身文本：短句、具体数字、第一人称、标点多样
FAKE_REWRITE = (
    "我用了三种方法：文献梳理、问卷、访谈。问卷回收有效样本 328 份；"
    "访谈 17 人，其中只有 3 人说得清学校用的是哪套检测系统。"
    "这个比例挺说明问题的——大家在被一个自己不了解的工具打分。"
)

requests_seen = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        requests_seen.append({
            "path": self.path,
            "auth": self.headers.get("Authorization", ""),
            "model": body.get("model"),
            "messages": len(body.get("messages", [])),
            "temperature": body.get("temperature"),
        })
        payload = {
            "id": "mock-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": FAKE_REWRITE},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print("[mock] 服务已启动 127.0.0.1:%d" % port)

    cfg = AppConfig(api_key="sk-mock-for-test", base_url="http://127.0.0.1:%d" % port,
                    model="deepseek-chat", high_threshold=55.0, rewrite_target=35.0,
                    max_rewrite_attempts=3, concurrency=3, min_rewrite_chars=40, timeout=10, retries=1)

    doc = read_docx(SAMPLE)
    report = scoring.score_document(doc)
    print("[1] 检测完成：整篇 %.1f（%s），高风险段 %d" % (
        report.overall, report.level, report.level_counts.get("高", 0)))

    client = DeepSeekClient(cfg)
    ok, msg = client.check_connection()
    print("[2] check_connection ->", ok, msg)
    assert ok, msg

    progress_msgs = []
    results = rewrite_document(doc, report, client, cfg, progress=lambda m: progress_msgs.append(m))
    print("[3] 改写完成：%d 段，成功 %d 段" % (
        len(results), sum(1 for r in results.values() if r.ok)))
    for r in sorted(results.values(), key=lambda x: x.index):
        print("    #%-3d %.1f -> %.1f  ok=%s tries=%d %s" % (
            r.index + 1, r.score_before, r.score_after, r.ok, r.attempts, r.error))

    print("[4] 收到 HTTP 请求 %d 次；鉴权头示例 = %s" % (
        len(requests_seen), requests_seen[0]["auth"] if requests_seen else "无"))
    print("    路径 =", requests_seen[0]["path"], "| model =", requests_seen[0]["model"])

    replacements = {i: r.revised for i, r in results.items() if r.ok and r.revised}
    revised_path = os.path.join(OUT, "_mock_改写版.docx")
    write_docx(SAMPLE, replacements, revised_path)
    d2 = read_docx(revised_path)
    for idx, text in replacements.items():
        got = next(p.text for p in d2.paragraphs if p.index == idx)
        assert got == text, idx
    print("[5] 回写 docx 并校验通过：%d 段已替换" % len(replacements))

    html_path = os.path.join(OUT, "_mock_报告.html")
    render_html(report, html_path, results)
    html = open(html_path, encoding="utf-8").read()
    assert "改写后" in html and "https://" not in html
    print("[6] 带改写对照的 HTML 报告 OK：%d 字节，%d 个等级徽章" % (
        len(html.encode("utf-8")), html.count("badge")))

    # 用改写版重新整篇评分，验证整篇指数确实下降
    report2 = scoring.score_document(d2)
    print("[7] 改写后整篇指数：%.1f（改写前 %.1f，下降 %.1f）" % (
        report2.overall, report.overall, report.overall - report2.overall))
    assert report2.overall < report.overall, "改写后整篇指数没有下降"

    server.shutdown()
    print("\nMOCK 端到端全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

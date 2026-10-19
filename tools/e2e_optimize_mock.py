# -*- coding: utf-8 -*-
"""全文优化端到端自测：本地 mock 服务把长句拆成短句（保留全部数字/引用），
验证 optimize -> 重评 -> 回写 docx -> 报告 的完整闭环，不消耗 API 额度。

    python tools/e2e_optimize_mock.py
"""
from __future__ import annotations

import json
import os
import re
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
from paper_aigc.rewrite.optimize import apply_replacements, optimize_document

SAMPLE = os.path.join(_ROOT, "samples", "样例论文_AI味混排.docx")
OUT = os.path.join(_ROOT, "out")
os.makedirs(OUT, exist_ok=True)

_ORIGINAL_RE = re.compile(r"原文(?:（[^）]*）)?[:：]\s*\n(.+?)\n\s*\n现在输出", re.S)


def humanize(text: str) -> str:
    """把逗号分隔的长句拆成短句；不改动任何字符，只替换标点。"""
    clauses = [part.strip() for part in re.split(r"[，,；;]", text) if part.strip()]
    if not clauses:
        return text
    body = "。".join(clauses)
    if not body.endswith("。"):
        body += "。"
    return body


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        prompt = body["messages"][-1]["content"]
        match = _ORIGINAL_RE.search(prompt)
        reply = humanize(match.group(1).strip()) if match else "无法解析原文。"
        payload = {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
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
    print("[mock] 服务已启动 127.0.0.1:%d（改写策略：逗号转句号，长句拆短）" % port)

    cfg = AppConfig(api_key="sk-mock", base_url="http://127.0.0.1:%d" % port,
                    model="deepseek-chat", optimize_target=18.0, optimize_rounds=2,
                    optimize_max_calls=80, concurrency=4, max_rewrite_attempts=3,
                    timeout=15, retries=1)

    doc = read_docx(SAMPLE)
    before = scoring.score_document(doc)
    print("[1] 优化前：整篇 %.1f（%s），正文 %d 段" % (
        before.overall, before.level, len(doc.body_indices)))

    results = optimize_document(doc, before, DeepSeekClient(cfg), cfg,
                                target=cfg.optimize_target,
                                progress=lambda m: print("    .. " + m))
    accepted = {i: r.revised for i, r in results.items() if r.ok and r.revised}
    print("[2] 改写成功 %d / 处理 %d 段" % (len(accepted), len(results)))

    new_doc = apply_replacements(doc, accepted)
    after = scoring.score_document(new_doc)
    info = {"target": cfg.optimize_target, "overall_before": before.overall,
            "overall_after": after.overall, "paragraphs_rewritten": len(accepted),
            "paragraphs_total": len(doc.body_indices)}
    after.meta["optimize"] = info
    print("[3] 优化后：整篇 %.1f（%s），下降 %.1f" % (
        after.overall, after.level, before.overall - after.overall))
    assert after.overall < before.overall, "整篇指数没有下降"
    assert len(new_doc.paragraphs) == len(doc.paragraphs), "段落数量变了"

    revised_path = os.path.join(OUT, "_opt_优化版.docx")
    write_docx(SAMPLE, accepted, revised_path)
    check = read_docx(revised_path)
    for index, text in accepted.items():
        got = next(p.text for p in check.paragraphs if p.index == index)
        assert got.strip() == text.strip(), index
    print("[4] 回写 docx 校验通过：%d 段，段落总数 %d（与原文一致）" % (
        len(accepted), len(check.paragraphs)))

    html_path = os.path.join(OUT, "_opt_报告.html")
    render_html(after, html_path, results)
    html = open(html_path, encoding="utf-8").read()
    assert "全文优化结果" in html, "报告缺少优化结果区块"
    assert "https://" not in html and "http://" not in html
    print("[5] 报告校验通过：含「全文优化结果」、零外链、%d 字节" % len(html.encode("utf-8")))

    # 结构护栏：标题/参考文献段落必须一字未动
    changed_heading = [p.index for p in doc.paragraphs
                       if (p.is_heading or p.is_reference)
                       and next(q.text for q in check.paragraphs if q.index == p.index) != p.text]
    print("[6] 标题/参考文献被改动数量：%d（应为 0）" % len(changed_heading))
    assert not changed_heading

    server.shutdown()
    print("\n优化端到端全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

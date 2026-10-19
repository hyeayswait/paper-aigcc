# -*- coding: utf-8 -*-
"""PaperAIGC 命令行入口。

    python run.py 论文.docx                          # 检测 + 打印摘要 + 写 HTML 报告
    python run.py 论文.docx --optimize               # 全文优化到目标指数（默认 18）
    python run.py 论文.docx --optimize --optimize-target 15
    python run.py 论文.docx --rewrite                # 只改写超过阈值的段落
    python run.py 论文.docx -o 报告.html --json 结果.json

分数是本地文体特征算出的启发式「AI 痕迹指数」，未与知网校准，
绝对值不可与知网 AIGC 率比较。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paper_aigc.config import AppConfig, load_config
from paper_aigc.core import scoring
from paper_aigc.core.types import DISCLAIMER_ZH, VERSION, Document, RewriteResult, ScoreReport
from paper_aigc.io.document_loader import read_document
from paper_aigc.io.docx_writer import write_docx
from paper_aigc.io.report import render_html, render_markdown
from paper_aigc.io.similarity import to_meta as similarity_to_meta
from paper_aigc.io.similarity_providers import check as similarity_check
from paper_aigc.rewrite.client import DeepSeekClient, RewriteError
from paper_aigc.rewrite.optimize import apply_replacements, document_quality, optimize_document
from paper_aigc.rewrite.pipeline import rewrite_document


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def _out_dir_for(docx_path: str, cfg: AppConfig) -> str:
    base = cfg.export_dir or os.path.join(os.path.dirname(os.path.abspath(docx_path)), "AIGC输出")
    os.makedirs(base, exist_ok=True)
    return base


def _brief(report: ScoreReport, top_n: int = 10) -> str:
    lines = [
        "整篇 AI 痕迹指数：%.1f / 100（等级：%s）" % (report.overall, report.level),
        "评分引擎：%s　|　校准状态：%s" % (
            (report.meta or {}).get("engine", "?"), report.calibration),
        "参与评分段落：%d    正文总字数：%d" % (len(report.paragraphs), report.document.total_words),
        "等级分布：" + "  ".join("%s %d" % (k, report.level_counts.get(k, 0))
                              for k in ("低", "中", "偏高", "高")),
        "",
        "【全文画像】" + str(((report.meta or {}).get("profile") or {}).get("note") or "—"),
        "",
        "风险最高的段落：",
    ]
    for ps in report.sorted_by_risk()[:top_n]:
        signals = "、".join(s[1] for s in ps.top_signals[:2]) or "—"
        lines.append("  #%-4d %5.1f  %-2s  %s ｜ %s" % (
            ps.index + 1, ps.score, ps.level, signals, ps.text[:36].replace("\n", " ")))
    if not report.paragraphs:
        lines.append("  （没有可评分的正文段落）")
    return "\n".join(lines)


def _json_payload(report: ScoreReport, rewrites: Optional[Dict[int, RewriteResult]]) -> dict:
    data = {
        "tool": "PaperAIGC",
        "version": VERSION,
        "calibration": report.calibration,
        "disclaimer": report.disclaimer,
        "document": {
            "path": report.document.path,
            "title": report.document.title,
            "total_words": report.document.total_words,
            "scored_paragraphs": len(report.paragraphs),
        },
        "overall": report.overall,
        "level": report.level,
        "level_counts": report.level_counts,
        "profile": (report.meta or {}).get("profile"),
        "optimize": (report.meta or {}).get("optimize"),
        "paragraphs": [
            {
                "index": p.index,
                "score": p.score,
                "level": p.level,
                "chars": len(p.text),
                "top_signals": [{"key": k, "name": n, "contribution": c} for k, n, c in p.top_signals],
                "suggestion": p.suggestion,
            }
            for p in report.sorted_by_risk()
        ],
    }
    if rewrites:
        data["rewrites"] = [
            {
                "index": r.index,
                "ok": r.ok,
                "score_before": r.score_before,
                "score_after": r.score_after,
                "attempts": r.attempts,
                "error": r.error,
                "notes": r.notes,
            }
            for r in sorted(rewrites.values(), key=lambda x: x.index)
        ]
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="论文 AI 痕迹自查与自然化改写（本地启发式指数，未与知网校准）",
    )
    p.add_argument("docx", help="待检测文件：.docx / .pdf / .txt / .md")
    p.add_argument("-o", "--out", default="", help="HTML 报告输出路径")
    p.add_argument("--no-html", action="store_true", help="不生成 HTML 报告")
    p.add_argument("--json", default="", dest="json_path", help="同时输出 JSON 摘要文件")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--rewrite", action="store_true", help="只改写超过阈值的段落（需 API Key）")
    mode.add_argument("--optimize", action="store_true",
                      help="全文优化：把所有段落压到目标指数以下（需 API Key）")
    mode.add_argument("--dedup", action="store_true",
                      help="降重专项：把重复率高的段落改写到目标重复率以下（需 API Key + 本地库）")
    p.add_argument("--optimize-target", type=float, default=None,
                   help="全文优化的目标指数，默认取配置（18）")
    p.add_argument("--threshold", type=float, default=None, help="改写阈值，默认取配置（45 = 偏高起点）")
    p.add_argument("--target", type=float, default=None, help="单段改写目标分数，默认取配置")
    p.add_argument("--target-rate", type=float, default=None,
                   help="全文优化目标百分比，例如 50 / 40 / 30（默认取配置，常用下限 30）")
    p.add_argument("--similarity", action="store_true", help="执行本地库查重（需 --corpus 或配置目录）")
    p.add_argument("--lint", action="store_true", help="执行论文规范自检（图表编号/参考文献/标点/术语等）")
    p.add_argument("--ledger-add", action="append", default=[], metavar="K=V,...",
                   help='写入分数台账，如 "system=朱雀,metric=疑似AI,value=34.47,version=V6"')
    p.add_argument("--ledger-file", default="", help="台账文件路径（默认 data/score_ledger.json）")
    p.add_argument("--ledger-show", action="store_true", help="打印分数台账")
    p.add_argument("--similarity-target", type=float, default=0.15,
                   help="降重目标重复率（0-1，默认 0.15）")
    p.add_argument("--corpus", default="", help="本地比对库目录（.txt/.md/.docx）")
    p.add_argument("--similarity-provider", default="",
                   help="查重 provider：local（默认）或 web:<已注册后端名>")
    p.add_argument("--heuristic", action="store_true",
                   help="强制使用手工锚点评分（默认走公开数据校准模型）")
    p.add_argument("--model", default="", help="覆盖模型名")
    p.add_argument("--api-key", default="", help="覆盖 API Key（不推荐，会留在命令历史）")
    p.add_argument("-q", "--quiet", action="store_true", help="只输出最终分数")
    return p


def _print_rewrite_summary(rewrites: Dict[int, RewriteResult]) -> None:
    ok = [r for r in rewrites.values() if r.ok]
    print("")
    print("改写结果：成功 %d / 共 %d 段" % (len(ok), len(rewrites)))
    for r in sorted(rewrites.values(), key=lambda x: x.index):
        flag = "OK  " if r.ok else "FAIL"
        note = r.error or ("；".join(r.notes) if r.notes else "")
        print("  [%s] #%-4d %.1f -> %.1f  (%d 次) %s" % (
            flag, r.index + 1, r.score_before, r.score_after, r.attempts, note[:80]))


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8_stdout()
    args = build_parser().parse_args(argv)
    cfg = load_config()
    if args.threshold is not None:
        cfg.high_threshold = args.threshold
    if args.target is not None:
        cfg.rewrite_target = args.target
    if args.optimize_target is not None:
        cfg.optimize_target = args.optimize_target
    if args.target_rate is not None:
        cfg.target_rate = args.target_rate
    if args.corpus:
        cfg.corpus_dir = args.corpus
    if args.similarity_provider:
        cfg.similarity_provider = args.similarity_provider
    if args.heuristic:
        cfg.score_engine = "heuristic"
    try:
        scoring.set_engine(cfg.score_engine)
    except Exception:
        scoring.set_engine("heuristic")
    if args.model:
        cfg.model = args.model
    if args.api_key:
        cfg.api_key = args.api_key

    path = os.path.abspath(args.docx)
    if not os.path.isfile(path):
        print("找不到文件：%s" % path, file=sys.stderr)
        return 2
    try:
        doc = read_document(path)
    except Exception as exc:  # noqa: BLE001
        print("读取文档失败：%s" % exc, file=sys.stderr)
        return 2
    if not doc.body_indices:
        print("这份文档没有可参与评分的正文段落（全是标题/参考文献/空行？）。", file=sys.stderr)
        return 2

    report = scoring.score_document(doc)
    rewrites: Dict[int, RewriteResult] = {}

    if not args.quiet:
        print("=" * 72)
        print("PaperAIGC  %s" % VERSION)
        print(_brief(report))
        print("=" * 72)
        print(DISCLAIMER_ZH)
        print("=" * 72)
    elif not (args.optimize or args.rewrite):
        print("%.1f" % report.overall)

    needs_api = args.rewrite or args.optimize or args.dedup
    client = None
    if needs_api:
        if not cfg.api_key:
            print("未配置 DeepSeek API Key：请设置环境变量 DEEPSEEK_API_KEY 或在设置里填写。",
                  file=sys.stderr)
            return 2
        try:
            client = DeepSeekClient(cfg)
        except RewriteError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    def progress(msg: str) -> None:
        if not args.quiet:
            print("  .. %s" % msg, file=sys.stderr, flush=True)

    if args.dedup and client is not None:
        corpus = cfg.corpus_dir
        if not corpus:
            print("降重需要本地比对库：请用 --corpus 指定目录（或设置里配置 corpus_dir）。", file=sys.stderr)
            return 2
        from paper_aigc.rewrite.dedup import dedup_document
        if not args.quiet:
            print("")
            print("降重开始：目标重复率 < %.0f%%，比对库 %s" % (args.similarity_target * 100, corpus))
        rewrites = dedup_document(doc, client, cfg, corpus,
                                  target_ratio=args.similarity_target, progress=progress)
        if not args.quiet:
            _print_rewrite_summary(rewrites)
        replacements = {i: r.revised for i, r in rewrites.items() if r.ok and r.revised}
        if replacements:
            stem = os.path.splitext(os.path.basename(path))[0]
            revised_path = os.path.join(_out_dir_for(path, cfg), stem + "_降重版.docx")
            write_docx(path, replacements, revised_path)
            if not args.quiet:
                print("降重版已保存：%s（改写 %d 段）" % (revised_path, len(replacements)))

    elif args.optimize and client is not None:
        goal = float(cfg.optimize_target)
        if not args.quiet:
            print("")
            print("全文优化开始：目标整篇指数 < %.0f，最多 %d 轮、%d 次调用"
                  % (goal, cfg.optimize_rounds, cfg.optimize_max_calls))
        rewrites = optimize_document(doc, report, client, cfg, target=goal, progress=progress)
        accepted = {index: r.revised for index, r in rewrites.items() if r.ok and r.revised}
        new_doc = apply_replacements(doc, accepted)
        report_after = scoring.score_document(new_doc)

        stem = os.path.splitext(os.path.basename(path))[0]
        revised_path = os.path.join(_out_dir_for(path, cfg), stem + "_优化版.docx")
        write_docx(path, accepted, revised_path)
        info = {
            "target": goal,
            "overall_before": report.overall,
            "overall_after": report_after.overall,
            "paragraphs_rewritten": len(accepted),
            "paragraphs_total": len(doc.body_indices),
        }
        report.meta["optimize"] = info
        report_after.meta["optimize"] = info
        texts = [new_doc.paragraphs[i].text for i in new_doc.body_indices]
        quality = document_quality(texts)
        report_after.meta["quality"] = quality
        report = report_after
        if args.quiet:
            print("%.1f" % report_after.overall)
        if not args.quiet:
            _print_rewrite_summary(rewrites)
            print("")
            print("节奏体检：句数 %d　平均句长 %.1f 字　短句占比 %.2f　长句占比 %.2f　"
                  "作者视角 %d 次　段首重复 %d" % (
                      quality["sentences"], quality["avg_len"], quality["short_ratio"],
                      quality["long_ratio"], quality["author_voice"], quality["repeated_openers"]))
            if quality["short_ratio"] > 0.30 or quality["long_ratio"] < 0.05:
                print("  警告：句子节奏偏单一（短句过多或几乎没有长句），可能仍带机器味。", file=sys.stderr)
            print("整篇 AI 痕迹指数：%.1f → %.1f（目标 < %.0f）" % (
                info["overall_before"], info["overall_after"], goal))
            print("优化版 docx：%s（替换 %d / %d 段，段落数不变）" % (
                revised_path, len(accepted), len(doc.body_indices)))
            if info["overall_after"] >= goal:
                print("注意：未达到目标，可提高轮数/次数或先手工改几句再跑一次。", file=sys.stderr)

    elif args.rewrite and client is not None:
        rewrites = rewrite_document(doc, report, client, cfg, progress=progress)
        if args.quiet:
            print("%.1f" % report.overall)
        if not args.quiet:
            _print_rewrite_summary(rewrites)
        replacements = {i: r.revised for i, r in rewrites.items() if r.ok and r.revised}
        if replacements:
            stem = os.path.splitext(os.path.basename(path))[0]
            revised_path = os.path.join(_out_dir_for(path, cfg), stem + "_改写版.docx")
            write_docx(path, replacements, revised_path)
            if not args.quiet:
                print("改写版已保存：%s（替换 %d 段）" % (revised_path, len(replacements)))

    # ---- 规范自检 ----
    if args.lint:
        try:
            from paper_aigc.core.lint import lint_document, to_meta as lint_to_meta
            lint_meta = lint_to_meta(lint_document(doc))
            report.meta["lint"] = lint_meta
            counts = lint_meta.get("counts") or {}
            if not args.quiet:
                print("")
                print("论文规范自检：错误 %d / 警告 %d / 提示 %d" % (
                    int(counts.get("error", 0)), int(counts.get("warn", 0)),
                    int(counts.get("info", 0))))
                for item in (lint_meta.get("issues") or [])[:10]:
                    where = "第%s段" % item.get("paragraph_index") if item.get("paragraph_index", -1) >= 0 else "-"
                    print("  [%s] %-10s %s  %s" % (
                        str(item.get("level")).upper(), item.get("code"), where,
                        str(item.get("message"))[:60]))
        except Exception as exc:  # noqa: BLE001
            print("规范自检失败：%s" % exc, file=sys.stderr)

    # ---- 分数台账 ----
    try:
        from paper_aigc.io.score_ledger import (
            LedgerRecord, add_record, ledger_path, load_records,
            to_meta as ledger_to_meta,
        )
        ledger_file = args.ledger_file or ""
        for spec in args.ledger_add:
            fields = {}
            for chunk in str(spec).split(","):
                if "=" in chunk:
                    key, value = chunk.split("=", 1)
                    fields[key.strip()] = value.strip()
            if not fields.get("system") or not fields.get("metric"):
                print("台账记录缺少 system 或 metric：%s" % spec, file=sys.stderr)
                continue
            try:
                numeric = float(fields.get("value", 0))
            except ValueError:
                numeric = 0.0
            path_written = add_record(LedgerRecord(
                version=fields.get("version", ""), system=fields["system"],
                metric=fields["metric"], value=numeric, note=fields.get("note", "")),
                path=ledger_file)
            if not args.quiet:
                print("台账已写入：%s（%s %s %.2f%%）" % (
                    path_written, fields["system"], fields["metric"], numeric))
        records = load_records(ledger_file)
        if records:
            report.meta["ledger"] = ledger_to_meta(records)
        if args.ledger_show and not args.quiet:
            print("")
            print("分数台账（%s）：" % (ledger_file or ledger_path()))
            for item in records[-20:]:
                print("  %-16s %-8s %-10s %-10s %6.2f%%  %s" % (
                    item.time, item.version, item.system, item.metric, item.value, item.note))
    except Exception as exc:  # noqa: BLE001
        print("台账处理失败：%s" % exc, file=sys.stderr)

    if args.similarity:
        try:
            sim_report = similarity_check(doc, cfg)
        except Exception as exc:  # noqa: BLE001
            print("查重失败：%s" % exc, file=sys.stderr)
            sim_report = None
        if sim_report is not None:
            report.meta["similarity"] = similarity_to_meta(sim_report)
            if not args.quiet:
                print("")
                print("本地库查重：总文字复制比 %.2f%%（比对库 %d 个文件 / %d 字）" % (
                    sim_report.overall_percent, sim_report.corpus_files, sim_report.corpus_chars))
                print("  %s" % sim_report.note)
                for item in sim_report.sorted_by_risk()[:8]:
                    if item.ratio <= 0:
                        continue
                    source = item.sources[0].source if item.sources else "—"
                    print("    #%-4d %5.1f%%  最相似来源：%s" % (
                        item.index + 1, item.ratio * 100, source))
                for a, b, overlap in sim_report.internal_duplicates[:5]:
                    print("    自我重复：第 %d 段与第 %d 段重合 %.0f%%" % (
                        a + 1, b + 1, overlap * 100))

# ---- 报告输出 ----
    if not args.no_html:
        stem = os.path.splitext(os.path.basename(path))[0]
        out = args.out or os.path.join(_out_dir_for(path, cfg), stem + "_AIGC自查报告.html")
        try:
            render_html(report, out, rewrites or None)
            if not args.quiet:
                print("HTML 报告已保存：%s" % os.path.abspath(out))
        except Exception as exc:  # noqa: BLE001
            print("生成报告失败：%s" % exc, file=sys.stderr)
            return 3

    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as f:
            json.dump(_json_payload(report, rewrites or None), f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print("JSON 摘要已保存：%s" % os.path.abspath(args.json_path))

    if not args.quiet:
        print("")
        print(render_markdown(report, rewrites or None))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""报告渲染：把 ScoreReport 渲染成单文件 HTML 或 Markdown。

定位说明：本工具没有接触过知网，也没有知网标注数据，报告中的分数是 0-100 的
启发式「AI 痕迹指数」，不是知网 AIGC 率。HTML 与 Markdown 都必须在开头/首屏
显著展示 disclaimer，且不得暗示分数与知网结果等价。

HTML 约束：单文件、零外部资源（无 CDN / 字体 / 图片 / 外链），全部用户文本
经 html.escape 转义；disclaimer 首屏可见、字号 >= 13px、不可折叠。
"""
from __future__ import annotations

import html
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.types import DISCLAIMER_ZH, LEVELS, VERSION, RewriteResult, ScoreReport

LEVEL_COLORS: Dict[str, str] = {
    "低": "#2e7d32",
    "中": "#f9a825",
    "偏高": "#ef6c00",
    "高": "#c62828",
}

_CSS = """
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 40px; background: #f4f6f8; color: #222;
       font-family: "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "Segoe UI", Arial, sans-serif;
       line-height: 1.65; font-size: 14px; }
.wrap { max-width: 1080px; margin: 0 auto; }
header { border-left: 6px solid #1e5aa8; padding: 4px 14px; margin-bottom: 16px; }
h1 { font-size: 24px; margin: 0 0 6px; }
.sub { color: #555; font-size: 13px; margin: 0; }
.disclaimer { background: #fff4e5; border: 1px solid #e0a04a; border-radius: 8px;
              padding: 14px 16px; margin: 0 0 18px; }
.disclaimer strong { display: block; font-size: 15px; margin-bottom: 6px; color: #8a4b00; }
.disclaimer p { margin: 6px 0 0; font-size: 14.5px; color: #5a3a00; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
         gap: 12px; margin-bottom: 20px; }
.card { background: #fff; border: 1px solid #dde3ea; border-radius: 8px; padding: 12px 14px; }
.card-label { color: #667788; font-size: 13px; margin-bottom: 6px; }
.card-value { font-size: 22px; font-weight: 700; }
.card-value.big { font-size: 40px; line-height: 1.1; }
.card-sub { color: #667788; font-size: 12.5px; margin-top: 4px; }
h2 { font-size: 18px; margin: 26px 0 10px; padding-bottom: 6px; border-bottom: 2px solid #dde3ea; }
.meta { color: #556677; font-size: 13px; margin: 0 0 10px; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dde3ea; }
th, td { border-bottom: 1px solid #e6ebf1; padding: 8px 9px; text-align: left;
         vertical-align: top; font-size: 13.5px; }
th { background: #eef2f7; font-size: 13px; white-space: nowrap; }
td.num { text-align: center; color: #667788; width: 40px; }
.score-num { font-weight: 700; }
.bar { height: 7px; background: #e8ecf2; border-radius: 4px; overflow: hidden;
       width: 96px; margin-top: 5px; }
.bar span { display: block; height: 100%; }
.badge { display: inline-block; color: #ffffff; padding: 2px 9px; border-radius: 10px;
         font-size: 12.5px; white-space: nowrap; }
.signals { font-size: 12.8px; color: #333333; }
.suggestion { font-size: 13.5px; }
.plain { white-space: pre-wrap; word-break: break-word; font-size: 13.5px; }
details summary { cursor: pointer; color: #1e5aa8; font-size: 13px; }
details[open] summary { margin-bottom: 6px; }
.empty { text-align: center; color: #778899; padding: 18px; }
.rewrite-item { background: #fff; border: 1px solid #dde3ea; border-radius: 8px;
                padding: 12px 14px; margin-bottom: 12px; }
.rewrite-item h3 { margin: 0 0 8px; font-size: 15px; }
.arrow { color: #1e5aa8; font-weight: 700; }
.attempts { color: #667788; font-weight: 400; font-size: 13px; }
.tag-ok { background: #e6f4ea; color: #1e7a3c; border-radius: 10px; padding: 2px 8px;
          font-size: 12px; }
.tag-fail { background: #fdecea; color: #b3261e; border-radius: 10px; padding: 2px 8px;
            font-size: 12px; }
.compare { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.col-title { font-size: 12.5px; color: #667788; margin-bottom: 4px; }
.note { margin-top: 8px; color: #b3261e; font-size: 13px; }
footer { margin-top: 30px; color: #667788; font-size: 12.5px; text-align: center;
         border-top: 1px solid #dde3ea; padding-top: 12px; }
@media (max-width: 720px) { .compare { grid-template-columns: 1fr; } }
"""


def _esc(value: object) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        num = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return num


def _fmt_score(value: object) -> str:
    return "%.1f" % _as_float(value)


def _bar_width(score: object) -> int:
    width = int(round(_as_float(score)))
    return max(0, min(100, width))


def _level_color(level: str) -> str:
    return LEVEL_COLORS.get(level or "", "#555555")


def _display_index(index: object) -> int:
    try:
        return int(index) + 1
    except (TypeError, ValueError):
        return 0


def _level_counts(report: ScoreReport) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for level in LEVELS:
        counts[level] = 0
    given = report.level_counts or {}
    for level in LEVELS:
        try:
            counts[level] = int(given.get(level, 0) or 0)
        except (TypeError, ValueError):
            counts[level] = 0
    if not any(counts.values()):
        for item in report.paragraphs or []:
            level = item.level if item.level in counts else "低"
            counts[level] += 1
    return counts


def _total_words(report: ScoreReport) -> int:
    total = 0
    try:
        total = int(report.document.total_words or 0)
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        for paragraph in report.document.paragraphs or []:
            total += int(paragraph.char_count or 0)
    return total


def _top_paragraph(report: ScoreReport):
    ranked = report.sorted_by_risk()
    return ranked[0] if ranked else None


def _signals_text(signals: Sequence[Tuple[str, str, float]]) -> str:
    parts: List[str] = []
    for signal in signals or []:
        try:
            name = signal[1]
            contribution = signal[2]
        except (TypeError, IndexError, KeyError):
            continue
        parts.append("%s（%s）" % (_esc(name), _esc(_fmt_score(contribution))))
    if not parts:
        return "—"
    return "<br>".join(parts)


def _card(label: str, value: str, sub: str, color: str, big: bool = False) -> str:
    size = " big" if big else ""
    return (
        '<div class="card"><div class="card-label">%s</div>'
        '<div class="card-value%s" style="color:%s">%s</div>'
        '<div class="card-sub">%s</div></div>'
        % (_esc(label), size, color, _esc(value), _esc(sub))
    )


def _header_html(report: ScoreReport, generated: str) -> str:
    title = report.document.title or "未命名文档"
    file_name = report.document.file_name or ""
    return (
        "<header><h1>论文 AI 痕迹自查报告</h1>"
        "<p class='sub'>文件：%s　|　标题：%s　|　工具版本：%s　|　生成时间：%s</p>"
        "</header>"
        % (_esc(file_name), _esc(title), _esc(VERSION), _esc(generated))
    )


def _disclaimer_html(report: ScoreReport) -> str:
    disclaimer = report.disclaimer or DISCLAIMER_ZH
    return (
        '<section class="disclaimer" id="disclaimer" style="font-size:14.5px;">'
        "<strong>重要说明（必读）：这不是知网 AIGC 率</strong>"
        '<p style="font-size:14.5px;">%s</p>'
        '<p style="font-size:13.5px;">'
        "分数的可成立用法只有：同一版本内改写前后对比、段落之间相对排序。"
        "绝对数值不可与知网或其他商业检测结果比较，也不能作为学术不端判定的依据。"
        "</p>"
        "</section>" % _esc(disclaimer)
    )


def _cards_html(report: ScoreReport) -> str:
    counts = _level_counts(report)
    top = _top_paragraph(report)
    if top is None:
        top_text = "—"
        top_sub = "没有参与评分的段落"
    else:
        top_text = "第 %d 段" % _display_index(top.index)
        top_sub = "本工具段序 · %s 分 · 等级 %s" % (_fmt_score(top.score), top.level or "低")
    level = report.level or "低"
    cards = [
        _card(
            "AI 痕迹指数（0-100）",
            _fmt_score(report.overall),
            "等级：%s　（启发式，非知网 AIGC 率）" % level,
            _level_color(level),
            big=True,
        ),
        _card("参与评分段数", str(len(report.paragraphs or [])), "只统计正文段落", "#1e5aa8"),
        _card("总字数", str(_total_words(report)), "正文段落字数合计", "#1e5aa8"),
        _card("高风险段数", str(counts.get("高", 0)), "等级为「高」的段落", _level_color("高")),
        _card("最高风险段落", top_text, top_sub, "#8a4b00"),
    ]
    return '<section class="cards">' + "".join(cards) + "</section>"


def _meta_html(report: ScoreReport) -> str:
    meta = report.meta or {}
    engine = str(meta.get("engine") or "stylometric-heuristic")
    feature_count = meta.get("feature_count")
    feature_text = str(feature_count) if feature_count is not None else "—"
    calibration = str(report.calibration or "uncalibrated")
    cal_info = meta.get("calibration") or {}
    if calibration.startswith("public-data"):
        try:
            cal_text = "公开数据校准 %s（测试集 AUC %.3f）" % (
                cal_info.get("model", "v1"), float(cal_info.get("test_auc", 0.0)))
        except (TypeError, ValueError):
            cal_text = "公开数据校准"
    else:
        cal_text = "未校准（%s）" % calibration
    thresholds = meta.get("level_thresholds") or [25.0, 45.0, 60.0]
    try:
        low, mid, high = [float(v) for v in thresholds]
    except (TypeError, ValueError):
        low, mid, high = 25.0, 45.0, 60.0
    return (
        "<p class='meta'>评分引擎：%s　|　校准状态：%s　|　特征维度：%s　|　"
        "等级口径：低 &lt;%.0f，中 &lt;%.0f，偏高 &lt;%.0f，高 ≥%.0f</p>"
        % (_esc(engine), _esc(cal_text), _esc(feature_text), low, mid, high, high)
    )


def _profile(report: ScoreReport) -> Dict[str, object]:
    profile = (report.meta or {}).get("profile")
    return profile if isinstance(profile, dict) else {}


def _profile_html(report: ScoreReport) -> str:
    profile = _profile(report)
    if not profile:
        return ""
    note = str(profile.get("note") or "")
    signals = profile.get("global_signals") or []
    rows: List[str] = []
    for item in signals[:5]:
        if not isinstance(item, dict):
            continue
        try:
            ratio = int(round(float(item.get("elevated_ratio", 0)) * 100))
            contribution = "%.1f" % float(item.get("mean_contribution", 0))
        except (TypeError, ValueError):
            continue
        rows.append("<tr><td>%s</td><td class='num'>%d%%</td><td class='num'>%s</td></tr>"
                    % (_esc(item.get("name")), ratio, _esc(contribution)))
    table = ""
    if rows:
        table = (
            "<table><thead><tr><th>共性特征</th><th>触顶段落占比</th><th>平均贡献分</th></tr>"
            "</thead><tbody>%s</tbody></table>" % "".join(rows)
        )
    homogeneous = "是" if profile.get("homogeneous") else "否"
    dispersion = profile.get("dispersion", 0)
    return (
        "<h2>全文画像（先看这里）</h2>"
        "<p style='background:#eef4ff;border-left:4px solid #1e5aa8;padding:10px 12px;"
        "margin:10px 0;font-size:14px;'>%s</p>"
        "<p class='meta'>文风是否高度同质：%s　|　分数标准差：%s</p>%s"
        % (_esc(note), homogeneous, _esc(dispersion), table)
    )


def _optimize_info(report: ScoreReport):
    info = (report.meta or {}).get("optimize")
    return info if isinstance(info, dict) else {}


def _optimize_html(report: ScoreReport) -> str:
    info = _optimize_info(report)
    if not info:
        return ""
    try:
        before = float(info.get("overall_before", 0))
        after = float(info.get("overall_after", 0))
        target = float(info.get("target", 0))
        rewritten = int(info.get("paragraphs_rewritten", 0))
        total = int(info.get("paragraphs_total", 0))
    except (TypeError, ValueError):
        return ""
    reached = after < target
    color = "#1b7a4b" if reached else "#b4231f"
    status = "已达标" if reached else "未达标"
    return (
        "<h2>全文优化结果</h2>"
        "<p style='font-size:15px;background:#f7f9fc;border:1px solid #dde3ea;"
        "border-radius:8px;padding:12px 14px;'>"
        "整篇 AI 痕迹指数：<b>%.1f</b> → <b style='color:%s'>%.1f</b>"
        "（目标 &lt; %.0f，%s）<br>"
        "改写段落：%d / %d　|　段落数量与文档结构未改变　|　数字、引用标记、表号已逐段校验"
        "</p>" % (before, color, after, target, status, rewritten, total)
    )


def _similarity_info(report: ScoreReport):
    info = (report.meta or {}).get("similarity")
    return info if isinstance(info, dict) else {}


def _similarity_html(report: ScoreReport) -> str:
    info = _similarity_info(report)
    if not info:
        return ""
    rows: List[str] = []
    for item in (info.get("paragraphs") or [])[:12]:
        if not isinstance(item, dict):
            continue
        sources = "；".join(
            "%s（%.0f%%）" % (_esc(src.get("source")), float(src.get("ratio_percent") or 0))
            for src in (item.get("sources") or [])[:2] if isinstance(src, dict)
        ) or "—"
        try:
            ratio = float(item.get("ratio_percent") or 0)
        except (TypeError, ValueError):
            ratio = 0.0
        color = "#b4231f" if ratio >= 40 else "#c25a12" if ratio >= 20 else "#1b7a4b"
        rows.append(
            "<tr><td class='num'>%s</td><td class='score-cell'><span class='score-num' "
            "style='color:%s'>%.1f%%</span></td><td class='signals'>%s</td></tr>"
            % (_esc(item.get("display")), color, ratio, sources)
        )
    table = ""
    if rows:
        table = ("<table><thead><tr><th>段落</th><th>重复率</th><th>相似来源</th></tr></thead>"
                 "<tbody>%s</tbody></table>" % "".join(rows))
    duplicates = info.get("internal_duplicates") or []
    dup_text = ""
    if duplicates:
        dup_text = "<p class='meta'>自我重复：" + "；".join(
            "第 %s 段与第 %s 段重合 %.0f%%" % (_esc(d.get("a")), _esc(d.get("b")),
                                            float(d.get("overlap_percent") or 0))
            for d in duplicates[:5] if isinstance(d, dict)) + "</p>"
    return (
        "<h2>本地库查重</h2>"
        "<p style='font-size:15px;background:#fff6e0;border:1px solid #f0d9a8;border-radius:8px;"
        "padding:12px 14px;'>总文字复制比：<b>%.2f%%</b>　|　比对库：%s 个文件 / %s 字"
        "　|　判据：连续 %s 字"
        "<br>%s</p>%s%s"
        % (float(info.get("overall_percent") or 0), _esc(info.get("corpus_files")),
           _esc(info.get("corpus_chars")), _esc(info.get("ngram")),
           _esc(info.get("note")), table, dup_text)
    )


def _overview_rows(report: ScoreReport) -> List[Tuple[str, str, str]]:
    """一页纸总览的数据：(项目, 数值, 说明)。"""
    meta = report.meta or {}
    rows: List[Tuple[str, str, str]] = []

    # AIGC
    engine = str(meta.get("engine") or "-")
    if engine.startswith("calibrated"):
        rows.append(("AIGC（校准概率）", "%.1f%%" % float(report.overall),
                     "等级 %s · 公开数据校准" % (report.level or "-")))
    else:
        rows.append(("AIGC（启发式指数）", "%.1f / 100" % float(report.overall),
                     "等级 %s · 未校准" % (report.level or "-")))

    # 规模
    top = _top_paragraph(report)
    rows.append(("参与评分段落", "%d 段" % len(report.paragraphs or []), "正文段数"))
    rows.append(("正文总字数", "%d" % _total_words(report), "参与评分段落字数合计"))
    counts = (report.level_counts or {})
    rows.append(("高风险段数", "%d 段" % int(counts.get("高", 0)), "等级为「高」的段落"))
    if top is not None:
        rows.append(("最高风险段落", "第 %d 段" % _display_index(top.index),
                     "%.1f 分 · 等级 %s" % (float(top.score), top.level or "低")))

    # 查重
    similarity = meta.get("similarity") or {}
    if isinstance(similarity, dict) and similarity:
        rows.append(("查重（本地库）", "%.2f%%" % float(similarity.get("overall_percent") or 0),
                     "比对库 %s 文件" % similarity.get("corpus_files", "-")))
    else:
        rows.append(("查重", "未执行", "用 --similarity 运行"))

    # 规范自检
    lint = meta.get("lint") or {}
    if isinstance(lint, dict) and lint:
        counts = lint.get("counts") or {}
        rows.append(("规范自检", "%d 错误 / %d 警告" % (
            int(counts.get("error", 0)), int(counts.get("warn", 0))), "详见下方自检表"))
    else:
        rows.append(("规范自检", "未执行", "用 --lint 运行"))

    # 优化结果
    optimize = meta.get("optimize") or {}
    if isinstance(optimize, dict) and optimize:
        rows.append(("优化", "%.1f → %.1f" % (
            float(optimize.get("overall_before") or 0),
            float(optimize.get("overall_after") or 0)),
            "目标 < %.0f · 改写 %s 段" % (float(optimize.get("target") or 0),
                                        optimize.get("paragraphs_rewritten", "-"))))

    # 台账
    ledger = meta.get("ledger") or {}
    latest = ledger.get("latest") if isinstance(ledger, dict) else None
    if isinstance(latest, dict) and latest:
        parts = []
        for system, metrics in latest.items():
            if not isinstance(metrics, dict):
                continue
            for metric, info in metrics.items():
                if isinstance(info, dict):
                    parts.append("%s %s %.2f%%" % (system, metric, float(info.get("value") or 0)))
        rows.append(("多系统台账", "见下方", "；".join(parts[:4]) or "-"))
    return rows


def _overview_html(report: ScoreReport) -> str:
    rows = _overview_rows(report)
    if not rows:
        return ""
    cells = []
    for label, value, sub in rows:
        cells.append(
            "<div class='card'><div class='card-label'>%s</div>"
            "<div class='card-value'>%s</div><div class='card-sub'>%s</div></div>"
            % (_esc(label), _esc(value), _esc(sub)))
    return ("<h2>一页纸总览</h2><section class='cards'>" + "".join(cells) + "</section>")


def _lint_html(report: ScoreReport) -> str:
    info = (report.meta or {}).get("lint")
    if not isinstance(info, dict) or not info:
        return ""
    counts = info.get("counts") or {}
    issues = info.get("issues") or []
    rows = []
    for item in issues[:40]:
        if not isinstance(item, dict):
            continue
        level = str(item.get("level") or "info")
        color = {"error": "#b4231f", "warn": "#c25a12"}.get(level, "#667788")
        where = "第 %s 段" % item.get("display") if item.get("display") else "-"
        rows.append(
            "<tr><td><span class='badge' style='background:%s'>%s</span></td>"
            "<td class='num'>%s</td><td class='signals'>%s</td>"
            "<td class='suggestion'>%s</td></tr>"
            % (color, _esc(level), _esc(where), _esc(item.get("message")),
               _esc(item.get("snippet"))))
    table = ""
    if rows:
        table = ("<table><thead><tr><th>级别</th><th>位置</th><th>问题</th><th>原文片段</th>"
                 "</tr></thead><tbody>%s</tbody></table>" % "".join(rows))
    return (
        "<h2>论文规范自检</h2>"
        "<p class='meta'>错误 %d 个 / 警告 %d 个 / 提示 %d 个（不涉及 AIGC，纯格式与规范检查）</p>%s"
        % (int(counts.get("error", 0)), int(counts.get("warn", 0)),
           int(counts.get("info", 0)), table)
    )


def _ledger_html(report: ScoreReport) -> str:
    info = (report.meta or {}).get("ledger")
    if not isinstance(info, dict) or not info:
        return ""
    rows = []
    for item in (info.get("rows") or [])[-20:]:
        if not isinstance(item, dict):
            continue
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td class='num'>%.2f%%</td><td>%s</td></tr>"
            % (_esc(item.get("time")), _esc(item.get("version")), _esc(item.get("system")),
               _esc(item.get("metric")), float(item.get("value") or 0), _esc(item.get("note"))))
    if not rows:
        return ""
    return (
        "<h2>多系统分数台账</h2>"
        "<table><thead><tr><th>时间</th><th>版本</th><th>系统</th><th>指标</th>"
        "<th>数值</th><th>备注</th></tr></thead><tbody>%s</tbody></table>" % "".join(rows)
    )


def _paragraph_rows(report: ScoreReport) -> str:
    rows: List[str] = []
    for rank, item in enumerate(report.sorted_by_risk(), 1):
        color = _level_color(item.level)
        level_text = item.level or "低"
        suggestion = item.suggestion or "—"
        original = item.text or "（无原文）"
        rows.append(
            (
                "<tr>"
                "<td class='num'>%d</td>"
                "<td class='score-cell'><span class='score-num'>%s</span>"
                "<div class='bar'><span style='width:%d%%;background:%s'></span></div></td>"
                "<td><span class='badge' style='background:%s'>%s</span></td>"
                "<td class='signals'>%s</td>"
                "<td class='suggestion'>%s</td>"
                "<td class='original'><details><summary>展开原文</summary>"
                "<div class='plain'>%s</div></details></td>"
                "</tr>"
            )
            % (
                rank,
                _fmt_score(item.score),
                _bar_width(item.score),
                color,
                color,
                _esc(level_text),
                _signals_text(item.top_signals),
                _esc(suggestion),
                _esc(original),
            )
        )
    if not rows:
        rows.append("<tr><td colspan='6' class='empty'>没有参与评分的段落。</td></tr>")
    return "\n".join(rows)


def _paragraph_table_html(report: ScoreReport) -> str:
    return (
        "<h2>段落风险排行（按启发式分数降序）</h2>"
        "<p class='meta'>分数越高，表示该段文体特征越接近本工具判定的 AI 生成倾向；"
        "排序只在这份文档内部有效。</p>"
        "<table><thead><tr><th>序号</th><th>分数</th><th>等级</th>"
        "<th>Top 信号</th><th>一句话建议</th><th>原文</th></tr></thead>"
        "<tbody>%s</tbody></table>" % _paragraph_rows(report)
    )


def _rewrite_html(rewrite: Optional[Dict[int, RewriteResult]]) -> str:
    if not rewrite:
        return ""
    blocks: List[str] = []
    for index in sorted(rewrite):
        result = rewrite[index]
        if result is None:
            continue
        ok = bool(result.ok)
        tag = (
            '<span class="tag-ok">已完成</span>'
            if ok
            else '<span class="tag-fail">未完成</span>'
        )
        title = (
            "<h3>第 %d 段　<span class='arrow'>%s → %s</span>　%s　"
            "<span class='attempts'>（尝试 %d 次）</span></h3>"
            % (
                _display_index(index),
                _fmt_score(result.score_before),
                _fmt_score(result.score_after),
                tag,
                _as_int(result.attempts),
            )
        )
        revised = result.revised or "（无改写结果）"
        block = [
            '<div class="rewrite-item">',
            title,
            '<div class="compare">',
            "<div class='col'><div class='col-title'>原文</div>"
            "<div class='plain'>%s</div></div>" % _esc(result.original),
            "<div class='col'><div class='col-title'>改写后</div>"
            "<div class='plain'>%s</div></div>" % _esc(revised),
            "</div>",
        ]
        notes = [str(note) for note in (result.notes or []) if note]
        if result.error:
            block.append("<div class='note'>失败原因：%s</div>" % _esc(result.error))
        if notes:
            block.append("<div class='note'>备注：%s</div>" % _esc("；".join(notes)))
        block.append("</div>")
        blocks.append("\n".join(block))
    if not blocks:
        return ""
    return (
        "<h2>改写前后对照</h2>"
        "<p class='meta'>分数变化只在本工具同一版本内可比；分数下降表示文体特征更接近"
        "本工具判定的自然写作，不等于通过任何官方检测。</p>" + "\n".join(blocks)
    )


def _as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _footer_html(generated: str) -> str:
    return (
        "<footer>PaperAIGC v%s　·　生成时间：%s<br>"
        "本报告由本地启发式模型生成，未接入知网、未用知网结果校准；"
        "分数不是知网 AIGC 率。</footer>" % (_esc(VERSION), _esc(generated))
    )


def render_html(
    report: ScoreReport,
    out_path: str,
    rewrite: Optional[Dict[int, RewriteResult]] = None,
) -> str:
    """渲染单文件 HTML 报告并写入 out_path，返回 out_path。"""
    target = os.fspath(out_path)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="zh-CN">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>%s</title>" % _esc("论文 AI 痕迹自查报告 - " + (report.document.title or "未命名文档")),
        "<style>" + _CSS + "</style>",
        "</head>",
        "<body>",
        '<div class="wrap">',
        _header_html(report, generated),
        _disclaimer_html(report),
        _overview_html(report),
        _meta_html(report),
        _profile_html(report),
        _optimize_html(report),
        _similarity_html(report),
        _lint_html(report),
        _ledger_html(report),
        _paragraph_table_html(report),
        _rewrite_html(rewrite),
        _footer_html(generated),
        "</div>",
        "</body>",
        "</html>",
    ]
    html_text = "\n".join(parts) + "\n"

    parent = os.path.dirname(os.path.abspath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(html_text)
    return target


def _md_cell(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("|", "\\|")
    text = text.strip()
    return text or "—"


def _md_signals(signals: Sequence[Tuple[str, str, float]]) -> str:
    parts: List[str] = []
    for signal in signals or []:
        try:
            name = signal[1]
            contribution = signal[2]
        except (TypeError, IndexError, KeyError):
            continue
        parts.append("%s（%s）" % (_md_cell(name), _md_cell(_fmt_score(contribution))))
    return "；".join(parts) if parts else "—"


def _md_blockquote(text: str) -> List[str]:
    lines = str(text or "").splitlines() or [""]
    return ["> %s" % line for line in lines]


def _profile_md(report: ScoreReport) -> List[str]:
    profile = _profile(report)
    if not profile:
        return []
    lines: List[str] = ["## 全文画像（先看这里）", ""]
    lines.append("> " + str(profile.get("note") or ""))
    lines.append("")
    signals = profile.get("global_signals") or []
    if signals:
        lines.append("| 共性特征 | 触顶段落占比 | 平均贡献分 |")
        lines.append("| --- | --- | --- |")
        for item in signals[:5]:
            if not isinstance(item, dict):
                continue
            try:
                ratio = int(round(float(item.get("elevated_ratio", 0)) * 100))
                contribution = "%.1f" % float(item.get("mean_contribution", 0))
            except (TypeError, ValueError):
                continue
            lines.append("| %s | %d%% | %s |" % (_md_cell(item.get("name")), ratio, _md_cell(contribution)))
        lines.append("")
    return lines


def _similarity_md(report: ScoreReport) -> List[str]:
    info = _similarity_info(report)
    if not info:
        return []
    lines = ["## 本地库查重", "",
             "总文字复制比：**%.2f%%**（比对库 %s 个文件 / %s 字，判据 连续 %s 字）"
             % (float(info.get("overall_percent") or 0), info.get("corpus_files"),
                info.get("corpus_chars"), info.get("ngram")),
             "", "> " + str(info.get("note") or ""), ""]
    rows = info.get("paragraphs") or []
    if rows:
        lines.append("| 段落 | 重复率 | 相似来源 |")
        lines.append("| --- | --- | --- |")
        for item in rows[:12]:
            if not isinstance(item, dict):
                continue
            sources = "；".join(str(src.get("source")) for src in (item.get("sources") or [])[:2]
                               if isinstance(src, dict)) or "—"
            lines.append("| %s | %.1f%% | %s |" % (_md_cell(item.get("display")),
                                                  float(item.get("ratio_percent") or 0),
                                                  _md_cell(sources)))
        lines.append("")
    duplicates = info.get("internal_duplicates") or []
    if duplicates:
        lines.append("自我重复：" + "；".join(
            "第 %s 段与第 %s 段重合 %.0f%%" % (d.get("a"), d.get("b"),
                                            float(d.get("overlap_percent") or 0))
            for d in duplicates[:5] if isinstance(d, dict)))
        lines.append("")
    return lines


def _optimize_md(report: ScoreReport) -> List[str]:
    info = _optimize_info(report)
    if not info:
        return []
    try:
        before = float(info.get("overall_before", 0))
        after = float(info.get("overall_after", 0))
        target = float(info.get("target", 0))
        rewritten = int(info.get("paragraphs_rewritten", 0))
        total = int(info.get("paragraphs_total", 0))
    except (TypeError, ValueError):
        return []
    status = "已达标" if after < target else "未达标"
    return [
        "## 全文优化结果",
        "",
        "整篇 AI 痕迹指数：%.1f → %.1f（目标 < %.0f，%s）；改写 %d / %d 段；段落数与结构未变。"
        % (before, after, target, status, rewritten, total),
        "",
    ]


def _overview_md(report: ScoreReport) -> List[str]:
    rows = _overview_rows(report)
    if not rows:
        return []
    lines = ["## 一页纸总览", "", "| 项目 | 数值 | 说明 |", "| --- | --- | --- |"]
    for label, value, sub in rows:
        lines.append("| %s | %s | %s |" % (_md_cell(label), _md_cell(value), _md_cell(sub)))
    lines.append("")
    return lines


def _lint_md(report: ScoreReport) -> List[str]:
    info = (report.meta or {}).get("lint")
    if not isinstance(info, dict) or not info:
        return []
    counts = info.get("counts") or {}
    issues = info.get("issues") or []
    lines = ["## 论文规范自检", "",
             "错误 %d 个 / 警告 %d 个 / 提示 %d 个" % (
                 int(counts.get("error", 0)), int(counts.get("warn", 0)),
                 int(counts.get("info", 0))), ""]
    if issues:
        lines.append("| 级别 | 位置 | 问题 | 原文片段 |")
        lines.append("| --- | --- | --- | --- |")
        for item in issues[:20]:
            if not isinstance(item, dict):
                continue
            lines.append("| %s | %s | %s | %s |" % (
                _md_cell(item.get("level")),
                _md_cell("第 %s 段" % item.get("display") if item.get("display") else "-"),
                _md_cell(item.get("message")), _md_cell(item.get("snippet"))))
        lines.append("")
    return lines


def _ledger_md(report: ScoreReport) -> List[str]:
    info = (report.meta or {}).get("ledger")
    if not isinstance(info, dict) or not info:
        return []
    rows = info.get("rows") or []
    if not rows:
        return []
    lines = ["## 多系统分数台账", "",
             "| 时间 | 版本 | 系统 | 指标 | 数值 |", "| --- | --- | --- | --- | --- |"]
    for item in rows[-20:]:
        if not isinstance(item, dict):
            continue
        lines.append("| %s | %s | %s | %s | %.2f%% |" % (
            _md_cell(item.get("time")), _md_cell(item.get("version")),
            _md_cell(item.get("system")), _md_cell(item.get("metric")),
            float(item.get("value") or 0)))
    lines.append("")
    return lines


def render_markdown(
    report: ScoreReport,
    rewrite: Optional[Dict[int, RewriteResult]] = None,
) -> str:
    """渲染 Markdown 报告字符串（供命令行输出）。"""
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    disclaimer = report.disclaimer or DISCLAIMER_ZH
    counts = _level_counts(report)
    top = _top_paragraph(report)
    meta = report.meta or {}
    engine = str(meta.get("engine") or "stylometric-heuristic")
    calibration = report.calibration or "uncalibrated"

    lines: List[str] = []
    lines.append("# 论文 AI 痕迹自查报告")
    lines.append("")
    lines.append("- 文件：%s" % _md_cell(report.document.file_name))
    lines.append("- 标题：%s" % _md_cell(report.document.title or "未命名文档"))
    lines.append("- 工具版本：%s" % VERSION)
    lines.append("- 评分引擎：%s" % _md_cell(engine))
    if str(calibration).startswith("public-data"):
        lines.append("- 校准状态：公开数据校准（%s）" % _md_cell(calibration.split(":", 1)[-1]))
    else:
        lines.append("- 校准状态：未校准（%s）" % _md_cell(calibration))
    lines.append("- 生成时间：%s" % generated)
    lines.append("")
    lines.append("> **重要说明（必读）：这不是知网 AIGC 率**")
    lines.append(">")
    for line in str(disclaimer).splitlines() or [""]:
        lines.append("> %s" % line)
    lines.append(">")
    lines.append("> 可成立的结论只有两类：同一版本内的改写前后对比、段落之间相对排序；")
    lines.append("> 绝对数值不可与知网或其他商业检测结果比较，也不能作为学术不端判定的依据。")
    lines.append("")
    lines.append("## 摘要")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append("| AI 痕迹指数 | %s |" % _fmt_score(report.overall))
    lines.append("| 等级 | %s |" % _md_cell(report.level or "低"))
    lines.append("| 参与评分段数 | %d |" % len(report.paragraphs or []))
    lines.append("| 总字数 | %d |" % _total_words(report))
    lines.append("| 高风险段数 | %d |" % counts.get("高", 0))
    if top is None:
        lines.append("| 最高风险段落 | — |")
    else:
        lines.append(
            "| 最高风险段落 | 第 %d 段 · %s 分 |"
            % (_display_index(top.index), _fmt_score(top.score))
        )
    distribution = " / ".join("%s %d" % (level, counts.get(level, 0)) for level in LEVELS)
    lines.append("| 等级分布 | %s |" % _md_cell(distribution))
    lines.append("")
    lines.extend(_overview_md(report))
    lines.extend(_profile_md(report))
    lines.extend(_optimize_md(report))
    lines.extend(_similarity_md(report))
    lines.extend(_lint_md(report))
    lines.extend(_ledger_md(report))
    lines.append("## 段落风险排行")
    lines.append("")
    ranked = report.sorted_by_risk()
    if not ranked:
        lines.append("（没有参与评分的段落。）")
    else:
        lines.append("| 序号 | 分数 | 等级 | Top 信号 | 一句话建议 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for rank, item in enumerate(ranked, 1):
            lines.append(
                "| %d | %s | %s | %s | %s |"
                % (
                    rank,
                    _fmt_score(item.score),
                    _md_cell(item.level or "低"),
                    _md_signals(item.top_signals),
                    _md_cell(item.suggestion),
                )
            )
    lines.append("")
    if ranked:
        lines.append("## 段落原文（按风险降序）")
        lines.append("")
        for item in ranked:
            lines.append("### 第 %d 段（%s 分 · %s）" % (
                _display_index(item.index),
                _fmt_score(item.score),
                _md_cell(item.level or "低"),
            ))
            lines.append("")
            lines.extend(_md_blockquote(item.text))
            lines.append("")
    if rewrite:
        lines.append("## 改写前后对照")
        lines.append("")
        for index in sorted(rewrite):
            result = rewrite[index]
            if result is None:
                continue
            status = "已完成" if result.ok else "未完成"
            lines.append(
                "### 第 %d 段（%s → %s，尝试 %d 次，%s）"
                % (
                    _display_index(index),
                    _fmt_score(result.score_before),
                    _fmt_score(result.score_after),
                    _as_int(result.attempts),
                    status,
                )
            )
            lines.append("")
            lines.append("**原文**")
            lines.append("")
            lines.extend(_md_blockquote(result.original))
            lines.append("")
            lines.append("**改写后**")
            lines.append("")
            lines.extend(_md_blockquote(result.revised or "（无改写结果）"))
            lines.append("")
            if result.error:
                lines.append("失败原因：%s" % _md_cell(result.error))
                lines.append("")
            if result.notes:
                lines.append("备注：%s" % _md_cell("；".join(str(note) for note in result.notes)))
                lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "PaperAIGC v%s　·　生成时间：%s　·　本地启发式模型，未接入知网、未用知网结果校准；"
        "分数不是知网 AIGC 率。" % (VERSION, generated)
    )
    return "\n".join(lines) + "\n"

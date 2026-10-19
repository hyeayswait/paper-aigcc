# -*- coding: utf-8 -*-
"""降重专项：把与本地比对库连续文字重合率偏高的正文段改写到目标重复率以下。

与 pipeline / optimize 的分工：
- pipeline 压「AI 痕迹指数」；optimize 全文迭代压指数；
- dedup 只处理「与本地比对库的 13-gram 重合率 > target_ratio」的正文段，
  做同义改写 + 语序调整，改写后重新计算重复率，取最低且通过事实校验的版本
  （最多 3 次尝试）。

边界：只与 corpus_dir 里的本地文档比对，不联网，结果不能等同于知网/PaperPass。
任何异常都写进 RewriteResult.error，绝不向外抛；cancel 返回 True 时尽快停。
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Tuple

from ..config import AppConfig
from ..core.types import Document, Paragraph, RewriteResult
from ..io.similarity import check_similarity, load_corpus
from .optimize import _clean_output, validate_rewrite

ProgressFn = Optional[Callable[[str], None]]
CancelFn = Optional[Callable[[], bool]]

DEDUP_NGRAM = 13
MAX_ATTEMPTS = 3

# 与 io.similarity 的 13-gram 口径保持一致：只保留汉字/字母/数字并小写。
_KEEP_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

DEDUP_SYSTEM = (
    "你是一位中文论文降重编辑。你的任务是在不改变任何事实的前提下，对段落做同义改写"
    "和语序调整，降低它与已有文献之间的连续文字重合。你不添加原文没有的任何信息，"
    "不评价内容，不解释你在做什么。你只输出改写后的正文本身。"
)

_DEDUP_RULES = """降重要求（按优先级）：
1. 事实零改动：原文中出现的所有数字、百分比、金额、年份、指标名、公司名、引用标记 [n]、表号图号（表4/图2）必须原样保留，一个都不能少、不能改，也不能新增。
2. 同义改写：把与原文连续相同的表述换成同义词或近义句式，打散越长越好的连续相同片段。
3. 语序调整：调整句子内部成分顺序和句子之间的先后顺序，可以拆句、合句、改变主被动，但不得改变逻辑关系与结论。
4. 不得新增事实：不得补充原文没有的数据、案例、观点、结论或引用。
5. 只输出正文：不要标题、不要编号、不要解释、不要 markdown 代码块，不要输出任何额外说明。"""


# ---------------------------------------------------------------------- 小工具
def _report(progress: ProgressFn, message: str) -> None:
    if progress is None:
        return
    try:
        progress(message)
    except Exception:
        pass


def _cancelled(cancel: CancelFn) -> bool:
    if cancel is None:
        return False
    try:
        return bool(cancel())
    except Exception:
        return False


def _attempts(cfg) -> int:
    """最多尝试次数：SPEC 要求 3 次，cfg 只能调小不能超过 3。"""
    try:
        value = int(getattr(cfg, "max_rewrite_attempts", MAX_ATTEMPTS) or MAX_ATTEMPTS)
    except (TypeError, ValueError):
        value = MAX_ATTEMPTS
    return max(1, min(MAX_ATTEMPTS, value))


def _grams(text: str, ngram: int) -> List[str]:
    if ngram <= 0:
        return []
    seq = "".join(_KEEP_RE.findall(text or "")).lower()
    if len(seq) < ngram:
        return []
    return [seq[i:i + ngram] for i in range(len(seq) - ngram + 1)]


def _corpus_grams(corpus: List[Tuple[str, str]], ngram: int) -> set:
    bag = set()
    for _name, text in corpus:
        bag.update(_grams(text, ngram))
    return bag


def _dup_ratio(text: str, corpus_grams: set, ngram: int) -> float:
    """单段对本地库的重复率：命中 13-gram 数 / 该段 13-gram 总数（0-1）。"""
    grams = _grams(text, ngram)
    if not grams:
        return 0.0
    matched = sum(1 for gram in grams if gram in corpus_grams)
    return matched / float(len(grams))


def _sources_text(sources) -> str:
    if not sources:
        return "（无）"
    return "、".join(
        "%s（%.1f%%）" % (getattr(item, "source", ""), float(getattr(item, "ratio", 0.0)) * 100.0)
        for item in sources[:3]
    )


# ---------------------------------------------------------------------- prompt
def build_dedup_prompt(original: str, ratio: float, target_ratio: float,
                       sources=None) -> str:
    parts = [
        "下面这段论文正文与本地比对库的连续文字重合率偏高，请做同义改写与语序调整。",
        "",
        "当前重复率：%.1f%%，目标：低于 %.1f%%。" % (ratio * 100.0, target_ratio * 100.0),
        "重合来源：%s" % _sources_text(sources),
        "",
        _DEDUP_RULES,
        "",
        "原文：",
        original,
        "",
        "现在输出改写后的正文：",
    ]
    return "\n".join(parts)


def build_dedup_retry_prompt(original: str, previous: str, prev_percent: float,
                             target_ratio: float, problems: List[str]) -> str:
    parts = [
        "上一版改写没有通过事实校验，或者重复率还不够低。请重做。",
        "",
        "上一版（当前重复率 %.1f%%，目标低于 %.1f%%）：" % (
            prev_percent, target_ratio * 100.0),
        previous or "（空）",
        "",
    ]
    if problems:
        parts.extend(["必须修正的硬性问题：", "；".join(problems), ""])
    parts.extend([
        _DEDUP_RULES,
        "",
        "原文（事实基准，所有数字/引用/表号以此为准）：",
        original,
        "",
        "现在输出改写后的正文：",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------- 单段降重
def _dedup_one(paragraph: Paragraph, ratio: float, client, cfg,
               corpus_grams: set, ngram: int, target_ratio: float,
               progress: ProgressFn, cancel: CancelFn) -> RewriteResult:
    before_percent = round(ratio * 100.0, 2)
    result = RewriteResult(index=paragraph.index, original=paragraph.text,
                           score_before=before_percent, score_after=before_percent)
    attempts_max = _attempts(cfg)
    best_text = ""
    best_ratio = ratio
    best_percent = before_percent
    previous_text = ""
    problems: List[str] = []
    last_error = ""

    for attempt in range(1, attempts_max + 1):
        if _cancelled(cancel):
            result.error = result.error or "已取消"
            break
        if attempt == 1:
            prompt = build_dedup_prompt(paragraph.text, ratio, target_ratio)
        else:
            prompt = build_dedup_retry_prompt(paragraph.text, previous_text,
                                              best_percent, target_ratio, problems)
        try:
            raw = client.chat(
                [{"role": "system", "content": DEDUP_SYSTEM},
                 {"role": "user", "content": prompt}],
                temperature=getattr(cfg, "temperature", 1.0),
                max_tokens=getattr(cfg, "max_tokens", 4096),
            )
        except Exception as exc:  # noqa: BLE001 - 网络/API 错误一律收进结果
            result.attempts = attempt
            last_error = str(exc) or exc.__class__.__name__
            result.error = "调用改写服务失败：%s" % last_error
            break

        result.attempts = attempt
        candidate = _clean_output(raw)
        if not candidate:
            problems = ["模型返回空内容"]
            continue
        passed, problems = validate_rewrite(paragraph.text, candidate)
        if not passed:
            previous_text = candidate
            continue

        candidate_ratio = _dup_ratio(candidate, corpus_grams, ngram)
        candidate_percent = round(candidate_ratio * 100.0, 2)
        previous_text = candidate
        if not best_text or candidate_ratio < best_ratio:
            best_text = candidate
            best_ratio = candidate_ratio
            best_percent = candidate_percent
        _report(progress, "第 %d 段：第 %d 次降重，重复率 %.1f%% → %.1f%%" % (
            paragraph.index + 1, attempt, before_percent, candidate_percent))
        if best_ratio <= target_ratio:
            break

    if best_text:
        result.revised = best_text
        result.score_after = best_percent
        result.ok = True
        if last_error:
            result.notes.append("后续尝试失败，已保留此前最好的一版：%s" % last_error)
        if best_ratio >= ratio:
            result.notes.append("改写已通过事实校验，但重复率没有下降（%.1f%%）" % best_percent)
        if best_ratio > target_ratio:
            result.notes.append("已到次数上限，重复率仍高于目标 %.1f%%" % (
                target_ratio * 100.0))
    else:
        result.ok = False
        result.score_after = before_percent
        if not result.error:
            result.error = "；".join(problems) if problems else "没有得到通过事实校验的改写"
    return result


# ---------------------------------------------------------------------- 整篇降重
def dedup_document(doc: Document, client, cfg: AppConfig, corpus_dir: str,
                   target_ratio: float = 0.15, progress=None,
                   cancel=None) -> Dict[int, RewriteResult]:
    """对重复率 > target_ratio 的正文段做降重改写。

    返回 {段落 index: RewriteResult}，其中 score_before/score_after 为「重复率×100」。
    任何异常都写进 RewriteResult.error，本函数不向外抛。
    """
    results: Dict[int, RewriteResult] = {}
    try:
        try:
            target = float(target_ratio)
        except (TypeError, ValueError):
            target = 0.15
        if target < 0:
            target = 0.0

        report = check_similarity(doc, corpus_dir)
        candidates = [item for item in report.paragraphs if item.ratio > target]
        if not candidates:
            _report(progress, "没有重复率高于 %.1f%% 的正文段，无需降重。"
                    % (target * 100.0))
            return results
        candidates.sort(key=lambda item: (-item.ratio, item.index))

        try:
            corpus = load_corpus(corpus_dir)
        except Exception:
            corpus = []
        ngram = int(getattr(report, "ngram", DEDUP_NGRAM) or DEDUP_NGRAM)
        corpus_grams = _corpus_grams(corpus, ngram)
        paragraphs = {paragraph.index: paragraph for paragraph in doc.paragraphs}

        _report(progress, "共 %d 段重复率高于 %.1f%%，按重复率降序开始降重。"
                % (len(candidates), target * 100.0))

        for item in candidates:
            if _cancelled(cancel):
                _report(progress, "收到取消请求，已停止降重（已完成 %d 段）。"
                        % len(results))
                break
            paragraph = paragraphs.get(item.index)
            if paragraph is None:
                paragraph = Paragraph(index=item.index, text=item.text)
            try:
                outcome = _dedup_one(paragraph, item.ratio, client, cfg,
                                     corpus_grams, ngram, target, progress, cancel)
            except Exception as exc:  # noqa: BLE001 - 单段异常不影响其它段
                before = round(item.ratio * 100.0, 2)
                outcome = RewriteResult(index=item.index, original=paragraph.text,
                                        score_before=before, score_after=before,
                                        ok=False, error="降重失败：%s" % exc)
            results[item.index] = outcome
            _report(progress, "第 %d 段降重完成：重复率 %.1f%% → %.1f%%" % (
                item.index + 1, outcome.score_before, outcome.score_after))
    except Exception as exc:  # noqa: BLE001 - 兜底：绝不外抛
        _report(progress, "降重流程异常终止：%s" % exc)
    return results

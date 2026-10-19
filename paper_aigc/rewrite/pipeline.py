"""迭代改写流水线。

定位：自然化改写 / 去模板化，让段落更像作者本人写的；保留事实、数据、术语与引用。
流程：取高风险段落 -> 送 DeepSeek 改写 -> 用评分器重新打分 -> 未达标就带反馈重试，
最多 cfg.max_rewrite_attempts 次，达到 cfg.rewrite_target 即停。
所有异常都写进 RewriteResult.error，绝不向外抛。
"""
from __future__ import annotations

import difflib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from ..config import AppConfig
from ..core.types import Document, Paragraph, ParagraphScore, RewriteResult, ScoreReport
from .client import DeepSeekClient, RewriteError
from .prompts import SYSTEM_PROMPT, build_retry_prompt, build_rewrite_prompt

try:  # 评分模块由 T1 提供；缺失时由测试注入替身，这里不复制评分逻辑
    from ..core import scoring  # type: ignore
except Exception:  # pragma: no cover - 集成阶段 scoring 一定存在
    scoring = None  # type: ignore

# 去空白后相似度高于该值视为「没改动」
_CHANGE_SIMILARITY_LIMIT = 0.98
_QUOTE_PAIRS = {"“": "”", "\"": "\"", "‘": "’", "'": "'", "「": "」", "『": "』"}
_LABEL_RE = re.compile(r"^(?:改写|修改|润色|自然化)(?:后|结果|版本)?\s*[:：]\s*")


# ---------------------------------------------------------------------- 小工具
def _notify(progress, message: str) -> None:
    """安全地调用 progress(msg)：回调异常一律吞掉。"""
    if progress is None:
        return
    try:
        progress(message)
    except Exception:
        pass


def _is_cancelled(cancel) -> bool:
    if cancel is None:
        return False
    try:
        return bool(cancel())
    except Exception:
        return False


def _scoring_module():
    """拿到评分模块；模块尚未落地时延迟导入一次，失败则抛中文 RewriteError。"""
    global scoring
    if scoring is None:
        try:
            from ..core import scoring as module  # type: ignore
        except Exception as exc:
            raise RewriteError("评分模块不可用，无法验证改写效果：{0}".format(exc))
        scoring = module
    return scoring


def _score_text(text: str, index: int) -> float:
    module = _scoring_module()
    paragraph_score = module.score_paragraph(text, index)
    try:
        return float(getattr(paragraph_score, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _clean_output(raw: str) -> str:
    """去掉模型可能附带的代码块围栏、前缀标签和包裹引号。"""
    text = (raw or "").replace("\ufeff", "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    text = _LABEL_RE.sub("", text).strip()
    if len(text) >= 2:
        first, last = text[0], text[-1]
        if _QUOTE_PAIRS.get(first) == last:
            text = text[1:-1].strip()
    # 合并模型偶尔插入的多余空行，但保留段落内部的正常换行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _similarity(left: str, right: str) -> float:
    a = _normalize_for_compare(left)
    b = _normalize_for_compare(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


# ---------------------------------------------------------------------- 单段改写
def rewrite_paragraph(paragraph: Paragraph, score: ParagraphScore,
                      client: DeepSeekClient, cfg: AppConfig,
                      context: str = "", progress=None) -> RewriteResult:
    """改写一个段落，迭代到目标分数或用尽次数；异常写进 result，不抛出。"""
    original = (getattr(paragraph, "text", "") or "") if paragraph is not None else ""
    index = int(getattr(paragraph, "index", 0) or 0) if paragraph is not None else 0
    try:
        score_before = float(getattr(score, "score", 0.0) or 0.0) if score is not None else 0.0
    except (TypeError, ValueError):
        score_before = 0.0
    result = RewriteResult(
        index=index, original=original, score_before=score_before, score_after=score_before
    )

    if not original.strip():
        result.error = "段落文本为空，无需改写"
        _notify(progress, "第 {0} 段：文本为空，跳过。".format(index))
        return result

    signals: List[Tuple[str, str, float]] = list(
        getattr(score, "top_signals", None) or []
    )
    try:
        target = float(getattr(cfg, "rewrite_target", 35.0) or 0.0)
    except (TypeError, ValueError):
        target = 35.0
    try:
        attempts_max = max(1, int(getattr(cfg, "max_rewrite_attempts", 1) or 1))
    except (TypeError, ValueError):
        attempts_max = 1
    temperature = getattr(cfg, "temperature", None)
    max_tokens = getattr(cfg, "max_tokens", None)

    best_revised = ""
    best_score = score_before
    previous = ""
    previous_score = score_before
    last_error = ""

    for attempt in range(1, attempts_max + 1):
        result.attempts = attempt
        _notify(
            progress,
            "第 {0} 段：第 {1}/{2} 次自然化改写…".format(index, attempt, attempts_max),
        )
        try:
            if attempt == 1 or not previous:
                prompt = build_rewrite_prompt(original, signals, context=context)
            else:
                prompt = build_retry_prompt(original, previous, previous_score, target, signals)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            raw = client.chat(messages, temperature=temperature, max_tokens=max_tokens)
            revised = _clean_output(raw)
            if not revised:
                raise RewriteError("模型返回了空内容，请稍后重试。")
            new_score = _score_text(revised, index)
            if not best_revised or new_score < best_score:
                best_revised = revised
                best_score = new_score
            previous = revised
            previous_score = new_score
            if new_score <= target:
                break
            if attempt < attempts_max:
                result.notes.append(
                    "第 {0} 次改写后 AI 痕迹指数 {1:.1f}，仍高于目标 {2:.1f}，继续改写。".format(
                        attempt, new_score, target
                    )
                )
        except RewriteError as exc:
            last_error = str(exc)
            result.notes.append("第 {0} 次改写失败：{1}".format(attempt, last_error))
            break
        except Exception as exc:
            last_error = "调用改写服务时发生异常：{0}".format(exc)
            result.notes.append("第 {0} 次改写失败：{1}".format(attempt, last_error))
            break

    result.revised = best_revised
    result.score_after = best_score

    if not best_revised:
        result.ok = False
        result.error = last_error or "改写失败，没有拿到有效结果"
        _notify(progress, "第 {0} 段改写失败：{1}".format(index, result.error))
        return result

    if _similarity(original, best_revised) > _CHANGE_SIMILARITY_LIMIT:
        result.ok = False
        result.error = "改写无变化"
        _notify(progress, "第 {0} 段改写无变化，保留原文。".format(index))
        return result

    result.ok = True
    if last_error:
        result.notes.append("后续尝试失败，已保留此前最好的一版：{0}".format(last_error))
    if best_score > target:
        result.notes.append(
            "已到改写次数上限，最好成绩 {0:.1f} 仍高于目标 {1:.1f}。".format(best_score, target)
        )
    _notify(
        progress,
        "第 {0} 段完成：AI 痕迹指数 {1:.1f} → {2:.1f}".format(index, score_before, best_score),
    )
    return result


# ---------------------------------------------------------------------- 整篇改写
def _select_targets(doc, report, cfg) -> List[Tuple[Paragraph, ParagraphScore]]:
    """按高风险阈值与最短字数挑出待改写段落，按风险从高到低排序。"""
    try:
        threshold = float(getattr(cfg, "high_threshold", 60.0) or 0.0)
    except (TypeError, ValueError):
        threshold = 60.0
    try:
        min_chars = int(getattr(cfg, "min_rewrite_chars", 0) or 0)
    except (TypeError, ValueError):
        min_chars = 0
    paragraphs = {}
    for paragraph in getattr(doc, "paragraphs", None) or []:
        paragraphs[paragraph.index] = paragraph
    picked: List[Tuple[Paragraph, ParagraphScore]] = []
    for paragraph_score in getattr(report, "paragraphs", None) or []:
        try:
            value = float(getattr(paragraph_score, "score", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if value < threshold:
            continue
        paragraph = paragraphs.get(paragraph_score.index)
        if paragraph is None:
            continue
        if getattr(paragraph, "is_heading", False) or getattr(paragraph, "is_reference", False):
            continue
        if int(getattr(paragraph, "char_count", 0) or 0) < min_chars:
            continue
        picked.append((paragraph, paragraph_score))
    picked.sort(key=lambda item: (-float(getattr(item[1], "score", 0.0) or 0.0), item[0].index))
    return picked


def rewrite_document(doc: Document, report: ScoreReport, client: DeepSeekClient,
                     cfg: AppConfig, progress=None, cancel=None) -> Dict[int, RewriteResult]:
    """并发改写整篇文档中达标的段落，返回 {段落 index: RewriteResult}。"""
    results: Dict[int, RewriteResult] = {}
    try:
        targets = _select_targets(doc, report, cfg)
    except Exception as exc:
        _notify(progress, "挑选待改写段落失败：{0}".format(exc))
        return results

    if not targets:
        _notify(progress, "没有达到改写阈值的段落，本次不需要改写。")
        return results

    try:
        threshold_text = "{0:g}".format(float(getattr(cfg, "high_threshold", 60.0) or 0.0))
    except (TypeError, ValueError):
        threshold_text = "60"
    _notify(
        progress,
        "共有 {0} 段需要自然化改写（高风险阈值 {1} 分）。".format(len(targets), threshold_text),
    )

    try:
        workers = max(1, int(getattr(cfg, "concurrency", 1) or 1))
    except (TypeError, ValueError):
        workers = 1
    workers = max(1, min(workers, len(targets)))
    context = (getattr(doc, "title", "") or "").strip()

    paragraph_by_index = {}
    for paragraph in getattr(doc, "paragraphs", None) or []:
        paragraph_by_index[paragraph.index] = paragraph
    score_by_index = {}
    for paragraph_score in getattr(report, "paragraphs", None) or []:
        score_by_index[paragraph_score.index] = paragraph_score

    def _process(target: Tuple[Paragraph, ParagraphScore]) -> Optional[RewriteResult]:
        if _is_cancelled(cancel):
            return None
        paragraph, paragraph_score = target
        return rewrite_paragraph(
            paragraph, paragraph_score, client, cfg, context=context, progress=progress
        )

    futures = {}
    total = len(targets)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for target in targets:
            if _is_cancelled(cancel):
                break
            futures[pool.submit(_process, target)] = target[0].index
        for future in as_completed(futures):
            index = futures[future]
            if _is_cancelled(cancel):
                for pending in futures:
                    pending.cancel()
                break
            try:
                result = future.result()
            except Exception as exc:
                paragraph = paragraph_by_index.get(index)
                original = getattr(paragraph, "text", "") if paragraph is not None else ""
                paragraph_score = score_by_index.get(index)
                try:
                    before = float(getattr(paragraph_score, "score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    before = 0.0
                result = RewriteResult(
                    index=index,
                    original=original,
                    score_before=before,
                    score_after=before,
                    ok=False,
                    error="改写失败：{0}".format(exc),
                )
            if result is None:
                continue
            results[index] = result
            done += 1
            _notify(
                progress,
                "进度 {0}/{1}：第 {2} 段 AI 痕迹指数 {3:.1f} → {4:.1f}".format(
                    done, total, index, result.score_before, result.score_after
                ),
            )

    cancelled = _is_cancelled(cancel)
    if cancelled:
        # 取消前已经跑完的段落也收进结果，避免丢失
        for future, index in futures.items():
            if index in results:
                continue
            try:
                if future.done() and not future.cancelled():
                    result = future.result()
                    if result is not None:
                        results[index] = result
            except Exception:
                continue
        _notify(progress, "收到取消请求，已停止改写（已完成 {0} 段）。".format(len(results)))

    return results

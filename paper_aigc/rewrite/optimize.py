# -*- coding: utf-8 -*-
"""全文优化：把整篇论文的 AI 痕迹指数压到目标值以下，同时守住结构与事实。

与 pipeline.py 的分工：
- pipeline 只挑「超过阈值」的高风险段，做一轮改写；
- optimize 处理「所有仍在目标之上的正文段落」，可多轮全局迭代，直到整篇
  指数低于 target 或预算/次数耗尽。

三道事实护栏（全部在本地校验，失败就重试）：
1. 原段里的每个数字/百分比都必须在改写后出现，且不得凭空新增数字；
2. 原段里的引用标记 [n]、表号/图号（表4、图2）必须保留；
3. 改写后字数不得塌缩（默认不低于原文汉字数的 55%）或暴涨（不超过 180%）。

结构护栏：只替换正文段落内部的文字，不增删段落、不动标题/表格/参考文献。
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, List as ListType, Optional, Tuple

from ..config import AppConfig
from ..core import scoring
from ..core.textnorm import count_words, normalize_text
from ..core.types import Document, Paragraph, ParagraphScore, RewriteResult, ScoreReport

ProgressFn = Optional[Callable[[str], None]]
CancelFn = Optional[Callable[[], bool]]

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_ASCII_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_ASCII_ALLOW = {"roc", "roa", "roe", "esg", "aigc", "ai", "bt", "it", "app", "cfo", "ceo", "io", "tv", "pc", "ocr", "nlp", "gpt", "llm", "url", "vip", "ceo"}
_CITATION_RE = re.compile(r"\[\s*\d{1,3}\s*\]")
_REF_RE = re.compile(r"(?:表|图)\s*\d{1,2}")

_QUOTE_WRAP_RE = re.compile(r"^[\s\"'“”「」『』]+|[\s\"'“”「」『』]+$")
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
_PREFIX_RE = re.compile(r"^\s*(改写后|改写|优化后|润色后|修改后)\s*[:：]\s*")

OPTIMIZE_SYSTEM = (
    "你是一位中文论文润色编辑。你的任务是把段落改写成作者本人的自然笔法，"
    "但必须逐字保留原文的事实、数据、术语、专有名词、引用标记与表图编号。"
    "你不添加原文没有的任何信息，不评价内容，不解释你在做什么。"
    "你只输出改写后的正文本身。"
)

_STYLE_RULES = """写作要求（按优先级）：
1. 事实零改动：数字、百分比、金额、年份、指标名、公司名、引用标记 [n]、表号图号（表4/图2）全部原样保留，一个都不能少、不能改、不能新增。
2. 节奏要自然：平均每句 22-32 个汉字。每段必须有明显的长短起伏——至少 1 句超过 40 字，至少 2 句少于 15 字。
3. 不要通篇碎句：连续三句都短于 15 字会显得像机器断句。长句里可以用插入语、括号、破折号展开。
4. 用数据说话：原文的数字、百分比要落在句子主干里，分散在各句，不要堆在同一句末尾。
5. 段首不要雷同：禁止用「……是什么？就是……」「……怎么破？」这类设问开头；各段开头句式必须彼此不同。
6. 作者视角要克制：全篇「笔者认为/本文认为/我判断」合计最多 3 次，不要每段都出现。
7. 去掉模板腔：不要用「首先/其次/再次/此外/另外/综上所述/总而言之/值得注意的是/从…角度来看/随着…的发展/本文旨在/具有重要意义」这类过渡与套话。
8. 不要排比：避免「不仅…而且」「一方面…另一方面」「既…又」的成串结构。
9. 标点有变化：允许分号、破折号、括号、问句自然出现，但不要每句都用。
10. 只输出正文：不要标题、不要编号、不要解释、不要 markdown 代码块，不要混入任何英文单词。"""


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


class _Budget:
    """全局调用次数预算，防止优化失控烧钱。"""

    def __init__(self, limit: int) -> None:
        self.limit = max(1, int(limit))
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self.used >= self.limit:
                return False
            self.used += 1
            return True


def _stray_ascii(original: str, revised: str) -> List[str]:
    """找出改写里出现、但原文没有的英文单词（防止模型混入 also 这类残句）。"""
    allowed = set(_ASCII_ALLOW)
    for word in _ASCII_WORD_RE.findall(original or ""):
        allowed.add(word.lower())
    stray: List[str] = []
    for word in _ASCII_WORD_RE.findall(revised or ""):
        low = word.lower()
        if low in allowed:
            continue
        if word not in stray:
            stray.append(word)
    return stray


_END_PUNCT = "。！？!?；;"


def _sentence_len(sentence: str) -> int:
    return len(sentence.rstrip(_END_PUNCT))


def rebalance_rhythm(text: str, long_target: int = 1, keep_short: int = 2,
                     max_merge: int = 8) -> str:
    """把「全短句」的段落调成长短混合：合并相邻短句成长句（只改标点，不动字词）。

    规则：
    - 制造至少 long_target 个 38-80 字的长句（用「；」连接）
    - 保留至少 keep_short 个 15 字以内的短句
    - 长句占比超过 1/3 时，把最长的句子在逗号处拆开
    """
    from ..core.textnorm import split_sentences

    sentences = split_sentences(text)
    current = [s.strip() for s in sentences if s.strip()]
    if len(current) < 4:
        return text

    def shorts(items) -> int:
        return sum(1 for item in items if _sentence_len(item) < 15)

    def longs(items) -> int:
        return sum(1 for item in items if _sentence_len(item) > 38)

    # 1) 合并出长句
    for _ in range(10):
        if longs(current) >= long_target:
            break
        chosen = None
        for i in range(len(current) - 1):
            total = _sentence_len(current[i])
            for k in range(2, max_merge + 1):
                if i + k > len(current):
                    break
                if current[i + k - 1].rstrip()[-1:] in "？！!?":
                    break
                total += _sentence_len(current[i + k - 1])
                if total > 80:
                    break
                if total < 38:
                    continue
                lost_short = sum(1 for item in current[i:i + k] if _sentence_len(item) < 15)
                if shorts(current) - lost_short < keep_short:
                    continue
                chosen = (i, k, total)
                break
            if chosen:
                break
        if chosen is None:
            break
        index, count, _total = chosen
        parts = [current[index].strip().rstrip(_END_PUNCT)]
        for offset in range(1, count):
            piece = current[index + offset].strip()
            if offset < count - 1:
                parts.append(piece.rstrip(_END_PUNCT))
            else:
                parts.append(piece)
        current[index:index + count] = ["，".join(parts)]

    # 2) 长句过多时拆开最长的句子
    guard = 0
    while len(current) >= 4 and longs(current) > max(1, len(current) // 3) and guard < 6:
        guard += 1
        target = max(current, key=_sentence_len)
        body = target.rstrip(_END_PUNCT)
        cut = -1
        for pos in range(len(body) // 2, 14, -1):
            if body[pos] in "，,；;":
                cut = pos
                break
        if cut < 0:
            break
        head, tail = body[:cut], body[cut + 1:]
        if _sentence_len(head) < 12 or _sentence_len(tail) < 12:
            break
        current[current.index(target)] = head + "。" + tail + "。"

    return "".join(current)


def document_quality(texts) -> Dict[str, float]:
    """改写后全文的节奏体检：句长分布、作者视角次数、段首重复度。"""
    from ..core.textnorm import split_sentences

    lengths: List[int] = []
    thoughts = 0
    openers: List[str] = []
    for text in texts:
        thoughts += text.count("笔者认为") + text.count("本文认为") + text.count("我判断")
        opened = False
        for sentence in split_sentences(text):
            size = len(sentence)
            if size:
                lengths.append(size)
                if not opened and size >= 6:
                    openers.append(sentence[:6])
                    opened = True
    if not lengths:
        return {"sentences": 0, "avg_len": 0.0, "short_ratio": 0.0, "long_ratio": 0.0,
                "author_voice": 0, "repeated_openers": 0}
    ordered = sorted(lengths)
    mid = ordered[len(ordered) // 2]
    return {
        "sentences": len(lengths),
        "avg_len": round(sum(lengths) / float(len(lengths)), 1),
        "median_len": float(mid),
        "short_ratio": round(sum(1 for x in lengths if x < 15) / float(len(lengths)), 2),
        "long_ratio": round(sum(1 for x in lengths if x > 40) / float(len(lengths)), 2),
        "author_voice": thoughts,
        "repeated_openers": len(openers) - len(set(openers)),
    }


def _numbers(text: str) -> List[float]:
    values: List[float] = []
    for token in _NUM_RE.findall(text or ""):
        try:
            values.append(round(float(token), 6))
        except ValueError:
            continue
    return values


def validate_rewrite(original: str, revised: str) -> Tuple[bool, List[str]]:
    """校验改写是否守住事实。返回 (是否通过, 问题列表)。"""
    problems: List[str] = []
    text = normalize_text(revised)
    if not text:
        return False, ["改写结果为空"]

    original = original or ""
    stray = _stray_ascii(original, text)
    if stray:
        problems.append("混入原文没有的英文：%s" % "、".join(stray[:5]))
    orig_numbers = _numbers(original)
    new_numbers = _numbers(text)
    missing = [value for value in orig_numbers if value not in new_numbers]
    if missing:
        problems.append("漏掉数字：%s" % "、".join(_format_number(v) for v in _unique(missing)))
    invented = [value for value in new_numbers if value not in orig_numbers]
    if invented:
        problems.append("凭空新增数字：%s" % "、".join(_format_number(v) for v in _unique(invented)))

    orig_citations = _unique(_CITATION_RE.findall(original))
    new_citations = set(_CITATION_RE.findall(text))
    lost_citations = [item for item in orig_citations if item not in new_citations]
    if lost_citations:
        problems.append("漏掉引用标记：%s" % "、".join(lost_citations))
    extra_citations = [item for item in new_citations if item not in set(orig_citations)]
    if extra_citations:
        problems.append("新增了原文没有的引用：%s" % "、".join(sorted(extra_citations)))

    orig_refs = _unique(_REF_RE.findall(original))
    new_refs = set(_REF_RE.findall(text))
    lost_refs = [item for item in orig_refs if item not in new_refs]
    if lost_refs:
        problems.append("漏掉表号/图号：%s" % "、".join(lost_refs))

    # 用「汉字+数字+英文词」的合计数做长度护栏：原文数字多时只数汉字会误判塌缩
    src_len = count_words(original)
    new_len = count_words(text)
    if src_len >= 40:
        ratio = new_len / float(src_len)
        if ratio < 0.55:
            problems.append("改写后字数塌缩到原文的 %.0f%%" % (ratio * 100))
        elif ratio > 1.8:
            problems.append("改写后字数膨胀到原文的 %.0f%%" % (ratio * 100))

    return (not problems), problems


def _unique(values: List[float]) -> List[float]:
    seen: List[float] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return ("%.6f" % value).rstrip("0").rstrip(".")


def _clean_output(raw: str) -> str:
    text = (raw or "").strip()
    text = _FENCE_RE.sub("", text)
    text = _PREFIX_RE.sub("", text)
    text = _QUOTE_WRAP_RE.sub("", text)
    text = text.replace("\r", "").strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "".join(lines) if lines else ""


def _signals_text(signals) -> str:
    if not signals:
        return "（无）"
    return "、".join("%s（%.1f 分）" % (name, contribution) for _key, name, contribution in signals[:4])


def build_optimize_prompt(original: str, signals, target: float, context: str = "") -> str:
    parts = [
        "把下面这段论文正文改写成更自然的作者笔法。",
        "",
        "本段被判定 AI 痕迹偏高的原因：" + _signals_text(signals),
        "目标：把本地 AI 痕迹指数压到 %.0f 分以下。" % target,
        "",
        _STYLE_RULES,
        "",
    ]
    if context:
        parts.extend(["上下文（仅供理解术语，不要改写这段）：", context[:400], ""])
    parts.extend(["原文：", original, "", "现在输出改写后的正文："])
    return "\n".join(parts)


def build_optimize_retry_prompt(original: str, previous: str, prev_score: float,
                                target: float, signals, problems: List[str]) -> str:
    parts = [
        "上一次改写没有通过，或者分数仍然偏高。请重做。",
        "",
        "上一次改写（分数 %.1f，目标 %.0f）：" % (prev_score, target),
        previous or "（空）",
        "",
    ]
    if problems:
        parts.extend(["必须修正的硬性问题：", "；".join(problems), ""])
    parts.extend([
        "本段仍然偏高的原因：" + _signals_text(signals),
        "",
        _STYLE_RULES,
        "",
        "原文（事实基准，所有数字/引用/表号以此为准）：",
        original,
        "",
        "现在输出改写后的正文：",
    ])
    return "\n".join(parts)


def _optimize_one(paragraph: Paragraph, score: ParagraphScore, client, cfg: AppConfig,
                  target: float, budget: _Budget, context: str,
                  progress: ProgressFn, cancel: CancelFn) -> RewriteResult:
    result = RewriteResult(index=paragraph.index, original=paragraph.text,
                           score_before=score.score, score_after=score.score)
    max_attempts = max(1, int(getattr(cfg, "max_rewrite_attempts", 3)))
    best_text = ""
    best_score = score.score
    previous_text = ""
    previous_score = score.score
    problems: List[str] = []

    for attempt in range(1, max_attempts + 1):
        if _cancelled(cancel):
            result.error = result.error or "已取消"
            break
        if not budget.take():
            result.error = result.error or "达到 API 调用预算上限"
            break
        if attempt == 1:
            prompt = build_optimize_prompt(paragraph.text, score.top_signals, target, context)
        else:
            prompt = build_optimize_retry_prompt(paragraph.text, previous_text, previous_score,
                                                 target, score.top_signals, problems)
        try:
            raw = client.chat(
                [{"role": "system", "content": OPTIMIZE_SYSTEM},
                 {"role": "user", "content": prompt}],
                temperature=getattr(cfg, "temperature", 1.15),
                max_tokens=getattr(cfg, "max_tokens", 4096),
            )
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            continue

        result.attempts = attempt
        candidate = _clean_output(raw)
        if not candidate:
            problems = ["模型返回空内容"]
            continue
        passed, problems = validate_rewrite(paragraph.text, candidate)
        if not passed:
            continue
        candidate_score = scoring.score_paragraph(candidate, paragraph.index).score
        previous_text, previous_score = candidate, candidate_score
        if candidate_score < best_score:
            best_text, best_score = candidate, candidate_score
        _report(progress, "第 %d 段：第 %d 次改写，%.1f → %.1f" % (
            paragraph.index + 1, attempt, score.score, candidate_score))
        if candidate_score <= target:
            break

    if best_text:
        result.revised = best_text
        result.score_after = best_score
        result.ok = True
        if best_score > target:
            result.notes.append("已到次数上限，未达目标 %.0f 分" % target)
    else:
        result.ok = False
        result.score_after = score.score
        if not result.error:
            result.error = "；".join(problems) if problems else "没有得到通过事实校验的改写"
    return result


def _better(new: RewriteResult, old: RewriteResult) -> bool:
    """多轮之间只保留更好的结果，避免后一轮把前一成功覆盖成失败。"""
    if new.ok and not old.ok:
        return True
    if old.ok and not new.ok:
        return False
    return new.score_after < old.score_after


def apply_replacements(doc: Document, replacements: Dict[int, str]) -> Document:
    """在不增删段落的前提下，返回替换了正文文字的新 Document。"""
    paragraphs: List[Paragraph] = []
    for paragraph in doc.paragraphs:
        text = replacements.get(paragraph.index)
        if text is None:
            paragraphs.append(paragraph)
            continue
        new_paragraph = Paragraph(
            index=paragraph.index,
            text=text,
            style=paragraph.style,
            is_heading=paragraph.is_heading,
            is_reference=paragraph.is_reference,
        )
        paragraphs.append(new_paragraph)
    total_words = sum(paragraphs[i].char_count for i in doc.body_indices if i < len(paragraphs))
    return Document(path=doc.path, title=doc.title, paragraphs=paragraphs,
                    body_indices=list(doc.body_indices), total_words=total_words)


def optimize_document(
    doc: Document,
    report: ScoreReport,
    client,
    cfg: AppConfig,
    target: Optional[float] = None,
    progress: ProgressFn = None,
    cancel: CancelFn = None,
) -> Dict[int, RewriteResult]:
    """把整篇指数压到 target 以下。任何异常都收进 RewriteResult，不向外抛。"""
    if target is not None:
        goal = float(target)
    else:
        rate = float(getattr(cfg, "target_rate", 0.0) or 0.0)
        goal = rate if rate > 0 else float(getattr(cfg, "optimize_target", 18.0))
    by_index = {paragraph.index: paragraph for paragraph in doc.paragraphs}
    current: Dict[int, str] = {}
    originals: Dict[int, str] = {}
    initial_scores: Dict[int, float] = {}
    for item in report.paragraphs:
        originals[item.index] = item.text
        initial_scores[item.index] = item.score
    for index in doc.body_indices:
        if index in by_index:
            current[index] = by_index[index].text

    if not current:
        return {}

    budget = _Budget(int(getattr(cfg, "optimize_max_calls", 300)))
    rounds = max(1, int(getattr(cfg, "optimize_rounds", 2)))
    results: Dict[int, RewriteResult] = {}
    accepted: Dict[int, str] = {}

    for round_no in range(1, rounds + 1):
        if _cancelled(cancel):
            break
        pending: List[Tuple[float, int, ParagraphScore]] = []
        for index, text in current.items():
            paragraph_score = scoring.score_paragraph(text, index)
            if paragraph_score.score > goal:
                pending.append((paragraph_score.score, index, paragraph_score))
        if not pending:
            _report(progress, "全部段落已低于目标 %.0f 分，停止迭代" % goal)
            break
        pending.sort(key=lambda item: (-item[0], item[1]))
        _report(progress, "第 %d/%d 轮：%d 段高于目标 %.0f 分，开始改写"
                % (round_no, rounds, len(pending), goal))

        improved = False

        def work(item: Tuple[float, int, ParagraphScore]) -> RewriteResult:
            _score_value, index, paragraph_score = item
            base = by_index.get(index)
            paragraph = Paragraph(index=index, text=current[index],
                                  style=base.style if base else "")
            return _optimize_one(paragraph, paragraph_score, client, cfg, goal,
                                 budget, "", progress, cancel)

        with ThreadPoolExecutor(max_workers=max(1, int(getattr(cfg, "concurrency", 3)))) as pool:
            futures = [pool.submit(work, item) for item in pending]
            for future in as_completed(futures):
                try:
                    outcome = future.result()
                except Exception as exc:  # noqa: BLE001
                    continue
                previous = results.get(outcome.index)
                if previous is None or _better(outcome, previous):
                    results[outcome.index] = outcome
                    if outcome.ok and outcome.revised:
                        if current.get(outcome.index) != outcome.revised:
                            improved = True
                        current[outcome.index] = outcome.revised
                        accepted[outcome.index] = outcome.revised

        if not improved:
            _report(progress, "第 %d 轮没有带来任何改进，提前结束" % round_no)
            break

    # 统一把 original / score_before 还原为最初版本，便于报告展示
    for index, outcome in results.items():
        outcome.original = originals.get(index, outcome.original)
        outcome.score_before = initial_scores.get(index, outcome.score_before)
        if outcome.attempts == 0 and outcome.ok:
            outcome.ok = False
    _report(progress, "优化结束：接受 %d 段改写，实际调用 %d 次"
            % (len(accepted), budget.used))
    return results

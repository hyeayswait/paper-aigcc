"""改写模块的提示词。

定位：让文字更贴近作者本人的写作习惯（自然化改写 / 去模板化），
保留事实、数据、术语与引用，改变句式和节奏。
不使用「规避检测」这类表述。
"""
from __future__ import annotations

from typing import List, Tuple

SYSTEM_PROMPT = (
    "你是一位中文论文润色编辑，任务是帮助作者把段落改写得更像作者本人写出来的文字，"
    "也就是自然化改写、去模板化。你要去掉的是模板化的痕迹：空泛的过渡句、堆砌的排比、"
    "机械的因果连接、平均到没有节奏的句长；而不是给文本换个说法继续套模板。\n"
    "铁律：\n"
    "1. 原文中的事实、数据、百分比、年份、术语、专有名词、引用标记和引号内的内容必须原样保留，"
    "不得改动，不得新增原文没有提供的事实或数据；\n"
    "2. 可以调整语序、拆分或合并句子、替换非关键表述，但不能改变原意，不能删掉论据；\n"
    "3. 不要写「综上所述」「值得注意的是」「随着……的快速发展」这类套话，除非原文有且删掉会丢失信息；\n"
    "4. 允许使用适度的第一人称限定（如「笔者认为」「在本文的语境下」），让判断有作者主体性；\n"
    "5. 只输出改写后的正文段落，不要解释、不要标题、不要编号、不要 markdown 代码块、不要用引号包裹。"
)

_REWRITE_RULES = (
    "1. 保留原文全部事实、数据、术语、专有名词、引用与引号内容，不得新增原文没有的事实；\n"
    "2. 拆开过长的句子，让长短句交替出现，句子节奏不要平均；\n"
    "3. 删除冗余过渡词、空泛套话和机械的因果连接；\n"
    "4. 减少排比与「不仅……而且……」「一方面……另一方面……」这类对称结构；\n"
    "5. 允许适度使用第一人称限定与作者判断语气；\n"
    "6. 专业术语保持原样，不要为了制造变化而替换术语。"
)


def _signal_lines(signals: List[Tuple[str, str, float]]) -> List[str]:
    """把评分器给出的 Top 信号翻译成可执行的改写提示。"""
    lines: List[str] = []
    for item in signals or []:
        try:
            key = str(item[0])
            name = str(item[1]) if len(item) > 1 and item[1] else key
            contribution = float(item[2]) if len(item) > 2 else 0.0
        except (TypeError, ValueError, IndexError):
            continue
        if contribution > 0:
            lines.append(
                "- {0}（贡献 {1:.1f} 分）：这一项偏模板化，请重点弱化。".format(name, contribution)
            )
        else:
            lines.append("- {0}：这一项还可以更自然一些。".format(name))
    if not lines:
        lines.append("- 没有明显的单项信号，请整体让表达更自然、更有个人节奏。")
    return lines


def _context_block(context: str) -> List[str]:
    ctx = (context or "").strip()
    if not ctx:
        return []
    return [
        "【上下文】\n" + ctx + "\n（上下文只用于理解语境，不要照抄，也不要改写它。）"
    ]


def build_rewrite_prompt(original: str, signals: List[Tuple[str, str, float]],
                         context: str = "") -> str:
    """构造首次改写用的用户提示词。"""
    body = (original or "").strip()
    parts: List[str] = []
    parts.extend(_context_block(context))
    parts.append("【需要自然化改写的段落】\n" + body)
    parts.append("【这一段偏模板化的具体信号】\n" + "\n".join(_signal_lines(signals)))
    parts.append("【改写要求】\n" + _REWRITE_RULES)
    parts.append("只输出改写后的这一段正文，不要任何解释或前后缀。")
    return "\n\n".join(parts)


def build_retry_prompt(original: str, previous: str, prev_score: float,
                       target: float, signals: List[Tuple[str, str, float]]) -> str:
    """构造未达标时的重试用提示词：告知上一版成绩，换一种改法。"""
    body = (original or "").strip()
    last = (previous or "").strip()
    try:
        score_text = "{0:.1f}".format(float(prev_score))
    except (TypeError, ValueError):
        score_text = str(prev_score)
    try:
        target_text = "{0:.1f}".format(float(target))
    except (TypeError, ValueError):
        target_text = str(target)
    parts: List[str] = []
    parts.append("【需要自然化改写的原文】\n" + body)
    parts.append("【上一版改写结果（仍有模板化痕迹）】\n" + last)
    parts.append(
        "上一版改写后，这段文字的 AI 痕迹指数为 {0}，高于本次希望达到的 {1}。"
        "请换一种句式结构重写，不要沿用上一版的分句顺序和连接方式。".format(score_text, target_text)
    )
    parts.append("【仍偏模板化的信号】\n" + "\n".join(_signal_lines(signals)))
    parts.append("【改写要求】\n" + _REWRITE_RULES)
    parts.append(
        "请只输出这次改写后的正文。务必保留原文的全部事实、数据、术语和引用，"
        "不要新增原文没有提供的内容。"
    )
    return "\n\n".join(parts)

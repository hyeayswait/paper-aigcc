# -*- coding: utf-8 -*-
"""导入 PaperPass（或同结构）AIGC 报告，把逐句疑似度映射到你的 docx 段落。

    python tools/import_paperpass.py --zip 报告.zip --docx 论文.docx --out out\pp_map.json

输出：整篇疑似度、逐段疑似度（按段落聚合）、最可疑句子清单。
用途：让「改写」有真实检测器的目标，而不是靠本工具自己的估分。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from paper_aigc.core.textnorm import normalize_text
from paper_aigc.io.docx_reader import read_docx


def _decode_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    try:
        return name.encode("cp437").decode("gbk")
    except Exception:
        return name


def extract_zip(zip_path: str, dest: str) -> str:
    if os.path.isdir(dest):
        import shutil

        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            target = os.path.join(dest, _decode_name(info).replace("\\", "/"))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
    return dest


def _find(root: str, suffix: str) -> str:
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(suffix):
                return os.path.join(base, name)
    return ""


def _load_var_json(path: str) -> dict:
    text = io.open(path, encoding="utf-8", errors="ignore").read()
    start = text.find("{")
    if start < 0:
        raise ValueError("文件里找不到 JSON：%s" % os.path.basename(path))
    payload = text[start:].strip()
    decoder = json.JSONDecoder()
    value, _end = decoder.raw_decode(payload)
    return value


def load_paperpass(root: str):
    sentence_file = _find(root, "texthtmldata_ai.js")
    score_file = _find(root, "reduceaigcpagedata.js")
    if not sentence_file:
        raise ValueError("报告里没有 texthtmldata_ai.js，可能不是 PaperPass 结构")
    sentences = _load_var_json(sentence_file).get("aiCheckSentenceList") or []
    totals = _load_var_json(score_file) if score_file else {}
    return sentences, totals


def map_to_paragraphs(sentences, docx_path: str):
    doc = read_docx(docx_path)
    paragraphs = [(p.index, normalize_text(p.text)) for p in doc.paragraphs if p.text]
    buckets = {}
    for item in sentences:
        content = normalize_text(str(item.get("content") or ""))
        if not content:
            continue
        score = float(item.get("score") or 0.0)
        matched = None
        for index, text in paragraphs:
            if content and content in text:
                matched = index
                break
        if matched is None:
            # 退化：按首 12 字匹配
            head = content[:12]
            for index, text in paragraphs:
                if head and head in text:
                    matched = index
                    break
        if matched is None:
            continue
        buckets.setdefault(matched, []).append((content, score))

    rows = []
    for index, items in buckets.items():
        items.sort(key=lambda pair: -pair[1])
        scores = [pair[1] for pair in items]
        rows.append({
            "index": index,
            "display": index + 1,
            "sentences": len(items),
            "mean_score": round(sum(scores) / len(scores), 1),
            "max_score": round(max(scores), 1),
            "high_count": sum(1 for s in scores if s >= 80),
            "mid_count": sum(1 for s in scores if 50 <= s < 80),
            "low_count": sum(1 for s in scores if 0 < s < 50),
            "clean_count": sum(1 for s in scores if s == 0),
            "worst_sentence": items[0][0][:80],
        })
    rows.sort(key=lambda row: -row["mean_score"])
    return doc, rows


def main() -> int:
    parser = argparse.ArgumentParser(description="导入 PaperPass AIGC 报告并映射到 docx 段落")
    parser.add_argument("--zip", required=True, help="报告 zip 路径")
    parser.add_argument("--docx", required=True, help="被检测的 docx")
    parser.add_argument("--out", default="", help="JSON 输出路径")
    parser.add_argument("--keep", default="", help="解压目录（默认临时目录）")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    dest = args.keep or os.path.join(_ROOT, "data", "report", "extracted")
    root = extract_zip(args.zip, dest)
    sentences, totals = load_paperpass(root)
    doc, rows = map_to_paragraphs(sentences, args.docx)

    print("PaperPass 报告：%s" % os.path.basename(args.zip))
    if totals:
        print("  AIGC 总体疑似度：%.2f%%" % float(totals.get("score", 0)))
        print("  高度疑似片段占比：%.2f%%  中度：%.2f%%  轻度：%.2f%%  不检测：%.2f%%" % (
            float(totals.get("highSuspectedTextRatio", 0)),
            float(totals.get("middleSuspectedTextRatio", 0)),
            float(totals.get("lowSuspectedTextRatio", 0)),
            float(totals.get("noAISuspectedTextRatio", 0))))
    print("  解析句子：%d 条，匹配到段落：%d 段" % (len(sentences), len(rows)))
    print("\n逐段 PaperPass 疑似度（降序）：")
    print("  段落    句数  平均   最高  高疑似  中  低  干净")
    for row in rows:
        print("  #%-5d %4d %6.1f %6.1f %6d %3d %3d %4d" % (
            row["display"], row["sentences"], row["mean_score"], row["max_score"],
            row["high_count"], row["mid_count"], row["low_count"], row["clean_count"]))

    payload = {"totals": totals, "paragraphs": rows,
               "sentence_count": len(sentences), "docx": os.path.abspath(args.docx)}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with io.open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("\n已写入 %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

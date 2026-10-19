# -*- coding: utf-8 -*-
"""PaperAIGC 桌面端入口。

用法：
    python app.py                 # 打开窗口
    python app.py 论文.docx       # 打开窗口并自动载入论文（可把 docx 拖到 .bat 上）

说明：分数是本地文体特征的启发式「AI 痕迹指数」，未与知网校准，
绝对值不可与知网 AIGC 率比较；只有同版本前后对比与段落排序有意义。
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from typing import Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from paper_aigc.config import AppConfig, load_config, mask_key, save_config
from paper_aigc.core import scoring
from paper_aigc.core.features import FEATURE_META
from paper_aigc.core.types import (
    DISCLAIMER_ZH,
    VERSION,
    Document,
    ParagraphScore,
    RewriteResult,
    ScoreReport,
)
from paper_aigc.io.document_loader import read_document
from paper_aigc.io.docx_writer import write_docx
from paper_aigc.io.report import render_html
from paper_aigc.io.similarity import to_meta as similarity_to_meta
from paper_aigc.io.similarity_providers import check as similarity_check
from paper_aigc.rewrite.client import DeepSeekClient, RewriteError
from paper_aigc.rewrite.optimize import apply_replacements, optimize_document
from paper_aigc.rewrite.pipeline import rewrite_document

APP_TITLE = "PaperAIGC · 论文 AI 痕迹自查 v" + VERSION
FONT_UI = ("Microsoft YaHei UI", 10)
FONT_UI_BOLD = ("Microsoft YaHei UI", 10, "bold")
FONT_H1 = ("Microsoft YaHei UI", 22, "bold")
FONT_SCORE = ("Microsoft YaHei UI", 40, "bold")
FONT_MONO = ("Consolas", 10)

LEVEL_FG = {"低": "#1b7a4b", "中": "#9a6b00", "偏高": "#c25a12", "高": "#b4231f"}
LEVEL_BG = {"低": "#e9f7ef", "中": "#fdf6e0", "偏高": "#fdeee1", "高": "#fbe9e8"}


class SettingsDialog(tk.Toplevel):
    """设置窗口：API Key 与改写阈值。"""

    FIELDS = [
        ("api_key", "DeepSeek API Key", "password"),
        ("base_url", "API Base URL", "text"),
        ("model", "模型名", "text"),
        ("temperature", "改写温度", "float"),
        ("timeout", "请求超时（秒）", "int"),
        ("retries", "失败重试次数", "int"),
        ("high_threshold", "改写阈值（0-100，45=偏高起点）", "float"),
        ("rewrite_target", "改写目标分数（0-100）", "float"),
        ("max_rewrite_attempts", "每段最多改写次数", "int"),
        ("concurrency", "并发线程数", "int"),
        ("score_engine", "评分引擎（calibrated/heuristic）", "text"),
        ("optimize_target", "全文优化目标指数（0-100）", "float"),
        ("target_rate", "目标值（压到该百分比以下，如 50/40/30）", "float"),
        ("corpus_dir", "本地查重库目录", "text"),
        ("similarity_provider", "查重方式（local / web:后端名）", "text"),
        ("optimize_rounds", "全文优化轮数", "int"),
        ("optimize_max_calls", "全文优化调用预算", "int"),
    ]

    def __init__(self, parent: "PaperAIGCApp"):
        super().__init__(parent)
        self.parent = parent
        self.transient(parent)
        self.title("设置")
        self.resizable(False, False)
        self.vars: Dict[str, tk.Variable] = {}
        self._build()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _build(self) -> None:
        body = ttk.Frame(self, padding=16)
        body.pack(fill="both", expand=True)
        for r, (key, label, kind) in enumerate(self.FIELDS):
            ttk.Label(body, text=label, font=FONT_UI).grid(row=r, column=0, sticky="w", pady=4, padx=(0, 12))
            var = tk.StringVar(value=str(getattr(self.parent.cfg, key)))
            if kind == "password":
                ent = ttk.Entry(body, textvariable=var, width=48, show="*")
            else:
                ent = ttk.Entry(body, textvariable=var, width=48)
            ent.grid(row=r, column=1, sticky="we", pady=4)
            self.vars[key] = var

        hint = ttk.Label(
            body,
            text="API Key 只保存在本机 %APPDATA%\\PaperAIGC\\config.json，不会上传到除 DeepSeek 之外的任何地方。\n"
                 "也可以用环境变量 DEEPSEEK_API_KEY 提供，环境变量优先级更高。",
            font=("Microsoft YaHei UI", 9),
            foreground="#666666",
            justify="left",
        )
        hint.grid(row=len(self.FIELDS), column=0, columnspan=2, sticky="w", pady=(10, 6))

        btns = ttk.Frame(body)
        btns.grid(row=len(self.FIELDS) + 1, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(btns, text="测试连接", command=self._test).pack(side="left", padx=6)
        ttk.Button(btns, text="保存", command=self._save).pack(side="left", padx=6)
        ttk.Button(btns, text="取消", command=self.destroy).pack(side="left")

    def _collect(self) -> Optional[AppConfig]:
        cfg = AppConfig(**{k: getattr(self.parent.cfg, k) for k in (
            "api_key", "base_url", "model", "temperature", "max_tokens", "timeout", "retries",
            "high_threshold", "rewrite_target", "max_rewrite_attempts", "min_rewrite_chars",
            "concurrency", "optimize_target", "optimize_rounds", "optimize_max_calls",
            "score_engine", "target_rate", "corpus_dir", "similarity_provider",
            "export_dir")})
        for key, _label, kind in self.FIELDS:
            raw = str(self.vars[key].get()).strip()
            try:
                if kind == "int":
                    setattr(cfg, key, int(raw))
                elif kind == "float":
                    setattr(cfg, key, float(raw))
                else:
                    setattr(cfg, key, raw)
            except ValueError:
                messagebox.showerror("参数错误", "「%s」不是合法数字：%s" % (_label, raw), parent=self)
                return None
        cfg.high_threshold = min(max(cfg.high_threshold, 0.0), 100.0)
        cfg.rewrite_target = min(max(cfg.rewrite_target, 0.0), 100.0)
        cfg.max_rewrite_attempts = max(1, cfg.max_rewrite_attempts)
        cfg.concurrency = max(1, min(cfg.concurrency, 8))
        cfg.optimize_target = min(max(cfg.optimize_target, 0.0), 100.0)
        cfg.target_rate = min(max(cfg.target_rate, 5.0), 95.0)
        cfg.optimize_rounds = max(1, min(cfg.optimize_rounds, 5))
        cfg.optimize_max_calls = max(1, min(cfg.optimize_max_calls, 2000))
        cfg.timeout = max(10, cfg.timeout)
        cfg.retries = max(0, cfg.retries)
        return cfg

    def _save(self) -> None:
        cfg = self._collect()
        if cfg is None:
            return
        try:
            path = save_config(cfg)
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc), parent=self)
            return
        self.parent.cfg = cfg
        try:
            scoring.set_engine(cfg.score_engine)
        except Exception:
            pass
        self.parent.refresh_settings_label()
        messagebox.showinfo("已保存", "配置已写入：\n" + path, parent=self)
        self.destroy()

    def _test(self) -> None:
        cfg = self._collect()
        if cfg is None:
            return
        self.parent.set_status("正在测试 DeepSeek 连接……")

        def work() -> None:
            try:
                client = DeepSeekClient(cfg)
                ok, msg = client.check_connection()
            except RewriteError as exc:
                ok, msg = False, str(exc)
            except Exception as exc:  # noqa: BLE001
                ok, msg = False, "未知错误：%s" % exc
            self.parent.q.put(("conn_test", (ok, msg)))

        threading.Thread(target=work, daemon=True).start()


class PaperAIGCApp(tk.Tk):
    def __init__(self, initial_file: str = "") -> None:
        super().__init__()
        self.cfg: AppConfig = load_config()
        try:
            scoring.set_engine(self.cfg.score_engine)
        except Exception:
            scoring.set_engine("heuristic")
        self.doc: Optional[Document] = None
        self.report: Optional[ScoreReport] = None
        self.rewrites: Dict[int, RewriteResult] = {}
        self.similarity_meta: Dict[str, object] = {}
        self.q: "queue.Queue" = queue.Queue()
        self.busy = False
        self.cancel_flag = threading.Event()
        self.current_index: Optional[int] = None
        self.source_path = ""

        self.title(APP_TITLE)
        self.geometry("1240x830")
        self.minsize(1020, 680)
        self._init_style()
        self._build()
        self.after(120, self._poll)
        if initial_file:
            self.after(250, lambda: self.open_file(initial_file))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI ----------------
    def _init_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Treeview", font=FONT_UI, rowheight=26)
        style.configure("Treeview.Heading", font=FONT_UI_BOLD)
        style.configure("Accent.TButton", font=FONT_UI_BOLD)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=(10, 8, 10, 6))
        root.pack(fill="both", expand=True)

        # 工具栏
        bar = ttk.Frame(root)
        bar.pack(fill="x", pady=(0, 6))
        self.btn_open = ttk.Button(bar, text="打开 docx…", command=self.open_file_dialog, width=12)
        self.btn_open.pack(side="left")
        self.btn_scan = ttk.Button(bar, text="开始检测", command=self.start_scan, style="Accent.TButton", width=12)
        self.btn_scan.pack(side="left", padx=(8, 0))
        self.btn_rewrite = ttk.Button(bar, text="改写「偏高」及以上", command=self.start_rewrite, width=18)
        self.btn_rewrite.pack(side="left", padx=(8, 0))
        self.btn_optimize = ttk.Button(bar, text="全文优化到目标", command=self.start_optimize, width=16)
        self.btn_optimize.pack(side="left", padx=(8, 0))
        self.btn_sim = ttk.Button(bar, text="查重（本地库）", command=self.start_similarity, width=14)
        self.btn_sim.pack(side="left", padx=(8, 0))
        self.btn_lint = ttk.Button(bar, text="规范自检", command=self.start_lint, width=10)
        self.btn_lint.pack(side="left", padx=(8, 0))
        self.btn_dedup = ttk.Button(bar, text="降重（本地库）", command=self.start_dedup, width=14)
        self.btn_dedup.pack(side="left", padx=(8, 0))
        self.btn_html = ttk.Button(bar, text="导出 HTML 报告", command=self.export_html, width=16)
        self.btn_html.pack(side="left", padx=(8, 0))
        self.btn_docx = ttk.Button(bar, text="导出改写后 docx", command=self.export_docx, width=18)
        self.btn_docx.pack(side="left", padx=(8, 0))
        ttk.Button(bar, text="设置", command=self.open_settings, width=8).pack(side="left", padx=(8, 0))
        self.btn_cancel = ttk.Button(bar, text="取消", command=self.request_cancel, width=8, state="disabled")
        self.btn_cancel.pack(side="left", padx=(8, 0))

        self.file_label = ttk.Label(root, text="尚未选择文件（提示：可以把 docx 拖到项目里的「拖入论文.bat」上）",
                                    font=FONT_UI, foreground="#444444")
        self.file_label.pack(fill="x", pady=(0, 4))

        # 摘要区
        summary = tk.Frame(root, bg="#ffffff", highlightbackground="#dddddd", highlightthickness=1)
        summary.pack(fill="x", pady=(0, 6))
        left = tk.Frame(summary, bg="#ffffff")
        left.pack(side="left", padx=14, pady=8)
        self.score_value = tk.Label(left, text="—", font=FONT_SCORE, bg="#ffffff", fg="#999999")
        self.score_value.pack(anchor="w")
        tk.Label(left, text="AI 痕迹指数（0-100，未校准）", font=("Microsoft YaHei UI", 9), bg="#ffffff", fg="#666666").pack(anchor="w")

        right = tk.Frame(summary, bg="#ffffff")
        right.pack(side="left", fill="both", expand=True, padx=(10, 14), pady=8)
        self.stats_label = tk.Label(right, text="载入论文后点「开始检测」", font=FONT_UI, bg="#ffffff",
                                    fg="#333333", justify="left", anchor="w", wraplength=860)
        self.stats_label.pack(anchor="w", fill="x")

        # 免责声明（首屏可见、不可折叠）
        disc = tk.Label(root, text=DISCLAIMER_ZH, font=("Microsoft YaHei UI", 9), bg="#fff6e0",
                        fg="#7a4b00", justify="left", wraplength=1180, anchor="w",
                        highlightbackground="#f0d9a8", highlightthickness=1, padx=8, pady=6)
        disc.pack(fill="x", pady=(0, 6))

        # 主体
        paned = ttk.Panedwindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left_box = ttk.Frame(paned)
        ttk.Label(left_box, text="段落列表（按风险降序，双击查看详情）", font=FONT_UI_BOLD).pack(anchor="w", pady=(0, 4))
        cols = ("idx", "score", "level", "chars", "preview")
        self.tree = ttk.Treeview(left_box, columns=cols, show="headings", selectmode="browse")
        for cid, title, width, anchor in (
            ("idx", "序号", 52, "center"),
            ("score", "分数", 58, "center"),
            ("level", "等级", 56, "center"),
            ("chars", "字数", 58, "center"),
            ("preview", "摘要", 250, "w"),
        ):
            self.tree.heading(cid, text=title)
            self.tree.column(cid, width=width, anchor=anchor, stretch=(cid == "preview"))
        vsb = ttk.Scrollbar(left_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        paned.add(left_box, weight=3)

        right_box = ttk.Frame(paned)
        self.nb = ttk.Notebook(right_box)
        self.nb.pack(fill="both", expand=True)
        self.detail_text = self._make_text_tab(self.nb, "段落详情")
        self.rewrite_text = self._make_text_tab(self.nb, "改写对照")
        paned.add(right_box, weight=4)

        # 状态栏
        status = ttk.Frame(root)
        status.pack(fill="x", pady=(6, 0))
        self.status_label = ttk.Label(status, text="就绪", font=("Microsoft YaHei UI", 9), foreground="#444444")
        self.status_label.pack(side="left")
        self.progress = ttk.Progressbar(status, mode="determinate", length=220)
        self.progress.pack(side="right")
        self.settings_label = ttk.Label(status, text="", font=("Microsoft YaHei UI", 9), foreground="#777777")
        self.settings_label.pack(side="right", padx=(0, 12))
        self.refresh_settings_label()

    def _make_text_tab(self, nb: ttk.Notebook, title: str) -> tk.Text:
        frame = ttk.Frame(nb)
        txt = tk.Text(frame, wrap="word", font=FONT_UI, state="disabled", bg="#ffffff",
                      relief="flat", padx=10, pady=8, spacing1=2, spacing3=4)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        txt.tag_configure("h1", font=FONT_UI_BOLD, foreground="#1f3a5f", spacing1=8, spacing3=6)
        txt.tag_configure("h2", font=FONT_UI_BOLD, foreground="#1f3a5f", spacing1=10, spacing3=4)
        txt.tag_configure("meta", font=("Microsoft YaHei UI", 9), foreground="#666666")
        txt.tag_configure("body", font=FONT_UI, spacing1=4)
        txt.tag_configure("good", foreground=LEVEL_FG["低"], font=FONT_UI_BOLD)
        txt.tag_configure("bad", foreground=LEVEL_FG["高"], font=FONT_UI_BOLD)
        txt.tag_configure("warn", foreground=LEVEL_FG["偏高"], font=FONT_UI_BOLD)
        txt.tag_configure("quote", font=FONT_MONO, lmargin1=14, lmargin2=14, foreground="#333333")
        nb.add(frame, text=title)
        return txt

    # ---------------- 状态 ----------------
    def set_status(self, msg: str) -> None:
        self.status_label.configure(text=msg)

    def refresh_settings_label(self) -> None:
        key_state = mask_key(self.cfg.api_key)
        self.settings_label.configure(
            text="模型 %s · Key %s · 改写目标 ≤%.0f" % (self.cfg.model, key_state, self.cfg.rewrite_target)
        )

    def _set_busy(self, busy: bool, msg: str = "") -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        for btn in (self.btn_open, self.btn_scan, self.btn_rewrite, self.btn_optimize, self.btn_sim,
                    self.btn_lint, self.btn_dedup,
                    self.btn_html, self.btn_docx):
            btn.configure(state=state)
        self.btn_cancel.configure(state="normal" if busy else "disabled")
        self.progress.configure(value=0)
        if msg:
            self.set_status(msg)

    def request_cancel(self) -> None:
        self.cancel_flag.set()
        self.set_status("正在取消……（已发出的请求会等它返回）")

    # ---------------- 文件与检测 ----------------
    def open_file_dialog(self) -> None:
        path = filedialog.askopenfilename(
            title="选择论文（.docx / .pdf / .txt / .md）",
            filetypes=[("论文文件", "*.docx *.pdf *.txt *.md"),
                       ("Word 文档", "*.docx"), ("PDF", "*.pdf"), ("所有文件", "*.*")],
        )
        if path:
            self.open_file(path)

    def open_file(self, path: str) -> None:
        if self.busy:
            messagebox.showinfo("请稍等", "当前任务尚未结束。")
            return
        try:
            doc = read_document(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("无法读取文档", str(exc))
            return
        self.doc = doc
        self.report = None
        self.rewrites = {}
        self.similarity_meta = {}
        self.current_index = None
        self.source_path = path
        self.file_label.configure(text="%s   ·   共 %d 段，参与评分 %d 段，正文 %d 字" % (
            path, len(doc.paragraphs), len(doc.body_indices), doc.total_words))
        self.tree.delete(*self.tree.get_children())
        self._write_text(self.detail_text, "点击「开始检测」生成报告。")
        self._write_text(self.rewrite_text, "检测完成后，可对高风险段落执行「改写高风险段落」。")
        self.score_value.configure(text="—", fg="#999999")
        self.stats_label.configure(text="点「开始检测」")
        self.set_status("已载入：%s" % os.path.basename(path))

    def start_scan(self) -> None:
        if not self.doc:
            messagebox.showinfo("提示", "请先打开一份 docx。")
            return
        if self.busy:
            return
        self._set_busy(True, "正在分析文体特征……")
        self.cancel_flag.clear()
        doc = self.doc

        def work() -> None:
            try:
                report = scoring.score_document(doc)
                self.q.put(("report", report))
            except Exception:  # noqa: BLE001
                self.q.put(("error", "检测失败：\n" + traceback.format_exc()))
            finally:
                self.q.put(("done", "scan"))

        threading.Thread(target=work, daemon=True).start()

    def start_rewrite(self) -> None:
        if not self.doc or not self.report:
            messagebox.showinfo("提示", "请先完成检测。")
            return
        if self.busy:
            return
        if not self.cfg.api_key:
            messagebox.showwarning("未配置 API Key", "改写需要 DeepSeek API Key，请在「设置」里填写。")
            self.open_settings()
            return
        targets = [p for p in self.report.paragraphs
                   if p.score >= self.cfg.high_threshold and p.index in self.doc.body_indices]
        if not targets:
            messagebox.showinfo("没有需要改写的段落",
                                "没有段落达到 %.0f 分（「偏高」起点）。" % self.cfg.high_threshold)
            return
        n = len(targets)
        if not messagebox.askyesno("确认改写",
                                   "将改写 %d 个高风险段落（阈值 %.0f），调用 DeepSeek 接口会产生费用。\n"
                                   "预计每段最多请求 %d 次。是否继续？" % (n, self.cfg.high_threshold,
                                                                         self.cfg.max_rewrite_attempts)):
            return
        self._set_busy(True, "正在改写 0/%d……" % n)
        self.cancel_flag.clear()
        doc, report, cfg = self.doc, self.report, self.cfg
        counter = {"n": 0}

        def progress(msg: str) -> None:
            self.q.put(("status", msg))

        def work() -> None:
            try:
                client = DeepSeekClient(cfg)
                results = rewrite_document(doc, report, client, cfg, progress=progress,
                                           cancel=self.cancel_flag.is_set)
                self.q.put(("rewrites", results))
            except Exception:  # noqa: BLE001
                self.q.put(("error", "改写失败：\n" + traceback.format_exc()))
            finally:
                self.q.put(("done", "rewrite"))

        threading.Thread(target=work, daemon=True).start()

    # ---------------- 结果渲染 ----------------
    def apply_report(self, report: ScoreReport) -> None:
        self.report = report
        if self.similarity_meta:
            self.report.meta["similarity"] = self.similarity_meta
        self.tree.delete(*self.tree.get_children())
        ordered = report.sorted_by_risk()
        for ps in ordered:
            preview = ps.text.replace("\n", " ")[:42]
            if len(ps.text) > 42:
                preview += "…"
            self.tree.insert("", "end", iid=str(ps.index), values=(
                ps.index + 1, "%.1f" % ps.score, ps.level, ps.text and len(ps.text) or 0, preview),
                tags=(ps.level,))
        for level, bg in LEVEL_BG.items():
            self.tree.tag_configure(level, background=bg, foreground=LEVEL_FG[level])

        self.score_value.configure(text="%.1f" % report.overall, fg=LEVEL_FG.get(report.level, "#333333"))
        high = report.level_counts.get("高", 0) + report.level_counts.get("偏高", 0)
        top = ordered[0] if ordered else None
        top_txt = "—" if top is None else "第 %d 段（%.1f 分，%s）" % (top.index + 1, top.score, top.level)
        self.stats_label.configure(text=(
            "等级判定：%s        参与评分段落：%d        正文总字数：%d\n"
            "高风险段（等级「高」）：%d        偏高及以上：%d        最高风险段：%s"
        ) % (report.level, len(report.paragraphs), report.document.total_words,
             report.level_counts.get("高", 0), high, top_txt))
        profile = (report.meta or {}).get("profile") or {}
        if profile.get("note"):
            self.stats_label.configure(
                text=str(self.stats_label.cget("text")) + "\n全文画像：" + str(profile["note"]))

        if ordered:
            self.tree.selection_set(str(ordered[0].index))
            self.tree.focus(str(ordered[0].index))
            self.show_paragraph(ordered[0].index)
        self.set_status("检测完成：整篇 AI 痕迹指数 %.1f（未校准，不可与知网数值比较）" % report.overall)

    def _on_select(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        try:
            self.show_paragraph(int(sel[0]))
        except (TypeError, ValueError):
            pass

    def show_paragraph(self, index: int) -> None:
        if not self.report:
            return
        ps = next((p for p in self.report.paragraphs if p.index == index), None)
        if ps is None:
            return
        self.current_index = index
        txt = self.detail_text
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        txt.insert("end", "第 %d 段   " % (index + 1), "h1")
        txt.insert("end", "分数 %.1f   " % ps.score, "h1")
        txt.insert("end", "等级 %s\n" % ps.level, "h1")
        txt.insert("end", "字数 %d   参与评分：%s\n" % (len(ps.text), "是" if index in self.report.document.body_indices else "否"), "meta")

        txt.insert("end", "高贡献信号", "h2")
        if ps.top_signals:
            for key, name, contrib in ps.top_signals:
                meta = FEATURE_META.get(key, {})
                txt.insert("end", "  · %s（%s）" % (name, key), "body")
                txt.insert("end", "    贡献 %.1f 分" % contrib, "warn")
                desc = str(meta.get("desc", "") or "")
                if desc:
                    txt.insert("end", "    %s" % desc, "meta")
                txt.insert("end", "\n")
        else:
            txt.insert("end", "  没有明显偏向 AI 的特征。\n", "good")

        txt.insert("end", "改写建议", "h2")
        txt.insert("end", (ps.suggestion or "—") + "\n", "body")

        txt.insert("end", "原文", "h2")
        txt.insert("end", ps.text + "\n", "quote")

        rw = self.rewrites.get(index)
        if rw is not None:
            txt.insert("end", "改写结果", "h2")
            if rw.ok and rw.revised:
                txt.insert("end", "分数 %.1f → %.1f（尝试 %d 次）\n" % (rw.score_before, rw.score_after, rw.attempts), "good")
                txt.insert("end", rw.revised + "\n", "quote")
            else:
                txt.insert("end", "未成功：%s（尝试 %d 次）\n" % (rw.error or "未知原因", rw.attempts), "bad")
        txt.configure(state="disabled")

    def render_rewrites(self) -> None:
        txt = self.rewrite_text
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        if not self.rewrites:
            txt.insert("end", "还没有改写结果。\n\n改写会保留事实、数据、术语与引用，只调整句式节奏、\n"
                              "删减模板化过渡词，让它更像作者本人写的。\n", "body")
            txt.configure(state="disabled")
            return
        ok = [r for r in self.rewrites.values() if r.ok]
        txt.insert("end", "改写完成 %d 段，成功 %d 段\n" % (len(self.rewrites), len(ok)), "h1")
        txt.insert("end", "注意：改写后的文字仍需你逐段核对事实与学术表述，工具只负责文风。\n", "meta")
        for r in sorted(self.rewrites.values(), key=lambda x: x.index):
            txt.insert("end", "\n第 %d 段   %.1f → %.1f   " % (r.index + 1, r.score_before, r.score_after), "h2")
            if r.ok:
                txt.insert("end", "✔ 成功（%d 次尝试）\n" % r.attempts, "good")
            else:
                txt.insert("end", "✘ %s（%d 次尝试）\n" % (r.error or "失败", r.attempts), "bad")
            txt.insert("end", "改写后：\n", "meta")
            txt.insert("end", (r.revised or "—") + "\n", "quote")
            txt.insert("end", "原文：\n", "meta")
            txt.insert("end", (r.original or "—") + "\n", "quote")
        txt.configure(state="disabled")

    def _write_text(self, widget: tk.Text, content: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", content, "body")
        widget.configure(state="disabled")

    # ---------------- 导出 ----------------
    def _out_dir(self) -> str:
        if not self.source_path:
            return os.getcwd()
        base = self.cfg.export_dir or os.path.join(os.path.dirname(self.source_path), "AIGC输出")
        os.makedirs(base, exist_ok=True)
        return base

    def _stem(self) -> str:
        return os.path.splitext(os.path.basename(self.source_path))[0] if self.source_path else "论文"

    def export_html(self) -> None:
        if not self.report:
            messagebox.showinfo("提示", "请先完成检测。")
            return
        default = os.path.join(self._out_dir(), self._stem() + "_AIGC自查报告.html")
        path = filedialog.asksaveasfilename(title="保存 HTML 报告", defaultextension=".html",
                                            initialfile=os.path.basename(default),
                                            initialdir=os.path.dirname(default),
                                            filetypes=[("HTML", "*.html")])
        if not path:
            return
        try:
            render_html(self.report, path, self.rewrites or None)
        except Exception:  # noqa: BLE001
            messagebox.showerror("导出失败", traceback.format_exc())
            return
        self.set_status("报告已保存：%s" % path)
        if messagebox.askyesno("已保存", "报告已保存到：\n%s\n\n是否现在打开？" % path):
            try:
                os.startfile(path)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    def export_docx(self) -> None:
        if not self.rewrites:
            messagebox.showinfo("提示", "还没有改写结果。")
            return
        replacements = {i: r.revised for i, r in self.rewrites.items() if r.ok and r.revised}
        if not replacements:
            messagebox.showinfo("提示", "没有任何成功改写的段落。")
            return
        default = os.path.join(self._out_dir(), self._stem() + "_改写版.docx")
        path = filedialog.asksaveasfilename(title="保存改写后的 docx", defaultextension=".docx",
                                            initialfile=os.path.basename(default),
                                            initialdir=os.path.dirname(default),
                                            filetypes=[("Word 文档", "*.docx")])
        if not path:
            return
        try:
            write_docx(self.source_path, replacements, path)
        except Exception:  # noqa: BLE001
            messagebox.showerror("导出失败", traceback.format_exc())
            return
        self.set_status("已保存改写版：%s（替换 %d 段）" % (path, len(replacements)))
        if messagebox.askyesno("已保存", "已写出 %d 段改写文本：\n%s\n\n请注意逐段核对内容。是否打开？"
                               % (len(replacements), path)):
            try:
                os.startfile(path)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    def start_optimize(self) -> None:
        """全文优化：把所有段落压到目标指数以下，然后重新评分。"""
        if not self.doc or not self.report:
            messagebox.showinfo("提示", "请先完成检测。")
            return
        if self.busy:
            return
        if not self.cfg.api_key:
            messagebox.showwarning("未配置 API Key", "全文优化需要 DeepSeek API Key，请在「设置」里填写。")
            self.open_settings()
            return
        goal = float(self.cfg.optimize_target)
        pending = sum(1 for item in self.report.paragraphs if item.score > goal)
        if pending == 0:
            messagebox.showinfo("已经达标", "所有段落都已低于目标指数 %.0f。" % goal)
            return
        if not messagebox.askyesno(
            "确认全文优化",
            "整篇当前 %.1f，目标压到 %.0f 以下。\n"
            "将改写 %d 段，最多 %d 轮、%d 次 API 调用（会产生费用）。\n\n"
            "改写只调整句式，数字/百分比/引用标记/表图编号会逐段本地校验，"
            "段落数量与文档结构不变。是否继续？"
            % (self.report.overall, goal, pending, self.cfg.optimize_rounds,
               self.cfg.optimize_max_calls),
        ):
            return

        self._set_busy(True, "正在全文优化……")
        self.cancel_flag.clear()
        doc, report, cfg = self.doc, self.report, self.cfg

        def progress(msg: str) -> None:
            self.q.put(("status", msg))

        def work() -> None:
            try:
                client = DeepSeekClient(cfg)
                results = optimize_document(doc, report, client, cfg, target=goal,
                                            progress=progress,
                                            cancel=self.cancel_flag.is_set)
            except Exception:  # noqa: BLE001
                self.q.put(("error", "全文优化失败：\n" + traceback.format_exc()))
                self.q.put(("done", "optimize"))
                return
            accepted = {i: r.revised for i, r in results.items() if r.ok and r.revised}
            new_doc = apply_replacements(doc, accepted)
            report_after = scoring.score_document(new_doc)
            info = {
                "target": goal,
                "overall_before": report.overall,
                "overall_after": report_after.overall,
                "paragraphs_rewritten": len(accepted),
                "paragraphs_total": len(doc.body_indices),
            }
            report_after.meta["optimize"] = info
            self.q.put(("optimized", (results, new_doc, report_after, info)))
            self.q.put(("done", "optimize"))

        threading.Thread(target=work, daemon=True).start()

    def start_similarity(self) -> None:
        """本地库查重：只与 corpus_dir 里的参考文档比对，不联网。"""
        if not self.doc:
            messagebox.showinfo("提示", "请先打开一份 docx。")
            return
        if self.busy:
            return
        corpus = self.cfg.corpus_dir
        if not corpus or not os.path.isdir(corpus):
            messagebox.showwarning("未配置查重库",
                                   "请先在「设置」里填写本地查重库目录，把参考文档（.txt/.md/.docx）放进去。\n\n"
                                   "注意：本功能只与本地库比对，不联网，结果不等同于 PaperPass/知网。")
            self.open_settings()
            return
        self._set_busy(True, "正在比对本地库……")
        doc = self.doc

        def work() -> None:
            try:
                sim = similarity_check(doc, self.cfg)
                self.q.put(("similarity", sim))
            except Exception:  # noqa: BLE001
                self.q.put(("error", "查重失败：\n" + traceback.format_exc()))
            finally:
                self.q.put(("done", "similarity"))

        threading.Thread(target=work, daemon=True).start()

    def start_lint(self) -> None:
        """论文规范自检（不涉及 AIGC，纯格式与规范检查）。"""
        if not self.doc:
            messagebox.showinfo("提示", "请先打开一份论文文件。")
            return
        if self.busy:
            return
        self._set_busy(True, "正在做规范自检……")

        def work() -> None:
            try:
                from paper_aigc.core.lint import lint_document, to_meta

                meta = to_meta(lint_document(self.doc))
                self.q.put(("lint", meta))
            except Exception:  # noqa: BLE001
                self.q.put(("error", "规范自检失败：\n" + traceback.format_exc()))
            finally:
                self.q.put(("done", "lint"))

        threading.Thread(target=work, daemon=True).start()

    def start_dedup(self) -> None:
        """降重：把重复率高的段落改写到目标重复率以下（需要本地库 + API Key）。"""
        if not self.doc:
            messagebox.showinfo("提示", "请先打开一份论文文件。")
            return
        if self.busy:
            return
        corpus = self.cfg.corpus_dir
        if not corpus or not os.path.isdir(corpus):
            messagebox.showwarning("未配置查重库",
                                   "降重需要本地比对库：请在「设置」里填写目录，并放入参考文档。")
            self.open_settings()
            return
        if not self.cfg.api_key:
            messagebox.showwarning("未配置 API Key", "降重需要 DeepSeek API Key。")
            self.open_settings()
            return
        if not messagebox.askyesno(
            "确认降重",
            "将把重复率超过 15% 的段落改写到 15% 以下（本地库口径）。\n"
            "改写会保留全部数字/引用/表号，并做本地事实校验。是否继续？"
        ):
            return
        self._set_busy(True, "正在降重……")
        self.cancel_flag.clear()
        doc, cfg = self.doc, self.cfg
        target = 0.15

        def progress(msg: str) -> None:
            self.q.put(("status", msg))

        def work() -> None:
            try:
                client = DeepSeekClient(cfg)
                from paper_aigc.rewrite.dedup import dedup_document

                results = dedup_document(doc, client, cfg, cfg.corpus_dir,
                                         target_ratio=target, progress=progress,
                                         cancel=self.cancel_flag.is_set)
                self.q.put(("deduped", results))
            except Exception:  # noqa: BLE001
                self.q.put(("error", "降重失败：\n" + traceback.format_exc()))
            finally:
                self.q.put(("done", "dedup"))

        threading.Thread(target=work, daemon=True).start()

    def open_settings(self) -> None:
        SettingsDialog(self)

    # ---------------- 消息泵 ----------------
    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    self.set_status(str(payload))
                elif kind == "report":
                    self.apply_report(payload)
                elif kind == "rewrites":
                    self.rewrites = payload or {}
                    self.render_rewrites()
                    self.nb.select(1)
                    ok = sum(1 for r in self.rewrites.values() if r.ok)
                    self.set_status("改写完成：成功 %d / 共 %d 段" % (ok, len(self.rewrites)))
                    if self.current_index is not None:
                        self.show_paragraph(self.current_index)
                elif kind == "conn_test":
                    ok, msg = payload
                    if ok:
                        messagebox.showinfo("连接正常", msg)
                    else:
                        messagebox.showerror("连接失败", msg)
                elif kind == "lint":
                    meta = payload or {}
                    if self.report is not None:
                        self.report.meta["lint"] = meta
                    counts = meta.get("counts") or {}
                    issues = (meta.get("issues") or [])[:6]
                    detail = "\n".join(
                        "  [%s] %s" % (str(item.get("level")).upper(), str(item.get("message"))[:50])
                        for item in issues) or "  没有发现明显问题。"
                    messagebox.showinfo(
                        "规范自检完成",
                        "错误 %d / 警告 %d / 提示 %d\n\n%s" % (
                            int(counts.get("error", 0)), int(counts.get("warn", 0)),
                            int(counts.get("info", 0)), detail))
                    self.set_status("规范自检完成：错误 %d / 警告 %d" % (
                        int(counts.get("error", 0)), int(counts.get("warn", 0))))
                elif kind == "deduped":
                    results = payload or {}
                    self.rewrites = results
                    ok = sum(1 for r in results.values() if r.ok)
                    messagebox.showinfo(
                        "降重完成",
                        "处理 %d 段，成功 %d 段。\n点「导出改写后 docx」保存。" % (len(results), ok))
                    self.set_status("降重完成：成功 %d / 共 %d 段" % (ok, len(results)))
                elif kind == "similarity":
                    sim = payload
                    self.similarity_meta = similarity_to_meta(sim)
                    if self.report is not None:
                        self.report.meta["similarity"] = self.similarity_meta
                    top = [item for item in sim.sorted_by_risk() if item.ratio > 0][:5]
                    detail = "\n".join(
                        "  #%d  %.1f%%" % (item.index + 1, item.ratio * 100) for item in top)
                    if not top:
                        detail = "  没有检测到明显重复。"
                    messagebox.showinfo(
                        "本地库查重完成",
                        "总文字复制比：%.2f%%\n比对库：%d 个文件 / %d 字\n\n%s\n\n%s"
                        % (sim.overall_percent, sim.corpus_files, sim.corpus_chars, detail, sim.note))
                    self.set_status("查重完成：总复制比 %.2f%%（仅本地库比对）" % sim.overall_percent)
                elif kind == "optimized":
                    results, new_doc, report_after, info = payload
                    self.rewrites = results
                    self.doc = new_doc
                    self.apply_report(report_after)
                    self.render_rewrites()
                    self.nb.select(1)
                    reached = "已达标" if info["overall_after"] < info["target"] else "未达标"
                    messagebox.showinfo(
                        "全文优化完成",
                        "整篇 AI 痕迹指数：%.1f → %.1f（目标 < %.0f，%s）\n"
                        "改写 %d / %d 段，下降 %.1f 分。\n\n"
                        "下一步：点「导出改写后 docx」保存，并逐段核对数字与引用标记。"
                        % (info["overall_before"], info["overall_after"], info["target"], reached,
                           info["paragraphs_rewritten"], info["paragraphs_total"],
                           info["overall_before"] - info["overall_after"]))
                elif kind == "error":
                    messagebox.showerror("出错了", str(payload)[:2000])
                elif kind == "done":
                    if payload == "rewrite":
                        self.set_status("改写流程结束")
                    elif payload == "scan":
                        pass
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def _on_close(self) -> None:
        if self.busy:
            if not messagebox.askyesno("确认退出", "当前任务还没结束，确定退出吗？"):
                return
            self.cancel_flag.set()
        self.destroy()


def main() -> None:
    initial = sys.argv[1] if len(sys.argv) > 1 else ""
    if initial and not os.path.isfile(initial.strip('"').strip("'")):
        initial = ""
    app = PaperAIGCApp(initial)
    app.mainloop()


if __name__ == "__main__":
    main()

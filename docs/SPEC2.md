# SPEC2 — 新增六功能的接口契约（v2）

> 环境同 SPEC：Python 3.9，禁止 pip install，write/edit 工具在 workspace 会 EISDIR，
> 必须用 pwsh + `[System.IO.File]::WriteAllText(路径, $t + "`n", UTF8NoBOM)` 落盘，写完 read 回读。
> 测试：`python -m unittest discover -s paper_aigc/tests -t . -v`（在 <project> 下）。

## 0. 文件归属（禁止越界）

| 文件 | 归属 |
|---|---|
| `paper_aigc/io/pdf_reader.py`、`paper_aigc/io/document_loader.py`、`tests/test_pdf_reader.py` | T4 |
| `paper_aigc/io/version_diff.py`、`tools/version_diff.py`、`tests/test_version_diff.py` | T5 |
| `paper_aigc/core/lint.py`、`tests/test_lint.py` | T6 |
| `paper_aigc/io/score_ledger.py`、`tests/test_ledger.py` | T7 |
| `paper_aigc/rewrite/dedup.py`、`tests/test_dedup.py` | T8 |
| `report.py`、`run.py`、`app.py`、`config.py`、`README.md`、`docs/SPEC2.md` | **Lead 独占，其他人禁止修改** |

## 1. T4 PDF 导入

```python
# paper_aigc/io/pdf_reader.py
def read_pdf(path: str) -> Document   # 复用 core/types.Document
def is_pdf_supported() -> bool        # pymupdf 是否可用
```

- 依赖 `pymupdf`（已装在 `<project>\vendor`）。文件顶部先 `from ._vendor import ensure_vendor; ensure_vendor()`。
- `from ._vendor import ensure_vendor` 由 Lead 提供；`vendor/` 不存在时 ensure_vendor 静默返回 False。
- 解析：每页 `page.get_text("blocks")` 或 `get_text("text")`，按空行/块切段落；去掉页眉页脚（每页重复的首尾行）、页码行（纯数字）。
- 段落规则与 docx_reader 一致：非空、`is_heading_like`、`is_furniture_like`、`is_reference_like`、`char_count >= 10` 才进 body_indices。
- 标题：第一个 heading 段；没有则文件名。
- 文件不存在/非 pdf/pymupdf 不可用 → 抛中文 `ValueError`。

```python
# paper_aigc/io/document_loader.py
def read_document(path: str) -> Document    # .docx -> docx_reader.read_docx；.pdf -> read_pdf；.txt/.md -> read_text
def supported_extensions() -> list
```
`.txt/.md`：整篇按空行切段，同样用 heading/furniture/reference 规则过滤。

## 2. T5 版本对比

```python
# paper_aigc/io/version_diff.py
@dataclass
class ParagraphDiff:
    status: str          # "added" | "removed" | "modified" | "same"
    old_index: int       # 无则 -1
    new_index: int       # 无则 -1
    old_text: str
    new_text: str
    similarity: float    # 0-1，去标点后的 13gram 重合 / min(len)
    char_delta: int

@dataclass
class DiffReport:
    old_path: str
    new_path: str
    items: List[ParagraphDiff]
    added: int
    removed: int
    modified: int
    unchanged: int
    old_chars: int
    new_chars: int

def diff_documents(old_path: str, new_path: str) -> DiffReport
def render_diff_html(diff: DiffReport, out_path: str) -> str
```
- 段落对齐：先按 `body_indices` 原文顺序，用 `difflib.SequenceMatcher` 对 `normalize_text` 后的段落文本做对齐；相似度 ≥0.6 视为 modified，否则 added/removed。
- HTML：单文件零外链、`html.escape`、内联 CSS；顶部摘要卡片（新增/删除/修改/未变、字数变化）；表格每行展示状态、序号、相似度、字数变化、原文（details 折叠）、新文（details 折叠）。
- `tools/version_diff.py` 命令行：`python tools/version_diff.py --old a.docx --new b.docx --out diff.html`，打印摘要。

## 3. T6 论文规范自检

```python
# paper_aigc/core/lint.py
@dataclass
class LintIssue:
    code: str        # 如 "table_seq" / "table_missing" / "ref_format" / "mixed_punct" / "dup_word" / "term_inconsistent" / "long_sentence" / "no_period"
    level: str       # "error" / "warn" / "info"
    message: str
    paragraph_index: int   # 无则 -1
    snippet: str

@dataclass
class LintReport:
    issues: List[LintIssue]
    counts: Dict[str, int]   # level -> 数量

def lint_document(doc: Document) -> LintReport
def to_meta(report: LintReport) -> dict
```
检查项（至少 7 类）：
1. 图表编号：正文出现的「表N」「图N」是否连续；表题（`^表\s*\d+`）是否有对应的「表N」引用；重复编号。
2. 参考文献：段落含 `[J]`/`[M]`/`[R]`/`[D]` 等但缺少年份(19xx/20xx)、缺少序号 `[n]` 开头、或序号不连续。
3. 中英文标点混用：中文段落里出现 `,` `.` `?` `!` `(` `)`（紧邻汉字）。
4. 重复词：同一汉字连续重复（如「的的」「了了」）、或 2-4 字词连续重复两次。
5. 术语不一致：同时出现「净资产收益率」与「ROE」、或「国内生产总值」与「GDP」这类中英混用（内置一个小词表）。
6. 超长句：单句汉字数 > 80。
7. 段落末尾无标点（非标题、非表格行）。

## 4. T7 多系统分数台账

```python
# paper_aigc/io/score_ledger.py
@dataclass
class LedgerRecord:
    time: str        # "YYYY-MM-DD HH:MM"
    version: str     # 如 "V6"
    system: str      # 朱雀 / PaperPass / 知网 / 本工具
    metric: str      # 如 疑似AI / 总复制比 / AI特征
    value: float     # 百分比数值，如 34.47
    note: str = ""

def ledger_path(base_dir: str = "") -> str      # 默认 data/score_ledger.json
def add_record(record: LedgerRecord, path: str = "") -> str
def load_records(path: str = "") -> List[LedgerRecord]
def to_meta(records: List[LedgerRecord]) -> dict   # 给报告用：latest_by_system + rows
```
- JSON 存储，文件不存在返回空列表；损坏文件不抛异常（返回空并忽略）。
- `to_meta` 结构：`{"rows":[{time,version,system,metric,value,note}], "latest": {system: {metric: value, version, time}}}`。

## 5. T8 降重专项

```python
# paper_aigc/rewrite/dedup.py
def dedup_document(doc: Document, client, cfg, corpus_dir: str,
                   target_ratio: float = 0.15,
                   progress=None, cancel=None) -> Dict[int, RewriteResult]
```
- 用 `paper_aigc.io.similarity.check_similarity` 找出重复率 > `target_ratio` 的正文段（按重复率降序）。
- 对每段：调 `client.chat` 做**同义改写/语序调整**，prompt 必须要求：保留全部数字/引用/表号、不得新增事实、只输出正文。
- 用 `paper_aigc.rewrite.optimize.validate_rewrite(original, candidate)` 校验事实；
- 改写后重新用 13gram 对比该段与库的重复率，取重复率最低且通过校验的版本（最多 3 次尝试）。
- 任何异常写入 `RewriteResult.error`，不向外抛；返回 `Dict[int, RewriteResult]`（index -> 结果，含 `score_before`/`score_after` 复用为「重复率×100」）。

## 6. 交付要求（所有人）

- `python -m compileall -q paper_aigc` 通过
- 只跑自己的测试模块：`python -m unittest paper_aigc.tests.test_xxx -v`
- 不许改 Lead 独占文件；需要新增共享文件先告诉 Lead
- 完成后 send_message 给 lead：文件、签名、测试结果、偏离说明

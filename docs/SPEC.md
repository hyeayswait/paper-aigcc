# SPEC — PaperAIGC 接口契约 v1

> 所有实现必须严格遵守本文件；类型定义以 `paper_aigc/core/types.py` 为准。

## 0. 定位与非目标（必须先理解）

本工具**没有接触过知网**，也没有知网标注数据。因此：

- 输出是 **0-100 的启发式「AI 痕迹指数」**，`calibration` 恒为 `"uncalibrated"`。
- **绝对数值不得与知网 AIGC 率比较**，报告里必须显著展示 `DISCLAIMER_ZH`。
- 能成立的结论：同版本内改写前后对比、段落之间相对排序。

禁止在代码/文案里把分数称作「知网 AIGC 率」「AIGC 率百分比」。

## 1. 环境约束

- Python **3.9.7**（系统 python）。禁止 3.10+ 语法（match、运行期 X|Y 注解）。
  每个文件都写 `from __future__ import annotations`。
- **pip 无网络，禁止安装任何包。** 可用：标准库、python-docx、numpy、pandas、requests、lxml。
- 编码：UTF-8 无 BOM。
- 测试：标准库 `unittest`（**pytest 未安装**），用例放 `paper_aigc/tests/`。
  运行方式（在 <project> 下）：`python -m unittest discover -s paper_aigc/tests -t . -v`
- 文件落盘：**write / edit 工具在本工作区有 bug（EISDIR）**，必须用 pwsh 写入。
  用 `[System.IO.File]::WriteAllText(路径, $t + "`n", $enc)`，其中 `$enc` 是
  `New-Object System.Text.UTF8Encoding($false)`。内容用 PowerShell 单引号 here-string 承载；
  **注意 here-string 的结束标记必须独占一行，因此写入的内容里不能出现独占一行的该标记。**

## 2. 目录与归属（禁止越界改他人文件）

```
paper_aigc/
  core/types.py        [Lead 已写]  共享契约，禁止修改
  core/textnorm.py     [Lead 已写]  共享工具，禁止修改
  config.py            [Lead 已写]  AppConfig / load_config / save_config / mask_key
  core/features.py     [T1]
  core/scoring.py      [T1]
  io/docx_reader.py    [T2]
  io/docx_writer.py    [T2]
  io/report.py         [T2]
  rewrite/prompts.py   [T3]
  rewrite/client.py    [T3]
  rewrite/pipeline.py  [T3]
  app.py run.py dnd.py README.md   [Lead]
  tests/test_features.py  tests/test_scoring.py   [T1]
  tests/test_io.py                                 [T2]
  tests/test_rewrite.py                            [T3]
```

## 3. 特征契约（T1 实现，键名固定，T3 会用到中文名）

`core/features.py` 暴露：

```python
FEATURE_META: Dict[str, Dict[str, object]]
# key -> {"name": 中文名, "weight": float, "desc": 含义, "anchors": List[Tuple[float, float]]}

def extract_features(text: str) -> FeatureSet
def extract_features_dict(text: str) -> Dict[str, float]
```

必须实现的 17 个键（值均为 float，不能 NaN/inf）：

| key | 中文名 | 量纲 / 计算 | AI 倾向 |
|---|---|---|---|
| sentence_len_cv | 句长变异系数 | 句长标准差/均值 | 低 |
| avg_sentence_len | 平均句长 | 每句汉字数均值 | 中偏长 |
| short_sentence_ratio | 短句占比 | 汉字数<12 的句子占比 | 低 |
| long_sentence_ratio | 长句占比 | 汉字数>45 的句子占比 | 高 |
| clause_len_cv | 分句长变异系数 | 分句长标准差/均值 | 低 |
| connective_density | 连接词密度 | 每百字过渡/连接词个数 | 高 |
| boilerplate_density | 套话密度 | 每百字学术套话命中数 | 高 |
| parallelism_density | 对仗排比密度 | 每百字「不仅…而且」类结构数 | 高 |
| hedge_density | 模糊限制语密度 | 每百字「可能/在一定程度上」等 | 高 |
| idiom_density | 四字格密度 | 四字连续汉字串 / 汉字总数 | 偏高 |
| num_ratio | 数字密度 | 每百字数字/百分数个数 | 低 |
| first_person_density | 第一人称密度 | 每百字我/我们/笔者/本人 | 低 |
| passive_density | 被动标记密度 | 每百字被/由/受到/得以/加以 | 高 |
| char_ttr | 词面丰富度 | 字符类 TTR（长文本用 MATTR，窗宽 50） | 低 |
| repetition_ratio | 三元组重复率 | 重复 3-gram 数 / 总 3-gram 数 | 高 |
| char_entropy | 汉字一元熵 | 汉字 unigram 香农熵（bits） | 偏低 |
| punct_diversity | 标点多样性 | 标点种类香农熵（bits） | 低 |

实现提示：分词用 `textnorm.tokenize_words`（汉字单字为 token）。切句/切分句用
`textnorm.split_sentences` / `textnorm.split_clauses`。空文本或超短文本返回全 0 且不抛异常。

## 4. 评分契约（T1）

`core/scoring.py` 暴露：

```python
LEVEL_THRESHOLDS = (25.0, 45.0, 60.0)   # <25 低, <45 中, <60 偏高, >=60 高
def level_of(score: float) -> str
def score_text(text: str) -> float
def score_paragraph(text: str, index: int = 0) -> ParagraphScore
def score_document(doc: Document) -> ScoreReport
```

算法：
1. 每个特征用**分段线性锚点**把原始值映射成 0-1 的 subscore（锚点表写在
   `FEATURE_META[key]["anchors"]`：`List[Tuple[float, float]]`，按 x 升序，超端点取端点值）。
2. raw = Σ(weight_i * sub_i) / Σ(weight_i)，score = round(raw*100, 1)。
3. top_signals：按 (sub_i - 0.5) * weight_i 降序，只保留 >0 的，取前 4 条，
   元素为 (key, FEATURE_META[key]["name"], round(weight_i*sub_i/Σweight*100, 1))。
4. suggestion：按命中的高贡献特征拼一句中文改写建议（每特征一句模板，最多拼 2 句）。
5. score_document：只统计 document.body_indices；overall 按 char_count 加权；
   level_counts 统计各等级段数；meta 写入
   {"version": VERSION, "feature_count": n, "engine": "stylometric-heuristic"}。
   空文档返回 overall=0.0、paragraphs=[]、level="低"，不抛异常。

必须满足：同一文本重复调用结果完全一致（无随机、无时间依赖）。

## 5. 文档解析契约（T2）

`io/docx_reader.py`：

```python
def read_docx(path: str) -> Document
```

- 用 python-docx。只收录 `normalize_text()` 后非空的段落，index 按收录顺序从 0 递增。
- style 取 `paragraph.style.name`；is_heading 用 `textnorm.is_heading_like(text, style)`；
  is_reference 用 `textnorm.is_reference_like(text)`。
- body_indices = 非标题、非参考文献、且 char_count >= 10 的段落 index。
- total_words = body 段落 char_count 之和。
- title = 第一个标题段文本；没有则用文件名去扩展名。
- 表格、文本框、页眉页脚不处理（README 注明）。文件不存在/非 docx 抛 ValueError，消息为中文。

`io/docx_writer.py`：

```python
def write_docx(source_path: str, replacements: Dict[int, str], out_path: str) -> str
```

- 用与 reader 完全相同的段落收录规则定位段落，替换 replacements 命中的段落文本。
- 保留原段落第一个 run 的格式：run[0].text = 新文本，其余 run 置空；无 run 时 add_run。
- 未命中的段落一字不动，原始文件不被修改。返回 out_path。

## 6. 报告契约（T2）

`io/report.py`：

```python
def render_html(report: ScoreReport, out_path: str,
                rewrite: Optional[Dict[int, RewriteResult]] = None) -> str
def render_markdown(report: ScoreReport,
                    rewrite: Optional[Dict[int, RewriteResult]] = None) -> str
```

HTML 要求：
- **单文件、零外部资源**（不引 CDN/字体/图片），内联 style，meta charset=utf-8。
- 顶部摘要卡片：AI 痕迹指数（大字）、参与评分段数、总字数、高风险段数、最高风险段落位置。
- **report.disclaimer 必须在首屏可见、字号 >= 13px、不可折叠**。
- 段落按风险降序表格：序号、分数（含内联进度条）、等级、Top 信号、一句话建议。
  原文用 details 折叠以免页面过长。
- 若传入 rewrite：展示原文/改写后对照与 score_before → score_after。
- 页脚含 VERSION 与生成时间。
- 用 html.escape 转义全部用户文本，防止论文里的 < 破坏页面。

Markdown：结构同上，管道表格，供命令行输出。

## 7. 改写契约（T3）

`rewrite/client.py`：

```python
class RewriteError(Exception): ...
class DeepSeekClient:
    def __init__(self, cfg: AppConfig) -> None
    def chat(self, messages: List[Dict[str, str]], temperature: float = None,
             max_tokens: int = None) -> str
    def check_connection(self) -> Tuple[bool, str]
```

- OpenAI 兼容：POST {base_url}/chat/completions，Authorization: Bearer <key>。
- 重试：网络错误/429/5xx 退避重试 cfg.retries 次；其余 4xx 不重试。
- 失败统一抛 RewriteError，消息为可直接展示的中文（API Key 无效 / 网络不可达 / 余额不足 等）。
- api_key 为空时构造函数抛 RewriteError("未配置 DeepSeek API Key")。
- 单测不得真实联网：用 unittest.mock.patch 替换 requests.post。

`rewrite/prompts.py`：

```python
SYSTEM_PROMPT: str
def build_rewrite_prompt(original: str, signals: List[Tuple[str, str, float]],
                         context: str = "") -> str
def build_retry_prompt(original: str, previous: str, prev_score: float,
                       target: float, signals: List[Tuple[str, str, float]]) -> str
```

改写规则（写进 prompt）：
- 保留全部事实、数据、术语、专有名词、引用与引号内容，不得新增未提供的事实；
- 拆长句、长短句交替；去掉冗余过渡词与套话；减少排比与「不仅…而且」结构；
- 允许适度第一人称限定与作者判断语气；
- 只输出改写后的正文，不要解释、不要标题、不要 markdown 代码块。

`rewrite/pipeline.py`：

```python
def rewrite_paragraph(paragraph: Paragraph, score: ParagraphScore,
                      client: DeepSeekClient, cfg: AppConfig,
                      context: str = "", progress=None) -> RewriteResult
def rewrite_document(doc: Document, report: ScoreReport, client: DeepSeekClient,
                     cfg: AppConfig, progress=None, cancel=None) -> Dict[int, RewriteResult]
```

- rewrite_document 只处理 score.score >= cfg.high_threshold 且
  char_count >= cfg.min_rewrite_chars 的段落；用 ThreadPoolExecutor(max_workers=cfg.concurrency)。
- 每段最多 cfg.max_rewrite_attempts 次，用 scoring.score_paragraph 重新评分，
  达到 score_after <= cfg.rewrite_target 即停止。
- progress 回调签名 progress(msg: str)，从工作线程调用，不抛异常即可。
  cancel 为 Callable[[], bool]，返回 True 时尽快停止。
- 任何异常写进 RewriteResult.error，不得向外抛。
- 改写结果去空白后与原文相同或相似度 >0.98 记 ok=False, error="改写无变化"。

## 8. 测试要求

每模块至少 6 个有意义的断言，覆盖边界：
- T1：空文本、纯英文、仅标点、超长段落、确定性（两次相等）、分数区间 [0,100]、等级映射。
- T2：代码生成临时 docx -> 读回校验段落数/body_indices/标题；写回只改目标段落；
  HTML 含 disclaimer 且转义 <script>；无 http 外链。
- T3：mock requests.post 验证 URL/请求体/鉴权头；mock 客户端验证重试直到达标；
  API 失败时 error 非空且不抛异常；短段落被跳过。

测试必须能在无网络、无 API Key 时 `python -m unittest discover` 全绿。

## 9. 交付标准

- `python -m compileall -q paper_aigc` 无错误。
- 上述 unittest 全绿。
- 无调试 print 残留（CLI/GUI 入口除外），无 TODO 占位。

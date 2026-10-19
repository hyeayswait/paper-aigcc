# PaperAIGC · 中文论文 AI 痕迹自查与本地查重工具箱

[English](README.md) | **简体中文**

一个本地运行的中文论文工具箱：逐段算「AI 痕迹」、做自然化改写、本地库查重、版本对比、论文规范自检、多系统分数台账。支持 `.docx` / `.pdf` / `.txt` / `.md`。

```bat
python run.py 论文.docx --lint --similarity --corpus data\corpus -o 报告.html
```

## ⚠️ 先读这一段：能力边界

**这不是知网 / PaperPass / 维普的检测器。**

- 这些厂商没有公开 API，且服务条款禁止自动化调用，本项目**不爬取**它们。
- 内置两个评分引擎：
  - `heuristic`：手工锚点，透明可改，但在真实数据上判别力接近随机（AUC ≈ 0.47）；
  - `calibrated`：在公开中文 AIGC 检测数据上训练的逻辑回归（HC3 中文 + THUCNews + GPT-4.1/Gemini 平行语料，测试集 **AUC ≈ 0.826**），默认使用。
- 两个引擎都**没有**与知网对齐，绝对数值不可与任何商业系统互换。
- 实测跨检测器相关性很弱（与某商业检测器逐句 Pearson r ≈ 0.30）。请把它当作**排序与前后对比工具**，不要拿去向学校证明「达标」。

## 功能

| 功能 | 说明 | 入口 |
|---|---|---|
| AIGC 痕迹检测 | 17 个文体特征 + 校准模型，逐段 0–100 分与命中信号 | `run.py` / GUI「开始检测」 |
| 段落改写 | 保事实（数字/引用/表号本地校验），改句式，最多多轮 | `--rewrite` / 「改写」 |
| 全文优化到目标值 | 多轮全局迭代，压到指定百分比以下 | `--optimize --target-rate 50` |
| 本地库查重 | 13 字连续片段重合，逐段重复率 + 相似来源 + 自我重复 | `--similarity --corpus 目录` |
| 降重专项 | 对高重复段落做同义改写，目标重复率可设 | `--dedup --similarity-target 0.15` |
| 版本对比 | 两版 docx 逐段 diff（新增/删除/修改/未变） | `tools/version_diff.py` |
| 规范自检 | 图表编号、参考文献格式、标点、术语、超长句等 8 类 | `--lint` |
| 分数台账 | 记录朱雀/PaperPass/知网/本工具的分数 | `--ledger-add ... --ledger-show` |
| PDF 导入 | 自动剔除页眉页脚与页码 | 直接传 `.pdf` |
| 一页纸总览 | 报告首屏汇总所有指标 | 报告自动生成 |

## 安装

```bat
git clone <repo-url>
cd paper-aigc
pip install -r requirements.txt
```

- Python 3.9+（Windows 自带 Tkinter 即可运行 GUI）
- `PyMuPDF` 仅检测 PDF 时需要
- 改写 / 优化 / 降重需要 DeepSeek API Key：`set DEEPSEEK_API_KEY=sk-...`

## 快速开始

**图形界面**：把论文拖到 `拖入论文.bat` 上（或 `python app.py`）。

**命令行**：

```bat
:: 1) 检测 + 规范自检 + 本地查重，生成报告
python run.py 论文.docx --lint --similarity --corpus data\corpus -o 报告.html

:: 2) 全文优化到 50% 以下（需 API Key）
python run.py 论文.docx --optimize --target-rate 50

:: 3) 两版对比
python tools\version_diff.py --old 原稿.docx --new 修改稿.docx --out 对比.html

:: 4) 记分数
python run.py 论文.docx --ledger-add "system=朱雀,metric=疑似AI,value=34.47,version=V6" --ledger-show
```

## 目录结构

```
paper_aigc/
  core/        types / textnorm / features / scoring / lint
  io/          docx&pdf 读取、报告、查重、版本对比、台账
  rewrite/     DeepSeek 客户端、改写流水线、全文优化、降重
  calibration/ 公开数据校准模型
app.py run.py  图形界面 / 命令行
tools/         诊断、评估、训练、对比等脚本
docs/          SPEC 接口契约
samples/       合成样例（非真实论文）
```

## 测试

```bat
python -m unittest discover -s paper_aigc/tests -t . -v
```

259 个离线用例，不需要网络与 API Key。

## 已知限制

1. 不能对齐知网，任何「与知网差值 ≤5%」的承诺都需要你提供知网标注数据做校准。
2. 校准模型的训练分布是问答/新闻语料，迁移到学位论文会打折。
3. 本地查重只比对你自己放进库的文档，不含学术库与互联网。
4. PDF 按页切段，跨页段落会被切开；表格内容不解析。
5. 「降 AIGC」与「降重」是两套方法，不要混用。

## 合理使用

本工具用于**自查与文字润色**。它不伪造数据、不替代研究内容，也不针对任何特定检测器做对抗优化。
论文内容的真实性与学术规范由作者本人负责。

## 许可证

MIT

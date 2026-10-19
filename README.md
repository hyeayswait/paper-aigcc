# PaperAIGC · Chinese Paper AI-Trace Self-Check & Local Similarity Toolkit

**English** | [简体中文](README.zh-CN.md)

A local Windows toolkit for Chinese academic papers: paragraph-level AI-trace scoring, naturalizing rewrites, local-corpus similarity checking, version diffing, format linting, and a multi-system score ledger. Works with `.docx`, `.pdf`, `.txt` and `.md`.

```bat
python run.py paper.docx --lint --similarity --corpus data\corpus -o report.html
```

## ⚠️ Read this first: scope and honesty

**This is NOT a CNKI / PaperPass / Weipu detector.**

- Those vendors have no public API and their terms forbid automation. This project does **not** scrape them.
- Two scoring engines ship with the tool:
  - `heuristic` — hand-tuned, fully transparent, but its discriminative power on real data is close to chance (AUC ≈ 0.47);
  - `calibrated` — a logistic regression trained on public Chinese AI-text detection data (HC3-zh + THUCNews + GPT-4.1/Gemini parallel corpora, **test AUC ≈ 0.826**); used by default.
- Neither engine is aligned with CNKI, and their absolute values are not interchangeable with any commercial system.
- Measured cross-detector correlation is weak (sentence-level Pearson r ≈ 0.30 against one commercial detector). Treat the output as a **ranking and before/after comparison tool**, not as proof of compliance.

## Features

| Feature | What it does | Entry point |
|---|---|---|
| AI-trace detection | 17 stylistic features + calibrated model, per-paragraph 0–100 with top signals | `run.py` / GUI |
| Paragraph rewriting | Fact-preserving (numbers/citations/table refs validated locally), style-only | `--rewrite` |
| Whole-document optimization | Multi-round iteration down to a target percentage | `--optimize --target-rate 50` |
| Local-corpus similarity | 13-character shingle overlap, per-paragraph ratio + sources + self-duplication | `--similarity --corpus DIR` |
| Dedup rewriting | Rewrites the most duplicated paragraphs toward a target ratio | `--dedup --similarity-target 0.15` |
| Version diff | Paragraph-level diff between two .docx (added/removed/modified/unchanged) | `tools/version_diff.py` |
| Format lint | Figure/table numbering, reference format, punctuation, terminology, overlong sentences (8 checks) | `--lint` |
| Score ledger | Keep scores from multiple detectors in one place | `--ledger-add ... --ledger-show` |
| PDF import | Header/footer and page-number stripping | pass a `.pdf` |
| One-page overview | Summary of every metric on the first screen of the report | generated automatically |

## Install

```bat
git clone <repo-url>
cd paper-aigc
pip install -r requirements.txt
```

- Python 3.9+ (Tkinter ships with Windows Python and is enough for the GUI)
- `PyMuPDF` is only needed for PDF input
- Rewriting / optimization / dedup need a DeepSeek API key: `set DEEPSEEK_API_KEY=sk-...`

## Quick start

**GUI**: drag a paper onto `拖入论文.bat`, or run `python app.py`.

**CLI**:

```bat
:: 1) detect + lint + local similarity check, write an HTML report
python run.py paper.docx --lint --similarity --corpus data\corpus -o report.html

:: 2) optimize the whole document below 50% (requires API key)
python run.py paper.docx --optimize --target-rate 50

:: 3) compare two versions
python tools\version_diff.py --old draft.docx --new revision.docx --out diff.html

:: 4) record detector scores
python run.py paper.docx --ledger-add "system=Tencent,metric=suspected-AI,value=34.47,version=V6" --ledger-show
```

## Layout

```
paper_aigc/
  core/        types / textnorm / features / scoring / lint
  io/          docx & pdf readers, report, similarity, version diff, ledger
  rewrite/     DeepSeek client, rewrite pipeline, optimization, dedup
  calibration/ public-data calibrated model
app.py run.py  GUI / CLI
tools/         diagnostics, evaluation, training, diffing
docs/          interface contracts (SPEC)
samples/       synthetic samples (not real papers)
```

## Tests

```bat
python -m unittest discover -s paper_aigc/tests -t . -v
```

259 offline test cases; no network or API key required.

## Limitations

1. Not aligned with CNKI. Any "within ±5% of CNKI" claim requires your own CNKI-labelled data for calibration.
2. The calibrated model was trained on Q&A/news corpora; transfer to theses degrades.
3. Local similarity only compares documents you put in your own corpus — no academic database, no internet.
4. PDF paragraphs are split at page boundaries; tables are not parsed.
5. "Lowering AI-trace" and "lowering duplication" are different pipelines — do not mix them up.

## Responsible use

This tool is for **self-checking and copy-editing**. It does not fabricate data, replace research content, or perform adversarial optimization against any specific detector. Responsibility for the accuracy and academic integrity of a paper remains with its author.

## License

MIT

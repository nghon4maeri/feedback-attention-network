# PDF Translation (VI-Translate integration)

This repository can translate research PDFs into Vietnamese (and 35 other
Latin-script languages) while preserving layout, formulas, tables and figures,
using a local integration of [VI-Translate](https://github.com/breslee1707/VI-Translate).

```
original research PDF
        ↓
    VI-Translate
        ↓
Vietnamese translated PDF
        ↓
papers/translated/
```

## Architecture

VI-Translate is brought in as a **git submodule** at `tools/vitranslate/`. It is
not copied into the tree, so it stays cleanly upgradable (`git submodule update
--remote tools/vitranslate`).

Because VI-Translate pins its own heavy dependencies (`babeldoc`, `onnxruntime`,
`pikepdf`, `pymupdf`, `opencv-python==5.0.0.93`, `numpy==2.2.6`) that conflict
with this repo's `torch`/`opencv`/`numpy` stack, it runs in its **own isolated
venv** at `papers/.vitranslate-venv` (git-ignored). This is exactly what the
upstream project's own `SKILL.md` recommends ("keep this environment separate
from the user's project").

The wrapper `tools/pdf_translate.py` locates that venv, creates/installs it on
first use, and delegates to `tools/vitranslate/scripts/translate_pdf.py`.

## Installation

```bash
# 1. Initialize the submodule (once)
git submodule update --init --recursive

# 2. Set up the isolated runtime (first run downloads deps + layout model/font,
#    so it needs network and takes a few minutes)
python tools/pdf_translate.py --setup
```

## Usage

Translate a paper to Vietnamese (default target, default output):

```bash
python tools/pdf_translate.py papers/original/attention.pdf
```

Result:

```
papers/
├── original/
│   └── attention.pdf
└── translated/
    └── attention-vi.pdf
```

The original PDF is never modified. The output directory is created
automatically.

### Options

```bash
# explicit output directory
python tools/pdf_translate.py papers/original/attention.pdf --output-dir out/

# different target language
python tools/pdf_translate.py papers/original/attention.pdf --target-language fr

# translate only pages 1 and 3-5
python tools/pdf_translate.py papers/original/attention.pdf --pages 1,3-5

# overwrite an existing translation
python tools/pdf_translate.py papers/original/attention.pdf --overwrite

# OCR for scanned pages (see below)
python tools/pdf_translate.py papers/original/scan.pdf --ocr standard

# recreate the runtime environment
python tools/pdf_translate.py --setup --reinstall
```

Configuration via environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `PDF_TRANSLATE_ENGINE` | `google` | translation engine |
| `PDF_TRANSLATE_OUTPUT_DIR` | `papers/translated` | output directory |
| `PDF_TRANSLATE_TARGET` | `vi` | target language |
| `PDF_TRANSLATE_THREADS` | `4` | worker threads (1–8) |
| `PDF_TRANSLATE_OCR` | `off` | OCR mode |
| `PDF_TRANSLATE_VENV` | `papers/.vitranslate-venv` | runtime venv location |

## Engines

| Engine | Notes | Use when |
| --- | --- | --- |
| **Google** (default) | `translate.google.com`; fast, free, **no API key**, needs network | Default; books, batches, first drafts |
| **Handoff** | extracts segments to JSONL, agent translates, rebuilds | Terminology/context/quality matters |

The Google engine requires no credentials. It does send document text to
Google — avoid it for sensitive material unless you authorize the disclosure.

Handoff is an agent-driven workflow; see `tools/vitranslate/SKILL.md` for the
full contract (segment JSONL format, formula/style placeholders, and the
verify-before-delivery checklist).

## OCR

Scanned (image-only) PDFs are not read by default. Enable OCR with
`--ocr standard` or `--ocr enhanced`. This needs the optional OCR dependencies:

```bash
& "papers/.vitranslate-venv/Scripts/python.exe" -m pip install -r "tools/vitranslate/requirements-ocr.txt"
```

OCR is safety-first: tables, formulas, figures and uncertain regions are kept in
the source and reported as a **partial** translation rather than being mangled.
With `--ocr off`, image-only pages are left untranslated and the CLI warns.

## Troubleshooting

| Error | Cause / fix |
| --- | --- |
| `VI-Translate submodule is not initialized` | Run `git submodule update --init --recursive` |
| `PDF core dependencies are missing` | Run `python tools/pdf_translate.py --setup` |
| `Input PDF does not exist` / `does not contain a PDF header` | Invalid or wrong path / not a PDF |
| `No text could be extracted ... image-only scans` | Scanned PDF; run with `--ocr standard` |
| `OCR is unavailable` | Install `requirements-ocr.txt` (see above) |
| `Output already exists` | Add `--overwrite` (only if you intend to replace it) |
| Translation failure / engine error | Google needs network; retry or switch engine |
| `unsupported target language` | Target must be a Latin-script code; see upstream CLI |

Exit codes: `0` success, `1` wrapper error (bad args / submodule / venv), `2`
translation error (propagated from the upstream CLI).

## OpenCode integration

A project-local OpenCode skill is registered at
`tools/opencode/skills/pdf-translate/SKILL.md` (added to `opencode.json`). You
can ask OpenCode:

> Translate this paper to Vietnamese: papers/original/my_paper.pdf

and it will locate the PDF, run the local integration, store the result in
`papers/translated/`, and report the output path — without modifying the
original.

## Tests

```bash
python tools/tests/test_pdf_translate.py
```

Network-free smoke tests (help, error handling, output-dir creation, source
preservation). The real end-to-end translation requires network and is a manual
smoke test:

```bash
python tools/pdf_translate.py papers/original/attention.pdf
```

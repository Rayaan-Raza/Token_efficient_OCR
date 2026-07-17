# RAVEN-Select + OCR-Protected Page Compression

**Separate experiment workspace.** Does **not** retune frozen RAVEN-Select v1.0.0.

Parent journal study (restore point): [`../raven_select_journal/RESTORE_POINT.md`](../raven_select_journal/RESTORE_POINT.md)

## Research hypothesis

Normal resizing wastes visual budget on whitespace and margins. OCR-protected
content-aware compression preserves text-bearing regions under the same VLM
input budget, improving the full-page reader and potentially strengthening
OCR-grounded output selection.

## Framing (Option A)

Keep RAVEN-Select as the selector. Seam / content-aware compression replaces
the **resize** reader only:

| Slot | Current | New |
|------|---------|-----|
| Full-page | Resize | OCR-SeamResize (or MarginCrop / WhitespaceCompress) |
| Lexical patches | BM25 | BM25 (unchanged) |
| Learned patches | LER-BOPS | LER-BOPS (unchanged) |
| Selector | RAVEN-Select v1.0.0 | **frozen** (unchanged) |

Cost remains **3 VLM calls**.

## Non-negotiables

- Do **not** change the frozen selector rule or tie-break order.
- Do **not** use gold answers or answer-bearing metadata in compression.
- Do **not** treat classic unprotected seam carving as production.
- Prefer OCR-protected / whitespace-aware variants; audit damaged layouts.
- Any adaptive ordering remains a separately named future method.

## Documents

| Doc | Role |
|-----|------|
| [docs/00-hypothesis.md](docs/00-hypothesis.md) | Why this experiment |
| [docs/01-variants.md](docs/01-variants.md) | MarginCrop / WhitespaceCompress / OCR-SeamResize |
| [docs/02-experiment-protocol.md](docs/02-experiment-protocol.md) | Runs, tables, CIs |
| [docs/03-gates.md](docs/03-gates.md) | PASS / PARTIAL / FAIL |
| [docs/04-status.md](docs/04-status.md) | Living results |

## Code

- `src/preprocessing/ocr_page_compress.py` — compression implementations
- `scripts/run_ocr_seam_resize_eval.py` — DocVQA driver for full-page + selector
- `tests/test_ocr_page_compress.py` — unit tests for protection invariants

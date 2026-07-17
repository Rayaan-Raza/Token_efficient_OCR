# RAVEN-Select Journal Workspace

**Title:** RAVEN-Select: Value and Limits of OCR-Grounded Answer Selection for Budgeted Document VQA

This folder is the **journal upgrade** workspace (frozen). The method-paper
snapshot remains in [`../raven_select_method_docvqa`](../raven_select_method_docvqa).

**Restore point:** [`RESTORE_POINT.md`](RESTORE_POINT.md) (git `0faa3c3`).

**Next experiment (separate):** OCR-protected page compression lives in
[`../raven_select_seam`](../raven_select_seam) and must not retune v1.0.0.

## Layout

| Path | Role |
|------|------|
| `main.tex` / `main.pdf` | IEEE journal draft (starts from the verified n=500 paper) |
| `references.bib` | Bibliography |
| `figures/` | Publication figures |
| `docs/README.md` | Phase index for the journal upgrade |
| `docs/P17-*.md` … `docs/P25-*.md` | Living phase documents |

## Frozen production method

RAVEN-Select (do not retune after seeing scaled / transfer results):

1. Run three readers: resize, BM25 patches, LER-BOPS patches
2. Normalize generated answers
3. Mark OCR-grounded if the normalized answer appears in page OCR or route-specific patch OCR
4. Return the shortest OCR-grounded answer
5. Else return the shortest nonempty answer
6. Break ties: resize → BM25 → LER-BOPS

Learned selectors are equal-cost comparators / ablations only.

## Phase documents

See [`docs/README.md`](docs/README.md) for gates, execution order, and status.

# P23 — Cost / Operating Point

## Scope

Honest cost subsection: RAVEN-Select is accuracy-oriented and uses 3 VLM calls.
Compare one-call readers separately from equal-cost three-output selectors.

## Gate

**COMPLETE** when calls and measured median runtime are reported for 1-call
readers and 3-call selectors, with OCR/cache assumptions stated.

## Reference table (n=500)

| Method | VLM calls | Median s | ANLS | EM |
|--------|-----------|----------|------|----|
| resize | 1 | 2.514 | 0.7840 | 0.676 |
| BM25 | 1 | 2.671 | 0.7555 | 0.658 |
| LER-BOPS | 1 | 3.519 | 0.7543 | 0.654 |
| RAVEN-Select | 3 | ~8.705 | 0.8053 | 0.706 |

## Framing

- Not a cheaper replacement for resize
- Appropriate when answer quality matters more than latency
- One-call pre-router did not beat resize → output information matters

## Results (n=1000)

| Method | VLM calls | Median s | ANLS | EM |
|--------|-----------|----------|------|----|
| resize | 1 | 2.609 | 0.8149 | 0.706 |
| BM25 | 1 | 3.141 | 0.7796 | 0.674 |
| LER-BOPS | 1 | 5.178 | 0.7873 | 0.681 |
| shortest nonempty | 3 | ~10.928 | 0.8173 | 0.708 |
| RAVEN-Select | 3 | ~10.928 | 0.8234 | 0.723 |

Status: **COMPLETE** for n=1000. Three-call time is the sum of per-reader
medians and excludes one-time OCR cache construction.

Artifact: `outputs/metrics/raven_select_cost_n1000.json`


## Commit

- TBD

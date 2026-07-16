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

## Results (scaled)

Status: **PENDING**

## Commit

- TBD

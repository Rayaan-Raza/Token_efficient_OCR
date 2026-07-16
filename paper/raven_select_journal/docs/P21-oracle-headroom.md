# P21 — Oracle Ceiling / Headroom

## Scope

Because RAVEN-Select only chooses among three generated outputs, accuracy is
bounded by the best-of-3 reader oracle.

## Metrics

- `oracle_anls` = mean max ANLS across resize / BM25 / LER-BOPS
- `available_headroom` = oracle − resize
- `recovered_headroom` = RAVEN-Select − resize
- `recovery_frac` = recovered / available

## Gate

**COMPLETE** when question-level vectors emit these metrics and they reconcile
with the main table.

## Reference (n=500, already known)

| Method | ANLS |
|--------|------|
| resize | 0.7840 |
| shortest nonempty | 0.7965 |
| RAVEN-Select | 0.8053 |
| best-of-3 oracle | 0.8333 |

Recovery ≈ (0.8053 − 0.7840) / (0.8333 − 0.7840) ≈ 0.43

## Results (n=1000)

Status: **PENDING**

## Commit

- TBD

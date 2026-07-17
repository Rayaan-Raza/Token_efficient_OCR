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

## Results (n=500, verified)

| Method | ANLS | EM |
|--------|------|----|
| resize | 0.7840 | 0.676 |
| RAVEN-Select | 0.8053 | 0.706 |
| best-of-3 oracle | 0.8333 | 0.740 |

- available_headroom = 0.0493
- recovered_headroom = 0.0214
- recovery_frac = 0.433

Status: **COMPLETE** for n=500.

## Commit

- pending in this feature commit

## Results (n=1000)

| Method | ANLS | EM |
|--------|------|----|
| resize | 0.8149 | 0.706 |
| RAVEN-Select | 0.8234 | 0.723 |
| best-of-3 oracle | 0.8547 | 0.759 |

- available_headroom = 0.0398
- recovered_headroom = 0.0086
- recovery_frac = 0.215

Status: **COMPLETE**. The frozen selector recovers 21.5% of the available
best-of-three gain over resize at n=1000.

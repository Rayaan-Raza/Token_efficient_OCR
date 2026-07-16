# P20 — Normalization Ablations

## Scope

Compare grounding normalizations on **identical** cached reader outputs.
The frozen conservative rule remains the production method.

## Variants (named; exploratory if selected post-hoc)

| Name | Rule |
|------|------|
| raw | Exact raw substring |
| conservative (production) | Lowercase + punctuation removal + whitespace collapse |
| num_norm / RAVEN-Select-NumNorm | Number normalization |
| date_norm / RAVEN-Select-DateNorm | Date normalization |
| fuzzy / RAVEN-Select-Fuzzy | Fuzzy match |

## Gate

**COMPLETE** only when all variants run and the production method is unchanged.
Any post-inspection choice keeps its distinct variant name.

## Results

| Grounding rule | ANLS | EM |
|----------------|------|----|
| RAVEN-Select-Raw (raw) | 0.8000 | 0.696 |
| RAVEN-Select (conservative) | 0.8053 | 0.706 |
| RAVEN-Select-NumNorm (num_norm) | 0.8064 | 0.704 |
| RAVEN-Select-DateNorm (date_norm) | 0.7900 | 0.684 |
| RAVEN-Select-Fuzzy (fuzzy) | 0.8067 | 0.700 |

Status: **COMPLETE** for n=500. Production conservative rule unchanged.
NumNorm/DateNorm/Fuzzy are exploratory named variants only.

## Commit

- see git history

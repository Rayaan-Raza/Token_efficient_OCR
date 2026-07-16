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
| raw | — | — |
| conservative (production) | — | — |
| number normalization | — | — |
| date normalization | — | — |
| fuzzy | — | — |

Status: **PENDING**

## Commit

- TBD

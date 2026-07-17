# P19 — VLM Robustness (Qwen2-VL-2B)

## Scope

Repeat the frozen RAVEN-Select three-reader + selection pipeline on DocVQA n=500
with a second backbone (`Qwen/Qwen2-VL-2B-Instruct`), tagged `qwen2vl2b`.

## Results (method 1.0.0)

| Method | ANLS | EM |
|--------|------|-----|
| resize | 0.5913 | 0.526 |
| BM25 | 0.4787 | 0.406 |
| LER-BOPS | 0.4937 | 0.418 |
| shortest nonempty | 0.6124 | 0.540 |
| RAVEN-Select | **0.6128** | **0.544** |

Paired 95% CI (RAVEN-Select − baseline):

| Contrast | Δ ANLS | 95% CI | CI lower > 0 |
|----------|--------|--------|--------------|
| vs resize | +0.0215 | [+0.0049, +0.0393] | **yes** |
| vs shortest nonempty | +0.0005 | [0.0000, +0.0013] | no (boundary) |

Status: **PARTIAL robustness** — significant gain vs resize on the weaker 2B backbone;
gain vs shortest is tiny / not strictly CI-positive. Artifact:
`outputs/metrics/raven_robustness_qwen2vl2b_n500.json`.

## Commit

- TBD

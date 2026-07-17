# P17 — DocVQA Scale

## Scope

Scale the frozen RAVEN-Select evaluation from DocVQA n=500 to n=1000, then to
the full available validation split **only if** the n=1000 gate PASSes.

## Gate

| Outcome | Condition |
|---------|-----------|
| **PASS** | Beats resize and shortest_nonempty; both paired 95% CI lower bounds > 0 |
| **PARTIAL** | Significantly beats exactly one of resize / shortest_nonempty |
| **FAIL** | Significantly beats neither |

## Commands / config

```text
# After download + manifest parameterization:
python scripts/download_docvqa_hf.py --num-samples 1000
python scripts/build_docvqa_manifests.py --source Data/manifests/docvqa_val_1000.jsonl --sizes 100 300 500 1000
python scripts/run_fullpage_ocr.py --manifest Data/manifests/docvqa_1000.jsonl
python scripts/run_raven_select_scale_vlm.py --limit 1000 --manifest Data/manifests/docvqa_1000.jsonl
python scripts/run_raven_select_build_features.py --n 1000
python scripts/run_raven_select_eval.py --n 1000 --write-gates
python scripts/run_raven_select_paper_ablations.py --n 1000
```

## Inputs

- Nested DocVQA manifests (preserve n=500 subset)
- Cached VLM outputs for resize / BM25 / LER-BOPS
- EasyOCR page + patch OCR

## Leakage safeguards

No gold answers, ANLS/EM, oracle routes, or gold-derived OCR flags at inference.

## Output artifacts

- `outputs/metrics/raven_select_overview_n1000.json`
- `outputs/metrics/raven_select_main_table_n1000.csv`
- `outputs/gates/P17_docvqa_scale_n1000.json`
- `outputs/gates/P17_docvqa_scale.json`

## Results (method 1.0.0, Qwen2.5-VL-3B)

| Method | ANLS | EM | VLM calls |
|--------|------|----|-----------|
| resize | 0.8149 | 0.706 | 1 |
| BM25 | 0.7796 | 0.674 | 1 |
| LER-BOPS | 0.7873 | 0.681 | 1 |
| shortest nonempty | 0.8173 | 0.708 | 3 |
| RAVEN-Select | **0.8234** | **0.723** | 3 |

Paired 95% CI deltas (RAVEN-Select − baseline):

| Contrast | Δ ANLS | 95% CI | CI lower > 0 |
|----------|--------|--------|--------------|
| vs resize | +0.0086 | [−0.0019, +0.0199] | no |
| vs shortest nonempty | +0.0061 | [+0.0008, +0.0123] | **yes** |

Route mix (RAVEN-Select): resize 915 / BM25 62 / LER-BOPS 23.

Status: **PARTIAL** — significant gain vs shortest nonempty; gain vs resize is positive but not significant at n=1000.
**Full validation is blocked** until a PASS (both CI lowers > 0).

Notes: 2 BM25 and 2 LER samples recorded empty predictions after CUDA device-side asserts (`vlm_error=True`).

## Commit

- TBD

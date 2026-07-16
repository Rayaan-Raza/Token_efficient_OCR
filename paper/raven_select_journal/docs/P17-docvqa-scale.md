# P17 — DocVQA Scale

## Scope

Scale the frozen RAVEN-Select evaluation from DocVQA n=500 to n=1000, then to
the full available validation split **only if** the n=1000 gate PASSes.

## Gate

| Outcome | Condition |
|---------|-----------|
| **PASS** | Beats resize and shortest_nonempty; both paired 95% CI lower bounds > 0 |
| **PARTIAL** | Significantly beats resize only |
| **FAIL** | Significantly beats neither |

## Commands / config

```text
# After download + manifest parameterization:
python scripts/download_docvqa_hf.py --num-samples 1000
python scripts/build_docvqa_manifests.py --source Data/manifests/docvqa_val_1000.jsonl --sizes 100 300 500 1000
python scripts/run_fullpage_ocr.py --manifest Data/manifests/docvqa_1000.jsonl
python scripts/run_raven_n500_driver.py --limit 1000 --manifest Data/manifests/docvqa_1000.jsonl
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
- `outputs/gates/P17_docvqa_scale.json` (when implemented)

## Results

| Method | ANLS | EM | VLM calls |
|--------|------|----|-----------|
| resize | — | — | 1 |
| BM25 | — | — | 1 |
| LER-BOPS | — | — | 1 |
| shortest nonempty | — | — | 3 |
| best learned selector | — | — | 3 |
| RAVEN-Select | — | — | 3 |

Status: **IN PROGRESS** — DocVQA n=1000 images+OCR ready; VLM three-reader eval resumed from seeded n=500 CSVs.

## Commit

- TBD

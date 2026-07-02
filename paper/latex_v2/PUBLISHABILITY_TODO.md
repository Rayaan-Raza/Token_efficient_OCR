# Publishability upgrade checklist for BOPS

## Completed (2026-07-02)
- [x] OCR word precision, word F1, predicted_token_count, duplicate_token_ratio (`src/ocr/ocr_metrics.py` + backfill script)
- [x] Budget/cost table (`scripts/make_budget_cost_table.py` → `paper/tables/table_budget_cost.csv`)
- [x] Answer coverage diagnostic for all patch selectors (random, uniform, overview_only, bops)
- [x] Failure panel split: 2 examples main (`failure_panel.pdf`) + 4 appendix (`failure_panel_appendix.pdf`)
- [x] Question-aware BOPS scoring (`src/preprocessing/patch_scoring_qa.py`, method `bops_qa`)
- [x] `paper/latex_v2/bops_publishable_ieee.tex` updated with precision table, budget table, coverage text

- [x] DocVQA n=100 question-aware BOPS (`bops_qa`) VLM eval — ANLS 0.225 vs bops 0.193; patch coverage still 5%

## Do not do yet
- [ ] DocVQA n=500 with broken default BOPS selector (scale only after selection improves)

## Key numbers for framing
| Metric | resize a0.25 | BOPS p8 |
|--------|--------------|---------|
| Word recall | 0.163 | 0.240 |
| Word precision | 0.296 | 0.217 |
| Word F1 | 0.193 | 0.199 |

Answer in selected-patch OCR: BOPS 6%, random 3%, uniform 2%, overview 0% (full-image OCR ~76% for all).

## Safe masters-application claim
"I designed and implemented a budget-aware multimodal preprocessing pipeline for text-rich images, evaluated OCR and VLM behavior under controlled visual budgets, and produced an IEEE-style empirical manuscript showing that OCR-guided patches improve text recall over aggressive resizing but fail for DocVQA unless patch selection becomes question-aware."

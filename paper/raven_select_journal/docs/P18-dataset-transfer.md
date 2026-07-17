# P18 — Dataset Transfer

## Scope

Evaluate the **frozen** RAVEN-Select rule on:

1. InfographicVQA n=300 (single-image pipeline unchanged)
2. MP-DocVQA **contact-sheet transfer setting** n=300 (budgeted adaptation; not standard MP-DocVQA)

Do not retune. Do not claim “we solve MP-DocVQA.”

## Gate (per dataset)

| Outcome | Condition |
|---------|-----------|
| **FULL TRANSFER** | Beats resize and shortest; both CI lower bounds > 0 |
| **PARTIAL TRANSFER** | Significantly beats resize only |
| **FAIL** | Significantly beats neither |

## Commands / config

```text
# After adapters + manifests exist:
python scripts/run_raven_select_eval.py --n 300 --dataset infographicvqa --models raven_select_rule --rebuild-ocr --write-gates
python scripts/run_raven_select_eval.py --n 300 --dataset mpdocvqa_contact --models raven_select_rule --rebuild-ocr --write-gates
```

## Leakage safeguards

No gold page labels, gold answers, or gold-derived OCR flags at inference.

## Results

| Dataset | resize | shortest | RAVEN-Select | Gate |
|---------|--------|----------|--------------|------|
| InfographicVQA n=300 | — | — | — | PENDING |
| MP-DocVQA contact-sheet n=300 | 0.3252 | 0.3244 | **0.3360** | **FAIL** |

MP contact-sheet RAVEN-Select improves by 0.0108 ANLS over resize, but the
paired 95% bootstrap CI crosses zero ([-0.0095, 0.0330]). It improves by
0.0116 over shortest nonempty with a positive CI ([0.0024, 0.0238]).
Under the preregistered transfer gate, shortest-only significance is still
**FAIL**: partial transfer requires significance over resize.

The MP contact-sheet result uses all 300 common examples. EM is 0.2800 for
RAVEN-Select versus 0.2567 for resize and 0.2600 for shortest nonempty.
The frozen rule routes 222 / 47 / 31 examples to resize / BM25 / LER-BOPS.

Artifacts:

- `outputs/metrics/raven_select_raven_select_rule_mpdocvqa_contact_n300.json`
- `outputs/gates/P18_dataset_transfer_mpdocvqa_contact_n300.json`



## Download status

- InfographicVQA n=300: **DOWNLOADED** (Data/manifests/infographicvqa_300.jsonl)
- MP-DocVQA contact-sheet n=300: **DOWNLOADED** (Data/manifests/mpdocvqa_contact_300.jsonl; images in data/raw/mpdocvqa_contact/images)

## Implementation status

- Scripts: scripts/download_infographicvqa_hf.py, scripts/download_mpdocvqa_contact.py, src/data/contact_sheet.py
- HF streaming download succeeded for n=300.
- Framing: MP-DocVQA contact-sheet transfer setting only.

## Commit

- TBD

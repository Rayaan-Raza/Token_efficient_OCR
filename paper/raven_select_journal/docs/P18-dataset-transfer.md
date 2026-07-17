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
| InfographicVQA n=300 | 0.3047 | 0.3053 | **0.3167** | **FAIL** |
| MP-DocVQA contact-sheet n=300 | 0.3252 | 0.3244 | **0.3360** | **FAIL** |

On InfographicVQA, the mean gains are +0.0120 over resize
(95% CI [−0.0125, +0.0377]) and +0.0114 over shortest nonempty
([−0.0049, +0.0275]); neither is significant.  The frozen rule routes
227 / 48 / 25 examples to resize / BM25 / LER-BOPS.  Notably, the BM25 reader
alone reaches 0.3713 ANLS and 0.3233 EM, exceeding the frozen selector's
0.3167 / 0.2833.  This is a negative transfer result and no rule is retuned.

MP contact-sheet RAVEN-Select improves by 0.0108 ANLS over resize, but the
paired 95% bootstrap CI crosses zero ([-0.0095, 0.0330]). It improves by
0.0116 over shortest nonempty with a positive CI ([0.0024, 0.0238]).
Under the preregistered transfer gate, shortest-only significance is still
**FAIL**: partial transfer requires significance over resize.

The MP contact-sheet result uses all 300 common examples. EM is 0.2800 for
RAVEN-Select versus 0.2567 for resize and 0.2600 for shortest nonempty.
The frozen rule routes 222 / 47 / 31 examples to resize / BM25 / LER-BOPS.

Artifacts:

- `outputs/metrics/raven_select_raven_select_rule_infographicvqa_n300.json`
- `outputs/gates/P18_dataset_transfer_infographicvqa_n300.json`
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

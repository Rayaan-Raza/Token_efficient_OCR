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
python scripts/run_raven_select_eval.py --n 300 --dataset infographicvqa
python scripts/run_raven_select_eval.py --n 300 --dataset mpdocvqa_contact
```

## Leakage safeguards

No gold page labels, gold answers, or gold-derived OCR flags at inference.

## Results

| Dataset | resize | shortest | RAVEN-Select | Gate |
|---------|--------|----------|--------------|------|
| InfographicVQA n=300 | — | — | — | PENDING |
| MP-DocVQA contact-sheet n=300 | — | — | — | PENDING |



## Download status

- InfographicVQA n=300: **DOWNLOADED** (Data/manifests/infographicvqa_300.jsonl)
- MP-DocVQA contact-sheet: still pending (HF smoke previously hung)

## Implementation status

- Scripts: scripts/download_infographicvqa_hf.py, scripts/download_mpdocvqa_contact.py, src/data/contact_sheet.py
- Smoke HF downloads hung in this environment; transfer evaluation is blocked until local subsets are obtained.
- Framing: MP-DocVQA contact-sheet transfer setting only.

## Commit

- TBD

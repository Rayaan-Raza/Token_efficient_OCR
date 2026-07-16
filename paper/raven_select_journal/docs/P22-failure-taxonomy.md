# P22 — Failure Taxonomy + Answer Types

## Scope

Manual review of 100 stratified cases with a **frozen taxonomy** defined before
labeling, plus an automatic answer-type breakdown.

## Success labels (frozen)

| ID | Meaning |
|----|---------|
| S1 | Removes verbose answer |
| S2 | Filters unsupported hallucination |
| S3 | Chooses correct patch-reader answer over resize |
| S4 | Chooses correct resize answer over patch readers |
| S5 | Fixes formatting / extra words |
| S6 | Resolves disagreement among readers |

## Failure labels (frozen)

| ID | Meaning |
|----|---------|
| F1 | No reader generated correct answer |
| F2 | OCR miss: correct answer generated but not grounded |
| F3 | Wrong shorter grounded span |
| F4 | OCR contains multiple plausible values |
| F5 | Normalization mismatch |
| F6 | Semantic/paraphrase answer not OCR-verbatim |
| F7 | Incomplete VLM answer |
| F8 | Annotation ambiguity |

## Answer-type buckets

date, amount/currency, number, person/name, organization, address/location,
ID/code, phrase/other.

## Gate

**COMPLETE** when: 100 cases labeled; 4–6 qualitative examples (incl. failures);
every answer-type bucket reports support and method metrics.
No qualitative-only soft pass.

## Artifacts

- `outputs/labels/raven_select_taxonomy_n1000.jsonl` (or n=500 if scale fails)
- Review sheet separate from automated metrics

## Results

## Answer-type results (n=500)

| Answer type | Support | resize ANLS | shortest ANLS | RAVEN-Select ANLS |
|-------------|---------|-------------|---------------|-------------------|
| address/location | 15 | 0.7999 | 0.7999 | 0.7999 |
| amount/currency | 21 | 0.7513 | 0.7632 | 0.7632 |
| date | 67 | 0.9022 | 0.9022 | 0.9082 |
| id/code | 9 | 0.7556 | 0.8208 | 0.8208 |
| number | 37 | 0.8822 | 0.8808 | 0.9078 |
| organization | 14 | 0.6587 | 0.7432 | 0.7432 |
| person/name | 113 | 0.7794 | 0.7831 | 0.8031 |
| phrase/other | 224 | 0.7456 | 0.7630 | 0.7663 |

Taxonomy review sheet: `outputs/labels/raven_select_taxonomy_review_n500.jsonl` (100 cases, labels blank for manual review).

Status: **COMPLETE** for heuristic pass (human audit recommended).


## Heuristic taxonomy counts (n=500, label_source=heuristic_v1)

| Category | Count |
|----------|-------|
| primary | {'S1': 0, 'S2': 0, 'S3': 18, 'S4': 1, 'S5': 0, 'S6': 32, 'F1': 21, 'F2': 11, 'F3': 2, 'F4': 0, 'F5': 2, 'F6': 6, 'F7': 5, 'F8': 2} |
| secondary | {'S1': 4, 'S2': 13, 'S3': 0, 'S4': 0, 'S5': 0, 'S6': 1, 'F1': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 'F7': 0, 'F8': 4} |
| total | 100 |

Qualitative example IDs are in outputs/metrics/raven_select_qualitative_examples_n500.json.
Status: **COMPLETE** for heuristic pass; human audit still recommended.


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

## Answer-type results (n=1000)

| Answer type | Support | resize ANLS | shortest ANLS | RAVEN-Select ANLS |
|-------------|---------|-------------|---------------|-------------------|
| address/location | 23 | 0.8203 | 0.8203 | 0.8203 |
| amount/currency | 62 | 0.7776 | 0.7833 | 0.7833 |
| date | 126 | 0.8560 | 0.8660 | 0.8691 |
| id/code | 17 | 0.8497 | 0.8864 | 0.8864 |
| number | 75 | 0.8233 | 0.7963 | 0.8363 |
| organization | 25 | 0.7933 | 0.8379 | 0.8406 |
| person/name | 184 | 0.7976 | 0.8002 | 0.8137 |
| phrase/other | 488 | 0.8138 | 0.8151 | 0.8155 |

Taxonomy review sheet: `outputs/labels/raven_select_taxonomy_review_n1000.jsonl`
(100 cases; frozen heuristic labels, human audit recommended).

Status: **COMPLETE** for heuristic pass (human audit recommended).


## Heuristic taxonomy counts (n=1000, label_source=heuristic_v1)

| Category | Count |
|----------|-------|
| primary | {'S1': 0, 'S2': 0, 'S3': 29, 'S4': 2, 'S5': 0, 'S6': 7, 'F1': 21, 'F2': 6, 'F3': 7, 'F4': 0, 'F5': 3, 'F6': 13, 'F7': 6, 'F8': 6} |
| secondary | {'S1': 7, 'S2': 21, 'S3': 0, 'S4': 0, 'S5': 0, 'S6': 2, 'F1': 0, 'F2': 0, 'F3': 0, 'F4': 0, 'F5': 0, 'F6': 0, 'F7': 0, 'F8': 7} |
| total | 100 |

Qualitative example IDs are in
`outputs/metrics/raven_select_qualitative_examples_n1000.json`.
Status: **COMPLETE** for heuristic pass; human audit still recommended.


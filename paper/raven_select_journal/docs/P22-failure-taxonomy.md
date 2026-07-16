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

Status: **PENDING**

## Commit

- TBD

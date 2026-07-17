# P25 — Journal Paper Audit

## Scope

Rewrite `paper/raven_select_journal/main.tex` into the journal structure once
scaled / robustness / analysis artifacts exist. Audit every number against its
source JSON/CSV.

## Required sections

1. Introduction
2. Related Work
3. Problem Formulation
4. Method: RAVEN-Select
5. Experimental Setup
6. Main Results
7. Robustness Results
8. Ablation Study
9. Analysis (oracle, taxonomy, answer types, cost)
10. Limitations
11. Conclusion

## Non-claims

- No SOTA claim
- Not cheaper than resize
- Contact-sheet ≠ standard MP-DocVQA
- Do not hide failed robustness / transfer

## Status

**COMPLETE** — journal text updated through P24/P18 results and rebuilt from
the audited artifacts.

## Audited in this update

- DocVQA n=1000 P17 **PARTIAL** result; full validation explicitly blocked.
- Qwen2-VL-2B n=500 partial robustness and Tesseract n=500 PASS.
- InfographicVQA and MP contact-sheet n=300 transfer **FAIL** results, including
  the stronger InfographicVQA BM25 reader and no post-result retuning.
- n=1000 oracle headroom, normalization, answer types, heuristic taxonomy,
  qualitative examples, and cost accounting.
- Abstract, setup, results, limitations, conclusion, and reproducibility
  commands reconciled with the scaled artifacts.

## Audit safeguards

- No full-validation result is claimed.
- No SOTA claim is made.
- Three-call selectors are separated from one-call reader operating points.
- The MP result is labeled contact-sheet transfer, not standard MP-DocVQA.
- The 100-case taxonomy is labeled heuristic and human audit is recommended.
- Named normalization variants remain exploratory; method 1.0.0 is unchanged.

## Build

Compiled successfully with bundled Tectonic:

```text
tools/tectonic/tectonic.exe paper/raven_select_journal/main.tex
```

## Commit

- See git history.

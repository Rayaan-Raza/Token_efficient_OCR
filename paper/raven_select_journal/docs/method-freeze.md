# Method Freeze

## Production rule (version 1.0.0)

Defined in `src/answer_selection/method_spec.py` and implemented by
`pick_raven_select_primary` / `pick_ocr_present_shortest`.

1. Readers: resize, BM25, LER-BOPS
2. Normalize answers (conservative lowercase + punctuation + whitespace)
3. OCR-ground if normalized answer appears in page OCR or route patch OCR
4. Return shortest OCR-grounded answer
5. Else return shortest nonempty answer
6. Break ties: resize → BM25 → LER-BOPS

## Non-negotiables

- Do not retune after seeing n=1000 / transfer / second-VLM results
- Learned selectors remain OOF comparators
- Normalization experiments must use distinct variant names if selected post-hoc
- Every result summary should include `method_version` / `method` stamp

## Verification

Re-run production rule on cached DocVQA n=500 after the freeze commit and confirm
ANLS/EM remain at the published baseline (0.8053 / 0.706) within floating error.

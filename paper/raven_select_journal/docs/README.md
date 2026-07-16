# Journal Upgrade Phase Index

Execution order (prespecified):

1. Workspace + docs (this folder)
2. Freeze / version the production method
3. **P17** DocVQA n=1000 (then full validation only on PASS)
4. **P21** Oracle / headroom
5. **P23** Cost analysis
6. **P20** Normalization ablations (production rule unchanged)
7. **P22** Failure taxonomy + answer-type breakdown
8. **P19** Qwen2-VL-2B n=500
9. **P24** EasyOCR vs Tesseract grounding
10. **P18** InfographicVQA + MP-DocVQA contact-sheet transfer
11. **P25** Journal rewrite / audit

## Documents

| Doc | Phase | Gate summary |
|-----|-------|--------------|
| [method-freeze.md](method-freeze.md) | Method lock | Production rule versioned at 1.0.0 |
| [P17-docvqa-scale.md](P17-docvqa-scale.md) | DocVQA scale | PASS / PARTIAL / FAIL vs resize & shortest |
| [P18-dataset-transfer.md](P18-dataset-transfer.md) | Second datasets | FULL / PARTIAL / FAIL transfer per dataset |
| [P19-vlm-robustness.md](P19-vlm-robustness.md) | Second VLM | FULL / PARTIAL / FAIL robustness |
| [P20-normalization.md](P20-normalization.md) | Normalization ablations | COMPLETE when variants run; method frozen |
| [P21-oracle-headroom.md](P21-oracle-headroom.md) | Oracle ceiling | COMPLETE when headroom metrics reconcile |
| [P22-failure-taxonomy.md](P22-failure-taxonomy.md) | Taxonomy + answer types | COMPLETE on 100 labels + type table |
| [P23-cost-analysis.md](P23-cost-analysis.md) | Cost / operating point | COMPLETE when 1-call vs 3-call table exists |
| [P24-ocr-robustness.md](P24-ocr-robustness.md) | OCR engine sensitivity | PASS ≤0.015 / PARTIAL ≤0.03 / FAIL >0.03 drop |
| [P25-paper-audit.md](P25-paper-audit.md) | Camera-ready rewrite | COMPLETE after number audit |

## Decision tree

- P17 n=1000 **PASS** → keep strong scaled claim; run full validation
- P17 **PARTIAL/FAIL** → subset-limited; journal-ready only if transfer/robustness is strong
- P18 full transfer → strengthen cross-dataset claim; both fail → limitation
- P19 partial → model-dependent robustness only
- P24 fail → OCR engine is a deployment limitation
- Strongest journal claim only if **P17 PASS** and at least one transfer/robustness phase fully succeeds

## Commit protocol

One small feature per commit. Include the matching phase-doc update in the
same commit. Do not commit bulky caches, checkpoints, or raw prediction dumps
unless they are compact summary JSON/CSV tables.

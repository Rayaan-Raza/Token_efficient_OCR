# Advisor Review Deck (10 slides)

1. **Problem** — text-rich images exceed visual budgets for OCR and VLMs
2. **Research gap** — compression vs reasoning not unified under fair budgets
3. **Proposed method** — BOPS: low-res overview + K OCR-guided high-res patches
4. **Baselines** — resize, JPEG, WebP, random/uniform tiling, overview-only
5. **Datasets** — TextOCR (OCR, n=200 pilot), DocVQA (QA, n=100 pilot)
6. **OCR results (pilot)** — BOPS `patches_8` word recall **0.245** vs resize `area_0.25` **0.168**; bootstrap CI **[+0.052, +0.102]** — **significant**
7. **VLM results (pilot)** — resize ANLS **0.788** >> BOPS **0.193** ≈ random **0.157**; **not a win** — direction is empirical study
8. **Diagnostics** — answer in full-image OCR **76%** but in selected patches only **~6%** → patch scoring is the bottleneck
9. **Failure analysis** — layout loss, missed answer regions, underutilized byte baselines at kb_500
10. **Next steps** — fix patch selection, re-run VLM pilot; paper-scale (`experiment_stage=paper`) only after VLM gate improves

# BOPS Experiment Results

Living log of dataset audits, preprocessing checks, OCR metrics, VLM QA scores, and method comparisons.

**Last updated:** 2026-07-15  
**Pipeline status:** G3-learned **PASS** (docvqa_500 OOF); **G4 VLM PASS** (docvqa_100 @ K=2); **G5 next** (docvqa_300 @ K=2 — key comparator is **BM25-only**, not just Q-BOPS).  
**Paper title (working):** *Learned Evidence Reranking for Budgeted Document VQA Patch Selection*  
**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU · PyTorch `2.12.1+cu126`  
**Unit tests:** passing (`python -m pytest tests/ -q`)

---

## QE-BOPS coverage gate (G3, docvqa_100, K=2)

**Gate rule:** `qe_bops` must beat Q-BOPS-fair on both `evidence_strict@2` and `evidence_any@2`, beat BOPS/BM25/OCR-confidence/uniform, and have bootstrap mean diff vs Q-BOPS ≥ 0.

| Method | strict@2 | any@2 | ocr_exact@2 | vs Q-BOPS |
|--------|----------|-------|-------------|-----------|
| `bops_qa_fair_pool` (Q-BOPS) | **0.26** | **0.33** | 0.12 | baseline |
| `qe_bops` v2 | 0.23 | 0.32 | 0.13 | **FAIL** (−3pp strict, −1pp any) |
| `qe_bops_safe_expand` | 0.24 | 0.31 | 0.11 | FAIL |
| `qe_bops_anchor_pair` | 0.23 | 0.32 | 0.11 | FAIL (neutral vs v2) |
| `qe_bops_node_pair` v3 | 0.22 | 0.28 | 0.09 | FAIL |
| `qe_bops_table_pair` (margin=0.08) | 0.22 | 0.27 | 0.11 | FAIL |
| `qe_bops_entity_row` (margin=0.05) | 0.23 | 0.28 | 0.11 | FAIL (final K=2 attempt) |
| random (fixed) | 0.05 | 0.07 | 0.03 | — |

**Diagnosis:** Gap is top-2 precision, not pool recall. K=2 structural tuning stopped after entity_row.

**G4 VLM:** Do not run until a QE-BOPS variant clearly beats Q-BOPS on coverage at the chosen K.

---

## K=4 auxiliary coverage (docvqa_100) — FAIL

**Pass rule:** QE strict@4 > Q-BOPS strict@4; QE any@4 ≥ Q-BOPS any@4; beat BM25/BOPS/uniform/OCR-confidence; bootstrap mean diff vs Q-BOPS ≥ 0.

Artifacts: `outputs/metrics/coverage_by_method_k4.csv`, `coverage_per_question_k4.csv`, `paper/tables/coverage_bootstrap_ci_k4.csv`.

| Method | strict@4 | any@4 | ocr_exact@4 | vs Q-BOPS |
|--------|----------|-------|-------------|-----------|
| `bops_qa_fair_pool` (Q-BOPS) | **0.35** | **0.45** | 0.20 | baseline (hits “good” bar alone) |
| `qe_bops_v2` | 0.31 | 0.42 | 0.19 | **FAIL** (−4pp strict, −3pp any) |
| `qe_bops_entity_row` | 0.33 | 0.40 | 0.21 | FAIL |
| `qe_bops_node_pair` | 0.29 | 0.36 | 0.15 | FAIL |
| `qe_bops_table_pair` | 0.25 | 0.30 | 0.14 | FAIL |
| `bm25_only` | 0.27 | 0.39 | 0.13 | — |
| `bops_fair_pool` | 0.23 | 0.27 | 0.12 | — |
| `ocr_confidence_topk` | 0.19 | 0.30 | 0.16 | — |
| `uniform` | 0.03 | 0.13 | 0.00 | — |
| `random` (seeds 0–9) | 0.09 | 0.12 | 0.06 | — |

**Bootstrap (`qe_bops_v2` − Q-BOPS @ K=4):** strict mean_diff=**−0.04** (CI [−0.11, +0.03]); any mean_diff=**−0.03** (CI [−0.12, +0.06]). Mean diffs vs Q-BOPS are **negative** → fail rule 4.

Targets not met for QE: good bar was strict≥0.35 / any≥0.45; strong was ≥0.40 / ≥0.50. Q-BOPS itself sits at the good bar; no QE variant reaches it.

---

## K sweep curves (1,2,3,4,6,8)

Artifact: `outputs/metrics/coverage_by_method_k_sweep.csv`.

| K | Q-BOPS strict/any | QE-v2 strict/any | entity_row strict/any | Notes |
|---|-------------------|------------------|----------------------|-------|
| 1 | 0.14 / 0.20 | 0.14 / 0.20 | 0.14 / 0.20 | tie |
| 2 | **0.26 / 0.33** | 0.23 / 0.32 | 0.23 / 0.28 | Q-BOPS wins |
| 3 | **0.30 / 0.38** | 0.28 / 0.38 | 0.30 / 0.37 | Q-BOPS ≥ |
| 4 | **0.35 / 0.45** | 0.31 / 0.42 | 0.33 / 0.40 | Q-BOPS wins |
| 6 | 0.37 / 0.50 | 0.37 / 0.51 | **0.38** / 0.46 | v2: tie strict, +1pp any; ER: +1pp strict, −4pp any |
| 8 | 0.40 / 0.55 | **0.41** / 0.55 | **0.44** / 0.54 | v2: +1pp strict, tie any; ER: +4pp strict, −1pp any |

**Interpretation (decision tree):**
1. **K=4:** QE does **not** beat Q-BOPS → do **not** reframe the paper around K=4; do **not** run gated VLM @ K=4.
2. **K=8:** Only a marginal v2 edge (+1pp strict, tie any). That is “better deep recall / weak efficiency,” **not** strong enough for the main method claim at a budgeted K.
3. **Neither K=4 nor a clear joint win at K=8** → **heuristic QE-BOPS track closed.** Q-BOPS-fair is the strongest hand-built selector at budgeted K.

### Paper framing (updated)

This is no longer “QE-BOPS heuristic improves evidence selection.” It is:

**A controlled study and learned reranking approach for budgeted evidence selection in Document VQA.**

Contributions:
1. OCR-density BOPS fails because evidence selection is harder than OCR preservation.
2. Q-BOPS is a strong lexical baseline at K=2/K=4.
3. Learned evidence ranker (LightGBM LambdaRank) using Q-BOPS + OCR late-interaction + layout/table/entity features.
4. Evaluate whether learned ranking beats hand-built selectors under fixed K.

**Next:** train learned ranker — debug on `docvqa_100` (OOF), claims on `docvqa_500` (held-out images). **G4 VLM remains blocked** until learned coverage@K beats Q-BOPS on both strict and any **on the paper path (docvqa_500 held-out)**.

---

## Learned evidence ranker — debug OOF (docvqa_100)

**Protocol:** 5-fold image-level OOF LambdaRank; gate on coverage@K (plain top-K, no MMR). **Not for paper claims** — too small; rerun on `docvqa_500` with `--final-train` / `--held-out`.

Artifacts: `outputs/ranker/ranker_dataset_100.parquet`, `outputs/metrics/learned_coverage_by_method.csv`, `outputs/gates/learned_ranker_gate.json`.

| Method | strict@2 | any@2 | strict@4 | any@4 |
|--------|----------|-------|----------|-------|
| `bops_qa_fair_pool` (Q-BOPS) | 0.26 | 0.33 | 0.35 | 0.45 |
| `lgbm_strict` (OOF) | **0.39** | **0.47** | 0.43 | 0.55 |
| `lgbm_any` (OOF) | 0.37 | 0.45 | **0.51** | **0.63** |
| `lgbm_combined` (OOF) | 0.38 | 0.46 | 0.48 | 0.59 |
| `lgbm_qbops_hybrid` (OOF) | **0.42** | **0.52** | 0.45 | 0.57 |
| `logreg_strict` (diagnostic) | 0.13 | 0.14 | 0.21 | 0.22 |
| `qe_bops_v2` | 0.23 | 0.32 | 0.31 | 0.42 |
| `bm25_only` | 0.15 | 0.22 | 0.27 | 0.39 |
| `bops_fair_pool` | 0.12 | 0.15 | 0.23 | 0.27 |

**Debug gate:** all LGBM variants and hybrid **PASS** vs Q-BOPS at K=2 and K=4 on this OOF setup (`G3_learned` PASS for `lgbm_combined`). Bootstrap mean_diff vs Q-BOPS positive.

**Caveats:** (1) n=100 only; (2) learned uses plain top-K while Q-BOPS uses MMR; (3) superseded by docvqa_500 OOF below for paper gate.

---

## Learned evidence ranker — docvqa_500 image-level OOF (paper gate)

**Protocol:** 5-fold CV by `image_id` (every image scored by a model that never trained on it). Lean eval: learned methods + `bops_qa_fair_pool` (secondary baselines skipped after hung `qe_bops_v2` run). Artifacts: `outputs/metrics/learned_coverage_by_method_500.csv`, `coverage_bootstrap_ci_learned.json`, `outputs/gates/learned_ranker_gate.json`.

| Method | strict@2 | any@2 | Δstrict | Δany | strict@4 | any@4 |
|--------|----------|-------|---------|------|----------|-------|
| `bops_qa_fair_pool` (Q-BOPS) | 0.406 | 0.492 | — | — | 0.480 | 0.598 |
| `lgbm_strict` (OOF) | **0.464** | **0.558** | **+0.058** | **+0.066** | 0.534 | 0.656 |
| `lgbm_any` (OOF) | 0.446 | 0.566 | +0.040 | +0.074 | 0.518 | 0.648 |
| `lgbm_combined` (OOF) | 0.452 | 0.552 | +0.046 | +0.060 | **0.540** | **0.662** |
| `lgbm_qbops_hybrid` (OOF) | 0.438 | 0.528 | +0.032 | +0.036 | 0.518 | 0.644 |

**Bootstrap @ K=2 vs Q-BOPS (n=500 pairs):** all four learned variants have mean_diff > 0 and **ci_low > 0** on both strict and any (e.g. `lgbm_strict` strict +0.058 CI [0.020, 0.094]).

**G3_learned:** **PASS** (headline `lgbm_combined` @ K=2: 0.452 / 0.552 vs 0.406 / 0.492). Strong bar (~+0.05) met by **`lgbm_strict`**.

**Winning method for G4 VLM:** `lgbm_strict` (or `learned_lgbm_strict`) at **K=2**. Prefer Option B inference (train on train images / OOF score for that image — never train on the eval image).

---

## G4 VLM — lean pilot (docvqa_100, K=2, OOF lgbm_strict)

**Protocol:** Qwen2.5-VL-3B-Instruct; same pool/OCR/K/overview/prompt; `learned_lgbm_strict` uses **image-level OOF scores** from `oof_scores_strict_500.parquet` (no same-image train). Runner: `scripts/run_g4_vlm_pilot.ps1`. Artifacts: `outputs/metrics/vlm_metrics_docvqa_100_*`, `vlm_metrics_merged.csv`, `outputs/logs/g4_vlm_pilot_20260714T234859Z.log`.

| Method | n | ANLS | EM | mean runtime (s) | total runtime (s) |
|--------|---|------|-----|------------------|-------------------|
| `learned_lgbm_strict` (OOF) | 100 | **0.8286** | **0.72** | 4.12 | 411.7 |
| `bm25_only` | 100 | 0.8240 | 0.71 | 3.59 | 358.8 |
| `resize` (full-image ref) | 100 | 0.7876 | 0.68 | 3.14 | 313.8 |
| `bops_qa_fair_pool` (Q-BOPS) | 100 | 0.7780 | 0.67 | 3.74 | 373.7 |
| `bops_fair_pool` | 100 | 0.7698 | 0.68 | 4.27 | 427.5 |
| `uniform` | 100 | 0.7525 | 0.69 | 3.29 | 329.0 |

**Paired Δ (`lgbm_strict` − Q-BOPS):** ANLS **+0.0506**, EM **+0.05** (9 wins / 87 ties / 4 losses on ANLS).

**G4 gate:** **PASS (strong)** — ANLS > Q-BOPS and EM ≥ Q-BOPS; paired Δ ANLS **+0.0506**, EM **+0.05**.

**Caution (paper-critical):** `lgbm_strict` (0.829) is only **+0.005 ANLS** above `bm25_only` (0.824) on n=100. Learned ranking clearly beats Q-BOPS, resize, BOPS-fair, and uniform, but **BM25-only is the headline baseline for G5**.

**Research arc (updated):**
1. BOPS: OCR preservation ≠ answer-evidence preservation.
2. Q-BOPS: strong lightweight lexical selector.
3. Heuristic QE-BOPS: hand-built rules do not consistently beat Q-BOPS.
4. Learned LambdaRank (OOF): improves strict/any coverage@2 on docvqa_500.
5. **G4:** coverage gains **transfer to VLM** ANLS/EM on docvqa_100 — method is no longer retrieval-only.

**Next (G5):** Run `scripts/run_g5_vlm_pilot.ps1` on **docvqa_300** before scaling to 500. Same fairness protocol (OOF scores, same pool/OCR/K/overview/prompt, cost logging). Bootstrap: `lgbm_strict` vs Q-BOPS, BM25, resize.

---

## G5 VLM — planned (docvqa_300, K=2)

**Gate (minimum):**
- `lgbm_strict` ANLS > Q-BOPS by ≥ **+0.03**
- `lgbm_strict` EM ≥ Q-BOPS
- `lgbm_strict` ANLS ≥ BM25-only (prefer **+0.01** or more)
- Bootstrap mean diff vs Q-BOPS > 0
- Cost table logged

**Strong result:** ANLS +0.04–0.06 over Q-BOPS; ANLS **+0.02+** over BM25; EM +0.03+ over Q-BOPS.

**Methods:** `learned_lgbm_strict`, `bm25_only`, `bops_qa_fair_pool`, `resize`, `bops_fair_pool`, `uniform`.

**Paper claim if G5 holds:** A leakage-safe LightGBM LambdaRank evidence reranker improves answer-evidence selection and VLM performance under fixed K=2, outperforming OCR-density, question-aware, BM25, uniform, and resize baselines on DocVQA.

---

## Pilot summary (2026-07-02)

## Real sanity run (2026-07-01) — n=10, not final

Commands executed:

```bash
python scripts/run_ocr_eval.py \
  --manifest data/manifests/textocr_debug.jsonl \
  --methods original resize jpeg webp bops \
  --budgets area_1.0 area_0.5 area_0.25 area_0.125 kb_500 kb_200 kb_100 kb_50 patches_2 patches_4 patches_8 \
  --limit 10 --experiment-stage sanity --overwrite

for method in resize overview_only random uniform bops; do
  python scripts/run_vlm_eval.py \
    --manifest data/manifests/docvqa_debug.jsonl \
    --method $method --num-patches 2 --limit 10 \
    --experiment-stage sanity --overwrite
done
python scripts/merge_vlm_metrics.py
python scripts/make_paper_assets.py
python scripts/generate_plots.py
python scripts/analyze_failures.py
```

| Track | Output | Rows | dry_run | Notes |
|-------|--------|------|---------|-------|
| OCR | `outputs/metrics/ocr_metrics_textocr_debug_sanity.csv` | 550 (160 applicable) | 0 | EasyOCR GPU; 390 `not_applicable` skipped |
| VLM | `outputs/metrics/vlm_metrics_merged.csv` | 50 (5 methods × 10) | 0 | Per-method CSVs under `outputs/metrics/vlm_metrics_docvqa_debug_*` |
| Paper tables | `paper/tables/*.csv` | 0 eligible | — | Filter requires `experiment_stage` ∈ {pilot, paper} |
| Plot | `outputs/plots/cer_vs_budget.png` | — | — | From sanity OCR CSV (includes sanity rows) |

### OCR sanity — word_recall (primary metric)

Mean word recall over 10 TextOCR samples (`not_applicable=false`, `dry_run=false`):

| method | budget | word_recall |
|--------|--------|-------------|
| original | area_1.0 | 0.317 |
| resize | area_1.0 | 0.317 |
| resize | area_0.5 | 0.269 |
| resize | area_0.25 | 0.177 |
| resize | area_0.125 | 0.102 |
| bops | patches_2 | 0.207 |
| bops | patches_4 | 0.235 |
| bops | patches_8 | 0.313 |

**Sanity pass criteria (primary):**

| Check | Result |
|-------|--------|
| `original` > `resize area_0.25` | ✅ 0.317 > 0.177 |
| `resize area_0.5` > `area_0.25` | ✅ 0.269 > 0.177 |
| OCR non-empty rate | ✅ 98.9% |
| BOPS `invalid_budget` | ✅ false for all patch rows |
| BOPS patches_2 → patches_8 trend | ✅ 0.207 → 0.313 (saturation OK) |

JPEG/WebP byte budgets are valid when `actual_bytes ≤ target` (kb_500 ≈ 0.328, kb_200 ≈ 0.325 word recall on n=10).

### VLM sanity — DocVQA n=10, K=2

Mean scores (`dry_run=false`, `experiment_stage=sanity`):

| method | EM | ANLS | mean runtime (s) |
|--------|-----|------|------------------|
| resize | 0.60 | 0.833 | 5.2 |
| overview_only | 0.20 | 0.329 | 3.9 |
| random | 0.20 | 0.200 | 4.8 |
| uniform | 0.20 | 0.374 | 8.2 |
| bops | 0.20 | 0.200 | 17.7 |

**Sanity pass criteria:**

| Check | Result |
|-------|--------|
| No dry_run rows | ✅ |
| `raw_prediction` / `parsed_prediction` populated | ✅ |
| `resize` not near-zero everywhere | ✅ EM=0.60 |
| `bops` not obviously worse than `random` | ✅ same EM/ANLS on n=10 |
| `overview_only` weaker on detail questions | ✅ observed on sample 3 (ITC vs ITC Limited) |
| Runtime tolerable (n=3 gate) | ✅ ~60–90 s first sample (model load), then 1–3 s/sample |

---

## Pilot run (2026-07-02) — OCR n=200, VLM n=100 — NOT final

Commands executed:

```bash
python scripts/run_ocr_eval.py \
  --manifest Data/manifests/textocr_pilot.jsonl \
  --methods original resize jpeg webp bops \
  --budgets area_1.0 area_0.5 area_0.25 area_0.125 kb_500 kb_200 kb_100 kb_50 patches_2 patches_4 patches_8 \
  --limit 200 --experiment-stage pilot --checkpoint-every 20

for method in resize overview_only random uniform bops; do
  python scripts/run_vlm_eval.py \
    --manifest Data/manifests/docvqa_pilot.jsonl \
    --method $method --num-patches 2 --limit 100 \
    --experiment-stage pilot --overwrite
done
python scripts/merge_vlm_metrics.py
python scripts/make_paper_assets.py
python scripts/generate_plots.py
python scripts/bootstrap_pilot_stats.py
```

| Track | Output | Rows | Paper-eligible | Notes |
|-------|--------|------|----------------|-------|
| OCR | `outputs/metrics/ocr_metrics_textocr_pilot_pilot.csv` | 11,000 (200 imgs) | 3,196 | 7,800 `not_applicable` skipped; resumable checkpoints every 20 |
| VLM | `outputs/metrics/vlm_metrics_merged.csv` | 500 pilot | 500 | 5 methods × 100, K=2 |
| Bootstrap | `paper/tables/bootstrap_ci.csv` | 5 comparisons | — | 95% paired CIs |

### OCR pilot — word_recall (primary metric, n=200)

| method | budget | word_recall | byte_util |
|--------|--------|-------------|-----------|
| original | area_1.0 | 0.280 | — |
| resize | area_0.5 | 0.235 | — |
| resize | area_0.25 | 0.168 | — |
| resize | area_0.125 | 0.105 | — |
| jpeg | kb_500 | 0.280 | 0.39 |
| jpeg | kb_200 | 0.279 | 0.83 |
| webp | kb_500 | 0.278 | 0.34 |
| bops | patches_2 | 0.177 | — |
| bops | patches_4 | 0.209 | — |
| bops | patches_8 | 0.245 | — |

**OCR gate — PASSED:**

| Criterion | Result |
|-----------|--------|
| BOPS `patches_8` > `resize area_0.25` | ✅ 0.245 > 0.168 |
| Word recall increases with patch count | ✅ 0.177 → 0.209 → 0.245 |
| BOPS `patches_8` approaches original | ⚠️ 0.245 vs 0.280 (close, still below) |
| JPEG/WebP rows valid | ✅ `invalid_budget=false`; `byte_utilization` reported |
| `ocr_nonempty` rate | ✅ high |

Note on byte utilization: `kb_500` compressions only use ~0.34–0.39 of the budget (flagged `underutilized_budget`), so they are conservative baselines, not exhausting the allowed size. Reported, not excluded.

### VLM pilot — DocVQA n=100, K=2

| method | EM | ANLS |
|--------|-----|------|
| resize | 0.68 | 0.788 |
| uniform | 0.14 | 0.212 |
| bops | 0.14 | 0.193 |
| overview_only | 0.11 | 0.190 |
| random | 0.11 | 0.157 |

**VLM gate — INCONCLUSIVE:**

| Criterion | Result |
|-----------|--------|
| BOPS > `overview_only` | ⚠️ marginal (0.193 vs 0.190) |
| BOPS >= `random` | ✅ mean (0.193 vs 0.157) but CI crosses 0 |
| BOPS competitive with `uniform` | ❌ below (0.193 vs 0.212) |
| `resize` strong baseline | ✅ dominates (0.788) |

**Patch-selection diagnostics (BOPS):** answer present in full-image OCR **76%** of the time, but in **selected** patch OCR only **~6%**. → BOPS is frequently *selecting the wrong patches* rather than the VLM failing on good patches. This is the concrete next thing to fix before scaling VLM.

### Bootstrap 95% CIs (paired)

| Comparison | Metric | Mean diff | CI | Significant? |
|-----------|--------|-----------|-----|--------------|
| BOPS p8 − resize a0.25 | word_recall | +0.077 | [0.052, 0.102] | ✅ yes (>0) |
| BOPS p8 − original | word_recall | −0.035 | [−0.054, −0.016] | ✅ yes (<0) |
| BOPS − random | ANLS | +0.036 | [−0.004, 0.083] | ❌ crosses 0 |
| BOPS − uniform | ANLS | −0.018 | [−0.088, 0.041] | ❌ crosses 0 |
| BOPS − overview_only | ANLS | +0.003 | [−0.048, 0.054] | ❌ crosses 0 |

### K=4 decision

K=4 sanity (n=10) ran stably (~19 s/sample for BOPS after model load). Paper scope stays **K=2** for the pilot; K=4 is available but not required, given VLM is not yet a win at K=2.

### Honest claim boundaries

- **Can say:** BOPS `patches_8` preserves significantly more OCR word recall than strict resize at low visual budget, and recall scales with patch count.
- **Cannot say:** BOPS improves VLM DocVQA accuracy. It ties random within noise and loses to uniform/resize.
- **Direction:** empirical study. To pursue a method paper, fix patch scoring (selected patches miss the answer 94% of the time) and re-run VLM pilot.

---

## Run log (2026-06-30)

Commands executed:

```bash
python scripts/run_full_experiment.py --phase pilot
python scripts/run_full_experiment.py --phase ablation
python scripts/run_full_experiment.py --phase paper
python scripts/audit_datasets.py
python scripts/make_paper_assets.py
python scripts/generate_plots.py
```

| Phase | OCR rows | VLM rows | Plot | Audit |
|-------|----------|----------|------|-------|
| pilot | 300 | — | ✅ | — |
| ablation | 15 (BOPS patches_4 × 3) | 50 (patch sweep 0/2/4/8/12) | ✅ | — |
| paper | 1,250 (50 samples × 5 methods × 5 budgets) | 10 (bops @ 4 patches, last write) | ✅ | ✅ |

**Outputs:**
- `outputs/metrics/ocr_metrics.csv` — 1,250 rows
- `outputs/metrics/vlm_metrics.csv` — 10 rows (last paper-phase method)
- `outputs/plots/cer_vs_budget.png`
- `paper/tables/table_ocr_budget.csv`
- `paper/tables/table_vlm_patches.csv`

---

## 1. Experiment design

### Methods compared

| Method | Type | Budget axis | Task |
|--------|------|-------------|------|
| `original` | Baseline | area / byte | OCR (TextOCR) |
| `resize` | Baseline | area ratio (±3% pixels) | OCR + VLM |
| `jpeg` | Baseline | byte target (`actual ≤ target`) | OCR |
| `webp` | Baseline | byte target (`actual ≤ target`) | OCR |
| `bops` | Proposed | patches + overview | OCR + VLM |
| `overview_only` | Ablation | 0 patches | VLM |
| `random` | Ablation | K patches | VLM |
| `uniform` | Ablation | K patches | VLM |

### Datasets & manifests

| Manifest | Samples | Purpose |
|----------|---------|---------|
| `data/manifests/textocr_debug.jsonl` | 50 | Smoke / debug OCR |
| `data/manifests/textocr_pilot.jsonl` | 200 | Pilot OCR curves |
| `data/manifests/docvqa_debug.jsonl` | 20 | Smoke / debug VLM |
| `data/manifests/docvqa_pilot.jsonl` | 100 | Pilot VLM + ablations |
| `data/manifests/docvqa_val_500.jsonl` | 500 | Full DocVQA validation export |

### Metrics

| Metric | Definition | Used for |
|--------|------------|----------|
| **CER** | Character error rate | OCR |
| **WER** | Word error rate | OCR |
| **Word recall (v1)** | matched normalized GT tokens / total GT tokens | OCR |
| **Exact Match (EM)** | Normalized string match to any reference answer | DocVQA |
| **ANLS** | Average normalized Levenshtein similarity | DocVQA |
| **invalid_budget** | Row excluded if pixel ±3%, byte over target, or patch count ≠ target | Fairness filter |
| **byte_utilization** | `actual_bytes / target_bytes` for compression baselines | Reporting |
| **underutilized_budget** | `byte_utilization < 0.70` — reported, not excluded | Reporting |
| **not_applicable** | Method does not use this budget axis (e.g. `jpeg` + `area_0.25`) | Compatibility filter |
| **experiment_stage** | `debug`, `sanity`, `pilot`, `paper` — paper tables use pilot/paper only | Run tagging |

---

## 2. Dataset audit (Phase 2A)

**Gate:** passed  
**Report:** `outputs/audit/dataset_audit_report.json`

| Dataset | Check | Result |
|---------|-------|--------|
| TextOCR | Total images | 21,778 |
| TextOCR | Missing on disk | 0 |
| TextOCR | Sample checks (n=10) | All pass |
| DocVQA | Manifest rows | 500 |
| DocVQA | PNG files on disk | 500 |
| DocVQA | Duplicate IDs | 0 |

---

## 3. Preprocessing smoke test (Phase 1)

```bash
python scripts/run_preprocessing.py --config configs/smoke_test.yaml
```

| Method | Area ratio | Target pixels | Actual pixels | invalid_budget |
|--------|------------|---------------|---------------|----------------|
| resize | 0.5 | 1,990,674 | 1,988,965 | false |

---

## 4. OCR results (TextOCR)

| Run | CSV | Rows | Paper-eligible |
|-----|-----|------|----------------|
| Sanity n=10 | `outputs/metrics/ocr_metrics_textocr_debug_sanity.csv` | 550 (160 applicable) | 0 (`experiment_stage=sanity`) |
| **Pilot n=200** | `outputs/metrics/ocr_metrics_textocr_pilot_pilot.csv` | 11,000 | **3,196** |
| Legacy dry-run | `outputs/metrics/ocr_metrics.csv` | 1,250 | 0 |

**Plot:** `outputs/plots/cer_vs_budget.png`  
**Paper table:** `paper/tables/table_ocr_budget.csv`

Real OCR uses **EasyOCR** (GPU when CUDA available). BOPS runs merge overview + per-patch OCR via `merge_patch_ocr`. See **Pilot run** section above for gate verdict and numbers.

**OCR gate: PASSED** — BOPS `patches_8` significantly beats resize `area_0.25` (bootstrap CI excludes 0).

---

## 5. VLM results (DocVQA)

| Run | CSV | Rows | Paper-eligible |
|-----|-----|------|----------------|
| Sanity n=10 | `outputs/metrics/vlm_metrics_merged.csv` | 50 | 0 |
| **Pilot n=100** | `outputs/metrics/vlm_metrics_merged.csv` | 500 pilot | **500** |
| Per-method | `outputs/metrics/vlm_metrics_docvqa_pilot_{method}_*.csv` | 100 each | — |

**Paper table:** `paper/tables/table_vlm_patches.csv`  
**Bootstrap:** `paper/tables/bootstrap_ci.csv`

Real VLM uses **Qwen2.5-VL-3B-Instruct** (4-bit) on RTX 3050. See **Pilot run** section above for gate verdict and numbers.

**VLM gate: INCONCLUSIVE** — BOPS ≈ random (CI crosses 0); loses to uniform and resize. Patch diagnostics show answer in selected patches only ~6% of the time.

**4 GB VRAM tips:** Close GPU-heavy apps before eval; use `--num-patches 2` for BOPS.

---

## 6. Failure analysis

**Latest:** `outputs/failure_cases/vlm_failures.csv` — 440 VLM failures from pilot (EM<1 cases across n=100 × 5 methods).  
Key pattern: for BOPS, the answer is in full-image OCR ~76% of the time but in the selected patches only ~6% — patch selection is the main failure mode, not the VLM.

---

## 7. Budget fairness

| Budget type | Validity rule | Notes |
|-------------|---------------|-------|
| Pixel (area) | within ±3% of target | `resize`, `original` |
| Byte (JPEG/WebP) | `actual_bytes <= target_bytes` | plus `byte_utilization`; `underutilized_budget` if <0.70 (reported, not excluded) |
| Patch count | exact match | BOPS |

Invalid rows are excluded from `paper/tables/table_ocr_budget.csv`.

---

## 8. Phase checklist

| Phase | Status |
|-------|--------|
| 1 — Smoke test | ✅ |
| 2A — Dataset audit | ✅ |
| 2B — Manifests | ✅ |
| 3–8 — OCR pipeline (dry-run) | ✅ |
| 9–10 — VLM pipeline (dry-run) | ✅ |
| 11 — Ablations (dry-run) | ✅ |
| 13–16 — Paper tables + plot | ✅ |
| Real OCR inference (sanity n=10) | ✅ EasyOCR GPU |
| Real Qwen inference (sanity n=10) | ✅ Qwen 4-bit on RTX 3050 |
| Pilot-scale OCR (n=200) | ✅ gate passed |
| Pilot-scale VLM (n=100, K=2) | ✅ complete — gate inconclusive |
| Bootstrap CIs | ✅ `paper/tables/bootstrap_ci.csv` |
| Patch-selection fix + VLM re-pilot | ⏳ next |
| Paper-scale runs | ⏳ needs `experiment_stage=paper` |

---

## 9. How to refresh this file

```bash
# OCR pilot (resumable)
python scripts/run_ocr_eval.py \
  --manifest data/manifests/textocr_pilot.jsonl \
  --methods original resize jpeg webp bops \
  --budgets area_1.0 area_0.5 area_0.25 area_0.125 kb_500 kb_200 kb_100 kb_50 patches_2 patches_4 patches_8 \
  --limit 200 --experiment-stage pilot --checkpoint-every 20

# VLM pilot (per-method CSVs)
for method in resize overview_only random uniform bops; do
  python scripts/run_vlm_eval.py \
    --manifest data/manifests/docvqa_pilot.jsonl \
    --method $method --num-patches 2 --limit 100 \
    --experiment-stage pilot --overwrite
done

python scripts/merge_vlm_metrics.py
python scripts/make_paper_assets.py
python scripts/generate_plots.py
python scripts/bootstrap_pilot_stats.py
python scripts/analyze_failures.py
```

Update tables from `outputs/metrics/*.csv` and `paper/tables/*.csv`.

---

## 10. Qwen local inference

**Status:** Verified on RTX 3050 Laptop GPU (4 GB VRAM).

### One-time GPU setup (this machine)

PyTorch was CPU-only (`2.10.0+cpu`). Reinstalled with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install accelerate bitsandbytes transformers
```

Verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 3050 Laptop GPU
```

### How inference works (`src/vlm/run_vlm.py`)

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Quantization | 4-bit via `BitsAndBytesConfig` |
| Device | `device_map="auto"` (CUDA) |
| Image cap | Longest side ≤ 768 px (4 GB VRAM) |

**CLI (no `--dry-run`):**

```bash
python scripts/run_vlm_eval.py \
  --manifest data/manifests/docvqa_debug.jsonl \
  --method resize --limit 2
```

First run downloads ~7 GB of model weights from Hugging Face, then caches locally.

**What `--dry-run` does:** Skips `load_vlm()` and writes `prediction = "dry-run"`.

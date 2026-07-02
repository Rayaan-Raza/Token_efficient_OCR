# BOPS Experiment Results

Living log of dataset audits, preprocessing checks, OCR metrics, VLM QA scores, and method comparisons.

**Last updated:** 2026-07-02  
**Pipeline status:** Pilot complete — OCR n=200 (TextOCR) + VLM n=100 (DocVQA), real EasyOCR GPU + real Qwen. **Pilot results, not final paper results.**  
**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU · PyTorch `2.12.1+cu126`  
**Unit tests:** 31/31 passing (`python -m pytest tests/ -q`)

> **Pilot verdict:** OCR gate **passed with statistical significance** (BOPS `patches_8` beats strict resize, bootstrap CI excludes 0). VLM is **inconclusive** — BOPS beats random on the mean but the CI crosses 0, and BOPS does not beat `uniform` or `resize`. Direction is an **empirical study** (clear OCR gains, VLM not a win) unless patch scoring is improved. Do **not** claim "BOPS improves VLM performance."

---

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

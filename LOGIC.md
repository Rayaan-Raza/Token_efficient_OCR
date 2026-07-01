# BOPS — Architectural & Logic Reference

Single source of truth for **how the system is designed**, **why decisions were made**, and **how data flows** through the pipeline. For experiment numbers see [RESULTS.md](RESULTS.md). For setup commands see [README.md](README.md).

---

## 1. Problem & research goal

**Problem:** Text-rich document images (scans, photos, slides) exceed the visual budget that downstream models (OCR, VLMs) can process at full resolution.

**Goal:** Compare preprocessing strategies **under equal declared budgets** and show that **BOPS** (Budget-Aware OCR-Guided Overview-Plus-Patch Selection) preserves more useful text signal than naive resize or compression.

**Two evaluation tracks:**

| Track | Dataset | Task | Downstream | Metrics |
|-------|---------|------|------------|---------|
| OCR | TextOCR | Read all text | PaddleOCR / EasyOCR | CER, WER, word recall v1 |
| VLM QA | DocVQA | Answer questions | Qwen2.5-VL-3B | Exact Match, ANLS |

**Core invariant:** Every method must declare and hit a budget (pixels, bytes, or patch count). Rows that miss the budget are flagged `invalid_budget=true` and **excluded from aggregates**.

---

## 2. High-level architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         scripts/ (CLI entry points)                      │
│  audit_datasets · run_preprocessing · run_ocr_eval · run_vlm_eval       │
│  run_full_experiment · make_paper_assets · generate_plots               │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│  src/data/    │           │ src/preproc/  │           │ src/ocr/ vlm/ │
│  manifests    │           │ baselines     │           │ inference     │
│  validation   │           │ BOPS          │           │ metrics       │
└───────┬───────┘           └───────┬───────┘           └───────┬───────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
                          ┌─────────────────┐
                          │  src/utils/     │
                          │  paths · budget │
                          │  image_io       │
                          └────────┬────────┘
                                   ▼
                          outputs/ · paper/tables/
```

**Design principles:**

1. **Library vs scripts** — Reusable logic lives in `src/`; `scripts/` only parse args and wire I/O.
2. **Manifest-driven** — All experiments iterate JSONL manifests; no hard-coded image lists in eval code.
3. **Single metric definitions** — OCR metrics only in `ocr_metrics.py`; QA metrics only in `qa_metrics.py`.
4. **Budget as first-class metadata** — Every transform attaches `budget_type`, `budget_target`, `budget_actual`, `invalid_budget`.
5. **Lazy model loading** — OCR backend and VLM are singletons initialized on first use.

---

## 3. Repository layout (logical)

| Path | Role |
|------|------|
| `data/manifests/*.jsonl` | Unified experiment index (one row = one sample) |
| `data/train_val_images/` | TextOCR JPEGs |
| `data/raw/docvqa_hf/images/` | DocVQA PNG exports |
| `data/raw/textocr/` | Converted TextOCR JSON + index files |
| `configs/*.yaml` | Smoke tests and experiment parameters |
| `outputs/metrics/*.csv` | Per-run metric tables |
| `outputs/ocr_results/` | Transformed images from OCR eval |
| `outputs/plots/` | Generated figures |
| `outputs/audit/` | Dataset integrity reports |
| `paper/tables/` | Aggregated CSVs for the paper |
| `src/utils/paths.py` | All path resolution (`data/` vs `Data/` on Windows) |

**Path helpers:**

- `repo_path(*parts)` → under project root
- `data_path(*parts)` → `data/` with fallback to `Data/`
- `outputs_path(*parts)` → `outputs/` (creates parent dirs; avoids treating filenames as directories)

---

## 4. Manifest schema (unified JSONL)

Every experiment row is one JSON object per line. **Required fields** (`validate_manifest.py`):

| Field | Type | TextOCR | DocVQA |
|-------|------|---------|--------|
| `image_id` | str, unique | `textocr_{id}` | `docvqa_val_{id}` |
| `dataset` | str | `"TextOCR"` | `"DocVQA"` |
| `split` | str | `train` / `val` | `val` |
| `image_path` | str | repo-relative path to JPEG | repo-relative path to PNG |
| `ocr_gt_text` | str | concatenated GT text | `""` |
| `question` | str | `""` | DocVQA question |
| `answer` | list[str] | `[]` | acceptable answers |
| `answer_type` | str / list | optional | DocVQA type |
| `metadata` | dict | width, height, docId, … | |

**Builders:**

- `build_textocr_manifest.py` — reads `textocr_imgs_index.json` + `textocr_img_text.json`
- `build_docvqa_manifest.py` — samples from `docvqa_val_500.jsonl`
- `convert_textocr_annotations.py` — one-time `.txt` → JSON + indices (Phase 2A)

**Loader:** `iter_manifest(path)` streams rows; `load_manifest(path)` loads all.

---

## 5. Budget model (fairness enforcement)

Implemented in `src/utils/budget_check.py`.

| Budget type | Used by | Target | Tolerance | `invalid_budget` when |
|-------------|---------|--------|-----------|------------------------|
| **pixel** | `resize`, `original` (implicit) | `area_ratio × original_pixels` | ±3% | `\|actual − target\| / target > 0.03` |
| **byte** | `jpeg`, `webp` | e.g. `kb_200` → 204,800 bytes | at-or-under target | `actual_bytes > target_bytes`; report `byte_utilization` and `underutilized_budget` if utilization < 0.70 (not auto-excluded) |
| **patch** | BOPS | exact K patches | **0%** (exact) | `actual_patches ≠ target_patches` |

**Budget token syntax** (CLI `--budgets`):

- `area_0.5` → 50% of original pixel area
- `area_0.25` → 25%
- `kb_200` → 200 KiB encoded file size
- `patches_4` → exactly 4 high-res patches (+ overview, not counted in patch budget check for OCR path)

**Aggregation rule:** `make_paper_assets.py`, `plot_budget_curves.py`, and `budget_metrics.py` filter with:

```python
df[df["invalid_budget"].fillna(False).astype(bool) == False]
```

---

## 6. Preprocessing methods

### 6.1 Baselines

| Method | Module | Mechanism | Budget axis |
|--------|--------|-----------|-------------|
| `original` | `run_ocr_eval.apply_method` | Save full-res PNG | none (reference) |
| `resize` | `resize.resize_to_area_ratio` | Uniform scale √ratio on W,H | pixel (area ratio) |
| `jpeg` | `compression.compress_image_to_file` | Binary search JPEG quality | byte |
| `webp` | same | Binary search WebP quality | byte |

**Resize math:** `scale = sqrt(area_ratio)`; new W,H = round(old × scale). Pixel budget checked via `check_pixel_budget`.

**Compression math:** Binary search quality ∈ [5, 95] until `len(bytes) ≤ target`; if impossible, use min quality.

### 6.2 BOPS (proposed method)

**Entry:** `preprocessing/bops.py` → `run_bops(image, num_patches, mode, ...)`

**Pipeline steps:**

```
Original image
    │
    ├─► [1] generate_overview()     → low-res overview (~50k pixels default)
    │
    ├─► [2] generate_grid_patches() → sliding window (256×256, stride 128)
    │
    ├─► [3] select patches by mode:
    │       • ocr_guided → score + NMS
    │       • random     → uniform sample
    │       • uniform    → evenly spaced grid indices
    │       • overview_only → K=0 patches
    │
    ├─► [4] crop_patch() × K        → high-res patch PIL images
    │
    └─► [5] check_patch_budget(K)   → metadata.invalid_budget
```

**Default hyperparameters:**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `overview_target_pixels` | 50,000 | Overview area cap |
| `patch_size` | 256 | Square patch side |
| `stride` | 128 | Grid step (50% overlap) |
| `nms iou_threshold` | 0.5 | Overlap suppression |

### 6.3 OCR-guided patch scoring

`patch_scoring.py` — weighted sum of four signals:

| Signal | Weight | Source |
|--------|--------|--------|
| `text_coverage` | 0.40 | Fraction of OCR box area inside patch |
| `text_confidence` | 0.30 | Mean OCR confidence for overlapping boxes |
| `edge_density` | 0.15 | Mean gradient magnitude in patch (grayscale) |
| `entropy` | 0.15 | Normalized histogram entropy in patch |

OCR boxes come from `run_ocr_with_boxes()` on a temp full-image PNG (PaddleOCR preferred, EasyOCR fallback).

**NMS** (`patch_nms.py`): Greedy by score; suppress candidates with IoU ≥ 0.5 to any kept patch; stop at `top_k = K`.

---

## 7. OCR subsystem

### 7.1 Backend selection (`ocr/run_ocr.py`)

```
get_ocr():
    try PaddleOCR (lang=en, angle cls)
    except → try EasyOCR (en, gpu=False)
    except → RuntimeError
```

- **PaddleOCR:** May fail on Python 3.14 / missing PaddlePaddle.
- **EasyOCR:** Portable fallback; CPU-only in current code (`gpu=False`).

### 7.2 APIs

| Function | Returns | Used for |
|----------|---------|----------|
| `run_ocr_on_image(path)` | concatenated line text | OCR eval on transformed image |
| `run_ocr_with_boxes(path)` | `[{box, text, confidence}, …]` | BOPS patch scoring |

### 7.3 OCR eval flow (`scripts/run_ocr_eval.py`)

```
for each manifest row:
  for each method in --methods:
    for each budget in --budgets:
      1. load_image(image_path)
      2. apply_method() → transformed path + meta
      3. run_ocr_on_image() OR empty if --dry-run
      4. cer(), wer(), word_recall() vs ocr_gt_text
      5. append row to ocr_metrics.csv
```

**BOPS quirk in OCR eval:** Saves **overview only** to disk for OCR (not patches). Patches are used in VLM track. `--dry-run` uses `mode="random"` for BOPS to skip OCR-for-scoring.

### 7.4 Metrics (`ocr/ocr_metrics.py`) — canonical

**Normalization** (`normalize_text.py`): lowercase → strip punctuation → collapse whitespace.

| Metric | Definition |
|--------|------------|
| **CER** | `jiwer.cer(gt, pred)` on normalized strings |
| **WER** | `jiwer.wer(gt, pred)` on normalized strings |
| **Word recall v1** | `matched_gt_tokens / total_gt_tokens` — each GT token matches at most one identical pred token (multiset / count-aware) |

Do not reimplement these elsewhere.

---

## 8. VLM subsystem

### 8.1 Model (`vlm/run_vlm.py`)

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Quantization | 4-bit `BitsAndBytesConfig` |
| Device | `device_map="auto"` (CUDA) |
| Dtype | float16 |
| Image cap | longest side ≤ 768 px (`_resize_for_vlm`) |
| Max new tokens | 64 |
| Caching | Global `_model`, `_processor` singletons |

### 8.2 Prompt layout (`vlm/prompt_templates.py`)

**Single image** (`resize`, `overview_only`):
```
[image] + "Question: {q}\nAnswer:"
```

**Overview + patches** (`bops`, `random`, `uniform`):
```
[overview, patch_1, patch_2, …] + multi-image instruction + question
```

Image order in the tensor **must** match prompt semantics: overview first, then patches.

### 8.3 Answer parsing (`vlm/parse_answers.py`)

Strips chat artifacts:
1. Text after last `assistant` marker (case-insensitive)
2. Text after `Answer:` / `answer:`
3. Whitespace normalization

### 8.4 VLM eval flow (`scripts/run_vlm_eval.py`)

| `--method` | Preprocessing | VLM call |
|------------|---------------|----------|
| `resize` | `resize_to_area_ratio(0.25)` | `run_vlm_single(resized, q)` |
| `overview_only` | `run_bops(K=0, overview_only)` | `run_vlm_single(overview, q)` |
| `random` | `run_bops(K, random)` | `run_vlm_overview_patches(overview, patches, q)` |
| `uniform` | `run_bops(K, uniform)` | same |
| `bops` | `run_bops(K, ocr_guided)` | same |

**`--dry-run`:** Skips model load; `prediction = "dry-run"`.

### 8.5 QA metrics (`vlm/qa_metrics.py`)

| Metric | Definition |
|--------|------------|
| **Exact Match** | 1.0 if normalized pred equals any normalized reference answer |
| **ANLS** | Best over references: `1 − lev_dist/max_len` if ratio < 0.5 threshold else 0 |

Normalization: same style as OCR (lowercase, no punctuation, collapsed spaces).

---

## 9. Experiment orchestration

### 9.1 Phase gates (`run_full_experiment.py`)

| Phase | OCR | VLM | Notes |
|-------|-----|-----|-------|
| `debug` | 5 samples, 2 methods | 2 samples | Smoke |
| `pilot` | 20 samples, 5 methods × 3 budgets | — | OCR curves |
| `ablation` | BOPS × 3 repeats | patch K ∈ {0,2,4,8,12} | |
| `paper` | 50 samples, full budget grid | 5 methods × 10 samples | + failure analysis + tables |

**Current limitation:** All phases pass `--dry-run` to OCR/VLM subprocesses. Remove for real inference.

**Current limitation:** Each `run_vlm_eval.py` invocation **overwrites** `vlm_metrics.csv` — multi-method comparisons require append logic or separate output files.

### 9.2 Downstream analysis

```
ocr_metrics.csv / vlm_metrics.csv
        │
        ├─► make_paper_assets.py → paper/tables/table_*.csv
        ├─► generate_plots.py     → outputs/plots/cer_vs_budget.png
        └─► analyze_failures.py   → outputs/failure_cases/
```

### 9.3 Statistical tests (`metrics/statistical_tests.py`)

`bootstrap_ci(diffs, n_boot=1000)` — paired difference confidence intervals. Holm / McNemar planned but not wired into scripts yet.

---

## 10. Data acquisition logic

### TextOCR

1. Source: `TextOCR_0.1_train.txt` (~280 MB JSON misnamed as `.txt`)
2. `convert_textocr_annotations.py` → `TextOCR_0.1_train.json` + indices
3. Images: `data/train_val_images/train_images/{id}.jpg`
4. `audit_datasets.py` — gate: ≤1% missing, sample checks pass

### DocVQA

1. `download_docvqa_hf.py` — **streaming** `validation[:500]` (avoids full ~9.5 GB download)
2. Exports PNGs + `docvqa_val_500.jsonl`
3. Subsets: `docvqa_debug` (20), `docvqa_pilot` (100)

---

## 11. Config YAML pattern

Example `configs/smoke_test.yaml`:

```yaml
input_image: data/raw/docvqa_hf/images/docvqa_val_49153.png
output_dir: transformed_images
metadata_csv: metrics/smoke_test_metadata.csv
area_ratio: 0.5
```

`run_preprocessing.py` loads YAML via `config.py`, resizes one image, writes metadata CSV — validates paths and I/O without full experiment cost.

---

## 12. Module dependency graph

```mermaid
flowchart TB
    subgraph scripts
        ROE[run_ocr_eval]
        RVE[run_vlm_eval]
        RFE[run_full_experiment]
    end

    subgraph data
        DL[dataset_loader]
        VM[validate_manifest]
    end

    subgraph preprocessing
        RS[resize]
        CP[compression]
        OV[overview]
        PG[patch_grid]
        PS[patch_scoring]
        NMS[patch_nms]
        BOPS[bops]
    end

    subgraph ocr
        OCR[run_ocr]
        OM[ocr_metrics]
    end

    subgraph vlm
        VLM[run_vlm]
        QM[qa_metrics]
        PT[prompt_templates]
    end

    subgraph utils
        PATHS[paths]
        BC[budget_check]
        IO[image_io]
    end

    ROE --> DL --> IO
    ROE --> RS --> BC
    ROE --> CP --> BC
    ROE --> BOPS --> OV
    BOPS --> PG --> PS --> NMS
    BOPS --> OCR
    ROE --> OCR --> OM

    RVE --> DL
    RVE --> BOPS
    RVE --> RS
    RVE --> VLM --> PT
    RVE --> QM

    RFE --> ROE
    RFE --> RVE
```

---

## 13. Key design decisions & tradeoffs

| Decision | Rationale | Tradeoff |
|----------|-----------|----------|
| Overview + K patches (not K full frames) | Separates global layout from local text | VLM must handle multi-image input; VRAM scales with K |
| OCR guides patch **selection**, not VLM text | Selection is cheap; VLM reads selected regions | OCR errors can mis-rank patches |
| OCR eval uses overview-only save for BOPS | Simplifies single-image OCR harness | OCR track doesn't test patch-merge OCR yet (`merge_patch_ocr.py` exists but unused in eval) |
| Exact patch budget (0% tolerance) | Prevents "BOPS with 3 patches vs 4" unfairness | Grid edge cases may yield fewer valid candidates |
| Word recall v1 (not F1) | Matches proposal; count-aware token match | May differ from standard IR recall |
| Qwen2.5-VL-3B 4-bit | Fits 4–8 GB consumer GPUs | Slower / lower quality than full precision |
| JSONL manifests | Streamable, git-friendly metadata | No schema enforcement beyond `validate_manifest` |
| `data/` vs `Data/` fallback | Windows case-insensitivity | Two possible paths; always use `data_path()` |

---

## 14. Known gaps & extension points

| Gap | Where to fix |
|-----|--------------|
| `run_full_experiment.py` always `--dry-run` | Add `--real` flag or invert default |
| VLM CSV overwrite | Append mode or `{method}_{timestamp}.csv` in `run_vlm_eval.py` |
| EasyOCR `gpu=False` hardcoded | `run_ocr.py` — enable GPU when CUDA available |
| BOPS OCR eval saves overview only | Wire `merge_patch_ocr.py` for patch-level OCR fusion |
| Paper-scale manifests (500–1000) | `build_*_manifest.py --limit` |
| Statistical tests not in CI | Script calling `bootstrap_ci` on paired method diffs |
| `invalid_budget` on resize+byte cross combos | `apply_method` raises/skips incompatible method×budget pairs (jpeg+area → n=0 in tables) |

---

## 15. File → responsibility quick index

| File | Responsibility |
|------|----------------|
| `src/preprocessing/bops.py` | Full BOPS pipeline orchestration |
| `src/preprocessing/patch_scoring.py` | OCR-guided patch importance |
| `src/preprocessing/patch_nms.py` | Overlap suppression |
| `src/preprocessing/patch_grid.py` | `Patch` dataclass, grid, crop |
| `src/preprocessing/overview.py` | Low-res global image |
| `src/preprocessing/resize.py` | Area-ratio baseline |
| `src/preprocessing/compression.py` | JPEG/WebP byte baseline |
| `src/utils/budget_check.py` | Budget fairness |
| `src/utils/paths.py` | Filesystem layout |
| `src/utils/image_io.py` | `load_image`, `write_metadata_csv` |
| `src/ocr/run_ocr.py` | OCR backends |
| `src/ocr/ocr_metrics.py` | CER, WER, word recall v1 |
| `src/vlm/run_vlm.py` | Qwen load + generate |
| `src/vlm/qa_metrics.py` | EM, ANLS |
| `src/data/validate_manifest.py` | Manifest gate |
| `src/data/dataset_loader.py` | JSONL iteration |
| `src/metrics/budget_metrics.py` | Valid-budget aggregation |
| `src/visualization/plot_budget_curves.py` | CER vs budget plot |
| `scripts/run_ocr_eval.py` | OCR experiment loop |
| `scripts/run_vlm_eval.py` | VLM experiment loop |
| `scripts/run_full_experiment.py` | Phase orchestration |
| `scripts/audit_datasets.py` | Phase 2A gate |
| `scripts/make_paper_assets.py` | Table aggregation |

---

## 16. Mental model (one paragraph)

A **manifest row** points to an image and ground truth. **Preprocessing** transforms the image under a **declared budget** and records whether the budget was met. For **OCR**, a single transformed image is read by PaddleOCR/EasyOCR and scored against `ocr_gt_text`. For **VLM**, the image is transformed into either one downscaled frame or a **BOPS bundle** (overview + K patches), fed to Qwen with a fixed prompt template, and scored against DocVQA answer lists. **Aggregates and plots** never include `invalid_budget` rows. The **research claim** is that at equal budget, BOPS preserves more semantically useful visual information than resize/compression/random tiling — measured by OCR degradation curves and DocVQA accuracy vs patch count.

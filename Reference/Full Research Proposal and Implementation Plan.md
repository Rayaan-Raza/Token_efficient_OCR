## Full Research Proposal and Implementation Plan 

Budget-Aware OCR-Guided Patch Selection for Text-Rich Image Understanding 

## 1. Executive Summary 

This project studies how to preprocess text-rich images before sending them to OCR engines or Vision-Language Models under strict visual budgets. 

Text-rich images include: 

- scanned documents, 

- screenshots, 

- slides, 

- receipts, 

- posters, 

- infographics, 

- forms, 

- tables, 

- whiteboard photos, 

- scene-text images. 

The problem is simple: when these images are resized or compressed, small text and layout cues often become unreadable. This damages both OCR accuracy and downstream VLM question-answering performance. 

The proposed solution is a lightweight, training-free method: 

## OCR-Guided Overview-Plus-Patch Selection 

The method creates: 

1. one low-resolution overview of the full image, and 

2. selected high-resolution patches from important text/layout regions. 

The overview preserves global context. The patches preserve small text. 

The project compares this method against simple and strong baselines under equal budgets. 

The paper’s contribution is not a new OCR model or a new VLM. The contribution is a reproducible budget-normalized evaluation and a simple preprocessing method for textrich multimodal inputs. 

## 2. Proposed Paper Title 

Main Title 

## **Budget-Aware OCR-Guided Patch Selection for Text-Rich Image Understanding** 

## Alternative Titles 

1. **OCR-Guided Overview-Plus-Patch Selection for Token-Efficient Multimodal Understanding** 

2. **Preserving Text Under Visual Budgets: A Training-Free Preprocessing Strategy for Vision-Language Models** 

3. **Budget-Normalized Preprocessing for OCR and VLM Understanding of Text-Rich Images** 

Recommended title: 

## **Budget-Aware OCR-Guided Patch Selection for Text-Rich Image Understanding** 

It is short, understandable, and defensible. 

## 3. Problem Statement 

Vision-Language Models and OCR systems are increasingly used on images containing dense text and layout structure. However, many real-world inputs are too large, too detailed, or too expensive to process directly. 

Common preprocessing methods such as resizing and compression reduce the image cost, but they often damage: 

- small text, 

- thin strokes, 

- table lines, 

- form structure, 

- reading order, 

- spatial relationships, 

- layout context. 

This creates a tradeoff: 

- high-resolution input improves readability but costs more, 

- low-resolution input saves cost but loses text and layout details. 

The research problem is: 

## **How can we reduce the visual cost of text-rich images while preserving OCR accuracy and VLM question-answering performance?** 

## 4. Research Gap 

Existing work addresses parts of this problem, but usually not in a unified way. 

## Gap 1: Standard compression is not optimized for machine reading 

JPEG/WebP are designed mainly for visual quality and file size, not OCR accuracy or VLM reasoning. 

## Gap 2: OCR-aware compression focuses on OCR, not downstream reasoning 

OCR-aware compression methods try to preserve readable text, but they often do not test whether the compressed image still supports document-level reasoning. 

## Gap 3: VLM token-reduction methods are usually model-side 

Visual token pruning methods reduce tokens after visual encoding. This project instead studies input-side preprocessing before the image reaches the VLM. 

## Gap 4: High-resolution slicing can fragment context 

Uniform slicing or tiling preserves local detail, but it may break global layout and crosspatch relationships. 

## Gap 5: OCR preservation does not guarantee VLM reasoning preservation 

A method may preserve characters but still damage the layout cues needed for answering document questions. 

## 5. Main Research Question 

**Under fixed pixel, byte, or patch/token budgets, does OCR-guided overview-pluspatch selection preserve OCR and VLM performance better than naive resizing, compression, random patching, and uniform tiling?** 

## 6. Secondary Research Questions 

## RQ1 

Under fixed pixel and byte budgets, how quickly do OCR accuracy and word recognition degrade for standard resizing and compression methods? 

## RQ2 

Does OCR-guided patch selection outperform random patches and uniform tiling under the same patch budget? 

## RQ3 

Does adding a low-resolution overview improve VLM QA compared with using highresolution patches alone? 

## RQ4 

Does improved OCR preservation transfer to improved VLM question-answering accuracy? 

## RQ5 

What are the dominant failure modes of budgeted preprocessing for text-rich images? 

## 7. Hypothesis 

## Main Hypothesis 

**OCR-guided overview-plus-patch selection will preserve OCR and VLM QA performance better than resizing, compression, random patching, and uniform tiling under strict visual budgets.** 

## Secondary Hypothesis 

## **OCR preservation and VLM reasoning preservation are related but not identical. Some methods may preserve text while still harming layout-sensitive reasoning.** 

This secondary hypothesis is important because it gives the paper value even if the method does not beat every baseline in every case. 

## 8. Paper Contributions 

The paper should claim exactly three contributions. 

## Contribution 1: Budget-Normalized Evaluation 

A reproducible comparison of preprocessing methods under equal: 

- pixel budgets, 

- byte budgets, 

- patch/token budgets. 

This prevents unfair comparisons where one method secretly uses more visual information than another. 

## Contribution 2: OCR-Guided Overview-Plus-Patch Method 

A simple training-free method that combines: 

- global overview for layout context, 

- OCR-guided high-resolution patches for small text. 

## Contribution 3: OCR-vs-Reasoning Analysis 

An analysis of when OCR preservation helps VLM QA and when it fails due to missing layout, patch fragmentation, or unselected answer regions. 

## 9. What This Project Is Not 

This is not: 

- a new OCR engine, 

- a new VLM, 

- a learned compression model, 

- a full high-resolution VLM architecture, 

- a UI/dashboard project, 

- a seam carving novelty paper. 

This is: 

- a preprocessing method, 

- a benchmark/evaluation paper, 

- a practical VLM efficiency study, 

- a text-rich image understanding paper. 

## 10. Proposed Method 

## Method Name 

Use one of these: 

1. **BOPS** — Budgeted OCR-Guided Patch Selection 

2. **OCR-OPS** — OCR-Guided Overview-Plus-Patch Selection 

3. **BOP-VLM** — Budgeted Overview-Patch Input for VLMs 

Recommended: 

BOPS: Budgeted OCR-Guided Patch Selection 

## 11. Method Overview 

Given a high-resolution image, BOPS produces: 

1. a low-resolution overview image, 

2. K selected high-resolution patches. 

The overview helps the model understand the full layout. 

The patches help the model read small text. 

## 12. BOPS Pipeline 

## Step 1: Input Image 

Input: 

High-resolution text-rich image 

Examples: 

- document, 

- receipt, 

- slide, 

- screenshot, 

- poster, 

- infographic. 

## Step 2: Overview Generation 

Create a low-resolution overview of the full image. 

Recommended overview size: 

longest side = 768 px 

Ablation sizes: 

512 px 768 px 1024 px 

## Step 3: OCR/Text Detection 

Run OCR or text detection on the original high-resolution image. 

Recommended first option: 

PaddleOCR 

Optional second option: 

EasyOCR 

The OCR output should include: 

- text boxes, 

- confidence scores, 

- recognized text, 

- bounding coordinates. 

## Step 4: Candidate Patch Generation 

Divide the original image into candidate patches. 

Default patch size: 

512 × 512 

Ablation patch sizes: 

448 × 448 512 × 512 768 × 768 

Default overlap: 

25% 

Ablation overlap: 

0% 25% 50% 

## Step 5: Patch Scoring 

Each candidate patch gets an importance score. 

Recommended scoring function: 

score = 0.60 × text_coverage + 0.20 × text_confidence + 0.10 × edge_density + 0.10 × entropy 

Where: 

## text_coverage 

How much detected text lies inside the patch. 

## text_confidence 

Average OCR/text detection confidence of text inside the patch. 

## edge_density 

Amount of edge/detail information. Useful for small text, tables, forms, and diagrams. 

## entropy 

General visual density of the patch. 

This is intentionally simple. A simple method is easier to defend than a complicated method that barely improves results. 

## Step 6: Non-Maximum Suppression 

Apply NMS so selected patches do not all come from the same region. 

Default IoU threshold: 

0.4 

Process: 

1. sort candidate patches by score, 

2. select highest scoring patch, 

3. remove patches with high overlap, 

4. repeat until K patches are selected. 

## Step 7: Patch Budget Selection 

Test these patch budgets: 

overview only overview + 2 patches overview + 4 patches overview + 8 patches overview + 12 patches 

For the first paper, the most important budgets are: 

overview only overview + 4 patches overview + 8 patches 

## Step 8: Patch Ordering 

For VLM input, order matters. 

Default order: 

overview first, then patches in reading order 

Ablation: 

importance order reading order random order 

Reading order means top-to-bottom, left-to-right. 

## Step 9: OCR/VLM Evaluation 

For OCR: 

- run OCR on the transformed image or patch set, 

- merge duplicate text, 

- compute OCR metrics. 

For VLM QA: 

- pass overview and patches to the VLM, 

- ask the dataset question, 

- compare answer with ground truth. 

## 13. Baselines 

You need strong baselines. Weak baselines make the paper look fake. 

## Required Baselines 

|Baseline|Why It Is Needed|
|---|---|
|Original image|Upper bound|



|Baseline|Why It Is Needed||
|---|---|---|
|Resize|Most common preprocessing||
|JPEG compression|Standard byte-budget baseline||
|WebP compression|Stronger common compression baseline||
|Overview-only|Tests whether global context alone is enough||
|Random patches|Tests whether patching alone helps||
|Uniform grid|Strong simple tiling baseline||
|patches|||
|OCR-guided patches|Proposed method||
|Optional Baselines|||
|Optional Baseline||Use Case|
|Vanilla seam carving||Classical retargeting comparison|
|OCR-guided seam carving||Useful failure case|
|OCR-confdence-only|patching|Ablation|
|Edge/entropy-only patching||Ablation|
|TFIC-style proxy||OCR-aware compression comparison|
|LLaVA-UHD-style uniform slicing||High-resolution VLM slicing comparison|



## 14. Why Uniform Tiling Is a Dangerous Baseline 

Uniform tiling is the strongest simple baseline. 

If uniform tiling performs as well as OCR-guided patching, your contribution becomes weak. 

Therefore, you must include it. 

Your method must show at least one of these: 

1. better performance than uniform tiling with the same number of patches, 

2. same performance with fewer patches, 

3. better runtime/cost tradeoff, 

4. better failure behavior on small-text regions. 

## 15. Datasets 

Use a small dataset plan first. Do not start with too many datasets. 

## Core Dataset 1: TextOCR 

Purpose: 

## OCR preservation 

Use it to test: 

- OCR degradation after resizing, 

- OCR degradation after compression, 

- whether OCR-guided patches preserve text better. 

## Metrics: 

- Character Error Rate, 

- Word Error Rate, 

- word recall, 

- detection F1 if using boxes. 

## Recommended first scale: 

debug: 50 images pilot: 200 images paper run: 500–1000 images 

## Core Dataset 2: DocVQA 

## Purpose: 

## VLM document question answering 

Use it to test: 

- whether overview-plus-patch helps document QA, 

- whether selected patches preserve answer regions, 

- whether global layout is needed. 

## Metrics: 

- Exact Match, 

- ANLS, 

- runtime, 

- patch count, 

- estimated visual token count. 

Recommended first scale: 

debug: 20 QA samples pilot: 100 QA samples paper run: 300–500 QA samples 

## Optional Dataset 3: OCRBench v2 

Purpose: 

OCR-centric VLM stress test 

Use it after DocVQA works. 

Recommended scale: 

200–500 samples 

This is useful for showing that the method generalizes beyond one document QA dataset. 

## Optional Dataset 4: HierText 

Purpose: 

dense text and layout stress test 

Use it only if time allows. 

Good for showing that seam carving and naive resizing struggle on dense text. 

## Optional Dataset 5: Self-Collected Real-World Set 

Purpose: 

qualitative external validity 

Collect 100–300 images: 

- slides, 

- posters, 

- receipts, 

- screenshots, 

- whiteboards, 

- scanned notes. 

Use only for qualitative analysis and failure examples, not main claims. 

## 16. Metrics 

## OCR Metrics 

Use: 

1. Character Error Rate 

2. Word Error Rate 

3. Word Recall 

4. Detection F1, if boxes are available 

## VLM QA Metrics 

Use: 

1. Exact Match 

2. ANLS 

3. Task accuracy, if benchmark provides it 

## Efficiency Metrics 

Use: 

1. output file size, 

2. output pixel count, 

3. number of patches, 

4. estimated visual-token count, 

5. preprocessing runtime, 

6. VLM inference runtime, 

7. total cost, if using hosted APIs. 

## 17. Budget Settings 

## Pixel Budgets 

Use: 

100% area 50% area 25% area 12.5% area 

Byte Budgets 

Use: 

500 KB 200 KB 100 KB 50 KB 

## Patch Budgets 

Use: 

overview only overview + 2 patches overview + 4 patches overview + 8 patches overview + 12 patches 

For the first paper, focus on: 

overview only overview + 4 patches overview + 8 patches 

## 18. Tools and Libraries 

## Programming Language 

Python 

## Image Processing 

OpenCV Pillow NumPy 

## OCR 

PaddleOCR EasyOCR 

Use PaddleOCR first. 

Use EasyOCR later for cross-OCR validation. 

## VLM 

Choose one option: 

1. Qwen2.5-VL 

2. InternVL 

3. LLaVA-NeXT 

4. GPT-class hosted multimodal model, if affordable 

5. Gemini/Claude-style hosted multimodal model, if available 

Do not compare five VLMs at the start. 

## Metrics 

jiwer python-Levenshtein scikit-learn scipy statsmodels 

## Plotting 

matplotlib 

## Experiment Tracking 

Start simple: 

CSV logs JSONL results YAML configs 

Do not start with MLflow or Weights & Biases unless you already use them comfortably. 

## 19. Repository Structure 

Use this from day one. 

ocr_bops/ 

│ ├── configs/ 

- │├── smoke_test.yaml 

- │├── resize.yaml 

- │├── compression.yaml 

- │├── patch_selection.yaml 

│├── ocr_eval.yaml │├── vlm_eval.yaml │└── paper_main.yaml │ ├── data/ │├── raw/ │├── processed/ │├── manifests/ │└── samples/ │ ├── outputs/ │├── transformed_images/ │├── patches/ │├── ocr_results/ │├── vlm_results/ │├── metrics/ │├── plots/ │└── failure_cases/ │ ├── src/ │├── data/ ││├── build_textocr_manifest.py ││├── build_docvqa_manifest.py ││├── validate_manifest.py ││└── dataset_loader.py ││ │├── preprocessing/ ││├── resize.py ││├── compression.py ││├── overview.py ││├── patch_grid.py ││├── patch_scoring.py ││├── patch_nms.py ││└── bops.py ││ │├── ocr/ ││├── run_ocr.py ││├── normalize_text.py ││├── merge_patch_ocr.py ││└── ocr_metrics.py ││ │├── vlm/ ││├── run_vlm.py 

││├── prompt_templates.py ││├── parse_answers.py ││└── qa_metrics.py ││ │├── metrics/ ││├── budget_metrics.py ││├── efficiency_metrics.py ││└── statistical_tests.py ││ │├── visualization/ ││├── draw_patches.py ││├── plot_budget_curves.py ││├── make_failure_panels.py ││└── make_paper_figures.py ││ │└── utils/ │ ├── config.py │ ├── image_io.py │ ├── logging_utils.py │ └── paths.py │ ├── scripts/ │├── run_preprocessing.py │├── run_ocr_eval.py │├── run_vlm_eval.py │├── run_full_experiment.py │├── generate_plots.py │└── make_paper_assets.py │ ├── notebooks/ │├── sanity_check.ipynb │├── inspect_patches.ipynb │└── analyze_results.ipynb │ ├── paper/ │├── figures/ │├── tables/ │└── draft.tex │ ├── requirements.txt ├── README.md └── LICENSE 

## 20. Data Manifest Format 

Every dataset should be converted into one common JSONL format. 

Example: 

{ "image_id": "docvqa_000001", "dataset": "DocVQA", "split": "val", "image_path": "data/raw/docvqa/images/000001.png", "ocr_gt_text": "", "question": "What is the invoice date?", "answer": "12 March 2024", "answer_type": "extractive", "boxes": [], "metadata": {} } 

For OCR-only samples: 

{ "image_id": "textocr_000001", "dataset": "TextOCR", "split": "val", "image_path": "data/raw/textocr/images/000001.jpg", "ocr_gt_text": "STOP MAIN STREET", "question": "", "answer": "", "answer_type": "", "boxes": [], "metadata": {} } 

## 21. Implementation Phases and Passing Gates 

## Phase 0 — Scope Lock 

## Goal 

Freeze the easiest publishable version. 

## What To Do 

1. Finalize title. 

2. Finalize research questions. 

3. Finalize methods. 

4. Finalize datasets. 

5. Finalize metrics. 

6. Create one-page professor pitch. 

## Output 

1-page proposal method list dataset list metric list timeline 

## Passing Gate 0 

You pass if you can explain the project in one sentence: 

We compare preprocessing methods for text-rich images under equal visual budgets and test whether OCR-guided overview-plus-patch selection preserves OCR and VLM QA better than resizing, compression, random patches, and uniform tiling. 

If the explanation requires five minutes, the scope is still too broad. 

## Phase 1 — Repository and Environment 

## Goal 

Create reproducible infrastructure. 

## What To Do 

1. Create repo. 

2. Create folder structure. 

3. Add requirements. 

4. Add config loader. 

5. Add logging. 

6. Add image I/O utilities. 

7. Add smoke-test script. 

## Output 

working repository smoke test output metadata CSV 

## Passing Gate 1 

Run: 

python scripts/run_preprocessing.py --config configs/smoke_test.yaml 

Pass only if: 

- image loads, 

- output saves, 

- metadata saves, 

- no manual path editing needed. 

## Phase 2 — Dataset Manifest Layer 

## Goal 

Make datasets usable through one interface. 

## What To Do 

1. Build TextOCR manifest. 

2. Build DocVQA manifest. 

3. Validate image paths. 

4. Create debug subsets. 

5. Create pilot subsets. 

6. Create paper-run subsets. 

## Output 

textocr_debug.jsonl textocr_pilot.jsonl docvqa_debug.jsonl docvqa_pilot.jsonl 

Passing Gate 2 

Run: 

python src/data/validate_manifest.py --manifest data/manifests/textocr_debug.jsonl python src/data/validate_manifest.py --manifest data/manifests/docvqa_debug.jsonl 

Pass only if: 

- all image paths exist, 

- all required fields exist, 

- no duplicate image IDs, 

- sample preview works. 

## Phase 3 — Resize and Compression Baselines 

## Goal 

Implement the baselines that every reviewer expects. 

## Methods 

1. Resize 

2. JPEG 

3. WebP 

## What To Do 

Implement: 

resize by area ratio JPEG quality search by byte budget WebP quality search by byte budget 

## Output 

Transformed images for: 

50% area 25% area 12.5% area 500 KB 200 KB 100 KB 50 KB 

## Passing Gate 3 

Pass only if: 

- resize hits target area within ±3%, 

- compression hits target byte budget within ±2%, 

- runtime is logged, 

- failed images are logged instead of crashing. 

## Phase 4 — OCR Evaluation Harness 

## Goal 

Measure OCR degradation. 

## What To Do 

1. Integrate PaddleOCR. 

2. Normalize text. 

3. Compute CER. 

4. Compute WER. 

5. Save OCR outputs. 

6. Save metrics. 

## Output 

ocr_results.csv ocr_metrics.csv 

## Passing Gate 4 

Pass only if: 

- OCR works on original images, 

- OCR works on resized/compressed images, 

- CER/WER calculations pass toy tests, 

- 50-image debug run completes. 

## Toy Metric Test 

ground truth: hello world prediction: hello world CER = 0 WER = 0 

If this fails, metrics are broken. 

## Phase 5 — First Baseline Benchmark 

## Goal 

Produce the first publishable result: OCR degradation curves. 

## What To Do 

Run TextOCR pilot subset with: 

1. original, 

2. resize, 

3. JPEG, 

4. WebP. 

Output 

Plots: 

CER vs pixel budget WER vs pixel budget CER vs byte budget WER vs byte budget runtime table 

## Passing Gate 5 

Pass only if: 

- plots regenerate from metric files, 

- OCR generally worsens under stricter budgets, 

- visual examples show text degradation, 

- professor can understand the motivation from one graph. 

This is your first serious professor checkpoint. 

## Phase 6 — Overview and Candidate Patch Extraction 

## Goal 

Build the skeleton of the proposed method. 

What To Do 

1. Generate overview. 

2. Generate candidate patches. 

3. Save patch coordinates. 

4. Save patch images. 

5. Draw patch grid visualization. 

## Output 

overview image candidate patches patch coordinate JSON patch visualization 

## Passing Gate 6 

Pass only if: 

- patch coordinates are valid, 

- crops match the original image, 

- visualization boxes align correctly, 

- no patch goes outside image boundaries. 

Coordinate bugs here will destroy the entire project. 

## Phase 7 — OCR-Guided Patch Scoring 

## Goal 

Select meaningful patches. 

## What To Do 

1. Run OCR/text detection on original image. 

2. Compute text coverage per patch. 

3. Compute OCR confidence per patch. 

4. Compute edge density. 

5. Compute entropy. 

6. Apply patch score. 

7. Apply NMS. 

8. Select top-K patches. 

## Output 

selected patch list patch scores patch visualizations score heatmaps 

## Passing Gate 7 

Pass only if: 

- selected patches contain visible text, 

- text-guided patches cover more OCR boxes than random patches, 

- NMS reduces redundant overlapping patches, 

- 20 visual examples look sensible. 

## Critical Test 

Compare: 

random patches vs OCR-guided patches 

Metric: 

percentage of OCR text boxes covered 

OCR-guided must beat random. 

If it does not, your method has no foundation. 

## Phase 8 — OCR Evaluation of BOPS 

## Goal 

Test whether the proposed method helps OCR. 

## Methods 

Compare: 

1. resize, 

2. JPEG, 

3. WebP, 

4. overview-only, 

5. random patches, 

6. uniform patches, 

7. OCR-guided patches. 

## What To Do 

1. Run TextOCR pilot subset. 

2. OCR all outputs. 

3. Merge OCR from patches. 

4. Compute CER/WER. 

5. Compare against baselines. 

## Output 

BOPS OCR results CER/WER comparison table patch budget curve 

## Passing Gate 8 

Pass only if: 

- OCR-guided patches beat random patches, 

- OCR-guided patches beat overview-only, 

- OCR-guided patches are competitive with or better than resize at strict budget, 

- budget fairness is logged. 

Budget fairness means you log: 

total pixels total bytes number of patches runtime 

If your method uses more budget than baselines, the comparison is invalid. 

## Phase 9 — VLM Evaluation Harness 

## Goal 

Evaluate downstream document QA. 

## What To Do 

1. Select one VLM. 

2. Implement fixed prompts. 

3. Run single-image QA. 

4. Run multi-image overview-plus-patch QA. 

5. Parse answers. 

6. Compute Exact Match. 

7. Compute ANLS. 

## Output 

vlm_results.csv qa_metrics.csv 

## Prompt for Single Image 

You are given an image. Answer the question using only the visual information. 

Question: {question} Answer: 

## Prompt for Overview Plus Patches 

You are given one low-resolution overview image followed by high-resolution patches from the same original image. Use the overview for global layout and the patches for small text. Answer the question using only the provided visual information. 

Question: {question} Answer: 

## Passing Gate 9 

Pass only if: 

- one DocVQA sample runs manually, 

- 20-sample debug run works, 

- answers are saved, 

- raw model outputs are saved, 

- metrics are computed automatically. 

Do not proceed without saving raw answers. 

## Phase 10 — VLM Patch-Budget Experiments 

## Goal 

Test whether the method improves QA under patch/token budgets. 

## Dataset 

DocVQA pilot subset. 

## Methods 

1. resized single image, 

2. overview-only, 

3. random patches, 

4. uniform patches, 

5. OCR-guided patches. 

## Budgets 

overview only overview + 2 patches overview + 4 patches overview + 8 patches 

## Output 

ANLS vs patch count Exact Match vs patch count failure examples 

## Passing Gate 10 

Pass only if: 

- OCR-guided patches beat random patches, 

- OCR-guided patches beat overview-only, 

- method is competitive with resized single image, 

- failures can be explained. 

If uniform tiling beats your method, do not hide it. Analyze why. 

## Phase 11 — Ablation Study 

## Goal 

Prove which parts of your method matter. 

## Required Ablations 

Ablation 1: Patch Count 

overview only overview + 2 overview + 4 overview + 8 overview + 12 

## Ablation 2: Patch Scoring 

random uniform text coverage only text coverage + confidence text coverage + confidence + edge text coverage + confidence + edge + entropy 

## Ablation 3: Overview 

patches only overview + patches 

## Ablation 4: Patch Order 

reading order importance order random order 

## Passing Gate 11 

Pass only if: 

- every ablation has a metric table, 

- every ablation has an interpretation, 

- full method is best or near-best, 

- if a simpler variant wins, you adopt the simpler variant honestly. 

## Phase 12 — Optional Seam Carving Baseline 

## Goal 

Add classical retargeting as a comparison/failure case. 

## What To Do 

1. Implement vanilla seam carving. 

2. Implement OCR-guided seam carving. 

3. Compare on small subset. 

4. Save layout distortion examples. 

## Passing Gate 12 

Pass only if: 

- seam carving runs reliably, 

- OCR-guided version protects text better than vanilla in some cases, 

- dense-layout failure cases are saved. 

This is optional. Do not delay the main paper for seam carving. 

## Phase 13 — Full Paper Experiments 

Goal 

Run the final experiments. 

Minimum Final Experiments 

## Experiment A: TextOCR OCR Preservation 

Methods: 

resize JPEG WebP overview-only random patches uniform patches BOPS 

Metrics: 

CER WER word recall runtime bytes pixels 

Experiment B: DocVQA QA Preservation 

Methods: 

resized single image overview-only random patches uniform patches BOPS 

Metrics: 

Exact Match ANLS patch count runtime estimated visual tokens 

## Experiment C: Failure Analysis 

Analyze: 

text unreadable answer patch not selected layout context missing table relation missed patch order confusion OCR detector failure VLM hallucination 

Passing Gate 13 

Pass only if: 

- all final methods run on selected subsets, 

- failure rate is below 5%, 

- failed samples are logged, 

- plots regenerate from result files, 

- no manual cherry-picking is used. 

Phase 14 — Statistical Testing 

Goal 

Make claims defensible. 

What To Do 

For OCR: 

paired bootstrap confidence intervals paired permutation tests 

For QA: 

paired bootstrap confidence intervals McNemar-style exact-match comparison 

For multiple comparisons: 

Holm correction 

Passing Gate 14 

Pass only if each main comparison has: 

mean score confidence interval sample count baseline comparison budget effect size 

Do not overclaim tiny gains. 

A 1% gain is weak. 

A 5–10% gain under strict budget is meaningful. 

## Phase 15 — Failure Analysis 

## Goal 

Explain why methods work or fail. 

## Required Failure Examples 

Save visual panels for: 

1. resize blurs small text, 

2. JPEG/WebP damages thin text, 

3. overview-only misses small text, 

4. random patch misses answer, 

5. uniform tiling wastes patches, 

6. BOPS selects useful text, 

7. BOPS misses answer region, 

8. BOPS loses layout context. 

## Passing Gate 15 

Pass only if: 

- at least 20 qualitative examples are saved, 

- at least 100 VLM failures are categorized, 

- failure chart is generated, 

- each example links to image ID and result row. 

Phase 16 — Paper Tables and Figures 

Required Tables 

Table 1: Dataset and Task Summary 

Dataset | Task | Samples Used | Metrics | Purpose 

Table 2: OCR Under Pixel Budget 

Method | 50% Area CER | 25% Area CER | 12.5% Area CER | Runtime 

Table 3: OCR Under Byte Budget 

Method | 500KB CER | 200KB CER | 100KB CER | Runtime 

Table 4: DocVQA Under Patch Budget 

Method | Patch Count | Exact Match | ANLS | Runtime 

Table 5: Ablation Study 

Variant | TextOCR CER | DocVQA ANLS | Runtime 

Table 6: Failure Breakdown 

Failure Type | Resize | Compression | Uniform Patches | BOPS 

Required Figures 

1. System pipeline 

2. BOPS overview-plus-patch diagram 

3. CER/WER vs budget curve 

4. QA accuracy vs patch count curve 

5. Patch selection visualization 

6. Failure-case panel 

## Passing Gate 16 

Pass only if: 

- all tables regenerate from CSV/JSONL, 

- all figures regenerate from scripts, 

- captions explain the result, 

- no screenshot-only plots are used. 

## Phase 17 — Professor Review 

## Goal 

Get approval before final paper writing. 

## Prepare 10 Slides 

1. Problem 

2. Research gap 

3. Proposed method 

4. Baselines 

5. Datasets 

6. OCR results 

7. VLM results 

8. Ablations 

9. Failure analysis 

- 10.Target venue 

## Passing Gate 17 

Professor should agree on: 

- final title, 

- final claim, 

- final datasets, 

- final baselines, 

- whether optional seam carving is included, 

- target venue. 

Ask directly: 

Is this contribution strong enough as a paper, or should it be positioned only as an FYP implementation? 

## Phase 18 — Paper Writing 

## Paper Structure 

Abstract Introduction Related Work Problem Formulation 

Method Experimental Setup Results Ablations Failure Analysis Limitations Conclusion 

## Main Claim Template 

Use this: 

We study input-side preprocessing for text-rich images under matched visual budgets. We compare resizing, JPEG/WebP compression, overview-only reduction, random patching, uniform tiling, and OCR-guided overview-plus-patch selection. Results show that OCR-guided patch selection improves the tradeoff between visual budget and textrich image understanding, while failure analysis shows that OCR preservation does not always guarantee layout-sensitive VLM reasoning. 

Only say “improves” if the data supports it. 

If results are mixed, use: 

We find that OCR-guided patching improves text preservation over random and overview-only baselines, but downstream QA remains sensitive to global layout, patch ordering, and missed answer regions. 

That is still a valid paper. 

Passing Gate 18 

Pass only if: 

- every claim maps to a result, 

- every result maps to a table/figure, 

- limitations are honest, 

- related work is connected to experiments, 

- repository can reproduce results. 

## 22. Literature Benchmark Map 

Use these benchmark categories in the related work. 

## Category 1: Standard Image Reduction 

Methods: 

resize JPEG WebP 

Shortcoming: 

Not optimized for OCR or VLM reasoning. 

How BOPS builds on it: 

Preserves global context while selecting high-resolution text patches. 

## Category 2: OCR-Aware Compression 

Representative idea: 

preserve text quality during compression 

Shortcoming: 

Usually focused on OCR readability, not VLM QA or layout reasoning. 

How BOPS builds on it: 

Evaluates both OCR and downstream VLM QA. 

## Category 3: High-Resolution VLM Slicing 

Representative idea: 

split high-resolution images into slices or patches 

Shortcoming: 

Uniform slicing may waste patches and fragment layout context. 

How BOPS builds on it: 

Uses OCR-guided selection plus an overview to preserve both text and context. 

## Category 4: Visual Token Reduction 

Representative idea: 

remove redundant visual tokens inside the model 

Shortcoming: 

Usually model-side and architecture-dependent. 

How BOPS builds on it: 

Works as input-side preprocessing and can be used before different OCR/VLM systems. 

## Category 5: Classical Retargeting 

Representative idea: 

seam carving 

Shortcoming: 

May distort reading order, tables, and dense text layout. 

How BOPS builds on it: 

Avoids geometry distortion by selecting patches instead of reshaping the whole image. 

## 23. What You Need To Do This 

## Minimum Hardware 

CPU with 16 GB RAM 

This is enough for: 

- resizing, 

- compression, 

- patch extraction, 

- small OCR runs. 

## Recommended Hardware 

GPU with 8–16 GB VRAM 

Useful for: 

- OCR acceleration, 

- local VLM evaluation. 

## If You Do Not Have GPU 

Use: 

- smaller dataset subsets, 

- hosted VLM API for limited evaluation, 

- OCR-only results as primary, 

- DocVQA evaluation on small sample. 

## Software Needed 

Python 3.10+ OpenCV Pillow NumPy Pandas PaddleOCR EasyOCR jiwer scipy statsmodels matplotlib PyYAML tqdm 

Optional: 

PyTorch Transformers Qwen-VL / InternVL / LLaVA-NeXT 

## 24. Minimum Viable Paper Version 

If time is short, do only this: 

Datasets 

TextOCR: 500 images DocVQA: 300 QA samples 

## Methods 

resize JPEG WebP overview-only random patches uniform patches BOPS 

## Metrics 

CER WER Exact Match ANLS runtime bytes pixels patch count 

## Figures 

pipeline patch visualization OCR error vs budget QA accuracy vs patch count failure panel 

This is enough for a 6-page FYP paper. 

## 25. Stronger Paper Version 

If the first version works, add: 

OCRBench v2 HierText OCR-guided seam carving second OCR engine second VLM statistical tests larger sample size 

This moves it toward ICDAR/WACV/ICIP quality. 

## 26. Risk Register 

## Risk 1: BOPS Does Not Beat Resize 

## Response: 

- check whether dataset has mostly large text, 

- test stricter budgets, 

- compare patch count fairly, 

- analyze cases where resize is enough. 

## Risk 2: BOPS Beats Random But Not Uniform 

## Response: 

- improve patch scoring, 

- add text box coverage objective, 

- use NMS better, 

- show BOPS uses fewer patches for similar performance. 

## Risk 3: OCR Improves But VLM QA Does Not 

## Response: 

- frame as OCR-vs-reasoning gap, 

- analyze missing global layout, 

- improve overview resolution, 

- test patch order. 

## Risk 4: VLM Evaluation Is Too Expensive 

## Response: 

- reduce sample size, 

- use one model, 

- report OCR evaluation more extensively, 

- use VLM QA as secondary validation. 

## Risk 5: OCR Detector Fails 

## Response: 

- use fallback edge/entropy scoring, 

- test second OCR engine, 

 classify OCR detector failure separately. 

## 27. Timeline 

## Week 1 

Scope lock, repo setup, dataset manifests. 

Week 2 

Resize, JPEG, WebP baselines. 

## Week 3 

OCR evaluation harness and first OCR curves. 

## Week 4 

Overview and patch extraction. 

Week 5 

OCR-guided patch scoring and NMS. 

## Week 6 

OCR evaluation of BOPS. 

Week 7 

DocVQA VLM evaluation setup. 

Week 8 

VLM patch-budget experiments. 

Week 9 

Ablations. 

Week 10 

Optional seam carving or OCRBench v2. 

Week 11 

Full experiments. 

Week 12 

Statistical testing and failure analysis. 

Week 13 

Tables and figures. 

Week 14 

Professor review. 

Week 15 

Paper draft. 

Week 16 

Final polish, code cleanup, submission preparation. 

## 28. First 7 Days: Exact Tasks 

Day 1 

Create repo, folder structure, requirements file. 

Day 2 

Build TextOCR manifest loader or use temporary image folder. 

Day 3 

Implement resize baseline. 

Day 4 

Implement JPEG/WebP byte-budget compression. 

Day 5 

Integrate PaddleOCR. 

Day 6 

Compute CER/WER. 

Day 7 

Generate first plot: 

OCR error vs budget 

At the end of week 1, you should have one graph proving that naive image reduction damages OCR. 

## 29. First Professor Update 

Say this: 

I narrowed the project into a paper-style study on budget-aware preprocessing for textrich images. The proposed method is OCR-guided overview-plus-patch selection. I will compare it against resize, JPEG/WebP, overview-only, random patches, and uniform tiling under equal visual budgets. The first milestone is to produce OCR degradation curves on TextOCR, then test whether OCR-guided patches improve OCR and DocVQA performance under patch budgets. 

Do not say: 

I am researching compression and VLMs. 

That sounds vague. 

## 30. Final Paper Target 

## First Target 

FYP paper / local IEEE-style conference / undergraduate research conference 

## Stronger Target 

ICDAR workshop DAS workshop WACV workshop ICIP-style image processing venue 

## Journal Later 

IJDAR IEEE Access IET Image Processing 

Do not target a serious journal until you have full experiments and strong ablations. 

## 31. Definition of Done 

The project is paper-ready when you have: 

1. reproducible repo, 

2. dataset manifests, 

3. resize/JPEG/WebP baselines, 

4. overview-only baseline, 

5. random patch baseline, 

6. uniform patch baseline, 

7. BOPS method, 

8. TextOCR OCR results, 

9. DocVQA QA results, 

- 10.ablations, 

- 11.failure analysis, 

- 12.statistical tests, 

- 13.paper tables, 

- 14.paper figures, 

15.complete 6-page draft. 

Anything beyond this is optional. 

## 32. Final Recommendation 

Build the easiest defensible paper: 

Budget-Aware OCR-Guided Patch Selection for Text-Rich Image Understanding 

Do not overcomplicate it. 

Your core claim should be: 

OCR-guided overview-plus-patch selection is a simple, training-free preprocessing strategy that improves the tradeoff between visual budget and text-rich image understanding compared with resizing, compression, random patching, and uniform tiling. 

That is the version most likely to become a real paper. 


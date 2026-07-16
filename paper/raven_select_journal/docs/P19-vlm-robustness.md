# P19 — Second-VLM Robustness

## Scope

Expand Qwen2-VL-2B from n=300 to n=500 using the frozen selector. Primary model
remains Qwen2.5-VL-3B.

## Gate

| Outcome | Condition |
|---------|-----------|
| **FULL ROBUSTNESS** | Beats resize and shortest; both CI lower bounds > 0 |
| **PARTIAL ROBUSTNESS** | Significantly beats resize only |
| **FAIL** | Significantly beats neither |

## Baseline (already measured)

Qwen2-VL-2B n=300: RAVEN-Select 0.6271 vs resize 0.5980 vs shortest 0.6269
→ **PARTIAL** (beats resize; ties shortest).

## Commands

```text
python scripts/run_raven_robustness_vlm.py --n 500 --manifest Data/manifests/docvqa_500.jsonl --model Qwen/Qwen2-VL-2B-Instruct --metrics-tag qwen2vl2b
python scripts/run_raven_select_build_features.py --n 500 --metrics-tag qwen2vl2b
python scripts/run_raven_select_eval.py --n 500 --metrics-tag qwen2vl2b
```

## Results

| VLM | n | resize | shortest | RAVEN-Select | Gate |
|-----|---|--------|----------|--------------|------|
| Qwen2.5-VL-3B | 500 | 0.7840 | 0.7965 | 0.8053 | main (PASS) |
| Qwen2-VL-2B | 300 | 0.5980 | 0.6269 | 0.6271 | PARTIAL |
| Qwen2-VL-2B | 500 | — | — | — | PENDING |

## Commit

- TBD

"""Vision-language model QA for DocVQA evaluation.

Submodules:
    prompt_templates: Overview+patch and single-image QA prompts
    parse_answers: Extract short answers from model text
    qa_metrics: Exact Match and ANLS for DocVQA
    run_vlm: Qwen2.5-VL-3B harness (4-bit, local inference)
"""

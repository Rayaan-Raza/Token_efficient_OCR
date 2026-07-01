"""Extract the final answer string from raw VLM decoder output."""

import re


def parse_answer(raw: str) -> str:
    """Strip chat template artifacts and normalize whitespace.

    If the model echoes an ``Answer:`` prefix, text after the last occurrence
    is returned.

    Args:
        raw: Full decoded model output.

    Returns:
        Cleaned answer string for metric computation.
    """
    raw = raw.strip()
    # Qwen chat templates often echo "assistant" before the final answer
    parts = re.split(r"(?i)\bassistant\b", raw)
    if len(parts) > 1:
        raw = parts[-1].strip()
    for prefix in ("Answer:", "answer:"):
        if prefix in raw:
            raw = raw.split(prefix, 1)[-1].strip()
    return re.sub(r"\s+", " ", raw).strip()

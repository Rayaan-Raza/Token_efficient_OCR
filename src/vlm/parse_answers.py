"""Extract the final answer string from raw VLM decoder output."""

import re


_PREFIX_PATTERNS = (
    r"^the answer is[:\s]*",
    r"^answer[:\s]*",
    r"^response[:\s]*",
    r"^final answer[:\s]*",
)


def normalize_answer_format(text: str) -> str:
    """Normalize VLM output for DocVQA ANLS/EM (format only, not semantics)."""
    raw = text.strip()
    if not raw:
        return raw

    # Chat-template artifacts may remain after parse_answer's first pass.
    parts = re.split(r"(?i)\bassistant\b", raw)
    if len(parts) > 1:
        raw = parts[-1].strip()

    # First line only — models often add explanation on later lines.
    raw = raw.splitlines()[0].strip()

    for pat in _PREFIX_PATTERNS:
        raw = re.sub(pat, "", raw, flags=re.IGNORECASE).strip()

    # Strip surrounding quotes.
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1].strip()

    # Trailing sentence punctuation (keep internal punctuation in dates/amounts).
    raw = re.sub(r"[.,;:!?]+$", "", raw).strip()
    return re.sub(r"\s+", " ", raw).strip()


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
    parts = re.split(r"(?i)\bassistant\b", raw)
    if len(parts) > 1:
        raw = parts[-1].strip()
    for prefix in ("Answer:", "answer:"):
        if prefix in raw:
            raw = raw.split(prefix, 1)[-1].strip()
    return normalize_answer_format(raw)

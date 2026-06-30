"""Text normalization and tokenization for OCR evaluation metrics.

All CER, WER, and word-recall computations use these helpers so scoring is
consistent across experiments. Normalization: lowercase, strip punctuation,
collapse whitespace.
"""

import re
import string


def normalize_text(text: str) -> str:
    """Normalize text for fair OCR string comparison.

    Args:
        text: Raw OCR or ground-truth string.

    Returns:
        Lowercased string with punctuation removed and whitespace collapsed.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split normalized text into whitespace-delimited word tokens.

    Args:
        text: Raw or normalized text.

    Returns:
        List of tokens (empty if text is blank after normalization).
    """
    norm = normalize_text(text)
    return norm.split() if norm else []

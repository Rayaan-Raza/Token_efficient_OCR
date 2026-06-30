"""Text normalization for OCR metrics."""

import re
import string


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    norm = normalize_text(text)
    return norm.split() if norm else []

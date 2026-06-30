"""Parse VLM answer strings."""

import re


def parse_answer(raw: str) -> str:
    raw = raw.strip()
    for prefix in ("Answer:", "answer:"):
        if prefix in raw:
            raw = raw.split(prefix, 1)[-1].strip()
    return re.sub(r"\s+", " ", raw).strip()

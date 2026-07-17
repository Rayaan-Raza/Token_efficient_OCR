"""Fixed prompt templates for DocVQA VLM evaluation.

Templates match the research plan (Phase 9):
    - Single image: one visual input + question
    - Overview + patches: low-res context plus high-res crops + question
    - Overview + patches + OCR: document-RAG style textual evidence
    - Constrained MCQ / binary verification for retrieve-extract-verify
"""

SINGLE_IMAGE = (
    "You are given an image. Answer the question using only the visual information.\n\n"
    "Question: {question}\nAnswer:"
)

OVERVIEW_PLUS_PATCHES = (
    "You are given one low-resolution overview image followed by high-resolution patches "
    "from the same original image. Use the overview for global layout and the patches for "
    "small text. Answer the question using only the provided visual information.\n\n"
    "Question: {question}\nAnswer:"
)

OVERVIEW_PATCHES_WITH_OCR = (
    "You are answering a document question.\n\n"
    "Use the provided document image patches and OCR evidence.\n"
    "Return only the final answer.\n"
    "Do not explain.\n"
    "Do not write a sentence.\n"
    "Do not include prefixes like \"The answer is\".\n"
    "If the answer is a name, number, date, amount, or phrase, output only that exact text.\n\n"
    "Question:\n{question}\n\n"
    "OCR evidence:\n{ocr_evidence}\n\n"
    "Answer:"
)

MCQ_VERIFIER = (
    "You are answering a document question.\n\n"
    "Question:\n{question}\n\n"
    "Candidate answers:\n{options}\n\n"
    "Look at the document evidence images.\n"
    "Choose the best answer.\n"
    "Return only one letter: {letters}.\n"
)

BINARY_VERIFIER = (
    "You are verifying a candidate answer for a document question.\n\n"
    "Question:\n{question}\n\n"
    "Candidate answer:\n{candidate}\n\n"
    "Supporting OCR snippet:\n{evidence}\n\n"
    "Is this candidate fully supported by the document evidence and does it answer "
    "the question exactly?\n"
    "Return only YES or NO.\n"
)


def format_single(question: str) -> str:
    """Format the single-image DocVQA prompt."""
    return SINGLE_IMAGE.format(question=question)


def format_overview_patches(question: str) -> str:
    """Format the overview-plus-patches DocVQA prompt (BOPS input layout)."""
    return OVERVIEW_PLUS_PATCHES.format(question=question)


def format_overview_patches_with_ocr(question: str, ocr_lines: list[str]) -> str:
    """Overview + patches prompt with strict short-answer OCR evidence block."""
    evidence = "\n".join(line.strip() for line in ocr_lines if line.strip()) or "(none)"
    return OVERVIEW_PATCHES_WITH_OCR.format(question=question, ocr_evidence=evidence)


def format_mcq_verifier(
    question: str, candidates: list[str], include_none: bool = True
) -> tuple[str, dict[str, str]]:
    """Build an option-letter verifier prompt.

    Returns:
        prompt text and mapping from option letter -> candidate text (or NONE).
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = []
    mapping: dict[str, str] = {}
    for i, cand in enumerate(candidates):
        letter = letters[i]
        mapping[letter] = cand
        options.append(f"{letter}. {cand}")
    if include_none:
        letter = letters[len(candidates)]
        mapping[letter] = "NONE"
        options.append(f"{letter}. none of the above")
    prompt = MCQ_VERIFIER.format(
        question=question,
        options="\n".join(options),
        letters="/".join(mapping.keys()),
    )
    return prompt, mapping


def format_binary_verifier(question: str, candidate: str, evidence: str) -> str:
    return BINARY_VERIFIER.format(
        question=question,
        candidate=candidate,
        evidence=evidence or "(none)",
    )

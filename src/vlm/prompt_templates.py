"""Fixed prompt templates for DocVQA VLM evaluation.

Templates match the research plan (Phase 9):
    - Single image: one visual input + question
    - Overview + patches: low-res context plus high-res crops + question
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


def format_single(question: str) -> str:
    """Format the single-image DocVQA prompt.

    Args:
        question: Natural-language question about the document.

    Returns:
        Complete prompt string for the VLM.
    """
    return SINGLE_IMAGE.format(question=question)


def format_overview_patches(question: str) -> str:
    """Format the overview-plus-patches DocVQA prompt (BOPS input layout).

    Args:
        question: Natural-language question about the document.

    Returns:
        Complete prompt string; images are passed separately in order
        [overview, patch_1, patch_2, ...].
    """
    return OVERVIEW_PLUS_PATCHES.format(question=question)

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
    return SINGLE_IMAGE.format(question=question)


def format_overview_patches(question: str) -> str:
    return OVERVIEW_PLUS_PATCHES.format(question=question)

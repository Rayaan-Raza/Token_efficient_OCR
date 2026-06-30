from src.vlm.qa_metrics import anls, exact_match


def test_exact_match():
    assert exact_match("March 12 2024", ["12 March 2024"]) == 0.0
    assert exact_match("hello", ["hello"]) == 1.0


def test_anls_perfect():
    assert anls("hello", ["hello"]) == 1.0


def test_anls_partial():
    score = anls("helo", ["hello"])
    assert 0.0 <= score <= 1.0

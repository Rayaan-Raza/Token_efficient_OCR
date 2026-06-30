from src.ocr.ocr_metrics import cer, wer, word_recall


def test_cer_wer_perfect_match():
    assert cer("hello world", "hello world") == 0.0
    assert wer("hello world", "hello world") == 0.0


def test_word_recall_perfect():
    assert word_recall("hello world", "hello world") == 1.0


def test_word_recall_partial():
    assert word_recall("hello", "hello world") == 0.5

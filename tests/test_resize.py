from PIL import Image

from src.preprocessing.resize import resize_to_area_ratio


def _solid(w: int, h: int) -> Image.Image:
    return Image.new("RGB", (w, h), color=(128, 128, 128))


def test_resize_area_ratio_half():
    img = _solid(1000, 1000)
    resized, meta = resize_to_area_ratio(img, 0.5)
    actual_ratio = (resized.width * resized.height) / (img.width * img.height)
    assert abs(actual_ratio - 0.5) <= 0.03
    assert meta["invalid_budget"] is False

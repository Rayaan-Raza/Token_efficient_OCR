import json
import tempfile
from pathlib import Path

from PIL import Image

from src.data.validate_manifest import validate_manifest


def test_manifest_validation_rejects_duplicate_ids():
    with tempfile.TemporaryDirectory() as td:
        img_path = Path(td) / "a.png"
        Image.new("RGB", (10, 10)).save(img_path)
        manifest = Path(td) / "m.jsonl"
        row = {
            "image_id": "x1",
            "dataset": "T",
            "split": "val",
            "image_path": str(img_path),
        }
        with open(manifest, "w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.write(json.dumps(row) + "\n")
        result = validate_manifest(manifest)
        assert result["valid"] is False

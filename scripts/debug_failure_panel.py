"""Debug failure panel green boxes."""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.dataset_loader import iter_manifest
from src.ocr.run_ocr import run_ocr_with_boxes
from src.utils.experiment_io import filter_paper_dataframe
from src.utils.image_io import load_image
from src.utils.paths import data_path, repo_path
from src.visualization.paper_figures import _answer_substring_in_text, _parse_answers

vlm = filter_paper_dataframe(pd.read_csv(REPO / "outputs/metrics/vlm_metrics_merged.csv"))
bops = vlm[vlm["method"] == "bops"]


def bcol(s):
    return s.map(lambda v: str(v).lower() in ("true", "1", "1.0"))


cand = bops[bcol(bops["answer_in_full_image_ocr"]) & ~bcol(bops["answer_in_selected_patch_ocr"]) & (bops["exact_match"] < 1)]
cand["q_len"] = cand["question"].astype(str).str.len()
cand = cand.sort_values("q_len")
manifest = {r["image_id"]: r for r in iter_manifest(data_path("manifests", "docvqa_pilot.jsonl"))}

for i in range(min(4, len(cand))):
    row = cand.iloc[i]
    image_id = row["image_id"]
    rec = manifest[image_id]
    img = load_image(repo_path(rec["image_path"]))
    answers = _parse_answers(str(row["ground_truth_answer"]))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        img.save(tmp.name)
        boxes = run_ocr_with_boxes(tmp.name)
    full_text = " ".join(b.get("text", "") for b in boxes)
    print("===", image_id, "img", img.size, "boxes", len(boxes))
    print("answers", answers)
    print("answer_in_full (diag)", row["answer_in_full_image_ocr"])
    print("full text match", _answer_substring_in_text(answers, full_text))
    matches = 0
    for b in boxes:
        if _answer_substring_in_text(answers, b.get("text", "")):
            matches += 1
            print("  MATCH:", repr(b.get("text")), "box", b.get("box"))
    print("matching boxes:", matches)
    if boxes:
        print("sample box structure:", boxes[0])

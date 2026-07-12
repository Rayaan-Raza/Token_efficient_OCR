"""OCR layout graph for document-structure features."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OCRNode:
    node_id: int
    box: list[list[int]]
    text: str
    confidence: float
    cx: float
    cy: float


@dataclass
class OCRLayoutGraph:
    nodes: list[OCRNode] = field(default_factory=list)
    edges: list[tuple[int, int, str]] = field(default_factory=list)

    def neighbors(self, node_id: int, rel: str | None = None) -> list[tuple[int, str]]:
        out = []
        for a, b, r in self.edges:
            if a == node_id and (rel is None or r == rel):
                out.append((b, r))
            elif b == node_id and (rel is None or r == rel):
                out.append((a, r))
        return out


def _center(box: list[list[int]]) -> tuple[float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def build_ocr_layout_graph(ocr_boxes: list[dict[str, Any]]) -> OCRLayoutGraph:
    nodes: list[OCRNode] = []
    for i, b in enumerate(ocr_boxes):
        cx, cy = _center(b["box"])
        nodes.append(
            OCRNode(i, b["box"], b.get("text", ""), float(b.get("confidence", 0.0)), cx, cy)
        )
    edges: list[tuple[int, int, str]] = []
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i >= j:
                continue
            dy = abs(a.cy - b.cy)
            dx = abs(a.cx - b.cx)
            if dy < 15 and dx < max(a.box[2][0] - a.box[0][0], 50):
                edges.append((i, j, "same_row"))
            if dx < 15 and dy < 40:
                edges.append((i, j, "same_column"))
            if b.cx > a.cx and dy < 20 and 0 < b.cx - a.cx < 200:
                edges.append((i, j, "nearby_right"))
            if b.cy > a.cy and dx < 40 and 0 < b.cy - a.cy < 80:
                edges.append((i, j, "nearby_below"))
    return OCRLayoutGraph(nodes=nodes, edges=edges)


def graph_features_for_patch(
    graph: OCRLayoutGraph,
    patch_x: int,
    patch_y: int,
    patch_w: int,
    patch_h: int,
    question_tokens: set[str],
) -> dict[str, float]:
    """Structural features using OCR document graph."""
    px0, py0, px1, py1 = patch_x, patch_y, patch_x + patch_w, patch_y + patch_h

    def _in_patch(n: OCRNode) -> bool:
        return px0 <= n.cx < px1 and py0 <= n.cy < py1

    patch_nodes = [n for n in graph.nodes if _in_patch(n)]
    q_nodes = [
        n for n in graph.nodes
        if question_tokens and any(t in n.text.lower() for t in question_tokens)
    ]

    same_row_lv = 0.0
    below_label = 0.0
    min_dist = 9999.0

    for qn in q_nodes:
        for nb_id, rel in graph.neighbors(qn.node_id):
            nb = graph.nodes[nb_id]
            if _in_patch(nb):
                if rel == "same_row":
                    same_row_lv = max(same_row_lv, 1.0)
                if rel == "nearby_below":
                    below_label = max(below_label, 1.0)
            d = math.hypot(nb.cx - qn.cx, nb.cy - qn.cy)
            min_dist = min(min_dist, d)

    if min_dist == 9999.0:
        min_dist = 0.0

    line_density = len(patch_nodes) / max(1, patch_h / 20.0)
    return {
        "same_row_label_value": same_row_lv,
        "below_label_relation": below_label,
        "dist_qnode_to_patch": 1.0 / (1.0 + min_dist),
        "patch_line_density": float(line_density),
        "patch_box_density": float(len(patch_nodes)),
    }

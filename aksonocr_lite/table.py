"""Table structure: Microsoft Table Transformer v1.1-all (rows, columns, headers, spanning cells), filled with OCR.

TATR finds the grid; the text in each cell comes from our own OCR lines, so Thai is read by the Thai recognizer.
Grid rules: rows/columns are de-duplicated (>50% 1-D overlap with a higher-scoring one) and snapped so neighbours
meet half way; cells are row x column intersections; a spanning cell merges the grid slots it covers by more than
half; a projected row header spans the full width; rows inside a column-header box are header rows.
"""
from __future__ import annotations

import html
import json

import cv2
import numpy as np

from .models import model_path, session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
LABELS = ["table", "table column", "table row", "table column header", "table projected row header",
          "table spanning cell"]


class TableStructure:
    def __init__(self, device: str = "auto", threshold: float = 0.5, pad: int = 20, longest: int = 1000):
        self.sess = session("tatr.onnx", device)
        self.threshold, self.pad, self.longest = threshold, pad, longest

    def detect(self, crop_bgr: np.ndarray) -> list[dict]:
        h, w = crop_bgr.shape[:2]
        k = self.longest / max(h, w)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_LINEAR)
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        logits, boxes = self.sess.run(None, {"pixel_values": x})
        e = np.exp(logits[0] - logits[0].max(-1, keepdims=True))
        prob = (e / e.sum(-1, keepdims=True))[:, :-1]  # last class is "no object"
        labels, scores = prob.argmax(-1), prob.max(-1)
        out = []
        for lab, sc, (cx, cy, bw, bh) in zip(labels, scores, boxes[0]):
            if sc <= self.threshold:
                continue
            out.append({"label": LABELS[int(lab)], "score": float(sc),
                        "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]})
        return out

    def __call__(self, crop_bgr: np.ndarray) -> tuple[list[dict], float]:
        """Table crop -> (grid cells in crop pixels, mean row/column score)."""
        h, w = crop_bgr.shape[:2]
        p = self.pad
        canvas = cv2.copyMakeBorder(crop_bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cells, score = grid_cells(self.detect(canvas))
        for c in cells:
            x0, y0, x1, y1 = c["bbox"]
            c["bbox"] = [max(0.0, x0 - p), max(0.0, y0 - p), min(w, x1 - p), min(h, y1 - p)]
        return cells, score


def _overlap_1d(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0)) / max(min(a1 - a0, b1 - b0), 1e-6)


def _dedupe(objs, axis):
    keep = []
    for o in sorted(objs, key=lambda o: -o["score"]):
        b = o["bbox"]
        if all(_overlap_1d(b[axis], b[axis + 2], q["bbox"][axis], q["bbox"][axis + 2]) < 0.5 for q in keep):
            keep.append(o)
    return sorted(keep, key=lambda o: o["bbox"][axis])


def grid_cells(objs):
    rows = _dedupe([o for o in objs if o["label"] == "table row"], 1)
    cols = _dedupe([o for o in objs if o["label"] == "table column"], 0)
    if not rows or not cols:
        return [], 0.0
    ry = [[r["bbox"][1], r["bbox"][3]] for r in rows]
    cx = [[c["bbox"][0], c["bbox"][2]] for c in cols]
    for edges in (ry, cx):
        for i in range(len(edges) - 1):
            m = (edges[i][1] + edges[i + 1][0]) / 2
            edges[i][1] = edges[i + 1][0] = m
    heads = [o["bbox"] for o in objs if o["label"] == "table column header"]
    header = [any(_overlap_1d(a, b, hb[1], hb[3]) > 0.5 for hb in heads) for a, b in ry]
    spans = []
    for o in objs:
        if o["label"] not in ("table spanning cell", "table projected row header"):
            continue
        b = o["bbox"]
        rs = [i for i, (a, c) in enumerate(ry) if max(0.0, min(c, b[3]) - max(a, b[1])) > 0.5 * (c - a)]
        cs = [j for j, (a, c) in enumerate(cx) if max(0.0, min(c, b[2]) - max(a, b[0])) > 0.5 * (c - a)]
        if o["label"] == "table projected row header":
            cs = list(range(len(cx)))
        if rs and cs and len(rs) * len(cs) > 1:
            spans.append((min(rs), max(rs), min(cs), max(cs)))
    taken, cells = set(), []
    for r0, r1, c0, c1 in sorted(spans, key=lambda t: -(t[1] - t[0] + 1) * (t[3] - t[2] + 1)):
        slots = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if slots & taken:
            continue
        taken |= slots
        cells.append((r0, r1, c0, c1))
    cells += [(r, r, c, c) for r in range(len(ry)) for c in range(len(cx)) if (r, c) not in taken]
    out = [{"bbox": [cx[c0][0], ry[r0][0], cx[c1][1], ry[r1][1]], "row": r0, "col": c0,
            "rowspan": r1 - r0 + 1, "colspan": c1 - c0 + 1, "is_header": bool(header[r0])}
           for r0, r1, c0, c1 in cells]
    return out, sum(o["score"] for o in rows + cols) / (len(rows) + len(cols))


def _distance(a, b):
    d = abs(b[0] - a[0]) + abs(b[1] - a[1]) + abs(b[2] - a[2]) + abs(b[3] - a[3])
    return d + min(abs(b[0] - a[0]) + abs(b[1] - a[1]), abs(b[2] - a[2]) + abs(b[3] - a[3]))


def _iou(a, b):
    l, r, t, btm = max(a[0], b[0]), min(a[2], b[2]), max(a[1], b[1]), min(a[3], b[3])
    if l >= r or t >= btm:
        return 0.0
    inter = (r - l) * (btm - t)
    union = max(0.0, (a[2] - a[0]) * (a[3] - a[1])) + max(0.0, (b[2] - b[0]) * (b[3] - b[1])) - inter
    return inter / union if union > 0 else 0.0


def match_lines_to_cells(cell_boxes, line_boxes):
    """Each OCR line goes to the cell with the highest IoU, ties broken by L1 corner distance
    (PaddleX table_recognition_post_processing.match_table_and_ocr, Apache-2.0)."""
    matched = {}
    for i, lb in enumerate(line_boxes):
        best = min(range(len(cell_boxes)), key=lambda j: (1.0 - _iou(cell_boxes[j], lb), _distance(cell_boxes[j], lb)))
        matched.setdefault(best, []).append(i)
    return matched


def grid_to_html(cells, lines):
    """cells (row/col/span, crop coords) + lines [{"text", "bbox"}] in crop coords -> <table> HTML."""
    if not cells:
        return ""
    matched = match_lines_to_cells([c["bbox"] for c in cells], [l["bbox"] for l in lines]) if lines else {}
    n_rows = max(c["row"] + c["rowspan"] for c in cells)
    n_cols = max(c["col"] + c["colspan"] for c in cells)
    occupied = [[False] * n_cols for _ in range(n_rows)]
    origin = {}
    for i, c in enumerate(cells):
        origin.setdefault((c["row"], c["col"]), i)
    parts = ["<table>"]
    for r in range(n_rows):
        parts.append("<tr>")
        c = 0
        while c < n_cols:
            if occupied[r][c]:
                c += 1
                continue
            idx = origin.get((r, c))
            if idx is None:
                parts.append("<td></td>")
                c += 1
                continue
            cell = cells[idx]
            for rr in range(r, min(n_rows, r + cell["rowspan"])):
                for cc in range(c, min(n_cols, c + cell["colspan"])):
                    occupied[rr][cc] = True
            text = " ".join(t for t in ((lines[k]["text"] or "").strip() for k in matched.get(idx, [])) if t)
            tag = "th" if cell["is_header"] else "td"
            attrs = (f' rowspan="{cell["rowspan"]}"' if cell["rowspan"] > 1 else "") + \
                    (f' colspan="{cell["colspan"]}"' if cell["colspan"] > 1 else "")
            parts.append(f"<{tag}{attrs}>{html.escape(text, quote=False)}</{tag}>")
            c += cell["colspan"]
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)

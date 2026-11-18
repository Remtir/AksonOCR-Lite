"""Text-line detection: PP-OCRv6_medium_det (DB) + Thai mark-fragment filter + line crops.

DB pre/post-processing follows PaddleX's DetResizeForTest / DBPostProcess
(Copyright (c) 2024 PaddlePaddle Authors, Apache License 2.0).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pyclipper

from .models import session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class TextDetector:
    def __init__(self, device: str = "auto", limit_side_len: int = 1280, limit_type: str = "max",
                 thresh: float = 0.3, box_thresh: float = 0.3, unclip_ratio: float = 2.0,
                 max_candidates: int = 3000, max_side_limit: int = 4000, drop_mark_fragments: bool = True):
        self.sess = session("det.onnx", device)
        self.input = self.sess.get_inputs()[0].name
        self.limit_side_len, self.limit_type, self.max_side_limit = limit_side_len, limit_type, max_side_limit
        self.thresh, self.box_thresh, self.unclip_ratio = thresh, box_thresh, unclip_ratio
        self.max_candidates, self.drop_marks = max_candidates, drop_mark_fragments

    def __call__(self, img_bgr: np.ndarray, unclip_ratio: float | None = None) -> list[np.ndarray]:
        """BGR page -> text-line quads (4x2 float32, clockwise from top-left) in reading-ish order."""
        h, w = img_bgr.shape[:2]
        if h + w < 64:
            pad = np.zeros((max(32, h), max(32, w), 3), np.uint8)
            pad[:h, :w] = img_bgr
            img_bgr = pad
        x, (rh, rw) = self._resize(img_bgr)
        x = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        pred = self.sess.run(None, {self.input: x})[0][0, 0]
        boxes = self._boxes(pred, pred > self.thresh, w, h, unclip_ratio or self.unclip_ratio)
        quads = sorted_boxes([b.astype(np.float32) for b in boxes])
        if self.drop_marks:
            quads = drop_mark_fragments(quads)
        return quads

    def _resize(self, img):
        h, w = img.shape[:2]
        L = self.limit_side_len
        if self.limit_type == "max":
            ratio = L / max(h, w) if max(h, w) > L else 1.0
        elif self.limit_type == "min":
            ratio = L / min(h, w) if min(h, w) < L else 1.0
        else:  # resize_long
            ratio = L / max(h, w)
        rh, rw = int(h * ratio), int(w * ratio)
        if max(rh, rw) > self.max_side_limit:
            r2 = self.max_side_limit / max(rh, rw)
            rh, rw = int(rh * r2), int(rw * r2)
        rh, rw = max(int(round(rh / 32) * 32), 32), max(int(round(rw / 32) * 32), 32)
        if (rh, rw) == (h, w):
            return img, (1.0, 1.0)
        return cv2.resize(img, (rw, rh)), (rh / h, rw / w)

    def _boxes(self, pred, bitmap, dest_w, dest_h, unclip_ratio):
        height, width = bitmap.shape
        sx, sy = dest_w / width, dest_h / height
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for contour in contours[: self.max_candidates]:
            pts, sside = _mini_box(contour)
            if sside < 3:
                continue
            pts = np.array(pts)
            if _box_score(pred, pts.reshape(-1, 2)) < self.box_thresh:
                continue
            box, sside = _mini_box(_unclip(pts, unclip_ratio).reshape(-1, 1, 2))
            if sside < 5:
                continue
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] * sx), 0, dest_w)
            box[:, 1] = np.clip(np.round(box[:, 1] * sy), 0, dest_h)
            out.append(box.astype(np.int16))
        return out


def _unclip(box, ratio):
    area, length = cv2.contourArea(box), cv2.arcLength(box, True)
    off = pyclipper.PyclipperOffset()
    off.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    try:
        return np.array(off.Execute(area * ratio / length))
    except ValueError:
        return np.array(off.Execute(area * ratio / length)[0])


def _mini_box(contour):
    rect = cv2.minAreaRect(contour)
    p = sorted(list(cv2.boxPoints(rect)), key=lambda x: x[0])
    i1, i4 = (0, 1) if p[1][1] > p[0][1] else (1, 0)
    i2, i3 = (2, 3) if p[3][1] > p[2][1] else (3, 2)
    return [p[i1], p[i2], p[i3], p[i4]], min(rect[1])


def _box_score(bitmap, box):
    h, w = bitmap.shape[:2]
    box = box.copy()
    x0 = max(0, min(math.floor(box[:, 0].min()), w - 1)); x1 = max(0, min(math.ceil(box[:, 0].max()), w - 1))
    y0 = max(0, min(math.floor(box[:, 1].min()), h - 1)); y1 = max(0, min(math.ceil(box[:, 1].max()), h - 1))
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    box[:, 0] -= x0
    box[:, 1] -= y0
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[y0:y1 + 1, x0:x1 + 1], mask)[0]


def sorted_boxes(quads):
    """Top-to-bottom, then left-to-right within ~10 px rows (PaddleOCR's sorted_boxes)."""
    boxes = sorted(quads, key=lambda b: (b[0][1], b[0][0]))
    for i in range(len(boxes) - 1):
        for j in range(i, -1, -1):
            if abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10 and boxes[j + 1][0][0] < boxes[j][0][0]:
                boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
            else:
                break
    return boxes


def _rect(q):
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def _area(q):
    return abs(float(cv2.contourArea(q.astype(np.float32))))


def drop_mark_fragments(quads):
    """Drop boxes around lone Thai marks (ั ้ ี ุ ...) next to or inside a real text line.

    The PP-OCR detector often boxes an upper or lower Thai vowel/tone mark on its own; the recognizer then reads the
    sliver as garbage. A box is a fragment of a neighbouring line when it is short in absolute terms (< 0.5x the
    page's median line height), much shorter (< 0.55x) and narrower (< 0.5x) than that line, the line is close to
    a rectangle (area >= 0.7 of its bbox), >= 80% of its width lies over the line, and its centre sits within
    0.8 line-heights outside to 0.3 inside the line's top/bottom edge, or >= 60% of its area is inside the line.
    """
    rects = [_rect(q) for q in quads]
    hs = sorted(r[3] - r[1] for r in rects if (r[2] - r[0]) >= 1.5 * (r[3] - r[1]))
    med_h = hs[len(hs) // 2] if hs else None
    boxy = [_area(q) >= 0.7 * max((r[2] - r[0]) * (r[3] - r[1]), 1.0) for q, r in zip(quads, rects)]
    keep = []
    for i, (x0, y0, x1, y1) in enumerate(rects):
        w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        if med_h is not None and h >= 0.5 * med_h:
            keep.append(i)
            continue
        cy, frag = (y0 + y1) / 2, False
        for j, (X0, Y0, X1, Y1) in enumerate(rects):
            if i == j or not boxy[j]:
                continue
            W, H = X1 - X0, Y1 - Y0
            if h > 0.55 * H or w > 0.5 * W:
                continue
            ix = max(0.0, min(x1, X1) - max(x0, X0))
            if ix < 0.8 * w:
                continue
            iy = max(0.0, min(y1, Y1) - max(y0, Y0))
            near = (Y0 - 0.8 * H) <= cy <= (Y0 + 0.3 * H) or (Y1 - 0.3 * H) <= cy <= (Y1 + 0.8 * H)
            if near or ix * iy >= 0.6 * w * h:
                frag = True
                break
        if not frag:
            keep.append(i)
    return [quads[i] for i in keep]


def crop_quad(img, quad):
    """Perspective-warp a quad to an upright crop; tall crops (h/w >= 1.5) are turned 90 degrees."""
    pts = np.asarray(quad, np.float32)
    w = max(1, int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))))
    h = max(1, int(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))))
    M = cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    dst = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    if dst.shape[0] / max(dst.shape[1], 1) >= 1.5:
        dst = np.rot90(dst)
    return np.ascontiguousarray(dst)

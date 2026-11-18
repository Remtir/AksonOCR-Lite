"""Orientation classifiers: PP-LCNet_x1_0_doc_ori (page, 0/90/180/270) and textline_ori (line, 0/180)."""
from __future__ import annotations

import cv2
import numpy as np

from .models import session

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def _norm(rgb):
    return ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)


class PageOrientation:
    ANGLES = (0, 90, 180, 270)

    def __init__(self, device: str = "auto"):
        self.sess = session("doc_ori.onnx", device)
        self.input = self.sess.get_inputs()[0].name

    def classify(self, img_bgr: np.ndarray) -> tuple[int, float]:
        """-> (label, score): label is the counter-clockwise rotation that makes the page upright."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        s = 256 / min(h, w)
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)))
        h, w = rgb.shape[:2]
        y, x = (h - 224) // 2, (w - 224) // 2
        p = self.sess.run(None, {self.input: _norm(rgb[y:y + 224, x:x + 224])[None]})[0][0]
        k = int(p.argmax())
        return self.ANGLES[k], float(p[k])


    def __call__(self, img_bgr: np.ndarray, single_pass_min: float = 0.92, min_conf: float = 0.85,
                 margin: float = 0.05) -> int:
        """-> counter-clockwise rotation (0/90/180/270) to apply with `rotate()`.

        One pass when the classifier is confident; otherwise score every candidate rotation by its "upright"
        probability, settle 90/270 first, then 0/180, and leave the page alone when it stays ambiguous."""
        probe = _downscale(img_bgr, 1280)
        label, conf = self.classify(probe)
        if conf >= single_pass_min:
            return label
        scores = self._upright_scores(probe)
        side = 90 if scores[90] >= scores[270] else 270
        stage1 = 0
        if scores[side] >= min_conf and scores[side] >= max(scores[0], scores[180]) + margin:
            stage1 = side
            scores = self._upright_scores(rotate(probe, side))
        flip = 180 if scores[180] >= scores[0] + margin and scores[180] >= min_conf else 0
        return (stage1 + flip) % 360

    def _upright_scores(self, img):
        out = {}
        for a in self.ANGLES:
            label, conf = self.classify(rotate(img, a))
            out[a] = conf if label == 0 else 0.0
        return out


def _downscale(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def rotate(img, angle: int):
    """Rotate counter-clockwise by `angle` (0/90/180/270), like PIL's Image.rotate(angle, expand=True)."""
    return {0: img, 90: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 180: cv2.rotate(img, cv2.ROTATE_180),
            270: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)}[angle % 360]


class LineOrientation:
    def __init__(self, device: str = "auto", batch: int = 64):
        self.sess = session("textline_ori.onnx", device)
        self.input, self.batch = self.sess.get_inputs()[0].name, batch

    def __call__(self, crops_bgr: list[np.ndarray]) -> list[bool]:
        """-> True for crops classified as upside down."""
        out = []
        for i in range(0, len(crops_bgr), self.batch):
            x = np.stack([_norm(cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (160, 80))) for c in crops_bgr[i:i + self.batch]])
            out += list(self.sess.run(None, {self.input: x})[0].argmax(-1) == 1)
        return out

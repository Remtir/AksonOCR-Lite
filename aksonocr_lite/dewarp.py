"""Page dewarping for photographed documents: the MonkeyOCRv2 document preprocessor (Apache-2.0).

A U2NETP net segments the page, the largest connected component masks the input, and a Swin-based net predicts
a backward sampling grid at 512x512 that is upsampled to the full image and applied with cv2.remap.
Off by default (`Config.dewarp`); it adds ~290 MB of weights and helps photos, not scans.
"""
from __future__ import annotations

import cv2
import numpy as np

from .models import session

S = 512


def _resize_align_corners(a: np.ndarray, H: int, W: int) -> np.ndarray:
    """Bilinear resize of a (h, w) map with align_corners=True semantics (torch F.interpolate)."""
    h, w = a.shape
    ys = np.linspace(0, h - 1, H, dtype=np.float32)
    xs = np.linspace(0, w - 1, W, dtype=np.float32)
    mx, my = np.meshgrid(xs, ys)
    return cv2.remap(a.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class Dewarper:
    def __init__(self, device: str = "auto", mask_thresh: float = 0.8, skip_coverage: float = 0.9):
        """skip_coverage: leave the image alone when the page fills at least this fraction of it. Scans and
        born-digital pages fill ~100% and only lose detail when resampled (MDPBench-th digital pages: 89.4
        unwarped vs 87.3 warped); photos with background around the page measure 0.26-0.85."""
        self.mask = session("dewarp_mask.onnx", device)
        self.grid = session("dewarp_grid.onnx", device)
        self.mask_thresh, self.skip_coverage = mask_thresh, skip_coverage

    def __call__(self, img_bgr: np.ndarray) -> tuple[np.ndarray, bool]:
        """-> (dewarped BGR image of the same size, applied)."""
        H, W = img_bgr.shape[:2]
        rgb = img_bgr[:, :, ::-1].astype(np.float32) / 255.0
        x = cv2.resize(rgb, (S, S)).transpose(2, 0, 1)[None].copy()
        m = (self.mask.run(None, {"x": x})[0][0, 0] > self.mask_thresh).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 1:
            m = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        if m.mean() >= self.skip_coverage:
            return img_bgr, False
        g = self.grid.run(None, {"x": x * m[None, None].astype(np.float32)})[0][0]
        if not np.isfinite(g).all():
            return img_bgr, False
        gx = _resize_align_corners(g[0], H, W)
        gy = _resize_align_corners(g[1], H, W)
        mapx, mapy = (gx + 1) * 0.5 * (W - 1), (gy + 1) * 0.5 * (H - 1)
        out = cv2.remap(img_bgr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return out, True

"""Layout analysis: PP-DocLayoutV3 (RT-DETR with mask and reading-order heads), PaddleX post-processing."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ._vendor.paddlex_layout import LayoutAnalysisProcess
from .models import session

LABELS = ["abstract", "algorithm", "aside_text", "chart", "content", "display_formula", "doc_title",
          "figure_title", "footer", "footer_image", "footnote", "formula_number", "header", "header_image",
          "image", "inline_formula", "number", "paragraph_title", "reference", "reference_content", "seal",
          "table", "text", "vertical_text", "vision_footnote"]
SIZE = 800


@dataclass
class Block:
    label: str
    score: float
    bbox: list[float]            # x0, y0, x1, y1 in page pixels
    order: float | None = None   # reading order (None for labels the model does not order)
    poly: list[list[float]] | None = None
    extra: dict = field(default_factory=dict)


class LayoutDetector:
    def __init__(self, device: str = "auto", threshold: float = 0.1, nms: bool = True,
                 merge_bboxes_mode: str = "large", unclip_ratio=None):
        self.sess = session("layout.onnx", device)
        self.names = [i.name for i in self.sess.get_inputs()]
        self.post = LayoutAnalysisProcess(labels=LABELS, scale_size=[SIZE, SIZE])
        self.kw = dict(threshold=threshold, layout_nms=nms, layout_unclip_ratio=unclip_ratio,
                       layout_merge_bboxes_mode=merge_bboxes_mode, layout_shape_mode="auto",
                       filter_overlap_boxes=True, skip_order_labels=None)

    def __call__(self, img_bgr: np.ndarray) -> list[Block]:
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
        feeds = {"im_shape": np.array([[SIZE, SIZE]], np.float32),
                 "image": x.transpose(2, 0, 1)[None],
                 "scale_factor": np.array([[SIZE / h, SIZE / w]], np.float32)}
        boxes, num, masks = self.sess.run(None, {k: feeds[k] for k in self.names})
        n = int(num[0])
        out = self.post([{"boxes": boxes[:n], "masks": masks[:n]}], [{"ori_img_size": [w, h]}], **self.kw)[0]
        blocks = []
        for b in out:
            poly = b.get("polygon_points")
            blocks.append(Block(label=b["label"], score=float(b["score"]),
                                bbox=[float(v) for v in b["coordinate"]], order=b.get("order"),
                                poly=[[float(p[0]), float(p[1])] for p in poly] if poly is not None and len(poly) >= 3 else None))
        fill_missing_order(blocks)
        return sorted(blocks, key=lambda b: b.order)


def fill_missing_order(blocks: list[Block]) -> None:
    """Give unordered blocks (headers, tables, figures...) an order between the ordered blocks above and below
    them, processing them top-to-bottom so two unordered blocks never tie."""
    ordered = [b for b in blocks if b.order is not None]
    real = list(ordered)
    for u in sorted((b for b in blocks if b.order is None), key=lambda b: (b.bbox[1], b.bbox[0])):
        base = max((b.order for b in ordered if b.bbox[1] <= u.bbox[1]), default=0)
        nxt = min((b.order for b in real if b.bbox[1] > u.bbox[1]), default=None)
        u.order = (base + nxt) / 2 if nxt is not None and nxt > base else base + 0.5
        ordered.append(u)

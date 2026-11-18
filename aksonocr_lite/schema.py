"""Output types."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Line:
    text: str
    confidence: float
    bbox: list[float]            # x0, y0, x1, y1
    poly: list[list[float]]      # 4 points, clockwise from top-left


@dataclass
class Block:
    id: int
    label: str                   # PP-DocLayoutV3 label (text, paragraph_title, table, image, ...)
    bbox: list[float]
    score: float
    order: float
    text: str = ""               # OCR text of the block (newline between lines)
    html: str | None = None      # tables: <table> HTML
    line_ids: list[int] = field(default_factory=list)


@dataclass
class Page:
    index: int
    width: int
    height: int
    rotation: int                # counter-clockwise rotation applied before OCR (0/90/180/270)
    dewarped: bool
    markdown: str
    lines: list[Line]
    blocks: list[Block]
    timings: dict

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text and b.label not in ("image", "chart", "seal"))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["text"] = self.text
        return d

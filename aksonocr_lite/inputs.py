"""Input loading: images (EXIF-corrected) and PDFs (rendered with pypdfium2)."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps

PDF_MAGIC = b"%PDF"


def _to_bgr(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.ascontiguousarray(np.asarray(img)[:, :, ::-1])


def load_pages(src, dpi: int = 150, max_pages: int | None = None, pages: list[int] | None = None) -> Iterator[np.ndarray]:
    """Yield BGR page images from a path, bytes, PIL image or numpy array (BGR)."""
    if isinstance(src, np.ndarray):
        yield src if src.ndim == 3 else np.repeat(src[:, :, None], 3, axis=2)
        return
    if isinstance(src, Image.Image):
        yield _to_bgr(src)
        return
    data = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    if data[:4] != PDF_MAGIC:
        yield _to_bgr(Image.open(io.BytesIO(data)))
        return
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(bytes(data))
    try:
        idx = pages if pages is not None else range(len(pdf))
        for n, i in enumerate(idx):
            if max_pages is not None and n >= max_pages:
                break
            page = pdf[i]
            bitmap = page.render(scale=dpi / 72)
            yield _to_bgr(bitmap.to_pil())
            page.close()
    finally:
        pdf.close()

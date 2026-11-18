"""End-to-end smoke test; downloads weights (or uses AKSONOCR_WEIGHTS). Skip with -m 'not e2e'."""
import os

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

aksonocr_lite = pytest.importorskip("aksonocr_lite")
FONT = next((p for p in [os.getenv("AKSONOCR_TEST_FONT", ""), "/usr/share/fonts/truetype/tlwg/Garuda.ttf", "/usr/share/fonts/truetype/freefont/FreeSerif.ttf", "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
                         "/usr/share/fonts/truetype/tlwg/Loma.ttf",
                         "/System/Library/Fonts/Thonburi.ttc", "/System/Library/Fonts/Supplemental/Tahoma.ttf",  # macOS
                         "C:/Windows/Fonts/tahoma.ttf"] if os.path.exists(p)), None)


def _load():
    try:
        return aksonocr_lite.load(device="cpu")
    except Exception as e:  # weights repo private/unreachable and no token: nothing to test against
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
        if isinstance(e, (RepositoryNotFoundError, GatedRepoError)) and not os.getenv("HF_TOKEN"):
            pytest.skip("model weights not accessible (private HF repo, HF_TOKEN not set)")
        raise


def _page(*lines):
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 40)
    for i, t in enumerate(lines):
        d.text((100, 120 + 100 * i), t, font=f, fill="black")
    return img


@pytest.mark.e2e
@pytest.mark.skipif(FONT is None, reason="no Thai font installed")
def test_reads_rendered_thai_page():
    img = Image.new("RGB", (1240, 800), "white")
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, 40)
    d.text((80, 80), "ภาษาไทยเป็นภาษาราชการ", font=f, fill="black")
    d.text((80, 180), "Thai OCR 2026", font=f, fill="black")
    conv = _load()
    page = next(conv.convert(img))
    text = page.markdown.replace(" ", "")
    assert "ภาษาไทยเป็นภาษาราชการ" in text
    assert "ThaiOCR2026" in text
    # upside-down page is turned back
    flipped = cv2.rotate(np.asarray(img)[:, :, ::-1].copy(), cv2.ROTATE_180)
    page2 = next(conv.convert(flipped))
    assert "ภาษาไทยเป็นภาษาราชการ" in page2.markdown.replace(" ", "")


@pytest.mark.e2e
@pytest.mark.skipif(FONT is None, reason="no Thai font installed")
def test_multipage_pdf_cli(tmp_path):
    """A 3-page PDF through the `aksonocr` command: Markdown with page breaks, JSON with every page, --max-pages."""
    import json, shutil, subprocess, sys

    _load()  # skip early when the weights are not reachable
    pages = [_page("หน้าที่หนึ่ง", "Page one"), _page("หน้าที่สอง", "Page two"), _page("หน้าที่สาม", "Page three")]
    pdf = tmp_path / "doc.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=150)
    exe = shutil.which("aksonocr")
    cmd = [exe] if exe else [sys.executable, "-m", "aksonocr_lite"]
    subprocess.run(cmd + [str(pdf), "--out", str(tmp_path), "--device", "cpu", "-q"], check=True)
    md = (tmp_path / "doc.md").read_text(encoding="utf-8")
    parts = md.split("<!-- page break -->")
    assert len(parts) == 3
    for part, word in zip(parts, ["หน้าที่หนึ่ง", "หน้าที่สอง", "หน้าที่สาม"]):
        assert word in part.replace(" ", "")
    subprocess.run(cmd + [str(pdf), "-f", "json", "--out", str(tmp_path), "--device", "cpu", "--max-pages", "2", "-q"],
                   check=True)
    doc = json.loads((tmp_path / "doc.json").read_text(encoding="utf-8"))
    assert [p["index"] for p in doc["pages"]] == [0, 1]

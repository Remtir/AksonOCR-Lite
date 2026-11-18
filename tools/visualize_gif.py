"""Animated GIF of what AksonOCR-Lite sees on a page: layout blocks (in reading order), table cells (TATR grid),
then text-line boxes.

    python tools/visualize_gif.py page.jpg [more.png ...] --out assets/viz [--size 600x800]

Only boxes are drawn (no recognised text). Every GIF gets the same canvas size (page fitted and centred), so a row
of them lines up. Coordinates are those of the rotated page the pipeline worked on.
"""
import argparse, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aksonocr_lite  # noqa: E402
from aksonocr_lite.inputs import load_pages  # noqa: E402
from aksonocr_lite.orient import rotate  # noqa: E402

PALETTE = {  # layout class -> RGB
    "text": (52, 120, 246), "paragraph_title": (235, 87, 150), "doc_title": (235, 87, 150),
    "table": (16, 163, 127), "image": (245, 158, 11), "chart": (245, 158, 11), "seal": (245, 158, 11),
    "header": (139, 92, 246), "footer": (139, 92, 246), "number": (139, 92, 246),
    "figure_title": (234, 88, 12), "table_title": (234, 88, 12), "chart_title": (234, 88, 12),
    "display_formula": (14, 165, 233), "inline_formula": (14, 165, 233), "formula_number": (14, 165, 233),
}
OTHER = (100, 116, 139)
LINE = (220, 38, 38)
CELL = (16, 163, 127)
BAR_H = 34
PAD_BG = (228, 228, 231)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def frame(base, tf, title, blocks=(), cells=(), polys=(), stage="layout"):
    """Draw on a copy of the padded page `base`; tf = (scale, x offset, y offset) maps page px -> canvas px."""
    sc, ox, oy = tf
    X = lambda x: x * sc + ox  # noqa: E731
    Y = lambda y: y * sc + oy  # noqa: E731
    im = base.convert("RGBA")
    over = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    f = font(12, bold=True)
    later = stage != "layout"
    for k, b in enumerate(blocks):
        c = PALETTE.get(b.label, OTHER)
        x0, y0, x1, y1 = X(b.bbox[0]), Y(b.bbox[1]), X(b.bbox[2]), Y(b.bbox[3])
        fill = c + ((0 if b.label == "table" else 16) if later else 50,)
        d.rectangle([x0, y0, x1, y1], fill=fill, outline=c + (150 if later else 255,), width=2)
        if not later:
            tag = f"{k + 1} {b.label}"
            tw = d.textlength(tag, font=f)
            ty = max(0, y0 - 16)
            d.rectangle([x0, ty, x0 + tw + 8, ty + 16], fill=c + (235,))
            d.text((x0 + 4, ty + 1), tag, font=f, fill=(255, 255, 255, 255))
    for cell in cells:
        x0, y0, x1, y1 = X(cell["bbox"][0]), Y(cell["bbox"][1]), X(cell["bbox"][2]), Y(cell["bbox"][3])
        a = 70 if stage == "cells" else 25
        fill = CELL + (a + 40,) if cell["is_header"] else (CELL + (a // 2,) if cell["rowspan"] * cell["colspan"] > 1 else (0, 0, 0, 0))
        d.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], fill=fill, outline=CELL + (255 if stage == "cells" else 120,), width=1)
    for p in polys:
        pts = [(X(x), Y(y)) for x, y in p]
        d.polygon(pts, outline=LINE + (255,), fill=LINE + (28,))
        d.line(pts + [pts[0]], fill=LINE + (255,), width=2)
    im = Image.alpha_composite(im, over).convert("RGB")
    ImageDraw.Draw(im).rectangle([0, 0, im.width, BAR_H], fill=(24, 24, 27))
    ImageDraw.Draw(im).text((12, 8), title, font=font(15, bold=True), fill=(244, 244, 245))
    return im


def table_cells(conv, img, blocks):
    """TATR grid cells of every table block, in page coordinates."""
    out = []
    H, W = img.shape[:2]
    for b in blocks:
        if b.label != "table" or conv.tatr is None:
            continue
        x0, y0, x1, y1 = [max(0, int(v)) for v in b.bbox]
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        cells, _ = conv.tatr(np.ascontiguousarray(img[y0:y1, x0:x1]))
        for c in cells:
            bx = c["bbox"]
            out.append({**c, "bbox": [bx[0] + x0, bx[1] + y0, bx[2] + x0, bx[3] + y0]})
    return sorted(out, key=lambda c: (c["row"], c["col"]))


def _steps(n, target):
    step = max(1, n // target)
    return list(range(step, n, step)) + [n] if n else []


def make_gif(src, out_dir, conv, size, ms):
    img = next(load_pages(str(src)))
    page = conv.page(img)
    img = rotate(img, page.rotation)
    rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    W, H = size
    avail_h = H - BAR_H - 12
    sc = min((W - 12) / rgb.width, avail_h / rgb.height)
    pw, ph = round(rgb.width * sc), round(rgb.height * sc)
    ox, oy = (W - pw) // 2, BAR_H + 6 + (avail_h - ph) // 2
    base = Image.new("RGB", (W, H), PAD_BG)
    base.paste(rgb.resize((pw, ph), Image.LANCZOS), (ox, oy))
    tf = (sc, ox, oy)

    blocks = sorted(page.blocks, key=lambda b: b.order)
    cells = table_cells(conv, img, blocks)
    polys = [l.poly for l in sorted(page.lines, key=lambda l: (l.bbox[1], l.bbox[0]))]

    frames, durs = [frame(base, tf, f"{src.name}  ·  input")], [900]
    for n in _steps(len(blocks), 24):
        frames.append(frame(base, tf, f"Layout + reading order  ·  {n}/{len(blocks)} blocks", blocks[:n]))
        durs.append(ms)
    durs[-1] = 1500
    if cells:
        rows = sorted({c["row"] for c in cells})
        for r in _steps(len(rows), 16):
            shown = [c for c in cells if c["row"] <= rows[r - 1]]
            frames.append(frame(base, tf, f"Table cells (TATR)  ·  {r}/{len(rows)} rows, {len(shown)} cells",
                                blocks, shown, stage="cells"))
            durs.append(ms)
        durs[-1] = 1800
    for n in _steps(len(polys), 20):
        frames.append(frame(base, tf, f"Text lines  ·  {n}/{len(polys)} lines", blocks, cells, polys[:n], stage="lines"))
        durs.append(ms)
    durs[-1] = 2500

    out = Path(out_dir) / f"{src.stem}.gif"
    # one shared palette (from the busiest frame) so unchanged pixels stay identical and GIF frame diffs stay small
    ref = frames[-1].quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    pal = [f.quantize(palette=ref, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True, disposal=1)
    print(f"{out}  {W}x{H}, {len(frames)} frames, {len(blocks)} blocks, {len(cells)} cells, {len(polys)} lines, "
          f"{out.stat().st_size / 1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--out", default="assets/viz")
    ap.add_argument("--size", default="600x834", help="canvas WxH shared by every GIF")
    ap.add_argument("--ms", type=int, default=120, help="milliseconds per animation frame")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    size = tuple(int(v) for v in a.size.lower().split("x"))
    conv = aksonocr_lite.load(device=a.device)
    for p in a.images:
        make_gif(p, a.out, conv, size, a.ms)


if __name__ == "__main__":
    main()

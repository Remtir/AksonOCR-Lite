"""Animated explainer: Thai grapheme-cluster tokens vs character tokens on one real line.

    python tools/cluster_gif.py LINE_IMAGE --out assets/viz/cluster_tokens.gif [--font-dir DIR]

The recognizer (no 180° pass) reads the crop; every CTC frame where it emits a token becomes one animation step.
The top row spells the same output as one token per Unicode code point (what a character-level CTC head has to
emit); the bottom row shows the cluster tokens AksonOCR-Lite actually emitted, at the horizontal position where it
emitted them. Needs a Thai font (Noto Sans Thai and Sarabun by default).
"""
import argparse, math, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aksonocr_lite.rec import Recognizer, _prep  # noqa: E402

BG, INK, MUTED, GRID = (255, 255, 255), (31, 35, 41), (107, 112, 121), (229, 231, 235)
PINK, PINK_SOFT = (235, 87, 150), (252, 228, 238)
ORANGE, ORANGE_SOFT = (214, 120, 30), (253, 238, 220)
GRAY_SOFT = (241, 242, 244)
COMBINING = set("ัิีึืฺุู็่้๊๋์ํ๎")


def fonts(d):
    d = Path(d)
    f = lambda name, s: ImageFont.truetype(str(d / name), s, layout_engine=ImageFont.Layout.RAQM)
    return {"title": f("sarabun__Sarabun-Bold.ttf", 30), "h": f("sarabun__Sarabun-SemiBold.ttf", 21),
            "body": f("sarabun__Sarabun-Regular.ttf", 20), "small": f("sarabun__Sarabun-Regular.ttf", 17),
            "tok": f("notosansthai__NotoSansThai[wdth,wght].ttf", 30),
            "cp": f("notosansthai__NotoSansThai[wdth,wght].ttf", 24),
            "code": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12),
            "sp": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)}


def emissions(img):
    """[(frame index, token)] for every CTC emission, plus the frame count."""
    rec = Recognizer("cpu", flip=False)
    x = _prep(img)[None]
    p = rec.sess.run(None, {rec.input: x})[0][0]
    ids = p.argmax(-1)
    out, prev = [], 0
    for t, c in enumerate(ids):
        if c != prev and c != 0:
            out.append((t, rec.chars[c]))
        prev = c
    return out, len(ids)


def show(cp):
    return ("◌" + cp) if cp in COMBINING else cp


def box(d, xy, fill, outline=None, r=8, w=2):
    d.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=w)


def frame(F, line, ems, T, k, final=False):
    """Frame after k emissions (k = 0 .. len(ems))."""
    W, H = 1280, 760
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((40, 26), "Why cluster tokens? One Thai line, two ways to spell the output", font=F["title"], fill=INK)
    # the line image, scaled
    sc = 150 / line.height
    li = line.resize((int(line.width * sc), 150), Image.LANCZOS)
    lx, ly = (W - li.width) // 2, 92
    im.paste(li, (lx, ly))
    d.rectangle((lx - 1, ly - 1, lx + li.width, ly + li.height), outline=GRID)
    xs = [lx + (t + 0.5) / T * li.width for t, _ in ems]
    for i in range(k):
        stacked = sum(ch in COMBINING for ch in ems[i][1]) >= 2
        d.line((xs[i], ly + li.height + 2, xs[i], ly + li.height + 14), fill=ORANGE if stacked else PINK, width=3)
    if 0 < k <= len(ems) and not final:
        cx = xs[k - 1]
        d.line((cx, ly - 8, cx, ly + li.height + 8), fill=PINK, width=3)
        d.text((cx, ly - 10), f"CTC frame {ems[k - 1][0]}", font=F["code"], fill=PINK, anchor="mb")
    # rows
    y1, y2 = 300, 520
    d.text((40, y1 - 40), "Character tokens: one Unicode code point per output symbol", font=F["h"], fill=INK)
    d.text((40, y2 - 40), "Cluster tokens (AksonOCR-Lite): one output symbol per glyph column", font=F["h"], fill=PINK)
    x1 = x2 = 40
    ncp = 0
    for i in range(k):
        tok = ems[i][1]
        cps = list(tok)
        ncp += len(cps)
        stacked = len(cps) >= 3 or (len(cps) >= 2 and all(c in COMBINING for c in cps[1:]) and len(cps) > 2)
        multi = len(cps) > 1
        # character row: one small box per code point, a group bracket for the cluster
        gx0 = x1
        for c in cps:
            wbox = 40
            box(d, (x1, y1, x1 + wbox, y1 + 58), ORANGE_SOFT if multi else GRAY_SOFT, ORANGE if multi else None)
            d.text((x1 + wbox / 2, y1 + 32), "␣" if c == " " else show(c), font=F["sp"] if c == " " else F["cp"],
                   fill=MUTED if c == " " else INK, anchor="mm")
            d.text((x1 + wbox / 2, y1 + 72), f"{ord(c):04X}", font=F["code"], fill=MUTED, anchor="mm")
            x1 += wbox + 4
        if multi:
            d.line((gx0, y1 + 86, x1 - 4, y1 + 86), fill=ORANGE, width=2)
            d.text(((gx0 + x1 - 4) / 2, y1 + 98), f"{len(cps)} in one spot", font=F["code"], fill=ORANGE,
                   anchor="mt")
        x1 += 10
        # cluster row
        tw = max(46, d.textlength(tok, font=F["tok"]) + 22)
        box(d, (x2, y2, x2 + tw, y2 + 70), PINK_SOFT, PINK)
        d.text((x2 + tw / 2, y2 + 40 if tok != " " else y2 + 35), "␣" if tok == " " else tok,
               font=F["sp"] if tok == " " else F["tok"], fill=MUTED if tok == " " else INK, anchor="mm")
        x2 += tw + 8
    # counters
    d.text((W - 40, y1 - 40), f"{ncp} symbols", font=F["h"], fill=ORANGE if k else MUTED, anchor="ra")
    d.text((W - 40, y2 - 40), f"{k} symbols", font=F["h"], fill=PINK, anchor="ra")
    if final:
        notes = [
            "Stacked vowels and tone marks sit in the same spot as their consonant. As characters they are 2–3 "
            "separate symbols that must come out in Unicode order;",
            "a swapped or dropped mark looks almost the same on screen but is a different string (search and exact "
            "match fail).",
            "As one cluster token, the order inside is fixed by the vocabulary, and only real clusters exist "
            "(2,149 of the 2,455 classes): no floating tone mark,",
            "no two vowels on one consonant. The output is also shorter, so narrow glyphs need fewer CTC frames.",
        ]
        y = 640
        for n in notes:
            d.text((40, y), n, font=F["small"], fill=INK)
            y += 26
    else:
        d.text((40, 650), "The recognizer emits one cluster token per glyph column (real CTC output on this crop, "
               "frames shown above the image).", font=F["small"], fill=MUTED)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default="assets/viz/cluster_tokens.gif")
    ap.add_argument("--font-dir", required=True, help="directory of Thai TTF fonts for the labels")
    a = ap.parse_args()
    F = fonts(a.font_dir)
    img = cv2.imread(a.image)
    ems, T = emissions(img)
    print("".join(t for _, t in ems), [t for _, t in ems])
    line = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    frames = [frame(F, line, ems, T, k) for k in range(len(ems) + 1)]
    frames.append(frame(F, line, ems, T, len(ems), final=True))
    durs = [1400] + [550] * len(ems) + [7000]
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(a.out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
    frames[-1].save(Path(a.out).with_suffix(".png"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

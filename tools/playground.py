"""Web playground: upload a page image or PDF, see the Markdown, the layout/line overlay and the JSON.

    pip install gradio
    python tools/playground.py [--host 127.0.0.1] [--port 7860] [--device auto]
"""
import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aksonocr_lite  # noqa: E402
from aksonocr_lite.inputs import load_pages  # noqa: E402
from aksonocr_lite.orient import rotate  # noqa: E402

PINK = (150, 87, 235)  # BGR
COLORS = {"table": (60, 160, 60), "image": (200, 140, 40), "chart": (200, 140, 40), "seal": (200, 140, 40),
          "doc_title": (40, 40, 200), "paragraph_title": (40, 90, 220), "header": (140, 140, 140),
          "footer": (140, 140, 140), "number": (140, 140, 140)}


def overlay(img, page, show_blocks, show_lines):
    vis = img.copy()
    s = max(1, round(max(img.shape[:2]) / 1200))
    if show_lines:
        for l in page.lines:
            cv2.polylines(vis, [np.int32(l.poly)], True, PINK, s)
    if show_blocks:
        for b in sorted(page.blocks, key=lambda b: b.order):
            c = COLORS.get(b.label, (200, 90, 30))
            x0, y0, x1, y1 = map(int, b.bbox)
            cv2.rectangle(vis, (x0, y0), (x1, y1), c, 2 * s)
            tag = f"{int(b.order) + 1} {b.label}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, s)
            cv2.rectangle(vis, (x0, max(0, y0 - th - 6 * s)), (x0 + tw + 6 * s, y0), c, -1)
            cv2.putText(vis, tag, (x0 + 3 * s, y0 - 4 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (255, 255, 255), s,
                        cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def build(device):
    import gradio as gr
    conv = {}
    dewarper = {}

    def get(tables):
        if tables not in conv:
            conv[tables] = aksonocr_lite.load(device=device, tables=tables)
        return conv[tables]

    def run(file, dewarp, tables, show_blocks, show_lines, max_pages):
        if file is None:
            raise gr.Error("Upload an image or a PDF first.")
        c = get(tables)
        if dewarp and "d" not in dewarper:
            from aksonocr_lite.dewarp import Dewarper
            dewarper["d"] = Dewarper(device)
        mds, pages, gallery, t0 = [], [], [], time.perf_counter()
        for i, img in enumerate(load_pages(file, max_pages=int(max_pages))):
            if dewarp:
                img, _ = dewarper["d"](img)
            page = c.page(img, index=i)
            gallery.append((overlay(rotate(img, page.rotation), page, show_blocks, show_lines), f"page {i + 1}"))
            mds.append(page.markdown)
            pages.append(page.to_dict())
        secs = time.perf_counter() - t0
        md = "\n\n---\n\n".join(mds)
        stats = (f"**{len(pages)} page(s) in {secs:.2f} s** ({secs / max(1, len(pages)):.2f} s/page) · "
                 f"{sum(len(p['lines']) for p in pages)} lines · {sum(len(p['blocks']) for p in pages)} blocks"
                 + (" · dewarped" if dewarp else ""))
        out = Path("/tmp/aksonocr_playground")
        out.mkdir(exist_ok=True)
        mdf = out / f"{Path(file).stem}.md"
        mdf.write_text(md, encoding="utf-8")
        return stats, md, md, gallery, json.dumps(pages, ensure_ascii=False, indent=1), str(mdf)

    css = ".md-out {font-family: 'Sarabun', 'Noto Sans Thai', sans-serif;} footer {display:none !important}"
    with gr.Blocks(title="AksonOCR-Lite playground") as app:
        gr.Markdown("# AksonOCR-Lite playground\nThai document OCR on ONNX Runtime: upload a page image or a PDF.")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                f = gr.File(label="Image or PDF", file_types=["image", ".pdf"], type="filepath")
                with gr.Accordion("Options", open=True):
                    dw = gr.Checkbox(label="Dewarp (photographed / curled pages)", value=False)
                    tb = gr.Checkbox(label="Tables to HTML", value=True)
                    sb = gr.Checkbox(label="Show layout blocks", value=True)
                    sl = gr.Checkbox(label="Show text lines", value=True)
                    mp = gr.Slider(1, 20, value=5, step=1, label="Max PDF pages")
                go = gr.Button("Run OCR", variant="primary")
                stats = gr.Markdown()
                dl = gr.File(label="Download Markdown")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Overlay"):
                        gal = gr.Gallery(label="Blocks (reading order) and lines", columns=1, height=760,
                                         object_fit="contain")
                    with gr.Tab("Rendered"):
                        rendered = gr.Markdown(elem_classes="md-out")
                    with gr.Tab("Markdown"):
                        raw = gr.Code(language="markdown", lines=30)
                    with gr.Tab("JSON"):
                        js = gr.Code(language="json", lines=30)
        go.click(run, [f, dw, tb, sb, sl, mp], [stats, rendered, raw, gal, js, dl])
        f.upload(run, [f, dw, tb, sb, sl, mp], [stats, rendered, raw, gal, js, dl])
    return app, dict(css=css, theme=gr.themes.Soft(primary_hue="pink"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    app, style = build(a.device)
    app.queue(default_concurrency_limit=2).launch(server_name=a.host, server_port=a.port, footer_links=[], **style)


if __name__ == "__main__":
    main()

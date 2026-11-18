"""Command line: aksonocr FILE [FILE ...] [--format md|text|json] [--out DIR]."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="aksonocr", description="Thai document OCR: PDF/image -> Markdown, text or JSON.")
    ap.add_argument("inputs", nargs="+", help="PDF or image files")
    ap.add_argument("--format", "-f", choices=["md", "text", "json"], default="md")
    ap.add_argument("--out", "-o", help="output directory (default: print to stdout)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--dpi", type=int, default=150, help="PDF render resolution")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--dewarp", action="store_true", help="flatten curled/photographed pages")
    ap.add_argument("--no-rotate", action="store_true", help="skip page orientation correction")
    ap.add_argument("--no-tables", action="store_true", help="emit tables as plain text (skip table structure)")
    ap.add_argument("--quiet", "-q", action="store_true")
    a = ap.parse_args(argv)

    from . import load

    conv = load(device=a.device, dpi=a.dpi, dewarp=a.dewarp, rotate=not a.no_rotate, tables=not a.no_tables)
    out_dir = Path(a.out) if a.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    ext = {"md": ".md", "text": ".txt", "json": ".json"}[a.format]
    for src in a.inputs:
        t0 = time.time()
        pages = list(conv.convert(src, max_pages=a.max_pages))
        if a.format == "json":
            body = json.dumps({"source": src, "pages": [p.to_dict() for p in pages]}, ensure_ascii=False, indent=1)
        else:
            sep = "\n\n<!-- page break -->\n\n" if a.format == "md" else "\n\n\f"
            body = sep.join(p.markdown if a.format == "md" else p.text for p in pages)
        if out_dir:
            (out_dir / (Path(src).stem + ext)).write_text(body, encoding="utf-8")
        else:
            sys.stdout.write(body + "\n")
        if not a.quiet:
            print(f"{src}: {len(pages)} page(s) in {time.time() - t0:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()

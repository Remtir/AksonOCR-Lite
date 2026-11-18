"""Run AksonOCR-Lite over a folder of page images/PDFs and write one <stem>.md (and optionally .json) per file.

Use it for any benchmark that scores per-page Markdown (e.g. SEA-Vision / OmniDocBench-style evaluators):
    python bench/run_folder.py IMAGES_DIR OUT_DIR [--json] [--dewarp]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aksonocr_lite  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".pdf"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images"); ap.add_argument("out")
    ap.add_argument("--json", action="store_true", help="also write the full JSON result")
    ap.add_argument("--dewarp", action="store_true"); ap.add_argument("--device", default="auto")
    ap.add_argument("--recursive", action="store_true")
    a = ap.parse_args()
    src, out = Path(a.images), Path(a.out)
    files = sorted(p for p in (src.rglob("*") if a.recursive else src.iterdir()) if p.suffix.lower() in EXTS)
    conv = aksonocr_lite.load(device=a.device, dewarp=a.dewarp)
    t0 = time.time()
    for k, p in enumerate(files):
        rel = p.relative_to(src).with_suffix("")
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        pages = list(conv.convert(str(p)))
        (out / rel).with_suffix(".md").write_text("\n\n".join(pg.markdown for pg in pages), encoding="utf-8")
        if a.json:
            (out / rel).with_suffix(".json").write_text(json.dumps([pg.to_dict() for pg in pages], ensure_ascii=False), encoding="utf-8")
        print(f"[{k + 1}/{len(files)}] {p.name}", flush=True)
    print(f"{len(files)} files in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

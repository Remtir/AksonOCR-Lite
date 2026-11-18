"""Run a CPU OCR system on every benchmark, with the same inputs and output layout as bench/vlm_serve_bench.py
(so the same scorers apply), one worker process per core, resumable item by item.

    python bench/cpu_ocr_bench.py --system indra --out runs/vlm/indra-ocr --workers 16

Systems (each run with its package defaults, as its card shows):
  indra           datalab-to/indra-ocr (`indra_ocr.load(threads=1)`; .convert(path) -> page.markdown)
  openthai-text   iapp/OpenThai2.0-OCR-Text (runtime/line_engine.LineOCR(threads=1).read(path)["text"]); plain text
  openthai-text-v2.1  the same from its `v2.1` branch (OpenThai2.1 OCR Text: new runtime + rec2.onnx re-reads)
  aksonocr-lite   this repo, default settings (page rotation on, no dewarp); `--lines-rec` also writes
                  lines_rec.jsonl = the recognizer alone on each line crop
  aksonocr-lite-dewarp  the same with dewarp on (MonkeyOCRv2 page flattening before rotation). Identical to
                  `load(dewarp=True)`: one process first runs the dewarp nets on the GPU over every input and saves
                  the result losslessly (out/_dewarped/*.png); the CPU workers then run the default pipeline on
                  those (the Swin dewarp net on CPU takes ~25 s and ~4 GB per page per process).
Passes: mdp, thai-parse (ThaiOCRBench Document parsing; these systems take no prompt), sea, lines.
"""
import argparse, json, os, sys, time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_serve_bench import Store, build_items, finalize_thai  # noqa: E402

_S = {}


def _init(system, lines_rec):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["AKSONOCR_THREADS"] = "1"
    if system == "indra":
        import indra_ocr
        conv = indra_ocr.load(threads=1)
        _S["f"] = lambda p: "\n\n".join((pg.markdown if pg.markdown is not None else "\n".join(l.text for l in pg.lines))
                                        for _, pg in conv.convert(p))
    elif system in ("openthai-text", "openthai-text-v2.1"):
        from huggingface_hub import snapshot_download
        d = snapshot_download("iapp/OpenThai2.0-OCR-Text", revision="v2.1" if system.endswith("v2.1") else None)
        sys.path.insert(0, d)
        from runtime.line_engine import LineOCR
        ocr = LineOCR(d, threads=1)
        _S["f"] = lambda p: ocr.read(p)["text"]
    elif system in ("aksonocr-lite", "aksonocr-lite-dewarp"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import cv2
        import aksonocr_lite
        conv = aksonocr_lite.load(device="cpu")  # -dewarp: workers read the pre-pass's dewarped pages
        _S["f"] = lambda p: next(conv.convert(p)).markdown
        if lines_rec:
            rec = conv.rec
            _S["rec"] = lambda p: rec([cv2.imread(p)])[0][0]
    else:
        raise SystemExit(f"unknown system {system}")


def _run(item):
    p, key, path = item[:3]
    t = time.time()
    try:
        text = _S["f"](path)
        err = None
    except Exception as e:  # noqa: BLE001 (a failed page scores as empty)
        text, err = "", f"{type(e).__name__}: {e}"
    out = {"p": p, "key": key, "text": text, "err": err, "s": time.time() - t}
    if p == "lines" and "rec" in _S:
        try:
            out["rec"] = _S["rec"](path)
        except Exception as e:  # noqa: BLE001
            out["rec"] = ""
    return out


def dewarp_prepass(items, cache: Path):
    """Dewarp every input once on the GPU (single process) and point the items at the lossless results."""
    import hashlib
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aksonocr_lite.dewarp import Dewarper
    from aksonocr_lite.inputs import load_pages
    cache.mkdir(parents=True, exist_ok=True)
    dw = Dewarper(os.getenv("AKSONOCR_DEWARP_DEVICE", "cuda"))
    new, secs = [], {}
    for i, it in enumerate(items):
        dst = cache / (hashlib.sha1(it[2].encode()).hexdigest()[:16] + ".png")
        if not dst.exists():
            t = time.time()
            img = next(iter(load_pages(it[2])))
            img, _ = dw(img)
            secs[(it[0], it[1])] = time.time() - t
            cv2.imwrite(str(dst) + ".tmp.png", img)
            os.replace(str(dst) + ".tmp.png", dst)
        new.append((it[0], it[1], str(dst), *it[3:]))
        if (i + 1) % 200 == 0:
            print(f"  dewarp {i + 1}/{len(items)}", flush=True)
    return new, secs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True, choices=["indra", "openthai-text", "openthai-text-v2.1", "aksonocr-lite", "aksonocr-lite-dewarp"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", nargs="+", default=["mdp", "thai-parse", "sea", "lines"])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--lines-rec", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(out)
    spec = {"prompt": "", "not_applicable": {"thai"}}
    items = [it for it in build_items(a.passes, spec) if not store.has(it[0], it[1])]
    print(f"{a.system}: {len(items)} items to run", flush=True)
    dewarp_s = {}
    if a.system.endswith("-dewarp") and items:
        items, dewarp_s = dewarp_prepass(items, out / "_dewarped")
    t0, n, secs = time.time(), 0, {}
    if items:
        rec_f = open(out / "lines_rec.jsonl", "a", encoding="utf-8") if a.lines_rec else None
        with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.system, a.lines_rec)) as pool:
            for r in pool.imap_unordered(_run, items, chunksize=1):
                store.put(r["p"], r["key"], "", r["text"], r["err"] or "stop", 0)
                if rec_f is not None and "rec" in r:
                    rec_f.write(json.dumps({"key": r["key"], "predict": r["rec"]}, ensure_ascii=False) + "\n")
                    rec_f.flush()
                secs.setdefault(r["p"], []).append(r["s"] + dewarp_s.get((r["p"], r["key"]), 0.0))
                n += 1
                if r["err"]:
                    print(f"[{r['p']}] {r['key']}: {r['err']}", flush=True)
                if n % 200 == 0 or n == len(items):
                    print(f"  {n}/{len(items)}", flush=True)
        dt = time.time() - t0
        tpath = out / "timing.json"
        timing = json.loads(tpath.read_text()) if tpath.exists() else {}
        timing[f"session_{int(t0)}"] = {"wall_s": round(dt, 1), "items": n, "workers": a.workers,
                                       "cpu_s_per_item": {k: round(sum(v) / len(v), 3) for k, v in secs.items()}}
        tpath.write_text(json.dumps(timing, indent=1))
    finalize_thai(out)


if __name__ == "__main__":
    main()

"""Text recognition only: line recognizers on the same pre-cut line crops, no detection, layout or tables.

    python bench/rec_bench.py --out-dir runs/rec --json bench/SPEED.json

Crops: the internal printed-line set (pv_hi_doc 184 + pv_hi_photo 285 = 469 line images, labels agreed by 3+ of 4
readers), scored by bench/lines_score.py (exact match, exact ignoring spaces, 1 - NED). Each recognizer gets the
image file and returns one string, with its own defaults:
  aksonocr-lite         aksonocr_lite.rec.Recognizer (CTC, H=48; also reads the crop rotated 180° and keeps the
                        more confident reading)
  aksonocr-lite-noflip  the same without the 180° reading (half the work; OpenThai has no such step)
  openthai-text         iapp/OpenThai2.0-OCR-Text `LineOCR(dir, threads=1).recognize([crop])` (its crop
                        preprocessing + CTC, H=48) on its default OpenVINO backend
  openthai-text-v2.1    the same from its `v2.1` branch (adds the rec2.onnx re-read of lines with Latin letters
                        inside Thai words)
Speed, on the CPU only:
  latency     one process, 1 thread, one crop at a time (image read included): ms per line; first 5 are warm-up
  throughput  16 single-thread processes, one crop each at a time: lines per second
"""
import argparse, json, os, statistics, subprocess, sys, time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lines_score import SPLITS  # noqa: E402

SYSTEMS = ["aksonocr-lite", "aksonocr-lite-noflip", "openthai-text", "openthai-text-v2.1"]


def crops():
    out = []
    for split, f in SPLITS.items():
        for k, line in enumerate(open(f, encoding="utf-8")):
            out.append((f"{split}/{k}", line.split("\t")[0]))
    return out


def make(system):
    """path -> text, single-threaded."""
    os.environ["AKSONOCR_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import cv2
    cv2.setNumThreads(1)
    if system.startswith("aksonocr-lite"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from aksonocr_lite.rec import Recognizer
        rec = Recognizer("cpu", flip=system == "aksonocr-lite")
        return lambda p: rec([cv2.imread(p)])[0][0]
    if system in ("openthai-text", "openthai-text-v2.1"):
        from huggingface_hub import snapshot_download
        d = snapshot_download("iapp/OpenThai2.0-OCR-Text", revision="v2.1" if system.endswith("v2.1") else None)
        sys.path.insert(0, d)
        from runtime.line_engine import LineOCR
        ocr = LineOCR(d, threads=1)
        return lambda p: ocr.recognize([cv2.imread(p)])[0][0]
    raise SystemExit(f"unknown system {system}")


_F = {}


def _init(system):
    _F["f"] = make(system)


def _one(item):
    return item[0], _F["f"](item[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    ap.add_argument("--out-dir", default="runs/rec")
    ap.add_argument("--json", default="bench/SPEED.json")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    items = crops()
    allres = json.loads(Path(a.json).read_text()) if Path(a.json).exists() else {}
    for system in a.systems:
        out = Path(a.out_dir) / system
        out.mkdir(parents=True, exist_ok=True)
        f = make(system)
        preds, secs = {}, []
        for i, (key, path) in enumerate(items[:5] + items):
            t = time.perf_counter()
            text = f(path)
            if i >= 5:
                secs.append(time.perf_counter() - t)
                preds[key] = text
        with open(out / "lines_rec.jsonl", "w", encoding="utf-8") as fh:
            for key, _ in items:
                fh.write(json.dumps({"key": key, "predict": preds[key]}, ensure_ascii=False) + "\n")
        score = json.loads(subprocess.run([sys.executable, str(Path(__file__).with_name("lines_score.py")),
                                           str(out / "lines_rec.jsonl")], capture_output=True, text=True,
                                          check=True).stdout.strip().splitlines()[-1])
        with get_context("spawn").Pool(a.workers, initializer=_init, initargs=(system,)) as pool:
            list(pool.imap_unordered(_one, items[: a.workers * 4]))  # warm-up
            t = time.perf_counter()
            n = sum(1 for _ in pool.imap_unordered(_one, items, chunksize=1))
            wall = time.perf_counter() - t
        ms = sorted(1000 * s for s in secs)
        r = {"lines": len(items), "score": score,
             "latency_1thread_ms": {"mean": round(statistics.mean(ms), 2), "median": round(statistics.median(ms), 2),
                                    "p90": round(ms[int(0.9 * (len(ms) - 1))], 2)},
             f"throughput_{a.workers}x1_lines_per_s": round(n / wall, 1)}
        print(system, json.dumps(r), flush=True)
        allres[f"rec_{system}"] = r
        Path(a.json).write_text(json.dumps(allres, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

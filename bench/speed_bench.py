"""AksonOCR-Lite speed on real benchmark pages (default settings: page rotation on, no dewarp).

    python bench/speed_bench.py --out bench/SPEED.json

Pages: the 160 Thai MDPBench pages (40 digital, 120 photographed) + every 10th Thai SEA-DocBench page (151), i.e. 311
document pages of mixed size. Each page is a file path, so decoding is included. Modes:
  latency     one process, AKSONOCR_THREADS threads (default 16): seconds per page (mean / median / p90) and the
              per-stage split from Page.timings; the first 3 pages are warm-up and not counted
  throughput  N single-thread worker processes (one page each at a time): pages per second over the whole set
  gpu         one process with device="cuda" (layout, detection, tables on the GPU; recognizer on CPU, the default)
--system openthai-text measures iapp/OpenThai2.0-OCR-Text (text lines only: detection + recognition + reading
order, plain text out) with the same pages and setups (its card's own API, `LineOCR(dir, threads=N).read(path)`, on
its default OpenVINO backend; `--backend onnxruntime` for the alternative). Its results go under
"cpu_openthai2.0-ocr-text" (or "..._onnxruntime"). It has no GPU path.
Run on an otherwise idle machine; the CPU model is recorded.
"""
import argparse, json, os, platform, statistics, sys, time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MDP_IMAGES as MDP, SEA_GT, SEA_IMAGES as SEA_IMG, THAI_CACHE  # noqa: E402


THAI_META = THAI_CACHE / "meta.json"  # ThaiOCRBench DP images (bench/vlm_serve_bench.py)
SETS = ["mdp", "thai", "sea"]


def pages(which="mix"):
    """mix: the original 311-page set (MDPBench-th + every 10th SEA-DocBench-th page). Per benchmark:
    mdp = the 160 Thai MDPBench pages, thai = the 211 ThaiOCRBench Document parsing images, sea = every 10th Thai
    SEA-DocBench page (151 of 1,506)."""
    mdp = [str(p) for p in sorted(MDP.iterdir()) if p.name.startswith("th_")]
    sea = [str(SEA_IMG / s["page_info"]["image_path"]) for s in json.loads(SEA_GT.read_text())][::10]
    if which == "mdp":
        return mdp
    if which == "sea":
        return sea
    if which == "thai":
        return [r["path"] for r in json.loads(THAI_META.read_text())]
    return mdp + sea


def cpu_name():
    try:
        return next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def stats(xs):
    xs = sorted(xs)
    return {"mean": round(statistics.mean(xs), 3), "median": round(statistics.median(xs), 3),
            "p90": round(xs[int(0.9 * (len(xs) - 1))], 3), "n": len(xs)}


def make(system, threads, device="cpu", backend="openvino"):
    """A function path -> (text, timings) for one system, run with `threads` threads."""
    if system == "openthai-text":
        from huggingface_hub import snapshot_download
        d = snapshot_download("iapp/OpenThai2.0-OCR-Text")
        sys.path.insert(0, d)
        from runtime.line_engine import LineOCR
        ocr = LineOCR(d, threads=threads, runtime={"backend": backend})
        return lambda p: (ocr.read(p)["text"], None)
    os.environ["AKSONOCR_THREADS"] = str(threads)
    import aksonocr_lite
    conv = aksonocr_lite.load(device=device)

    def f(p):
        page = next(conv.convert(p))
        return page.markdown, page.timings
    return f


def latency(ps, device="cpu", system="aksonocr-lite", threads=16, backend="openvino"):
    f = make(system, threads, device, backend)
    secs, stages = [], {}
    for i, p in enumerate(ps):
        t = time.perf_counter()
        _, timings = f(p)
        dt = time.perf_counter() - t
        if i < 3:
            continue
        secs.append(dt)
        for k, v in (timings or {}).items():
            stages.setdefault(k, []).append(v)
    r = {"s_per_page": stats(secs), "pages_per_s": round(len(secs) / sum(secs), 3)}
    if stages:
        r["stage_mean_s"] = {k: round(statistics.mean(v), 3) for k, v in stages.items()}
    return r


_C = {}


def _init(system, backend):
    _C["f"] = make(system, 1, backend=backend)


def _one(p):
    _C["f"](p)
    return 1


def throughput(ps, n, system="aksonocr-lite", backend="openvino"):
    os.environ["AKSONOCR_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    with get_context("spawn").Pool(n, initializer=_init, initargs=(system, backend)) as pool:
        list(pool.imap_unordered(_one, ps[: n * 2]))  # warm-up: every worker has loaded and run
        t = time.perf_counter()
        done = sum(pool.imap_unordered(_one, ps, chunksize=1))
        dt = time.perf_counter() - t
    return {"workers": n, "pages": done, "wall_s": round(dt, 1), "pages_per_s": round(done / dt, 3),
            "s_per_page_per_worker": round(dt * n / done, 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bench/SPEED.json")
    ap.add_argument("--modes", nargs="+", default=["latency", "throughput", "gpu"])
    ap.add_argument("--threads", type=int, default=16, help="threads for the single-process latency run")
    ap.add_argument("--workers", type=int, nargs="+", default=[16])
    ap.add_argument("--system", default="aksonocr-lite", choices=["aksonocr-lite", "openthai-text"])
    ap.add_argument("--backend", default="openvino", choices=["openvino", "onnxruntime"],
                    help="openthai-text only")
    ap.add_argument("--set", default="mix", choices=["mix"] + SETS,
                    help="page set; a per-benchmark set is stored under SPEED.json['sets'][set]")
    a = ap.parse_args()
    ps = pages(a.set)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    if a.system == "openthai-text":
        key = "cpu_openthai2.0-ocr-text" + ("" if a.backend == "openvino" else "_onnxruntime")
        r = {"model": "iapp/OpenThai2.0-OCR-Text", "backend": a.backend, "output": "plain text (no layout, no tables)"}
        if "latency" in a.modes:
            r[f"latency_cpu_{a.threads}threads"] = latency(ps, system=a.system, threads=a.threads, backend=a.backend)
            print("latency", r[f"latency_cpu_{a.threads}threads"], flush=True)
        if "throughput" in a.modes:
            for n in a.workers:
                r[f"throughput_cpu_{n}x1"] = throughput(ps, n, a.system, a.backend)
                print("throughput", r[f"throughput_cpu_{n}x1"], flush=True)
        res[key] = r
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        return
    res.update({"cpu": cpu_name(), "logical_cpus": os.cpu_count(),
                "settings": "default (page rotation on, no dewarp, tables on)"})
    tgt = res if a.set == "mix" else res.setdefault("sets", {}).setdefault(a.set, {})
    tgt["pages"] = len(ps)
    if "latency" in a.modes:
        tgt[f"latency_cpu_{a.threads}threads"] = r = latency(ps, threads=a.threads)
        print("latency", r, flush=True)
    if "gpu" in a.modes:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            tgt["latency_gpu"] = r = latency(ps, device="cuda", threads=a.threads)
            res["gpu"] = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip()
            print("gpu", r, flush=True)
    if "throughput" in a.modes:
        for n in a.workers:
            tgt[f"throughput_cpu_{n}x1"] = r = throughput(ps, n)
            print("throughput", r, flush=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

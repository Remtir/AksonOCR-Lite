"""Surya OCR 2.5 (datalab-to/surya-ocr-2.5) on every benchmark, same inputs/outputs as bench/vlm_serve_bench.py.

Surya's package normally spawns vLLM in Docker (no Docker here), so this starts `vllm serve` natively with the
exact arguments surya/inference/backends/vllm.py passes to its container (max-model-len 24576, prefix caching,
mm max_pixels 6.29 MP, MTP speculative decoding when the checkpoint has an MTP head, gpu util 0.85) and points
the surya client at it (SURYA_INFERENCE_URL). Pages go through surya's own RecognitionPredictor in its default
full-page mode. Blocks -> Markdown with Datalab's own converter (chandra.output.parse_markdown: page headers and
footers dropped, as chandra does), after mapping Surya's labels to Chandra's.

    python bench/surya_bench.py --out runs/vlm/surya-ocr-2.5 [--vllm /path/to/vllm]
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_serve_bench import Store, build_items, finalize_thai  # noqa: E402

CKPT = "datalab-to/surya-ocr-2.5"
LABEL = {"PageHeader": "Page-Header", "PageFooter": "Page-Footer", "Picture": "Image", "Figure": "Image"}


def has_mtp():
    from huggingface_hub import hf_hub_download
    c = json.load(open(hf_hub_download(CKPT, "config.json")))
    for cfg in (c, c.get("text_config") or {}):
        for k in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
            if cfg.get(k):
                return True
    return False


def to_markdown(page):
    from chandra.output import parse_markdown
    html = "".join(f'<div data-label="{LABEL.get(b.label, b.label)}">{b.html}</div>'
                   for b in sorted(page.blocks, key=lambda b: b.reading_order) if not b.skipped)
    return parse_markdown(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vllm", default="vllm", help="vllm executable (an env with vLLM installed)")
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--passes", nargs="+", default=["mdp", "thai-parse", "sea", "lines"])
    ap.add_argument("--chunk", type=int, default=192, help="pages per RecognitionPredictor call")
    ap.add_argument("--parallel", type=int, default=128, help="concurrent requests from the surya client")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(out)
    items = [it for it in build_items(a.passes, {"prompt": "", "not_applicable": {"thai"}}) if not store.has(it[0], it[1])]
    print(f"surya: {len(items)} items to run", flush=True)
    if not items:
        finalize_thai(out)
        return

    mtp = has_mtp()
    cmd = [a.vllm, "serve", CKPT, "--port", str(a.port), "--no-enforce-eager", "--max-num-seqs", "256",
           "--dtype", "bfloat16", "--max-model-len", "24576", "--max-num-batched-tokens", "32768",
           "--gpu-memory-utilization", "0.85", "--enable-prefix-caching",
           "--mm-processor-kwargs", json.dumps({"min_pixels": 3136, "max_pixels": 6291456}),
           "--served-model-name", CKPT, "--api-server-count", "6"]
    if mtp:
        cmd += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": 2})]
    print(" ".join(cmd), flush=True)
    srv = subprocess.Popen(cmd, stdout=open(out / "server.log", "a"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        import httpx
        for _ in range(3600):
            if srv.poll() is not None:
                raise SystemExit(f"vllm serve exited ({srv.returncode}); see {out / 'server.log'}")
            try:
                if httpx.get(f"http://127.0.0.1:{a.port}/health", timeout=2).status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        print("server ready", flush=True)
        os.environ.update(SURYA_INFERENCE_URL=f"http://127.0.0.1:{a.port}/v1", SURYA_INFERENCE_BACKEND="vllm",
                          SURYA_INFERENCE_PARALLEL=str(a.parallel), SURYA_INFERENCE_AUTOSTART="0")
        from PIL import Image
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
        rec = RecognitionPredictor(SuryaInferenceManager())
        t0, n = time.time(), 0
        for i in range(0, len(items), a.chunk):
            chunk = items[i:i + a.chunk]
            imgs = [Image.open(it[2]).convert("RGB") for it in chunk]
            try:
                pages = rec(imgs, full_page=True)
            except Exception as e:  # noqa: BLE001 (retry page by page so one bad page does not sink the chunk)
                print(f"chunk failed ({e!r}); retrying one by one", flush=True)
                pages = []
                for im in imgs:
                    try:
                        pages.append(rec([im], full_page=True)[0])
                    except Exception as e2:  # noqa: BLE001
                        print(f"page failed: {e2!r}", flush=True)
                        pages.append(None)
            for it, pg in zip(chunk, pages):
                raw = json.dumps(pg.model_dump(), ensure_ascii=False) if pg is not None else ""
                md = to_markdown(pg) if pg is not None else ""
                store.put(it[0], it[1], raw, md, "stop" if pg is not None else "error", 0)
            n += len(chunk)
            print(f"  {n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
        dt = time.time() - t0
        tpath = out / "timing.json"
        timing = json.loads(tpath.read_text()) if tpath.exists() else {}
        timing[f"session_{int(t0)}"] = {"wall_s": round(dt, 1), "requests": len(items), "req_per_s": round(len(items) / dt, 3),
                                       "mtp": mtp, "parallel": a.parallel}
        tpath.write_text(json.dumps(timing, indent=1))
        finalize_thai(out)
    finally:
        import signal
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(90)
        except Exception:  # noqa: BLE001
            os.killpg(srv.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()

"""Speed of a VLM OCR model under `vllm serve`, on the same 311 pages as bench/speed_bench.py.

    python bench/vlm_speed.py --harness /path/to/evaluation_thai_ocr --model-key typhoon-ocr2-2b \
        --upscale 1600 --out bench/SPEED.json

Same server and per-model recipe as bench/vlm_serve_bench.py (prompt, preprocessing, decoding, token limit,
Markdown conversion). Each page is timed end to end: image load + preprocessing + encoding, the request, and the
model's output conversion. Modes:
  latency     one request at a time over the first 40 pages (the first 3 are warm-up): seconds per page
  throughput  every page at once with up to 256 in flight (6 API-server processes, 12 client workers): pages per
              second over the whole set, after a 16-page warm-up
"""
import argparse, asyncio, json, os, statistics, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speed_bench import SETS, pages, stats  # noqa: E402
from vlm_serve_bench import _encode, _init, _post, payload, server_cmd  # noqa: E402


async def run(ps, spec, pool, port, concurrency, prompt, system):
    import httpx
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)
    secs, ntok, finish = [], 0, {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(7200.0), limits=httpx.Limits(max_connections=concurrency)) as cl:
        async def one(path):
            nonlocal ntok
            async with sem:
                t = time.perf_counter()
                url = await loop.run_in_executor(pool, _encode, path)
                r = await cl.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload(spec, url, prompt, system))
                r.raise_for_status()
                j = r.json()
                raw = j["choices"][0]["message"].get("content") or ""
                try:
                    await loop.run_in_executor(pool, _post, path, raw)
                except Exception:  # noqa: BLE001 (a converter crash still costs the time spent)
                    pass
                secs.append(time.perf_counter() - t)
                ntok += j.get("usage", {}).get("completion_tokens", 0)
                f = j["choices"][0].get("finish_reason")
                finish[f] = finish.get(f, 0) + 1
        t0 = time.perf_counter()
        await asyncio.gather(*(one(p) for p in ps))
        return time.perf_counter() - t0, secs, ntok, finish


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--upscale", type=int, default=0)
    ap.add_argument("--out", default="bench/SPEED.json")
    ap.add_argument("--port", type=int, default=18100)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--api-servers", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--log", default="vlm_speed_server.log")
    ap.add_argument("--sets", nargs="+", default=["mix"], choices=["mix"] + SETS,
                    help="page sets to time with one server start; per-benchmark sets go under ['sets'][set]")
    a = ap.parse_args()

    harness = str(Path(a.harness).resolve())
    sys.path.insert(0, harness)
    from ocr_eval.models import resolve
    spec = resolve(a.model_key, upscale=a.upscale)
    model = spec["repo"]
    prompt, system = spec["prompt"], spec.get("system_prompt")
    pool = ProcessPoolExecutor(a.workers, initializer=_init, initargs=(harness, a.model_key, a.upscale, model))
    env = dict(os.environ, **spec.get("vllm_env", {}))
    srv = subprocess.Popen(server_cmd(spec, model, a.port, a.api_servers, None), stdout=open(a.log, "a"),
                           stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        import httpx
        for _ in range(3600):
            if srv.poll() is not None:
                raise SystemExit(f"vllm serve exited ({srv.returncode}); see {a.log}")
            try:
                if httpx.get(f"http://127.0.0.1:{a.port}/health", timeout=2).status_code == 200:
                    break
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1)
        print("server ready", flush=True)
        for which in a.sets:
            ps = pages(which)
            _, lat, _, _ = asyncio.run(run(ps[:40], spec, pool, a.port, 1, prompt, system))
            lat = lat[3:]
            print(which, "latency", stats(lat), flush=True)
            asyncio.run(run(ps[-16:], spec, pool, a.port, 16, prompt, system))  # warm-up at batch size
            wall, secs, ntok, finish = asyncio.run(run(ps, spec, pool, a.port, a.concurrency, prompt, system))
            res = {"model": model, "pages": len(ps),
                   "gpu": os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip(),
                   "latency_1_request": {"s_per_page": stats(lat), "pages_per_s": round(len(lat) / sum(lat), 3)},
                   "throughput": {"concurrency": a.concurrency, "api_servers": a.api_servers, "wall_s": round(wall, 1),
                                  "pages_per_s": round(len(ps) / wall, 3), "output_tokens": ntok,
                                  "output_tok_per_s": round(ntok / wall), "finish": finish}}
            print(which, "throughput", res["throughput"], flush=True)
            out = Path(a.out)
            allres = json.loads(out.read_text()) if out.exists() else {}
            if which == "mix":
                allres[f"vlm_{a.model_key}"] = res
            else:
                allres.setdefault("sets", {}).setdefault(which, {})[f"vlm_{a.model_key}"] = res
            out.write_text(json.dumps(allres, indent=1, ensure_ascii=False))
    finally:
        import signal
        try:
            os.killpg(srv.pid, signal.SIGTERM)
            srv.wait(90)
        except Exception:  # noqa: BLE001
            os.killpg(srv.pid, signal.SIGKILL)
        pool.shutdown(cancel_futures=True)


if __name__ == "__main__":
    main()

"""Run a third-party OCR VLM on every benchmark with `vllm serve` + many concurrent requests (GPU kept busy),
exactly as its authors run it, and save outputs item by item so an interrupted run resumes where it stopped.

Each model's recipe (prompt, image preprocessing, decoding, max tokens, output -> Markdown conversion) is
imported unchanged from the public Typhoon OCR V2 evaluation harness
(https://github.com/srn-source/evaluation_thai_ocr, pinned commit e2512e3), which follows each model's card.
Requests are built the way each model's own client builds them (PNG data URLs; Nemotron Parse / OpenThai2.0
like NVIDIA's DocumentParser: prompt then image, its GenerationSettings, special tokens kept).

    python bench/vlm_serve_bench.py --harness /path/to/evaluation_thai_ocr --model-key typhoon-ocr2-2b \
        --upscale 1600 --out runs/vlm/typhoon-ocr2-2b

Passes (all submitted together in one server session; outputs under --out):
  mdp         MDPBench Thai pages (th_*), page-parsing prompt            -> mdp/<stem>.md
  thai-q      ThaiOCRBench Document parsing, item question as the prompt -> thai_q.json   (prompt-capable models)
  thai-parse  ThaiOCRBench Document parsing, page-parsing prompt          -> thai_parse.json
  sea         SEA-DocBench Thai pages, page-parsing prompt                -> sea/<stem>.md
  lines       internal printed-line set (pv_hi_doc + pv_hi_photo), page-parsing prompt on each line crop
                                                                          -> lines.jsonl
Timing: <out>/timing.json (wall time of the session, requests, output tokens, finish reasons per pass).
"""
import argparse, asyncio, base64, io, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TASK = "Document parsing"
QSUFFIX = "\nตอบเฉพาะข้อความตามภาพตรง ๆ อย่างกระชับ โดยไม่ใส่คำอธิบายเพิ่มเติม"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import LINE_SPLITS, MDP_IMAGES, SEA_GT, SEA_IMAGES, THAI_CACHE  # noqa: E402
_W = {}


# ------------------------------------------------------------------ worker processes (CPU side)
def _init(harness, key, upscale, model):
    sys.path.insert(0, harness)
    from ocr_eval.models import resolve
    spec = resolve(key, upscale=upscale)
    if spec.get("setup"):
        spec["setup"](model)
    _W["spec"] = spec


def _load(path):
    from PIL import Image
    origin = Image.open(path).convert("RGB")
    return origin, _W["spec"]["preprocess"](origin)


def _encode(path):
    _, im = _load(path)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=1)  # lossless; low compression = fast
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _post(path, raw):
    spec = _W["spec"]
    if spec.get("postprocess_ctx"):
        origin, im = _load(path)
        return spec["postprocess_ctx"](raw, origin, im)
    return spec["postprocess"](raw)


# ------------------------------------------------------------------ inputs
def export_thai():
    meta = Path(THAI_CACHE) / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text())
    from datasets import load_dataset
    Path(THAI_CACHE).mkdir(parents=True, exist_ok=True)
    rows = []
    for r in load_dataset("scb10x/ThaiOCRBench")["test"]:
        if r["Task"] != TASK:
            continue
        p = Path(THAI_CACHE) / f"{r['Id']}.png"
        r["image"].convert("RGB").save(p)
        rows.append({"id": r["Id"], "path": str(p), "question": r["question"],
                     "answer": r["answer"].strip() if isinstance(r["answer"], str) else r["answer"]})
    meta.write_text(json.dumps(rows, ensure_ascii=False))
    return rows


def build_items(passes, spec):
    """[(pass, key, image path, prompt text, system prompt)] for every item not yet done."""
    items = []
    if "mdp" in passes:
        for p in sorted(Path(MDP_IMAGES).iterdir()):
            if p.name.startswith("th_"):
                items.append(("mdp", p.stem, str(p), spec["prompt"], spec.get("system_prompt")))
    if "sea" in passes:
        for s in json.loads(Path(SEA_GT).read_text()):
            name = s["page_info"]["image_path"]
            items.append(("sea", Path(name).stem, f"{SEA_IMAGES}/{name}", spec["prompt"], spec.get("system_prompt")))
    if "thai-q" in passes or "thai-parse" in passes:
        rows = export_thai()
        for r in rows:
            if "thai-q" in passes and not (spec.get("runner") or "thai" in spec.get("not_applicable", ())):
                q = "Instruction: " + r["question"].strip() + QSUFFIX
                items.append(("thai-q", r["id"], r["path"], q, spec.get("vqa_system_prompt")))
            if "thai-parse" in passes:
                items.append(("thai-parse", r["id"], r["path"], spec["prompt"], spec.get("system_prompt")))
    if "lines" in passes:
        for split, f in LINE_SPLITS.items():
            for k, line in enumerate(open(f, encoding="utf-8")):
                path = line.split("\t")[0]
                items.append(("lines", f"{split}/{k}", path, spec["prompt"], spec.get("system_prompt")))
    return items


class Store:
    """Item-level outputs: .md files for page benchmarks, JSONL for the rest (resume = skip what exists)."""

    def __init__(self, out):
        self.out = Path(out)
        self.done = {}
        for p in ("thai-q", "thai-parse", "lines"):
            f = self.jsonl(p)
            self.done[p] = set()
            if f.exists():
                for l in open(f, encoding="utf-8"):
                    try:
                        self.done[p].add(json.loads(l)["key"])
                    except Exception:  # noqa: BLE001 (half-written last line after a crash)
                        pass

    def jsonl(self, p):
        return self.out / f"{p.replace('-', '_')}.jsonl"

    def has(self, p, key):
        if p in ("mdp", "sea"):
            return (self.out / p / f"{key}.md").exists()
        return key in self.done[p]

    def put(self, p, key, raw, text, finish, ntok):
        if p in ("mdp", "sea"):
            d = self.out / p
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{key}.raw.txt").write_text(raw, encoding="utf-8")
            (d / f"{key}.md").write_text(text, encoding="utf-8")  # written last = the "done" marker
        else:
            with open(self.jsonl(p), "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "predict": text, "raw": raw, "finish": finish, "tokens": ntok},
                                   ensure_ascii=False) + "\n")


def finalize_thai(out):
    """thai_q.jsonl / thai_parse.jsonl -> the prediction JSON ThaiOCRBench's eval.py reads."""
    rows = {r["id"]: r for r in export_thai()}
    for p in ("thai_q", "thai_parse"):
        f = Path(out) / f"{p}.jsonl"
        if not f.exists():
            continue
        got = {}
        for l in open(f, encoding="utf-8"):
            try:
                x = json.loads(l)
                got[x["key"]] = x
            except Exception:  # noqa: BLE001
                pass
        if len(got) < len(rows):
            continue
        preds = [{"type": TASK, "id": i, "question": "Instruction: " + r["question"].strip() + QSUFFIX,
                  "answers": [r["answer"]], "predict": got[i]["predict"], "raw": got[i]["raw"], "finish": got[i]["finish"]}
                 for i, r in rows.items()]
        (Path(out) / f"{p}.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ server
def server_cmd(spec, model, port, api_servers, gpu_mem):
    cmd = ["vllm", "serve", model, "--port", str(port), "--served-model-name", "m",
           "--max-model-len", str(spec["max_model_len"]), "--limit-mm-per-prompt", json.dumps({"image": 1}),
           "--gpu-memory-utilization", str(gpu_mem or 0.92), "--api-server-count", str(api_servers),
           "--max-num-seqs", "256"]
    if spec.get("mm_processor_kwargs") is not None:
        cmd += ["--mm-processor-kwargs", json.dumps(spec["mm_processor_kwargs"])]
    if spec.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if spec.get("llm_extra", {}).get("enable_prefix_caching") is False:
        cmd.append("--no-enable-prefix-caching")
    if spec.get("chat_template_content_format"):
        cmd += ["--chat-template-content-format", spec["chat_template_content_format"]]
    return cmd


def payload(spec, data_url, text, system):
    img = {"type": "image_url", "image_url": {"url": data_url}}
    body = {"model": "m", "max_tokens": spec["max_tokens"], "temperature": spec["temperature"], "top_p": spec["top_p"]}
    if spec.get("raw_prompt"):  # Nemotron Parse family: DocumentParser's request
        content = [{"type": "text", "text": text}, img]
    else:
        content = [img, {"type": "text", "text": spec.get("text_prefix", "") + text}]
    if spec.get("stop"):
        body["stop"] = spec["stop"]
    if spec.get("seed") is not None:
        body["seed"] = spec["seed"]
    body.update(spec.get("sampling_extra", {}))
    if spec.get("chat_template_kwargs"):
        body["chat_template_kwargs"] = spec["chat_template_kwargs"]
    msgs = [{"role": "system", "content": system}] if system else []
    body["messages"] = msgs + [{"role": "user", "content": content}]
    return body


async def run_all(items, spec, store, pool, port, concurrency, stats):
    import httpx
    loop = asyncio.get_running_loop()
    sem = asyncio.Semaphore(concurrency)
    total, done = len(items), 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(7200.0), limits=httpx.Limits(max_connections=concurrency)) as cl:
        async def one(p, key, path, text, system):
            nonlocal done
            async with sem:
                try:
                    url = await loop.run_in_executor(pool, _encode, path)
                except Exception as e:  # noqa: BLE001 (unreadable input: recorded as an empty output)
                    print(f"[{p}] {key} encode failed: {e!r}", flush=True)
                    store.put(p, key, "", "", "encode_error", 0)
                    return
                raw, fin, ntok = "", None, 0
                for attempt in range(3):
                    try:
                        r = await cl.post(f"http://127.0.0.1:{port}/v1/chat/completions",
                                          json=payload(spec, url, text, system))
                        r.raise_for_status()
                        j = r.json()
                        raw = j["choices"][0]["message"].get("content") or ""
                        fin = j["choices"][0].get("finish_reason")
                        ntok = j.get("usage", {}).get("completion_tokens", 0)
                        break
                    except Exception as e:  # noqa: BLE001
                        if attempt == 2:
                            print(f"[{p}] {key} failed: {e!r}", flush=True)
                        else:
                            await asyncio.sleep(5)
                del url
            if fin is None:  # request never succeeded: not stored, so a resumed run retries it
                return
            if p == "thai-q":
                text_out = raw
            else:
                try:
                    text_out = await loop.run_in_executor(pool, _post, path, raw)
                except Exception as e:  # noqa: BLE001 (the recipe's converter crashed on this output)
                    print(f"[{p}] {key} postprocess failed: {e!r}; keeping the raw text", flush=True)
                    text_out, fin = raw, "post_error"
            if p == "thai-q":
                text_out = raw.strip()
                if text_out.startswith("<think>"):  # vqa_clean of the harness
                    e = text_out.find("</think>")
                    text_out = text_out[e + 8:].strip() if e != -1 else text_out
            store.put(p, key, raw, text_out, fin, ntok)
            s = stats.setdefault(p, {"items": 0, "output_tokens": 0, "finish": {}})
            s["items"] += 1
            s["output_tokens"] += ntok
            s["finish"][fin] = s["finish"].get(fin, 0) + 1
            done += 1
            if done % 100 == 0 or done == total:
                print(f"  {done}/{total}  {json.dumps({k: v['items'] for k, v in stats.items()})}", flush=True)

        res = await asyncio.gather(*(one(*it) for it in items), return_exceptions=True)
        for it, r in zip(items, res):
            if isinstance(r, Exception):
                print(f"[{it[0]}] {it[1]} crashed: {r!r}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--upscale", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", nargs="+", default=["mdp", "thai-q", "thai-parse", "sea", "lines"])
    ap.add_argument("--port", type=int, default=18000)
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--api-servers", type=int, default=6, help="vLLM API-server processes (parallel image processing)")
    ap.add_argument("--gpu-mem", type=float, default=None)
    ap.add_argument("--workers", type=int, default=12, help="client processes: PNG encoding + output conversion")
    a = ap.parse_args()

    harness = str(Path(a.harness).resolve())
    sys.path.insert(0, harness)
    from ocr_eval.models import resolve
    spec = resolve(a.model_key, upscale=a.upscale)
    model = a.model or spec["repo"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(out)
    items = [it for it in build_items(a.passes, spec) if not store.has(it[0], it[1])]
    print(f"{a.model_key}: {len(items)} items to run", flush=True)
    if not items:
        finalize_thai(out)
        return

    pool = ProcessPoolExecutor(a.workers, initializer=_init, initargs=(harness, a.model_key, a.upscale, model))
    env = dict(os.environ, **spec.get("vllm_env", {}))
    cmd = server_cmd(spec, model, a.port, a.api_servers, a.gpu_mem)
    print(" ".join(cmd), flush=True)
    srv = subprocess.Popen(cmd, stdout=open(out / "server.log", "a"), stderr=subprocess.STDOUT, env=env,
                           start_new_session=True)
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
        stats, t = {}, time.time()
        asyncio.run(run_all(items, spec, store, pool, a.port, a.concurrency, stats))
        dt = time.time() - t
        tpath = out / "timing.json"
        timing = json.loads(tpath.read_text()) if tpath.exists() else {}
        ntok = sum(s["output_tokens"] for s in stats.values())
        timing[f"session_{int(t)}"] = {"wall_s": round(dt, 1), "requests": len(items), "req_per_s": round(len(items) / dt, 3),
                                       "output_tokens": ntok, "output_tok_per_s": round(ntok / dt),
                                       "concurrency": a.concurrency, "api_servers": a.api_servers, "passes": stats}
        tpath.write_text(json.dumps(timing, indent=1))
        print(f"done: {len(items)} requests in {dt:.0f}s ({ntok / dt:.0f} tok/s)", flush=True)
        finalize_thai(out)
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

"""A general VLM behind an OpenAI-compatible API (OpenRouter) on every benchmark, same inputs/outputs as
bench/vlm_serve_bench.py. Used for Gemini 4 Hydrogen and Claude Haiku 5.5, both with reasoning effort "low".

Prompts (a general model has no OCR recipe of its own):
  page benchmarks (mdp, sea, thai-parse, lines): one generic Markdown-transcription prompt (PROMPT below)
  thai-q: ThaiOCRBench's own item question, wrapped like its run_inference.py
Images: JPEG quality 95 data URLs; temperature 0.

    OPENROUTER_API_KEY=... python bench/api_vlm_bench.py --model google/gemini-4-hydrogen --out runs/vlm/gemini-4-hydrogen
"""
import argparse, asyncio, base64, io, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_serve_bench import Store, build_items, finalize_thai  # noqa: E402

PROMPT = ("Extract this document image into Markdown. Transcribe text exactly as shown, "
          "preserve reading order, headings, line breaks, lists, formulas, and tables. "
          "Output only the Markdown document with no explanation.")


def jpeg_url(path):
    from PIL import Image
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def main_async(a, items, store):
    import httpx
    key = os.environ["OPENROUTER_API_KEY"]
    sem = asyncio.Semaphore(a.concurrency)
    loop = asyncio.get_running_loop()
    done = 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as cl:
        async def one(p, k, path, text, _system):
            nonlocal done
            async with sem:
                url = await loop.run_in_executor(None, jpeg_url, path)
                body = {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens,
                        "reasoning": {"effort": a.reasoning},
                        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                                  {"type": "text", "text": text}]}]}
                raw, fin = "", "error"
                for attempt in range(5):
                    try:
                        r = await cl.post("https://openrouter.ai/api/v1/chat/completions", json=body,
                                          headers={"Authorization": f"Bearer {key}"})
                        r.raise_for_status()
                        j = r.json()
                        raw = j["choices"][0]["message"].get("content") or ""
                        fin = j["choices"][0].get("finish_reason")
                        break
                    except Exception as e:  # noqa: BLE001
                        await asyncio.sleep(2 ** attempt)
                        if attempt == 4:
                            print(f"[{p}] {k} failed: {e!r}", flush=True)
            md = raw.strip()
            if md.startswith("```"):
                md = md.split("\n", 1)[1] if "\n" in md else ""
                md = md.rsplit("```", 1)[0]
            store.put(p, k, raw, md.strip(), fin, 0)
            done += 1
            if done % 100 == 0 or done == len(items):
                print(f"  {done}/{len(items)}", flush=True)
        await asyncio.gather(*(one(*it) for it in items))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="google/gemini-4-hydrogen")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", nargs="+", default=["mdp", "thai-q", "thai-parse", "sea", "lines"])
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8192)
    a = ap.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is not set")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(out)
    spec = {"prompt": PROMPT}
    items = [it for it in build_items(a.passes, spec) if not store.has(it[0], it[1])]
    print(f"{a.model}: {len(items)} items to run", flush=True)
    t0 = time.time()
    if items:
        asyncio.run(main_async(a, items, store))
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    finalize_thai(out)


if __name__ == "__main__":
    main()

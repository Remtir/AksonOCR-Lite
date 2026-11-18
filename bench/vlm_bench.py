"""Run third-party OCR VLMs on MDPBench-th and ThaiOCRBench "Document parsing" with vLLM, exactly as their
authors run them, so the scores can sit next to AksonOCR-Lite's.

Each model's recipe (prompt, image preprocessing, decoding, max tokens, output -> Markdown conversion) is
imported unchanged from the public Typhoon OCR V2 evaluation harness
(https://github.com/srn-source/evaluation_thai_ocr, pinned commit e2512e3), which in turn follows each
model's card. Nothing model-specific is defined here.

    python bench/vlm_bench.py --harness /path/to/evaluation_thai_ocr --model-key typhoon-ocr2-2b --upscale 1600 \
        --mdp-images /path/to/MDPBench_img_public --out runs/vlm/typhoon-ocr2-2b

Passes (one model load for all):
  mdp         MDPBench Thai pages (th_*), page-parsing prompt -> <out>/mdp/<stem>.md
  thai-q      ThaiOCRBench Document parsing, the item's own question as the prompt (ThaiOCRBench's protocol;
              only for models that take free-form prompts)  -> <out>/thai_q.json
  thai-parse  ThaiOCRBench Document parsing, the model's own page-parsing prompt and conversion (the same
              mode AksonOCR-Lite and every model use on MDPBench) -> <out>/thai_parse.json
Scoring is done afterwards with bench/mdpbench.py --eval-only and bench/thaiocrbench_docparse.py --eval-only,
the same scorers used for every system. Timings (generation only, model load excluded) go to <out>/timing.json.
"""
import argparse, json, os, sys, time
from pathlib import Path

TASK = "Document parsing"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", required=True, help="clone of srn-source/evaluation_thai_ocr")
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--model", default=None, help="weights (default: the recipe's HF repo)")
    ap.add_argument("--upscale", type=int, default=0, help="enlarge images whose long side is under PX (Typhoon OCR V2: 1600)")
    ap.add_argument("--mdp-images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", nargs="+", default=["mdp", "thai-q", "thai-parse"])
    ap.add_argument("--chunk", type=int, default=256, help="requests per llm.chat/generate call")
    ap.add_argument("--spec-model", default=None, help="optional draft head for speculative decoding (EAGLE-3)")
    a = ap.parse_args()

    sys.path.insert(0, str(Path(a.harness).resolve()))
    from ocr_eval import engine
    from ocr_eval.infer import vqa_clean
    from ocr_eval.models import resolve
    from PIL import Image

    spec = resolve(a.model_key, upscale=a.upscale)
    a.model = a.model or spec["repo"]
    if spec.get("setup"):
        spec["setup"](a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    timing_path = out / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}

    eng_args = argparse.Namespace(model=a.model, tensor_parallel_size=1, gpu_memory_utilization=None,
                                  enforce_eager=False, spec_model=a.spec_model, spec_method="eagle3", num_spec_tokens=3)
    llm = engine.build_engine(spec, eng_args)
    sampling = engine.build_sampling(spec)
    extra = engine.chat_kwargs(spec)

    def run(requests):
        texts, n_tok = [], 0
        for i in range(0, len(requests), a.chunk):
            for r in engine.generate(llm, spec, requests[i:i + a.chunk], sampling, extra):
                texts.append(r.outputs[0].text)
                n_tok += len(r.outputs[0].token_ids)
        return texts, n_tok

    def record(name, n, t, n_tok):
        timing[name] = {"items": n, "seconds": round(t, 1), "items_per_s": round(n / t, 3),
                        "s_per_item": round(t / n, 3), "output_tokens": n_tok}
        timing_path.write_text(json.dumps(timing, indent=1))
        print(f"[{name}] {n} items in {t:.1f}s ({t / n:.2f}s/item, {n_tok} output tokens)", flush=True)

    if "mdp" in a.passes:
        paths = sorted(p for p in Path(a.mdp_images).iterdir() if p.name.startswith("th_"))
        origins = [Image.open(p).convert("RGB") for p in paths]
        imgs = [spec["preprocess"](o) for o in origins]
        reqs = [engine.build_request(spec, im, spec["prompt"], spec.get("system_prompt")) for im in imgs]
        t = time.time()
        raws, n_tok = run(reqs)
        record("mdp", len(reqs), time.time() - t, n_tok)
        d = out / "mdp"
        d.mkdir(exist_ok=True)
        for p, raw, o, im in zip(paths, raws, origins, imgs):
            (d / f"{p.stem}.md").write_text(engine.postprocess(spec, raw, o, im), encoding="utf-8")
            (d / f"{p.stem}.raw.txt").write_text(raw, encoding="utf-8")

    thai_passes = [p for p in a.passes if p.startswith("thai")]
    if thai_passes:
        from datasets import load_dataset
        rows = [r for r in load_dataset("scb10x/ThaiOCRBench")["test"] if r["Task"] == TASK]
        origins = [r["image"].convert("RGB") for r in rows]
        imgs = [spec["preprocess"](o) for o in origins]
        # the question wrapping of ThaiOCRBench's run_inference.py, as in the harness's run_thai()
        qs = ["Instruction: " + r["question"].strip() + "\nตอบเฉพาะข้อความตามภาพตรง ๆ อย่างกระชับ โดยไม่ใส่คำอธิบายเพิ่มเติม"
              for r in rows]
        for name in thai_passes:
            if name == "thai-q" and (spec.get("runner") or "thai" in spec.get("not_applicable", ())):
                print(f"[thai-q] skipped: {a.model_key} takes no free-form prompts")
                continue
            if name == "thai-q":
                reqs = [engine.build_request(spec, im, q, spec.get("vqa_system_prompt")) for im, q in zip(imgs, qs)]
            else:
                reqs = [engine.build_request(spec, im, spec["prompt"], spec.get("system_prompt")) for im in imgs]
            t = time.time()
            raws, n_tok = run(reqs)
            record(name, len(reqs), time.time() - t, n_tok)
            preds = []
            for r, q, raw, o, im in zip(rows, qs, raws, origins, imgs):
                pred = vqa_clean(raw) if name == "thai-q" else engine.postprocess(spec, raw, o, im)
                preds.append({"type": TASK, "id": r["Id"], "question": q,
                              "answers": [r["answer"].strip() if isinstance(r["answer"], str) else r["answer"]],
                              "predict": pred, "raw": raw})
            (out / f"{name.replace('-', '_')}.json").write_text(json.dumps(preds, ensure_ascii=False, indent=1),
                                                                encoding="utf-8")


if __name__ == "__main__":
    main()

"""ThaiOCRBench "Document parsing" task (211 samples, scb10x/ThaiOCRBench on the HF Hub).

    python bench/thaiocrbench_docparse.py --bench-root /path/to/ThaiOCRBench --out runs/thaiocrbench_dp.json

Writes predictions in the format ThaiOCRBench's eval_scripts/eval.py reads, then runs that evaluator unchanged
(TEDS-based doc_parsing_evaluation) with --python. Needs `datasets` for loading the benchmark.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aksonocr_lite  # noqa: E402

TASK = "Document parsing"
SUFFIX = "\nตอบเฉพาะข้อความตามภาพตรง ๆ อย่างกระชับ โดยไม่ใส่คำอธิบายเพิ่มเติม"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench-root", required=True, help="checkout of the ThaiOCRBench repo (for eval_scripts/eval.py)")
    ap.add_argument("--out", required=True, help="predictions JSON path")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the official evaluator")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dewarp", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not a.eval_only:
        from datasets import load_dataset

        rows = [r for r in load_dataset("scb10x/ThaiOCRBench")["test"] if r["Task"] == TASK]
        rows = rows[: a.limit] if a.limit else rows
        conv = aksonocr_lite.load(device=a.device, dewarp=a.dewarp)
        preds, t0 = [], time.time()
        for k, r in enumerate(rows):
            t = time.time()
            md = next(conv.convert(r["image"].convert("RGB"))).markdown
            q = (r["question"] or "").strip()
            preds.append({"type": TASK, "id": r["Id"], "question": f"Instruction: {q}{SUFFIX}" if q else SUFFIX.strip(),
                          "answers": [r["answer"].strip() if isinstance(r["answer"], str) else r["answer"]], "predict": md})
            print(f"[{k + 1}/{len(rows)}] {r['Id']} {time.time() - t:.2f}s", flush=True)
        out.write_text(json.dumps(preds, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(rows)} samples in {time.time() - t0:.1f}s -> {out}")
    res = out.with_suffix(".eval.json")
    root = Path(a.bench_root).resolve()
    # the official scorer prints every ThaiOCRBench task; only Document parsing items are sent, so show only that one
    r = subprocess.run([a.python, str(root / "eval_scripts" / "eval.py"), "--input_path", str(out.resolve()),
                        "--output_path", str(res.resolve())], cwd=root, capture_output=True, text=True)
    if r.returncode:
        sys.exit(r.stdout + r.stderr)
    print("\n".join(l for l in r.stdout.splitlines() if TASK in l))
    scores = [x["score"] for x in json.loads(res.read_text(encoding="utf-8")) if x.get("type") == TASK]
    print(f"{TASK}: {sum(scores) / len(scores):.4f} over {len(scores)} samples")


if __name__ == "__main__":
    main()

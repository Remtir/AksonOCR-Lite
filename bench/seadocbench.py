"""SEA-DocBench (the document-parsing part of SEA-Vision, arXiv 2603.15409) for one language (default Thai).

Data: the Hugging Face dataset ``xingranzhao/SEA-Vision`` (``ground_truth.json`` + ``SEA-DocBench-images.tar.gz``).
The ground truth is in OmniDocBench format, so it is scored with the OmniDocBench end-to-end evaluator that ships
with MDPBench (unchanged; pass its checkout as ``--scorer-root``).

    python bench/seadocbench.py --gt ground_truth.json --image-dir images/ --scorer-root MDPBench \
        --python <env with the scorer deps>/bin/python --out runs/seadoc_th [--shards 4]

Pages are processed by ``--shards`` worker processes. ``--eval-only`` re-scores existing Markdown files.
"""
import argparse, json, os, subprocess, sys, time
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EVAL_CFG = """end2end_eval:
  metrics:
    text_block: {{metric: [Edit_dist]}}
    display_formula: {{metric: [Edit_dist]}}
    table: {{metric: [TEDS, Edit_dist]}}
    reading_order: {{metric: [Edit_dist]}}
  dataset:
    dataset_name: end2end_dataset
    ground_truth: {{data_path: {gt}}}
    prediction: {{data_path: {pred}}}
    match_method: quick_match
"""


def _worker(args):
    shard, paths, out, device, dewarp, threads = args
    os.environ["AKSONOCR_THREADS"] = str(threads)
    import aksonocr_lite

    conv = aksonocr_lite.load(device=device, dewarp=dewarp)
    times = {}
    for k, p in enumerate(paths):
        p = Path(p)
        dst = out / f"{p.stem}.md"
        if dst.exists():
            continue
        t = time.time()
        try:
            md = next(conv.convert(str(p))).markdown
        except Exception as e:  # keep going; an empty page scores as a miss
            print(f"[shard {shard}] {p.name}: {e!r}", flush=True)
            md = ""
        dst.write_text(md, encoding="utf-8")
        times[p.name] = round(time.time() - t, 3)
        if k % 25 == 0:
            print(f"[shard {shard}] {k + 1}/{len(paths)} {p.name} {times[p.name]:.2f}s", flush=True)
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="SEA-Vision ground_truth.json (all languages)")
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--scorer-root", required=True, help="MDPBench checkout (OmniDocBench evaluator)")
    ap.add_argument("--python", default=sys.executable, help="interpreter with the scorer's dependencies")
    ap.add_argument("--language", default="th")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dewarp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8, help="CPU threads per shard")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()

    out = Path(a.out).absolute()  # not resolve(): a symlinked run dir keeps its own (unique) name
    out.mkdir(parents=True, exist_ok=True)
    gt = [s for s in json.loads(Path(a.gt).read_text(encoding="utf-8"))
          if s["page_info"]["page_attribute"].get("language") == a.language]
    # Minimal GT normalisation so the unchanged OmniDocBench matcher can read SEA-Vision's annotations:
    # formula LaTeX lives in "text" (matcher reads "latex"); a few items use the category "text" (= text_block);
    # text items without any transcription (~4% of Thai items) are marked ignore instead of crashing the matcher.
    textual = {"text_block", "title", "figure_caption", "figure_footnote", "table_caption", "table_footnote"}
    for s in gt:
        for d in s["layout_dets"]:
            if d["category_type"] == "text":
                d["category_type"] = "text_block"
            if d["category_type"] == "equation_isolated" and "latex" not in d:
                d["latex"] = d.get("text", "")
            if d["category_type"] == "table" and "latex" not in d:  # GT tables carry HTML only; the matcher reads
                d["latex"] = d.get("html", "")                         # "latex" when a prediction writes LaTeX tables
            if d["category_type"] in textual | {"equation_isolated"} and "text" not in d:
                d["text"], d["ignore"] = "", True
    if a.limit:
        gt = gt[: a.limit]
    imgs = [str(Path(a.image_dir) / s["page_info"]["image_path"]) for s in gt]

    if not a.eval_only:
        t0 = time.time()
        jobs = [(i, imgs[i:: a.shards], out, a.device, a.dewarp, a.threads) for i in range(a.shards)]
        with get_context("spawn").Pool(a.shards) as pool:
            times = {k: v for d in pool.map(_worker, jobs) for k, v in d.items()}
        (out.parent / f"{out.name}_timings.json").write_text(json.dumps(times, indent=1))
        print(f"{len(times)} pages in {time.time() - t0:.1f}s ({a.shards} shards)")

    root = Path(a.scorer_root).resolve()
    gt_path = out.parent / f"{out.name}_gt.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    cfg = root / "configs" / f"end2end_seadoc_{out.name}.yaml"
    cfg.write_text(EVAL_CFG.format(gt=gt_path, pred=out), encoding="utf-8")
    subprocess.run([a.python, "pdf_validation.py", "--config", str(cfg), "--slim"], cwd=root, check=True)
    summarize(root / "result" / f"{out.name}_result" / f"{out.name}_result_metric_result.json")


def summarize(path: Path):
    """Print the OmniDocBench-style numbers SEA-Vision reports (NED: lower is better; TEDS: higher is better)."""
    m = json.loads(path.read_text())
    g = lambda *k: (lambda d: [d := d.get(x, {}) for x in k][-1])(m)  # noqa: E731
    rows = [("text NED (page avg)", g("text_block", "all", "Edit_dist", "ALL_page_avg")),
            ("formula NED", g("display_formula", "all", "Edit_dist", "ALL_page_avg")),
            ("table TEDS", g("table", "all", "TEDS", "all")),
            ("table NED", g("table", "all", "Edit_dist", "ALL_page_avg")),
            ("reading-order NED", g("reading_order", "all", "Edit_dist", "ALL_page_avg"))]
    print("\nSEA-DocBench summary")
    for k, v in rows:
        print(f"  {k:22s} {v:.4f}" if isinstance(v, float) else f"  {k:22s} n/a")
    for k, v in sorted(g("text_block", "page", "Edit_dist").items()):
        if k.startswith("data_source"):
            print(f"  text NED {k[13:]:20s} {v:.4f}")


if __name__ == "__main__":
    main()

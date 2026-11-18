"""MDPBench end-to-end evaluation for one language (default Thai).

    python bench/mdpbench.py --mdpbench-root /path/to/MDPBench --out runs/mdpbench_th [--dewarp]

Writes one Markdown file per page image (the layout MDPBench's evaluator expects), then runs the unchanged
official scorer (results land in <root>/result/<out name>_result) (pdf_validation.py + tools/calculate_scores.py) with --python (an env that has MDPBench's deps).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import aksonocr_lite  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mdpbench-root", required=True)
    ap.add_argument("--image-dir", default=None, help="default: <root>/MDPBench_dataset/MDPBench_img_public")
    ap.add_argument("--language", default="th")
    ap.add_argument("--out", required=True)
    ap.add_argument("--python", default=sys.executable, help="interpreter for the official MDPBench scorer")
    ap.add_argument("--dewarp", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    root = Path(a.mdpbench_root).resolve()
    img_dir = Path(a.image_dir or root / "MDPBench_dataset" / "MDPBench_img_public")
    out = Path(a.out).absolute()  # not resolve(): a symlinked run dir keeps its own (unique) name
    out.mkdir(parents=True, exist_ok=True)
    pre = a.language.lower() + "_"
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in EXTS and p.stem.lower().startswith(pre))
    if a.limit:
        imgs = imgs[: a.limit]
    if not a.eval_only:
        conv = aksonocr_lite.load(device=a.device, dewarp=a.dewarp)
        t0, times = time.time(), {}
        for k, p in enumerate(imgs):
            t = time.time()
            page = next(conv.convert(str(p)))
            (out / f"{p.stem}.md").write_text(page.markdown, encoding="utf-8")
            times[p.name] = round(time.time() - t, 3)
            print(f"[{k + 1}/{len(imgs)}] {p.name} {times[p.name]:.2f}s", flush=True)
        (out.parent / f"{out.name}_timings.json").write_text(json.dumps(times, indent=1))
        print(f"{len(imgs)} pages in {time.time() - t0:.1f}s")

    gt_all = json.loads((root / "MDPBench_dataset" / "MDPBench_public.json").read_text(encoding="utf-8"))
    gt = [s for s in gt_all if Path(s["page_info"]["image_path"]).stem.lower().startswith(pre)]
    if a.limit:
        names = {p.name for p in imgs}
        gt = [s for s in gt if Path(s["page_info"]["image_path"]).name in names]
    gt_path = out.parent / f"{out.name}_gt.json"
    gt_path.write_text(json.dumps(gt, ensure_ascii=False), encoding="utf-8")
    cfg = root / "configs" / f"end2end_{out.name}_{a.language}.yaml"  # unique per run (parallel scoring)
    cfg.write_text(EVAL_CFG.format(gt=gt_path, pred=out), encoding="utf-8")
    subprocess.run([a.python, "pdf_validation.py", "--config", str(cfg), "--slim"], cwd=root, check=True)
    subprocess.run([a.python, "tools/calculate_scores.py", "--result_folder", str(root / "result" / f"{out.name}_result")],
                   cwd=root, check=True)


if __name__ == "__main__":
    main()

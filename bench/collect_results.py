"""Collect every system's scores (written by the official scorers) into one Markdown table + CSV.

    python bench/collect_results.py runs/vlm > results.md
"""
import csv, io, json, re, sys
from pathlib import Path

NAMES = {  # run dir -> (display name, params, how it runs)
    "aksonocr-lite": ("**AksonOCR-Lite** (this repo)", "81M", "CPU, ONNX"),
    "aksonocr-lite-dewarp": ("**AksonOCR-Lite** `--dewarp`", "81M + dewarp nets", "CPU (dewarp on GPU here), ONNX"),
    "typhoon-ocr2-2b": ("Typhoon OCR V2 2B", "2B", "vLLM"),
    "typhoon-ocr2-0.8b": ("Typhoon OCR V2 0.8B", "0.8B", "vLLM"),
    "typhoon-ocr1.5-2b": ("Typhoon OCR 1.5 2B", "2B", "vLLM"),
    "openthai2.0-ocr-markdown": ("OpenThai2.0-OCR-Markdown", "0.63B", "vLLM"),
    "nemotron-3.5-parse": ("NVIDIA Nemotron 3.5 Parse", "0.63B", "vLLM"),
    "chandra-ocr-2": ("Chandra OCR 2", "5B", "vLLM"),
    "dots.mocr": ("dots.mocr", "3B", "vLLM"),
    "surya-ocr-2.5": ("Surya OCR 2.5", "0.69B", "vLLM"),
    "indra-ocr": ("Indra OCR", "~1M", "CPU, ONNX"),
    "openthai2.0-ocr-text": ("OpenThai2.0-OCR-Text", "1M", "CPU, OpenVINO"),
    "gemini-4-hydrogen": ("Gemini 4 Hydrogen (reasoning low)", "?", "API"),
    "claude-haiku-5.5": ("Claude Haiku 5.5 (reasoning low)", "?", "API"),
}


def num(pat, text, cast=float):
    m = re.search(pat, text)
    return cast(m.group(1)) if m else None


def read(p):
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def collect(d: Path):
    r = {}
    mdp = read(d / "mdp.eval.log")
    r["mdp"] = num(r"All\s*:\s*([\d.]+)", mdp)
    r["mdp_digital"] = num(r"Digit\.\s*:\s*([\d.]+)", mdp)
    r["mdp_photo"] = num(r"Photo\.\s*:\s*([\d.]+)", mdp)
    r["thai_q"] = num(r"Document parsing: ([\d.]+)", read(d / "thai_q.eval.log"))
    r["thai_parse"] = num(r"Document parsing: ([\d.]+)", read(d / "thai_parse.eval.log"))
    sea = read(d / "sea.eval.log")
    r["sea_text_ned"] = num(r"text NED \(page avg\)\s+([\d.]+)", sea)
    r["sea_table_teds"] = num(r"table TEDS\s+([\d.]+)", sea)
    r["sea_ro_ned"] = num(r"reading-order NED\s+([\d.]+)", sea)
    for f, k in (("lines.eval.json", "lines"), ("lines_rec.eval.json", "lines_rec")):
        if (d / f).exists():
            j = json.loads((d / f).read_text())
            r[f"{k}_doc"] = j["pv_hi_doc"]["exact"]
            r[f"{k}_photo"] = j["pv_hi_photo"]["exact"]
            r[f"{k}_mean"] = j["mean_exact"]
            r[f"{k}_nospace"] = j.get("mean_exact_nospace")
    if (d / "timing.json").exists():
        t = json.loads((d / "timing.json").read_text())
        reqs = sum(s.get("requests", s.get("items", 0)) for s in t.values())
        wall = sum(s.get("wall_s", s.get("seconds", 0)) for s in t.values())
        r["items_per_s"] = round(reqs / wall, 2) if wall else None
    return r


def fmt(v, nd=1):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    root = Path(sys.argv[1])
    rows = []
    for key, (name, params, how) in NAMES.items():
        d = root / key
        if d.exists():
            rows.append((key, name, params, how, collect(d)))
    rows.sort(key=lambda x: -(x[4].get("mdp") or 0))
    print("| System | Params | Runs on | MDPBench-th ↑ | digital | photo | ThaiOCRBench DP ↑ (question) | (page parse) "
          "| SEA-DocBench-th text NED ↓ | table TEDS ↑ | pv_hi lines exact ↑ (doc / photo) |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, name, params, how, r in rows:
        lines = f"{fmt(r.get('lines_doc'))} / {fmt(r.get('lines_photo'))}" if r.get("lines_doc") is not None else "–"
        if r.get("lines_rec_doc") is not None:
            lines += f" (recognizer only: {fmt(r['lines_rec_doc'])} / {fmt(r['lines_rec_photo'])})"
        print(f"| {name} | {params} | {how} | {fmt(r.get('mdp'))} | {fmt(r.get('mdp_digital'))} | {fmt(r.get('mdp_photo'))} "
              f"| {fmt(r.get('thai_q'), 3)} | {fmt(r.get('thai_parse'), 3)} | {fmt(r.get('sea_text_ned'), 4)} "
              f"| {fmt(r.get('sea_table_teds'), 3)} | {lines} |")
    buf = io.StringIO()
    keys = sorted({k for *_, r in rows for k in r})
    w = csv.writer(buf)
    w.writerow(["system"] + keys)
    for key, *_, r in rows:
        w.writerow([key] + [r.get(k) for k in keys])
    (root / "results.csv").write_text(buf.getvalue())


if __name__ == "__main__":
    main()

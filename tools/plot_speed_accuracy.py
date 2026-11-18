"""README charts (light + dark, PNG + SVG): accuracy bars, speed bars, speed vs accuracy scatter, from the benchmark run directories.

    python tools/plot_speed_accuracy.py runs/vlm --out assets/charts

Speed = measured pages per second on one fixed set of 311 document pages (160 MDPBench-th + every 10th SEA-DocBench-th
page), timed end to end (image load to Markdown), from bench/SPEED.json:
  - AksonOCR-Lite: 16 single-thread worker processes, CPU only (bench/speed_bench.py)
  - VLMs: `vllm serve` on one RTX 5090 with the eval recipe, up to 256 requests in flight (bench/vlm_speed.py)
Systems without a measurement show "not measured"; API models have no comparable speed.
"""
import argparse, json, math, re, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from collect_results import collect  # noqa: E402

CPU_CORES = 16
LABELS = {
    "aksonocr-lite": "AksonOCR-Lite",
    "aksonocr-lite-dewarp": "AksonOCR-Lite --dewarp",
    "typhoon-ocr2-2b": "Typhoon OCR V2 2B",
    "typhoon-ocr2-0.8b": "Typhoon OCR V2 0.8B",
    "typhoon-ocr1.5-2b": "Typhoon OCR 1.5 2B",
    "openthai2.0-ocr-markdown": "OpenThai-Markdown",
    "nemotron-3.5-parse": "Nemotron 3.5 Parse",
    "chandra-ocr-2": "Chandra OCR 2",
    "dots.mocr": "dots.mocr",
    "surya-ocr-2.5": "Surya OCR 2.5",
    "indra-ocr": "Indra OCR",
    "openthai2.0-ocr-text": "OpenThai-Text",
    "gemini-4-hydrogen": "Gemini 4 Hydrogen (API)",
    "claude-haiku-5.5": "Claude Haiku 5.5 (API)",
}
# hand-placed labels per panel: key -> (dx, dy in points, horizontal alignment)
R, L, C = "left", "right", "center"
NUDGE = {
    "mdp": {"aksonocr-lite": (-12, -4, L), "aksonocr-lite-dewarp": (-12, -4, L), "surya-ocr-2.5": (0, -16, C),
            "chandra-ocr-2": (-9, -4, L), "typhoon-ocr1.5-2b": (-9, -4, L), "dots.mocr": (0, -16, C),
            "indra-ocr": (-10, -4, L)},
    "sea": {"aksonocr-lite": (-12, 3, L), "aksonocr-lite-dewarp": (-12, 3, L), "typhoon-ocr2-2b": (14, -7, R),
            "typhoon-ocr2-0.8b": (9, -14, R), "chandra-ocr-2": (-9, -4, L), "typhoon-ocr1.5-2b": (9, -8, R),
            "dots.mocr": (0, -17, C),
            "surya-ocr-2.5": (-9, -4, L), "openthai2.0-ocr-text": (-9, -4, L), "indra-ocr": (0, 9, C)},
}

THEMES = {
    "light": dict(bg="#ffffff", ink="#1f2329", muted="#6b7079", grid="#e9eaec", ctx="#b4b8bf", ctx_edge="#8f949c",
                  accent="#EB5796", accent_soft="#fbe3ed"),
    "dark": dict(bg="#0d1117", ink="#e6e8eb", muted="#9aa0a8", grid="#262c35", ctx="#5b616b", ctx_edge="#7d838d",
                 accent="#F07BAA", accent_soft="#3a2230"),
}


SPEED_JSON = Path(__file__).resolve().parents[1] / "bench" / "SPEED.json"


def speed(key):
    """Measured pages/s on the 311-page speed set (bench/speed_bench.py, bench/vlm_speed.py), or None."""
    m = json.loads(SPEED_JSON.read_text()) if SPEED_JSON.exists() else {}
    if key == "aksonocr-lite" and "throughput_cpu_16x1" in m:
        return m["throughput_cpu_16x1"]["pages_per_s"], "cpu"
    if f"vlm_{key}" in m:
        return m[f"vlm_{key}"]["throughput"]["pages_per_s"], "gpu"
    return None, "nm"


def points(root: Path):
    out = []
    for key, label in LABELS.items():
        d = root / key
        if not d.exists():
            continue
        r = collect(d)
        if r.get("mdp") is None or r.get("sea_text_ned") is None:
            continue
        if "gemini" in key or "claude" in key:
            sp, hw = None, "api"
        elif key.endswith("-dewarp"):  # dewarp ran as a GPU pre-pass here; its speed is not a CPU-only number
            sp, hw = None, "dewarp"
        else:
            sp, hw = speed(key)
        out.append(dict(key=key, label=label, speed=sp, hw=hw, mdp=r["mdp"], sea=100 * (1 - r["sea_text_ned"]),
                        mdp_digital=r.get("mdp_digital"),
                        # ThaiOCRBench Document parsing: the better of the two modes (item question / page parse)
                        thai=max([v for v in (r.get("thai_q"), r.get("thai_parse")) if v is not None], default=None)))
    return out


def draw(pts, theme, path):
    pts = [p for p in pts if p["speed"] is not None]
    c = THEMES[theme]
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 10.5, "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), facecolor=c["bg"])
    panels = [("mdp", "MDPBench, Thai — overall ↑", None),
              ("sea", "SEA-DocBench, Thai — text accuracy, 1 − NED ↑", None)]
    for ax, (k, title, _) in zip(axes, panels):
        ax.set_facecolor(c["bg"])
        ax.set_xscale("log")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(c["grid"])
        ax.grid(True, color=c["grid"], linewidth=0.8, zorder=0)
        ax.tick_params(colors=c["muted"], length=0, pad=6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        sp = [p["speed"] for p in pts]
        ax.set_xlim(min(sp) / 1.6, max(sp) * 2.2)
        ax.set_xticks([t for t in (0.25, 0.5, 1, 2, 5, 10, 20) if min(sp) / 1.6 <= t <= max(sp) * 2.2])
        ax.set_title(title, loc="left", color=c["ink"], fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("pages per second (log scale) →", color=c["muted"], labelpad=8)
        for p in sorted(pts, key=lambda p: p["key"].startswith("aksonocr")):
            mine = p["key"].startswith("aksonocr")
            marker = "s" if p["hw"] == "cpu" else "o"
            ax.scatter(p["speed"], p[k], s=150 if mine else 70, marker=marker, zorder=4 if mine else 3,
                       color=c["accent"] if mine else c["ctx"], edgecolors=c["bg"], linewidths=2)
            dx, dy, ha = NUDGE[k].get(p["key"], (9, -4, R))
            ax.annotate(p["label"], (p["speed"], p[k]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        color=c["ink"] if mine else c["muted"], fontsize=10.5 if mine else 9.5,
                        fontweight="bold" if mine else "normal", zorder=5)
        lo = min(p[k] for p in pts)
        ax.set_ylim(lo - (lo % 5) - 2, 100 if k == "sea" else 92)
    # legend: hardware shape (identity of AksonOCR is carried by its bold direct label)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c["ctx"], markersize=8, label="VLM on 1× RTX 5090 (vLLM)"),
         Line2D([], [], marker="s", ls="", color=c["accent"], markersize=9, label=f"AksonOCR-Lite (CPU only, {CPU_CORES} processes)")]
    leg = fig.legend(handles=h, loc="upper right", ncol=2, frameon=False, bbox_to_anchor=(0.985, 0.995))
    for t in leg.get_texts():
        t.set_color(c["muted"])
    fig.text(0.012, 0.965, "Speed vs accuracy", color=c["ink"], fontsize=15, fontweight="bold")
    fig.text(0.012, 0.018, "Measured on the same 311 pages (160 MDPBench-th + 151 SEA-DocBench-th). GPU = vLLM on one RTX 5090; "
             "CPU = 16 single-thread processes on an EPYC 7763. See bench/SPEED.json", color=c["muted"], fontsize=8.5)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.83, bottom=0.17, wspace=0.18)
    fig.savefig(path, facecolor=c["bg"], dpi=200)
    plt.close(fig)


LOGO_DIR = Path(__file__).resolve().parents[1] / "assets" / "logos"
LOGO = {"aksonocr": "aksonocr", "typhoon": "typhoon", "openthai": "openthai", "nemotron": "nvidia", "dots": "dots",
        "chandra": "datalab", "surya": "datalab", "indra": "datalab", "gemini": "google", "claude": "anthropic"}
_logo_cache = {}


def logo(key):
    """The maker's logo, fitted in a square, as an RGBA array; None if unknown."""
    name = next((v for k, v in LOGO.items() if key.startswith(k)), None)
    if name is None:
        return None
    if name not in _logo_cache:
        from PIL import Image
        import numpy as np
        s = 124
        tile = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        im = Image.open(LOGO_DIR / f"{name}.png").convert("RGBA")
        im.thumbnail((s, s), Image.LANCZOS)
        tile.alpha_composite(im, ((s - im.width) // 2, (s - im.height) // 2))
        _logo_cache[name] = np.asarray(tile)
    return _logo_cache[name]


def draw_panels(rows, panels, theme, path, title, footer, col_w=0.95, panel_h=4.0, ncols=1):
    """Vertical bar panels stacked top to bottom, each sorted best-to-worst from left to right on its own metric.
    Above every bar: the maker's logo, with the value (boxed) between the logo and the bar; the system name below.
    rows: dicts with key, label and one value per panel key (None -> the row's note[k] or "no result").
    panels: (key, title, limits, format, lower_is_better); limits = (lo, hi), None (0 to max) or "zoom" (the axis
    starts just below the lowest value so close scores stay apart; the axis says where it starts)."""
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    c = THEMES[theme]
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 10.5, "svg.fonttype": "path"})
    n, m = len(rows), len(panels)
    label_h = 1.35  # rotated names under each panel
    lpad = 1.9  # room for the leftmost rotated name
    pw = n * col_w  # one panel's plot width
    nrows = math.ceil(m / ncols)
    w = max(8.0, ncols * (lpad + pw) + 0.3)
    h = 0.9 + nrows * (panel_h + label_h) + 0.85
    fig = plt.figure(figsize=(w, h), facecolor=c["bg"])
    for i, (k, ptitle, lim, fmt, low_best) in enumerate(panels):
        top_in = 0.9 + (i // ncols) * (panel_h + label_h)
        left_in = lpad + (i % ncols) * (lpad + pw)
        ax = fig.add_axes([left_in / w, 1 - (top_in + panel_h) / h, (w - 0.3 - ncols * lpad) / ncols / w if ncols > 1 else 1 - (lpad + 0.3) / w, (panel_h - 0.35) / h])
        order = sorted(rows, key=lambda p: (p[k] is None, (p[k] if low_best else -p[k]) if p[k] is not None else 0))
        ax.set_facecolor(c["bg"])
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ax.spines["bottom"].set_color(c["grid"])
        ax.tick_params(colors=c["muted"], length=0)
        ax.grid(True, axis="y", color=c["grid"], linewidth=0.8, zorder=0)
        ax.set_title(ptitle, loc="left", color=c["ink"], fontsize=12, fontweight="bold", pad=12)
        vals = [p[k] for p in order if p[k] is not None]
        top = max(vals)
        if lim == "zoom":
            span = top - min(vals)
            step = 5 if span > 10 else 1 if span > 2 else 0.1
            lo = max(0, math.floor((min(vals) - 0.25 * span) / step) * step)
        else:
            lo = lim[0] if lim else 0
            top = max(top, lim[1]) if lim else top
        f = 56 / ((panel_h - 0.35) * 72)  # headroom for logo + value above the tallest bar, in points
        hi = (top - f * lo) / (1 - f)
        for x, p in enumerate(order):
            mine = p["key"].startswith("aksonocr")
            v = p[k]
            if v is None:
                ax.annotate(p.get("note", {}).get(k, "no result"), (x, lo), xytext=(0, 40), textcoords="offset points",
                            ha="center", va="bottom", rotation=90, color=c["muted"], fontsize=9, style="italic")
            else:
                ax.bar(x, v - lo, bottom=lo, width=0.66, color=c["accent"] if mine else c["ctx"], zorder=3,
                       edgecolor=c["bg"], linewidth=1)
            img = logo(p["key"])
            base = v if v is not None else lo
            if v is not None:
                ax.annotate(fmt.format(v), (x, base), xytext=(0, 6), textcoords="offset points", ha="center",
                            va="bottom", color=c["ink"], fontsize=9.5, fontweight="bold" if mine else "normal",
                            zorder=5, bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.25", fc=c["bg"],
                                                ec="#1f2329", lw=1))
            if img is not None:
                ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.19), (x, base), xybox=(0, 33 if v is not None else 14),
                                             boxcoords="offset points", frameon=False, annotation_clip=False,
                                             zorder=4))
        ax.set_xlim(-0.6, n - 0.4)
        ax.set_ylim(lo, hi)
        if lo > 0:  # truncated axis: say so next to the title
            ax.annotate(f"axis starts at {lo:g}", xy=(1, 1), xycoords="axes fraction", xytext=(0, 12),
                        textcoords="offset points", ha="right", va="baseline", color=c["muted"], fontsize=9.5,
                        style="italic")
        ax.set_xticks(range(n))
        ax.set_xticklabels([p["label"] for p in order], rotation=32, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=6, labelsize=9.5)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        for t, p in zip(ax.get_xticklabels(), order):
            mine = p["key"].startswith("aksonocr")
            t.set_color(c["ink"] if mine else c["muted"])
            t.set_fontweight("bold" if mine else "normal")
    fig.text(0.3 / w, 1 - 0.3 / h, title, color=c["ink"], fontsize=15, fontweight="bold", va="top")
    fig.text(0.3 / w, 0.12 / h, footer, linespacing=1.5, color=c["muted"], fontsize=8.5)
    fig.savefig(path, facecolor=c["bg"], dpi=200)
    plt.close(fig)


ACC_NOTE = ("ThaiOCRBench: better of its two modes (item question / page parse), ×100. SEA-DocBench-th: text accuracy "
            "= 100 − NED×100 (page average). MDPBench-th: overall (digital + photographed pages).")
BENCH_NAMES = {"mdp": "MDPBench-th", "thai": "ThaiOCRBench DP", "sea": "SEA-DocBench-th"}


def accuracy_rows(pts):
    rows = []
    for p in pts:
        r = {**p, "sea_err": 100 - p["sea"], "thai100": None if p["thai"] is None else 100 * p["thai"]}
        parts = [p["mdp"], r["thai100"], p["sea"]]
        r["avg"] = None if None in parts else sum(parts) / 3
        rows.append(r)
    return rows


def draw_accuracy(pts, theme, out):
    """accuracy_<theme>: the average of the three benchmarks; accuracy_<bench>_<theme>: one chart per benchmark."""
    rows = accuracy_rows(pts)
    one = dict(col_w=0.95, panel_h=4.0)
    for ext in ("svg", "png"):
        draw_panels(rows, [("avg", "Average of MDPBench-th, ThaiOCRBench DP and SEA-DocBench-th (0–100) ↑", "zoom",
                            "{:.1f}", False)], theme, out / f"accuracy_{theme}.{ext}",
                    "Accuracy: AksonOCR-Lite vs other Thai OCR systems",
                    ACC_NOTE + "\nPer benchmark: BENCHMARKS.md. "
                    "Full tables: bench/RESULTS.md", **one)
        draw_panels(rows, [("mdp", "MDPBench-th, overall (digital + photo) ↑", "zoom", "{:.1f}", False)], theme,
                    out / f"accuracy_mdp_{theme}.{ext}", "Accuracy: MDPBench-th (Thai)",
                    "160 Thai pages (40 digital, 120 photographed), official overall score; digital / photo split in bench/RESULTS.md. "
                    "Official MDPBench scorer.", **one)
        draw_panels(rows, [("thai", "ThaiOCRBench, Document parsing ↑", "zoom", "{:.3f}", False)], theme,
                    out / f"accuracy_thai_{theme}.{ext}", "Accuracy: ThaiOCRBench, Document parsing",
                    "211 items, official eval.py. Better of its two modes (item question / page parse); both in "
                    "bench/RESULTS.md.", **one)
        draw_panels(rows, [("sea_err", "SEA-DocBench-th, text error (NED×100) ↓", "zoom", "{:.1f}", True)], theme,
                    out / f"accuracy_sea_{theme}.{ext}", "Accuracy: SEA-DocBench-th (Thai)",
                    "1,506 pages, OmniDocBench end-to-end evaluator; page-average text NED. Table TEDS in "
                    "bench/RESULTS.md.", **one)


def speed_rows(m):
    """One page at a time (s/page) and batched (pages/s) from one block of bench/SPEED.json."""
    rows = [dict(key="aksonocr-lite", label="AksonOCR-Lite, CPU", lat=m["latency_cpu_16threads"]["s_per_page"]["mean"],
                 thr=m["throughput_cpu_16x1"]["pages_per_s"]),
            dict(key="aksonocr-lite-gpu", label="AksonOCR-Lite, GPU", lat=m["latency_gpu"]["s_per_page"]["mean"],
                 thr=None, note={"thr": "not measured"})]
    # OpenThai2.0-OCR-Text reads text lines only; it is compared on text recognition in draw_rec
    for k, v in m.items():
        if k.startswith("vlm_"):
            rows.append(dict(key=k[4:], label=LABELS.get(k[4:], k[4:]) + ", GPU",
                             lat=v["latency_1_request"]["s_per_page"]["mean"], thr=v["throughput"]["pages_per_s"]))
    return rows


SPEED_NOTE = ("One at a time: AksonOCR-Lite 1 process × 16 threads (GPU: --device cuda); VLMs 1 request (first 40 pages). "
              "\nBatched: AksonOCR-Lite 16 single-thread processes, CPU only; VLMs vLLM, up to 256 requests in flight. "
              "CPU = EPYC 7763, GPU = RTX 5090.\n")


def draw_speed(theme, out):
    """speed_<theme>: the average over the three benchmarks' page sets; speed_<bench>_<theme>: one per benchmark.
    Falls back to the mixed 311-page set until the per-benchmark runs exist."""
    m = json.loads(SPEED_JSON.read_text())
    panels = [("lat", "One page at a time, s / page ↓", None, "{:.2f}", True),
              ("thr", "Batched, pages / s ↑", None, "{:.2f}", False)]
    sets = m.get("sets", {})
    per = {b: speed_rows(sets[b]) for b in BENCH_NAMES if b in sets and all(
        k in sets[b] for k in ("latency_cpu_16threads", "latency_gpu", "throughput_cpu_16x1"))}
    if len(per) == 3:
        avg = []
        for r in per["mdp"]:
            same = [next((x for x in per[b] if x["key"] == r["key"]), None) for b in per]
            if None in same:
                continue
            a = dict(r)
            for k in ("lat", "thr"):
                v = [x[k] for x in same]
                a[k] = None if None in v else sum(v) / 3
            avg.append(a)
        sizes = {b: sets[b]["pages"] for b in per}
        title, rows = "Speed: average over the three benchmarks", avg
        note = SPEED_NOTE + (f"Mean of the per-benchmark results: MDPBench-th {sizes['mdp']} pages, ThaiOCRBench DP "
                             f"{sizes['thai']}, SEA-DocBench-th {sizes['sea']} (every 10th page).\n"
                             "Per benchmark: BENCHMARKS.md; tables: bench/SPEED.md")
    else:
        title, rows = "Speed: same 311 pages, end to end", speed_rows(m)
        note = SPEED_NOTE + "Pages: 160 MDPBench-th + 151 SEA-DocBench-th. Full tables: bench/SPEED.md"
    for ext in ("svg", "png"):
        draw_panels(rows, panels, theme, out / f"speed_{theme}.{ext}", title, note, col_w=1.25, ncols=2)
        for b, rws in per.items():
            draw_panels(rws, panels, theme, out / f"speed_{b}_{theme}.{ext}",
                        f"Speed: {BENCH_NAMES[b]} pages ({sets[b]['pages']})", SPEED_NOTE + "Tables: bench/SPEED.md",
                        col_w=1.25, ncols=2)


def draw_rec(theme, path):
    """Text recognition only (bench/rec_bench.py): the same 469 line crops, no detection or layout."""
    m = json.loads(SPEED_JSON.read_text())
    # the recognizer on upright crops, without the 180° re-read (that pass is for upside-down lines in full pages;
    # its numbers are in bench/SPEED.md)
    names = {"aksonocr-lite-noflip": "AksonOCR-Lite", "openthai-text": "OpenThai2.0-OCR-Text"}
    rows = [dict(key=k, label=v, exact=m[f"rec_{k}"]["score"]["mean_exact"],
                 ned=round(100 - (m[f"rec_{k}"]["score"]["pv_hi_doc"]["1-NED"] +
                                  m[f"rec_{k}"]["score"]["pv_hi_photo"]["1-NED"]) / 2, 2),
                 ms=m[f"rec_{k}"]["latency_1thread_ms"]["mean"], lps=m[f"rec_{k}"]["throughput_16x1_lines_per_s"])
            for k, v in names.items() if f"rec_{k}" in m]
    panels = [("exact", "Exact match, % ↑", (0, 100), "{:.1f}", False),
              ("ned", "Character error (NED×100) ↓", None, "{:.2f}", True),
              ("ms", "One line at a time, ms ↓", None, "{:.1f}", True),
              ("lps", "Batched, lines / s ↑", None, "{:,.0f}", False)]
    draw_panels(rows, panels, theme, path, "Text recognition only: same 469 line crops",
                "Internal printed-line set (184 document + 285 photo lines; labels agreed by 3+ of 4 readers). Each "
                "recognizer reads the pre-cut crop, no detection or layout (AksonOCR-Lite without its 180° re-read).\n"
                "CPU only (EPYC 7763): one line at a time = 1 process, 1 thread; batched = 16 single-thread "
                "processes. OpenThai on OpenVINO. bench/rec_bench.py, bench/SPEED.md", col_w=1.25, panel_h=3.2, ncols=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out", default="assets/charts")
    a = ap.parse_args()
    pts = points(Path(a.runs))
    for p in sorted(pts, key=lambda p: -p["mdp"]):
        print(f"{p['label']:28s} {p['hw']}  {p['speed'] or 0:6.2f} p/s  mdp {p['mdp']:.1f}  sea {p['sea']:.1f}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for th in ("light",):  # charts on a white background
        draw(pts, th, out / f"speed_accuracy_{th}.svg")
        draw(pts, th, out / f"speed_accuracy_{th}.png")
        draw_accuracy(pts, th, out)
        draw_speed(th, out)
        for ext in ("svg", "png"):
            draw_rec(th, out / f"rec_{th}.{ext}")


if __name__ == "__main__":
    main()

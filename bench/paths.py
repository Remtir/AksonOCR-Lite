"""Where the benchmark data lives. Every path can be overridden with an environment variable; the defaults
assume the layout below (relative to the working directory):

    data/MDPBench/                  MDPBench checkout (github.com/Yuliang-Liu/MultimodalOCR, MDPBench/)
    data/SEA-Vision/gt_th.json      Thai items of SEA-Vision's ground_truth.json
    data/SEA-Vision/images/         SEA-DocBench page images
    data/pv_hi/pv_hi_doc.txt        internal printed-line set ("<image path>\\t<label>" per line; not released)
    data/pv_hi/pv_hi_photo.txt
    runs/vlm/                       outputs of every system
"""
import os
from pathlib import Path


def _p(env, default):
    return Path(os.environ.get(env, default))


DATA = _p("AKSONOCR_BENCH_DATA", "data")
RUNS = _p("AKSONOCR_BENCH_RUNS", "runs/vlm")
MDPBENCH_ROOT = _p("MDPBENCH_ROOT", DATA / "MDPBench")
MDP_IMAGES = _p("MDPBENCH_IMAGES", MDPBENCH_ROOT / "MDPBench_dataset" / "MDPBench_img_public")
SEA_GT = _p("SEA_DOCBENCH_GT", DATA / "SEA-Vision" / "gt_th.json")
SEA_IMAGES = _p("SEA_DOCBENCH_IMAGES", DATA / "SEA-Vision" / "images")
PV_HI = _p("PV_HI_DIR", DATA / "pv_hi")
LINE_SPLITS = {"pv_hi_doc": PV_HI / "pv_hi_doc.txt", "pv_hi_photo": PV_HI / "pv_hi_photo.txt"}
THAI_CACHE = _p("THAIOCRBENCH_CACHE", RUNS / "_thai_dp")  # ThaiOCRBench DP images + meta, exported once

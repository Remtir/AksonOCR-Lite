"""Model files and ONNX Runtime sessions.

Weights are fetched from the Hugging Face Hub on first use and cached by huggingface_hub.
Set AKSONOCR_WEIGHTS=/path/to/dir to use a local directory with the same file names instead.
Set AKSONOCR_THREADS=N to fix the CPU threads per ONNX session (default: min(usable cores, 16)).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import onnxruntime as ort

ort.set_default_logger_severity(3)  # silence per-node fallback warnings from the CUDA provider

# file name -> (hf repo, file in repo). layout/det/doc_ori/textline_ori are unchanged copies of the official
# PaddlePaddle ONNX exports (Apache-2.0), mirrored so every file loads from one repo.
MODEL_FILES = {
    "layout.onnx": ("Remtir/AksonOCR-Lite", "layout.onnx"),
    "det.onnx": ("Remtir/AksonOCR-Lite", "det.onnx"),
    "doc_ori.onnx": ("Remtir/AksonOCR-Lite", "doc_ori.onnx"),
    "textline_ori.onnx": ("Remtir/AksonOCR-Lite", "textline_ori.onnx"),
    "rec.onnx": ("Remtir/AksonOCR-Lite", "rec.onnx"),
    "rec_charset.json": ("Remtir/AksonOCR-Lite", "rec_charset.json"),
    "tatr.onnx": ("Remtir/AksonOCR-Lite", "tatr.onnx"),
    "dewarp_mask.onnx": ("Remtir/AksonOCR-Lite", "dewarp_mask.onnx"),
    "dewarp_grid.onnx": ("Remtir/AksonOCR-Lite", "dewarp_grid.onnx"),
    "config.json": ("Remtir/AksonOCR-Lite", "config.json"),  # pipeline manifest (also what the Hub counts downloads by)
}


def model_path(name: str) -> str:
    local = os.getenv("AKSONOCR_WEIGHTS")
    if local:
        p = Path(local) / name
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (AKSONOCR_WEIGHTS={local})")
        return str(p)
    from huggingface_hub import hf_hub_download

    repo, fname = MODEL_FILES[name]
    return hf_hub_download(repo, fname)


def touch_manifest() -> None:
    """Fetch config.json from the model repo once per load() (cached; skipped offline or with local weights)."""
    if os.getenv("AKSONOCR_WEIGHTS"):
        return
    try:
        model_path("config.json")
    except Exception:  # noqa: BLE001 (offline, no network: the weights are already cached)
        pass


def providers(device: str = "auto") -> list:
    avail = ort.get_available_providers()
    if device == "cpu" or (device == "auto" and "CUDAExecutionProvider" not in avail):
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in avail:
        raise RuntimeError("device='cuda' requested but onnxruntime-gpu with CUDA is not installed")
    # inputs have a dynamic width: cuDNN EXHAUSTIVE costs ~1 s per new shape, HEURISTIC ~0.5 s, DEFAULT ~50 ms
    opts = {"cudnn_conv_algo_search": "DEFAULT"}
    if os.getenv("AKSONOCR_GPU_MEM_GB"):  # cap each session's CUDA memory pool (several processes sharing one GPU)
        opts.update(gpu_mem_limit=int(float(os.environ["AKSONOCR_GPU_MEM_GB"]) * 2**30),
                    arena_extend_strategy="kSameAsRequested")
    return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]


def cpu_threads() -> int:
    """Usable cores: honours the cgroup CPU quota (containers often report every host core in os.cpu_count())."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


@lru_cache(maxsize=None)
def session(name: str, device: str = "auto", threads: int = 0) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads or int(os.getenv("AKSONOCR_THREADS", 0)) or min(cpu_threads(), 16)
    so.log_severity_level = 3
    return ort.InferenceSession(model_path(name), so, providers=providers(device))

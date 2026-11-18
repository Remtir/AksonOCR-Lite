# Speed benchmark

Every system is timed on each benchmark's own pages: the 160 Thai MDPBench pages (40 digital, 120 photographed), the
211 ThaiOCRBench Document parsing images, and every 10th Thai SEA-DocBench page (151 of 1,506). Each page is timed
end to end, from the image file to the final Markdown (decode, preprocessing, model, post-processing).
AksonOCR-Lite runs with default settings (page rotation on, tables on, no dewarp). The VLMs run under `vllm serve`
with the same per-model recipe (prompt, image size, decoding, token limit) as in the accuracy runs. Machine: AMD
EPYC 7763 (128 logical CPUs), NVIDIA RTX 5090. Raw numbers: [SPEED.json](SPEED.json) (`sets`).

"One at a time": one request or page at a time, after warm-up (AksonOCR-Lite: one process with 16 threads; VLMs:
concurrency 1 on the first 40 pages of the set, 3 of them warm-up). "Batched": AksonOCR-Lite with 16 single-thread
processes; the VLMs with up to 256 requests in flight (6 API-server processes), over every page of the set.

## Average of the three benchmarks

| System | Hardware | s / page, one at a time | pages / s, batched | MDPBench-th ↑ | SEA-DocBench-th text NED ↓ |
|---|---|---:|---:|---:|---:|
| **AksonOCR-Lite** | CPU only | **1.48** | **1.76** | **79.5** | **0.0580** |
| **AksonOCR-Lite** | RTX 5090 + CPU | **1.08** | – | **79.5** | **0.0580** |
| OpenThai2.0-OCR-Markdown | RTX 5090, vLLM | 4.14 | 4.88 | 70.3 | 0.1035 |
| Typhoon OCR V2 0.8B | RTX 5090, vLLM | 4.17 | 2.14 | 78.7 | 0.0725 |
| Typhoon OCR V2 2B | RTX 5090, vLLM | 5.11 | 1.23 | 84.3 | 0.0626 |

With `--dewarp`, AksonOCR-Lite reaches 85.6 on MDPBench-th, at the cost of one extra dewarp pass per page.

## Per benchmark

s / page one at a time, pages / s batched (pages that hit the token limit in brackets: they are usually repetition
loops and cost the full generation budget). Charts: [MDPBench-th](../assets/charts/speed_mdp_light.png) ·
[ThaiOCRBench DP](../assets/charts/speed_thai_light.png) · [SEA-DocBench-th](../assets/charts/speed_sea_light.png).

| System | MDPBench-th (160) | ThaiOCRBench DP (211) | SEA-DocBench-th (151) |
|---|---|---|---|
| AksonOCR-Lite, CPU | 1.71 s · 1.57 p/s | 1.31 s · 2.03 p/s | 1.42 s · 1.68 p/s |
| AksonOCR-Lite, RTX 5090 | 1.15 s | 1.14 s | 0.97 s |
| OpenThai2.0-OCR-Markdown | 3.91 s · 3.80 p/s | 3.40 s · 7.64 p/s | 5.12 s · 3.21 p/s |
| Typhoon OCR V2 0.8B | 5.02 s · 0.79 p/s (17) | 3.58 s · 3.09 p/s (1) | 3.90 s · 2.54 p/s (1) |
| Typhoon OCR V2 2B | 5.29 s · 0.46 p/s (11) | 4.50 s · 1.78 p/s (2) | 5.56 s · 1.46 p/s (2) |

MDPBench-th is mostly photographed pages: AksonOCR-Lite spends more on page rotation there, and both Typhoon models
loop on 11–17 of its pages, which stretches their batched runs.

## Earlier mixed set (311 pages)

The tables below are the first measurement, on one mixed set: the 160 MDPBench-th pages + the 151 SEA-DocBench-th
pages above.

### One page at a time

| System | Hardware | mean s | median s | p90 s | pages |
|---|---|---:|---:|---:|---:|
| AksonOCR-Lite | CPU, 1 process × 16 threads | 1.53 | 1.49 | 1.96 | 308 |
| AksonOCR-Lite | RTX 5090 (`--device cuda`, recognizer on CPU) | 1.08 | 1.07 | 1.48 | 308 |
| OpenThai2.0-OCR-Markdown | RTX 5090, vLLM, concurrency 1 | 4.01 | 3.91 | 4.97 | 37 |
| Typhoon OCR V2 0.8B | RTX 5090, vLLM, concurrency 1 | 4.99 | 3.79 | 6.05 | 37 |
| Typhoon OCR V2 2B | RTX 5090, vLLM, concurrency 1 | 5.34 | 5.18 | 8.19 | 37 |

The first 3 pages are warm-up and are not counted. The VLMs are timed on the first 40 pages.

AksonOCR-Lite mean time per stage (s):

| Stage | CPU | RTX 5090 |
|---|---:|---:|
| page rotation | 0.06 | 0.06 |
| layout | 0.45 | 0.23 |
| text detection + recognition | 0.81 | 0.60 |
| tables (structure + cell OCR) | 0.09 | 0.08 |

### Batched throughput (all 311 pages)

| System | Setup | wall s | pages / s | output tok / s | hit token limit |
|---|---|---:|---:|---:|---:|
| AksonOCR-Lite | CPU, 16 processes × 1 thread | 178.9 | 1.74 | – | – |
| OpenThai2.0-OCR-Markdown | RTX 5090, vLLM, ≤256 in flight | 87.3 | 3.56 | 5,692 | 0 |
| Typhoon OCR V2 0.8B | RTX 5090, vLLM, ≤256 in flight | 245.0 | 1.27 | 2,622 | 15 |
| Typhoon OCR V2 2B | RTX 5090, vLLM, ≤256 in flight | 400.2 | 0.78 | 1,224 | 11 |

AksonOCR-Lite uses 16 of the 128 logical CPUs and no GPU. Throughput grows with more processes. Pages that hit the
token limit are usually repetition loops; they cost the full generation budget.

## Text recognition only

The recognizer alone, on the same 469 pre-cut line crops (internal printed-line set: 184 document + 285 photo
lines, labels agreed by 3+ of 4 readers), against the other Thai CPU line recognizer, OpenThai2.0-OCR-Text. No
detection, layout or tables: each recognizer reads one crop file and returns one string, with its own defaults
(AksonOCR-Lite `aksonocr_lite.rec.Recognizer`; OpenThai `LineOCR(dir, threads=1).recognize([crop])`, its crop
preprocessing included, OpenVINO). Scored with `bench/lines_score.py`. CPU only, EPYC 7763.

| Recognizer | exact % (doc / photo) | exact, ignoring spaces % | character error (NED×100) | ms / line, 1 thread (mean / median / p90) | lines / s, 16 × 1 thread |
|---|---|---:|---:|---|---:|
| AksonOCR-Lite (default, with 180° pass) | 88.1 (88.0 / 88.1) | 91.0 | 1.63 | 15.6 / 13.1 / 32.8 | 929 |
| AksonOCR-Lite, no 180° pass | 88.6 (88.0 / 89.1) | 91.6 | 1.15 | 8.1 / 6.7 / 16.6 | 1,800 |
| OpenThai2.0-OCR-Text | 38.1 (38.0 / 38.3) | 40.3 | 12.10 | 10.9 / 9.8 / 19.2 | 1,187 |

The 180° pass reads every line a second time upside down and keeps the more confident reading: it rescues
upside-down lines in full pages, costs twice the recognizer time, and on these upright crops changes a few photo
lines for the worse. The crops were cut by a DB-style detector with some padding; a recognizer trained on tighter
or looser crops may lose a little here. OpenThai's errors are mostly misread characters
(e.g. ลักทรัพย์ → ลัคทรัพย์, bullets read as ๑), not formatting. Reproduce:
`python bench/rec_bench.py --out-dir runs/rec --json bench/SPEED.json`.

Page level, for reference: OpenThai2.0-OCR-Text's full pipeline (line detection + recognition + reading order, plain
text, no layout or tables) in the same CPU setups as above takes 0.33 s per page (OpenVINO; 0.40 s ONNX Runtime) and
reads 10.2 pages/s with 16 processes (9.9 ONNX Runtime). It does much less work per page than AksonOCR-Lite, whose
own detection + recognition stage is 0.81 s of its 1.53 s.

Reproduce:

```bash
python bench/speed_bench.py --set mdp --out bench/SPEED.json             # AksonOCR-Lite on one set: latency, gpu, throughput
python bench/vlm_speed.py --harness <evaluation_thai_ocr> --model-key typhoon-ocr2-2b --upscale 1600 --sets mdp thai sea --out bench/SPEED.json
# (typhoon-ocr2-0.8b: --upscale 1600; openthai2.0-ocr-markdown: --upscale 0)
```

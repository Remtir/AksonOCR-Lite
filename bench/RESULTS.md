# Thai OCR comparison

Every system below read the same pages and was scored by the same official scorers. Scripts: `bench/`.

## How each system was run

- **VLM parsers** (Typhoon OCR V2 2B / 0.8B, Typhoon OCR 1.5 2B, OpenThai2.0-OCR-Markdown, NVIDIA Nemotron 3.5 Parse,
  dots.mocr, Chandra OCR 2): each model's own recipe (prompt, image preprocessing, decoding, token limit, output to
  Markdown) imported unchanged from the public Typhoon OCR V2 evaluation harness
  ([srn-source/evaluation_thai_ocr](https://github.com/srn-source/evaluation_thai_ocr), commit `e2512e3`), which
  follows each model card. Served with vLLM 0.33.0 (`vllm serve`, CUDA graphs, 6 API-server processes, up to 256
  requests in flight) on one RTX 5090. No speculative decoding. Typhoon OCR V2 enlarges images whose long side is
  under 1,600 px, as its card requires. No dewarping or page-orientation step was added to any VLM.
  `bench/vlm_serve_bench.py`.
- **Surya OCR 2.5**: its own package (`surya-ocr` 0.24, full-page mode) against `vllm serve` started with the
  arguments the package uses for its own server; blocks converted to Markdown with Datalab's converter
  (`chandra.output.parse_markdown`). `bench/surya_bench.py`.
- **Indra OCR**, **OpenThai2.0-OCR-Text**: their packages' defaults on CPU (`bench/cpu_ocr_bench.py`).
- **AksonOCR-Lite**: default settings (page rotation on, no dewarp), CPU. The `--dewarp` row is the same pipeline
  with MonkeyOCRv2 page flattening first; for this run the dewarp nets ran as one GPU pass over all inputs (saved
  losslessly, identical to `load(dewarp=True)`), the rest on CPU.
- **Gemini 4 Hydrogen** and **Claude Haiku 5.5** (OpenRouter, reasoning effort low): a generic Markdown prompt;
  ThaiOCRBench with its own item questions (`bench/api_vlm_bench.py`).

## Benchmarks

| Benchmark | Items | Score | Scorer |
|---|---:|---|---|
| MDPBench, Thai | 160 pages (40 digital, 120 photographed) | Overall ↑ | official `pdf_validation.py` + `calculate_scores.py` |
| ThaiOCRBench, Document parsing | 211 | STEDS-based score ↑ | official `eval_scripts/eval.py` |
| SEA-DocBench, Thai | 1,506 pages | text NED ↓ (page average), table TEDS ↑ | OmniDocBench end-to-end evaluator (as in SEA-Vision); `bench/seadocbench.py` |
| pv_hi (internal) | 469 printed text-line crops (184 documents, 285 photos) | exact match % ↑ | `bench/lines_score.py` |

Notes:
- **ThaiOCRBench, two columns.** "question" = the benchmark's own protocol (each item's instruction as the prompt), for
  models that accept a prompt. "page parse" = the model's page-parsing mode, the only mode for systems without a
  prompt. The scorer compares document structure (headings, lists), so converters that write titles without `#`
  (Typhoon OCR V2's doc2json→Markdown) lose most of the score in page-parse mode even with the right text.
- **SEA-DocBench ground truth** was minimally normalised so the evaluator can read it (formula LaTeX copied to the
  field it reads, category `text` → `text_block`, the 4% of text items with no transcription marked ignore; tables
  carry HTML only, so a prediction written as a LaTeX table is compared with that HTML).
- **pv_hi** labels are lines on which 3+ of 4 independent readers agreed; one of the readers was an earlier version of
  the AksonOCR recognizer, so this set can favour AksonOCR-Lite. Page parsers read each line crop as a page, which is
  not what they are built for; "recognizer only" is AksonOCR-Lite's recognizer applied to the crop directly.
- Long generation tails: some VLM outputs repeat until the token limit (counted in `timing.json` per run); they are
  scored as returned.

## Results

| System | Params | Runs on | MDPBench-th ↑ | digital | photo | ThaiOCRBench DP ↑ (question) | (page parse) | SEA-DocBench-th text NED ↓ | table TEDS ↑ | pv_hi lines exact ↑ (doc / photo) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemini 4 Hydrogen (reasoning low) | ? | API | 86.5 | 91.3 | 84.9 | 0.576 | 0.372 | 0.0724 | 0.810 | 75.5 / 83.5 |
| **AksonOCR-Lite** `--dewarp` | 81M + dewarp nets | CPU (dewarp on GPU here), ONNX | 85.6 | 89.5 | 84.3 | – | 0.641 | 0.0595 | 0.757 | 73.9 / 64.6 |
| Typhoon OCR V2 2B | 2B | vLLM | 84.3 | 85.9 | 83.8 | 0.394 | 0.224 | 0.0626 | 0.888 | 43.5 / 44.9 |
| **AksonOCR-Lite** (this repo) | 81M | CPU, ONNX | 79.5 | 89.5 | 76.2 | – | 0.640 | 0.0580 | 0.749 | 73.9 / 64.2 (recognizer only: 88.0 / 88.1) |
| Typhoon OCR V2 0.8B | 0.8B | vLLM | 78.7 | 83.4 | 77.2 | 0.453 | 0.217 | 0.0725 | 0.860 | 33.1 / 35.4 |
| NVIDIA Nemotron 3.5 Parse | 0.63B | vLLM | 76.5 | 89.3 | 72.2 | – | 0.434 | 0.0494 | 0.865 | 27.2 / 24.6 |
| Surya OCR 2.5 | 0.69B | vLLM | 75.6 | 85.3 | 72.4 | – | 0.185 | 0.1017 | 0.876 | 17.9 / 14.7 |
| Chandra OCR 2 | 5B | vLLM | 75.5 | 87.1 | 71.7 | 0.353 | 0.465 | 0.0802 | 0.892 | 12.5 / 21.8 |
| Typhoon OCR 1.5 2B | 2B | vLLM | 72.9 | 78.0 | 71.2 | 0.252 | 0.292 | 0.0850 | 0.840 | 57.1 / 77.2 |
| Claude Haiku 5.5 (reasoning low) | ? | API | 70.4 | 89.2 | 64.2 | 0.502 | 0.301 | 0.0684 | 0.818 | 50.0 / 41.4 |
| OpenThai2.0-OCR-Markdown | 0.63B | vLLM | 70.3 | 83.2 | 66.0 | – | 0.335 | 0.1035 | 0.852 | 32.6 / 30.5 |
| dots.mocr | 3B | vLLM | 70.3 | 82.3 | 66.3 | 0.257 | 0.219 | 0.0893 | 0.858 | 23.4 / 44.6 |
| OpenThai2.0-OCR-Text | 1M | CPU, OpenVINO | 58.7 | 72.1 | 54.2 | – | 0.417 | 0.1178 | 0.000 | 32.6 / 59.6 |
| Indra OCR | ~1M | CPU, ONNX | 36.1 | 75.5 | 22.9 | – | 0.285 | 0.1176 | 0.520 | 4.9 / 14.7 |

Generated 2026-11-17 03:26.

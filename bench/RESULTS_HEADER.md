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

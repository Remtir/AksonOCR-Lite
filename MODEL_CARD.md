---
license: apache-2.0
language:
- th
- en
library_name: onnx
pipeline_tag: image-to-text
tags:
- ocr
- thai
- document-parsing
- text-recognition
---

# AksonOCR-Lite models

Weights used by the [AksonOCR-Lite](https://github.com/Remtir/AksonOCR-Lite) package
(`pip install git+https://github.com/Remtir/AksonOCR-Lite`). Everything here runs on ONNX Runtime; nothing needs PyTorch or Paddle.

| File | What it is | Origin | License |
|---|---|---|---|
| `rec.onnx`, `rec_charset.json` | Thai + English text-line recognizer (ours) | fine-tuned from PaddlePaddle `PP-OCRv6_tiny_rec` | Apache-2.0 |
| `tatr.onnx` | table structure (rows, columns, headers, spans) | `microsoft/table-transformer-structure-recognition-v1.1-all`, converted | MIT |
| `dewarp_mask.onnx`, `dewarp_grid.onnx` | page segmentation + dewarping grid (optional) | MonkeyOCRv2 document preprocessor, converted | Apache-2.0 |
| `layout.onnx`, `det.onnx`, `doc_ori.onnx`, `textline_ori.onnx` | layout, text detection, page / line orientation | PaddlePaddle `PP-DocLayoutV3`, `PP-OCRv6_medium_det`, `PP-LCNet_x1_0_doc_ori` / `textline_ori` ONNX exports, unmodified copies | Apache-2.0 |

## Recognizer (`rec.onnx`)

- **Architecture:** PP-OCRv6 tiny (PPLCNetV4 backbone + SVTR neck) with a CTC head. About 1.5M parameters, 4.5 MB.
- **Output vocabulary:** 2,455 classes. Thai is predicted as *grapheme clusters* (a consonant together with its
  upper/lower vowels and tone marks is one class), so stacked marks cannot come out in the wrong order;
  `rec_charset.json` maps each class index straight to its output string.
- **Input:** BGR line crop resized to height 48 (width kept proportional, max 3200), scaled to [-1, 1].
- **Training data:** Thai and English text lines. The training data and recipe are not released.

### Line-level accuracy (exact match, %)

| Split | Lines | Exact | Notes |
|---|---:|---:|---|
| printed, documents (`pv_hi_doc`) | 184 | 87.5 | in-house, labels agreed by 3+ of 4 independent readers |
| printed, photos (`pv_hi_photo`) | 285 | 89.1 | same |
| SROIE receipts (test) | 1,500 | 48.9 | labels are ALL CAPS, images are mixed case, so case errors count as misses |

Measured on the ONNX export with the package's own batching (`Recognizer`, bucket 16, no 180° flip).

### Limitations

- Handwriting is weak; the model targets printed text.
- Common remaining confusions are Thai look-alikes: ต/ด, ค/ด, ่/้, ั/ิ, ส/ล, น/บ, ข/บ, ช/ซ.
- Vertical text and non-Thai/English scripts are not supported.

## Conversion

`tools/export_tatr.py` and `tools/export_dewarp.py` in the code repository reproduce the TATR and dewarp ONNX
files. Every export, the recognizer's included, was checked against the original framework on real inputs
(max abs difference on probabilities ~1e-4 for the recognizer, ~1e-5 for TATR; dewarped pixels within 1 grey
level on average).

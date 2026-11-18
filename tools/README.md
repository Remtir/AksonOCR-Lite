# Export tools (maintainers only)

These scripts produced the ONNX files published in `Remtir/AksonOCR-Lite`. They need the original
frameworks and are not installed with the package.

| Script | Needs | Output |
|---|---|---|
| `export_tatr.py` | torch, transformers, onnx | `tatr.onnx` |
| `export_dewarp.py` | torch, timm, onnx, onnxscript, MonkeyOCRv2's `modeling_preprocessor.py` | `dewarp_mask.onnx`, `dewarp_grid.onnx` |

Notes from the exports:

- **TATR:** the TorchScript exporter is fine; inputs are dynamic in height and width.
- **Dewarp:** the Swin warp net needs the dynamo exporter (`dynamo=True`); the TorchScript exporter hits an
  internal shape-inference assert.

Verify every export against the source framework before publishing (see the parity numbers in `MODEL_CARD.md`).

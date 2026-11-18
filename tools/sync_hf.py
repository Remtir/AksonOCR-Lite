"""Publish the GitHub README (and the pages and images it links to) as the Hugging Face model card.

    HF_TOKEN=... python tools/sync_hf.py [--repo Remtir/AksonOCR-Lite] [--dry-run]

The card is README.md with the Hub front matter on top, the CI badge dropped (it only works on GitHub), links to
code turned into GitHub URLs, and a "Files in this repository" section listing the weight files. BENCHMARKS.md,
MODEL_CARD.md, bench/RESULTS.md, bench/SPEED.md, LICENSE, NOTICE and every image they reference are uploaded
next to it, so relative links work on the Hub exactly as on GitHub. Weight files are not touched.
"""
import argparse, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GH = "https://github.com/Remtir/AksonOCR-Lite"
DOCS = ["BENCHMARKS.md", "MODEL_CARD.md", "bench/RESULTS.md", "bench/SPEED.md", "LICENSE", "NOTICE"]

FRONT = """---
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
- table-recognition
- layout-analysis
---

"""

FILES = """## Files in this repository

| File | What it is | Origin | License |
|---|---|---|---|
| `rec.onnx`, `rec_charset.json` | Thai + English text-line recognizer (ours, [model card](MODEL_CARD.md)) | fine-tuned from PaddlePaddle `PP-OCRv6_tiny_rec` | Apache-2.0 |
| `layout.onnx` | layout blocks + reading order | [PaddlePaddle/PP-DocLayoutV3_onnx](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_onnx), unmodified copy | Apache-2.0 |
| `det.onnx` | text-line detection | [PaddlePaddle/PP-OCRv6_medium_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_onnx), unmodified copy | Apache-2.0 |
| `doc_ori.onnx`, `textline_ori.onnx` | page / text-line orientation | [PP-LCNet_x1_0_doc_ori_onnx](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_doc_ori_onnx), [PP-LCNet_x1_0_textline_ori_onnx](https://huggingface.co/PaddlePaddle/PP-LCNet_x1_0_textline_ori_onnx), unmodified copies | Apache-2.0 |
| `tatr.onnx` | table structure (rows, columns, headers, spans) | `microsoft/table-transformer-structure-recognition-v1.1-all`, converted | MIT |
| `dewarp_mask.onnx`, `dewarp_grid.onnx` | page segmentation + dewarping grid (optional) | MonkeyOCRv2 document preprocessor, converted | Apache-2.0 |
| `config.json` | pipeline manifest (components, source revisions) | ours | Apache-2.0 |

The `aksonocr-lite` package downloads these on first use. The four PaddlePaddle files are byte-identical copies of
the official `inference.onnx` exports, kept here so the whole pipeline loads from one repository.

"""

LINK = re.compile(r'(\]\(|src=")([^)"#\s]+)([^)"]*)([)"])')


def fix_links(text: str, doc: str) -> str:
    base = (ROOT / doc).parent

    def sub(m):
        pre, path, frag, end = m.groups()
        if re.match(r"[a-z]+:", path):
            return m.group(0)
        rel = (base / path).resolve().relative_to(ROOT).as_posix()
        if rel in DOCS or rel == "README.md" or rel.startswith("assets/"):
            return m.group(0)  # uploaded alongside: relative link works on the Hub
        kind = "tree" if (ROOT / rel).is_dir() else "blob"
        return f"{pre}{GH}/{kind}/main/{rel}{frag}{end}"

    return LINK.sub(sub, text)


def images(text: str, doc: str):
    base = (ROOT / doc).parent
    for _, path, _, _ in LINK.findall(text):
        if not re.match(r"[a-z]+:", path):
            rel = (base / path).resolve().relative_to(ROOT).as_posix()
            if rel.startswith("assets/") and (ROOT / rel).is_file():
                yield rel


def card() -> str:
    md = (ROOT / "README.md").read_text(encoding="utf-8")
    # the big "live demo" button is a shields.io badge too: on the Hub it becomes a centred heading link instead
    md = re.sub(r"<p align=\"center\">\s*<a href=\"(https://aksonocr\.com/lite-demo)\"><img [^>]*></a>\s*</p>",
                r'<h2 align="center"><a href="\1">▶ Try the live demo</a></h2>\n\n'
                f'<p align="center"><b>Code, docs and issues: <a href="{GH}">github.com/Remtir/AksonOCR-Lite</a></b></p>', md)
    # GitHub alert syntax ("> [!TIP]") is not rendered by the Hub: keep the quote, drop the marker
    md = re.sub(r"^> \[!(?:NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\n", "", md, flags=re.M)
    # badge rows (shields.io / CI): the Hub renders them as one run-on line of alt text, so they stay GitHub-only
    md = re.sub(r"<p align=\"center\">(?:(?!</p>).)*img\.shields\.io(?:(?!</p>).)*</p>\n", "", md, flags=re.S)
    # rows of side-by-side images (width="32%"): the Hub stacks them in one column, so lay them out as a table row
    def row(m):
        imgs = re.findall(r"<img [^>]*>", m.group(0))
        if len(imgs) < 2:
            return m.group(0)
        cells = "".join(f'<td width="{100 // len(imgs)}%">{re.sub(r" width=\"[^\"]*\"", " width=\"100%\"", i)}</td>'
                        for i in imgs)
        return f"<table><tr>{cells}</tr></table>\n"

    md = re.sub(r"<p align=\"center\">\s*(?:<img [^>]*>\s*){2,}</p>\n", row, md)
    md = md.replace("\n## Development\n", "\n" + FILES + "## Development\n", 1)
    return FRONT + fix_links(md, "README.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="Remtir/AksonOCR-Lite")
    ap.add_argument("--dry-run", action="store_true", help="write the card to /dev/stdout and list the uploads")
    a = ap.parse_args()
    from huggingface_hub import CommitOperationAdd, HfApi

    readme = card()
    ops = [CommitOperationAdd("README.md", readme.encode())]
    assets = set(images((ROOT / "README.md").read_text(encoding="utf-8"), "README.md"))
    for doc in DOCS:
        text = (ROOT / doc).read_text(encoding="utf-8")
        if doc.endswith(".md"):
            assets |= set(images(text, doc))
            text = fix_links(text, doc)
        ops.append(CommitOperationAdd(doc, text.encode()))
    ops += [CommitOperationAdd(p, str(ROOT / p)) for p in sorted(assets)]
    if a.dry_run:
        print(readme)
        print("\n".join(o.path_in_repo for o in ops))
        return
    HfApi().create_commit(a.repo, ops, commit_message="Sync model card and docs with the GitHub README")
    print(f"uploaded {len(ops)} files to {a.repo}")


if __name__ == "__main__":
    main()

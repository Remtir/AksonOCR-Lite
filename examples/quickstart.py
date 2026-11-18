"""python examples/quickstart.py path/to/file.pdf"""
import sys

import aksonocr_lite

conv = aksonocr_lite.load()
for page in conv.convert(sys.argv[1]):
    print(f"--- page {page.index + 1} ({page.width}x{page.height}, rotated {page.rotation}) ---")
    print(page.markdown)
    tables = [b for b in page.blocks if b.html]
    print(f"[{len(page.blocks)} blocks, {len(page.lines)} lines, {len(tables)} tables, {page.timings}]")

"""Score line-level OCR on the internal printed-line set (pv_hi_doc, pv_hi_photo): exact match and 1 - NED.

    python bench/lines_score.py run_dir/lines.jsonl

Every system is scored with the same normalisation, applied to prediction and label alike:
  - Markdown / HTML markup removed (page parsers wrap a line in headings, tables, bold, ...)
  - Unicode NFC; Thai sara am written as U+0E33 (U+0E4D U+0E32 -> U+0E33)
  - all whitespace runs collapsed to one space, ends stripped
Exact match = identical strings after that; exact_nospace also ignores spaces (word spacing is a convention). Labels: 3+ of 4 independent readers agreed (see MODEL_CARD.md).
"""
import json, re, sys, unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import LINE_SPLITS as SPLITS  # noqa: E402  (set PV_HI_DIR to the directory holding the split files)
_TAG = re.compile(r"<[^>]+>")
_MD = re.compile(r"(^|\s)#{1,6}\s+|\*\*|__|`|^\s*[-*]\s+(?=\S)|\$+", re.M)


def norm(s: str) -> str:
    s = s or ""
    s = re.sub(r"```[a-z]*", "", s)
    s = _TAG.sub(" ", s)
    s = _MD.sub(" ", s)
    s = s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    s = unicodedata.normalize("NFC", s).replace("ํา", "ำ")
    return re.sub(r"\s+", " ", s).strip()


def main():
    preds = {}
    for l in open(sys.argv[1], encoding="utf-8"):
        try:
            x = json.loads(l)
            preds[x["key"]] = x.get("predict", "")
        except Exception:  # noqa: BLE001
            pass
    res = {}
    for split, f in SPLITS.items():
        em = emns = ned = n = miss = 0
        for k, line in enumerate(open(f, encoding="utf-8")):
            gt = norm(line.split("\t")[1])
            key = f"{split}/{k}"
            if key not in preds:
                miss += 1
            p = norm(preds.get(key, ""))
            em += p == gt
            emns += p.replace(" ", "") == gt.replace(" ", "")
            ned += 1 - Levenshtein.distance(p, gt) / max(len(p), len(gt), 1)
            n += 1
        res[split] = {"exact": round(100 * em / n, 2), "exact_nospace": round(100 * emns / n, 2), "1-NED": round(100 * ned / n, 2), "lines": n, "missing": miss}
    res["mean_exact"] = round(sum(v["exact"] for v in res.values()) / 2, 2)
    res["mean_exact_nospace"] = round(sum(v["exact_nospace"] for k, v in res.items() if k in SPLITS) / 2, 2)
    out = Path(sys.argv[1]).with_suffix(".eval.json")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()

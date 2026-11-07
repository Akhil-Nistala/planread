"""End-to-end evaluation on synthetic scanned sheets with oracle (PlanScope-format) regions.

Per category (table cell, header, notes, callout, rot90/270 callout, vertical callout, title
block): CER / exact / top-k recall, canonical-header accuracy, orientation accuracy, timing.
Also writes the worst failures as image crops to eval/results/failures/ for error analysis.

Usage::

    python -m eval.evaluate_docs --engine ppocr_v6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import TextScorer
from src.crnn import levenshtein
from src.crops import crop_bbox, rectify
from src.pipeline import OCRPipeline, load_config, load_page_images
from src.textnorm import normalize

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ORIENT = {"callout": "0", "callout_rot90": "90", "callout_rot270": "270", "callout_vertical": "vertical",
                   "notes": "0", "title_block": "0"}


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", type=Path, default=ROOT / "data" / "docs")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    ap.add_argument("--n-failures", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config()
    pipe = OCRPipeline(cfg, args.engine)
    sc = TextScorer()
    orient_ok, orient_n, hdr_ok, hdr_n = 0, 0, 0, 0
    failures = []
    secs, n_items = 0.0, 0
    for layout_path in sorted(args.docs.glob("*.layout.json")):
        stem = layout_path.name[: -len(".layout.json")]
        layout_doc = json.loads(layout_path.read_text(encoding="utf-8"))
        gt = json.loads((args.docs / f"{stem}.gt.json").read_text(encoding="utf-8"))
        imgs = load_page_images(args.docs / f"{stem}.png", layout_doc)
        t0 = time.perf_counter()
        out = pipe.process(layout_doc, imgs)
        secs += time.perf_counter() - t0
        page = out["pages"][0]
        for r in page["regions"]:
            g = gt["regions"][r["id"]]
            if r["table"] is not None:
                for i, row in enumerate(r["table"]["rows"]):
                    for j, c in enumerate(row["cells"]):
                        cat = "table_header" if c.get("is_header") else "table_cell"
                        alts = [a["text"] for ln in c.get("ocr_lines", []) for a in ln["alternatives"]]
                        sc.add(c["text"] or "", g["cells"][i][j], alts, group=cat)
                        n_items += 1
                        if normalize(c["text"] or "") != normalize(g["cells"][i][j]):
                            failures.append((stem, cat, c["bbox"], None, g["cells"][i][j], c["text"], c.get("ocr_conf", 0)))
                hdr_n += len(g["canonical_headers"])
                hdr_ok += sum(a == b for a, b in zip(r["table"]["canonical_headers"], g["canonical_headers"]))
                continue
            cat = g["category"]
            alts = [a["text"] for ln in r["ocr_lines"] for a in ln["alternatives"]] if len(r["ocr_lines"]) == 1 else []
            sc.add(r["text"] or "", g["text"], alts, group=cat)
            n_items += 1
            orient_n += 1
            orient_ok += r.get("ocr_orientation") == EXPECTED_ORIENT[cat]
            if normalize(r["text"] or "") != normalize(g["text"]):
                failures.append((stem, cat, None, r["polygon"], g["text"], r["text"], r.get("ocr_conf", 0),
                                 r.get("ocr_orientation"), r.get("ocr_flags")))
    res = {"engine": pipe.engine, **sc.summary(), "orientation_acc": orient_ok / max(1, orient_n),
           "canonical_header_acc": hdr_ok / max(1, hdr_n), "items": n_items, "sec_per_item": secs / max(1, n_items),
           "sec_total": secs}
    # error analysis gallery: largest normalised edit distance first
    failures.sort(key=lambda f: -levenshtein(normalize(f[5] or ""), normalize(f[4])) / max(1, len(normalize(f[4]))))
    fdir = args.out / f"failures_{pipe.engine}"
    fdir.mkdir(parents=True, exist_ok=True)
    gallery = []
    page_cache: dict[str, np.ndarray] = {}
    for k, f in enumerate(failures[: args.n_failures]):
        stem = f[0]
        if stem not in page_cache:
            page_cache[stem] = cv2.imread(str(args.docs / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        img = page_cache[stem]
        crop = crop_bbox(img, f[2], 4).image if f[2] is not None else rectify(img, f[3], 4).image
        name = f"{k:02d}_{f[1]}.png"
        cv2.imwrite(str(fdir / name), crop)
        gallery.append({"image": f"{fdir.name}/{name}", "category": f[1], "gt": f[4], "pred": f[5], "conf": f[6],
                        "orientation": f[7] if len(f) > 7 else None, "flags": f[8] if len(f) > 8 else None})
    res["failures"] = gallery
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"docs_{pipe.engine}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "failures"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

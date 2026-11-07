"""Text-line detection (DBNet) on synthetic multi-line blocks with lines crossing text.

Reports precision / recall / hmean at IoU >= 0.5, overall and by number of lines per block.

Usage::

    python -m eval.evaluate_det
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from eval.metrics import det_prf
from src.detect_lines import DBNetDetector

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "det" / "test")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "det.json")
    args = ap.parse_args()
    det = DBNetDetector()
    tot = [0, 0, 0]
    by_n: dict[str, list[int]] = {}
    for ln in (args.data / "label.txt").read_text(encoding="utf-8").splitlines():
        path, js = ln.split("\t")
        gts = [a["points"] for a in json.loads(js)]
        img = cv2.imread(str(args.data / path), cv2.IMREAD_GRAYSCALE)
        preds = [p.tolist() for p in det.detect(img)]
        tp, n_p, n_g = det_prf(preds, gts)
        for acc in (tot, by_n.setdefault("1 line" if len(gts) == 1 else "2-3 lines" if len(gts) <= 3 else "4-6 lines", [0, 0, 0])):
            acc[0] += tp
            acc[1] += n_p
            acc[2] += n_g

    def prf(t: list[int]) -> dict:
        p, r = t[0] / max(1, t[1]), t[0] / max(1, t[2])
        return {"precision": p, "recall": r, "hmean": 2 * p * r / max(1e-9, p + r), "n_gt": t[2]}

    res = {"overall": prf(tot), "by_lines": {k: prf(v) for k, v in sorted(by_n.items())}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""Grid-search DBNet post-processing (unclip ratio, thresholds) on a TRAIN subset (never on test).

Usage::

    python -m eval.tune_det --n 200
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2

from eval.metrics import det_prf
from src.detect_lines import DBNetDetector

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "det" / "train")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    rows = (args.data / "label.txt").read_text(encoding="utf-8").splitlines()[: args.n]
    samples = [(cv2.imread(str(args.data / r.split("\t")[0]), cv2.IMREAD_GRAYSCALE),
                [a["points"] for a in json.loads(r.split("\t")[1])]) for r in rows]
    det = DBNetDetector()
    maps = [(det.prob_map(img), img.shape, gts) for img, gts in samples]  # network runs once per image
    results = []
    for unclip, thresh, box_thresh in itertools.product([0.6, 0.8, 1.0, 1.2, 1.4, 1.6], [0.2, 0.3], [0.4, 0.5, 0.6]):
        det.unclip, det.thresh, det.box_thresh = unclip, thresh, box_thresh
        tp = n_p = n_g = 0
        for (prob, sy, sx), shape, gts in maps:
            a, b, c = det_prf([p.tolist() for p in det.postprocess(prob, sy, sx, shape)], gts)
            tp, n_p, n_g = tp + a, n_p + b, n_g + c
        p, r = tp / max(1, n_p), tp / max(1, n_g)
        results.append((2 * p * r / max(1e-9, p + r), unclip, thresh, box_thresh, p, r))
    for h, u, t, bt, p, r in sorted(results, reverse=True)[:5]:
        print(f"hmean={h:.3f} P={p:.3f} R={r:.3f}  unclip={u} thresh={t} box_thresh={bt}")


if __name__ == "__main__":
    main()

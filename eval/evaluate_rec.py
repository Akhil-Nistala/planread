"""Line-recognition evaluation + confidence calibration (fit on val, report on test).

For each engine: CER / exact / top-k recall on test_clean and test_hard, per-script CER,
per-text-source exact match, speed, out-of-charset (OOV) characters w.r.t. the model's
own charset, ECE before/after calibration. Saves models/calibrator_<engine>.json and
eval/results/rec_<engine>.json.

Usage::

    python -m eval.evaluate_rec --engines ppocr_v6,ppocr_japan_v4,crnn
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from eval.metrics import TextScorer
from src.calibration import Calibrator, ece, reliability
from src.charset import oov_report
from src.recognize import build_recognizer
from src.textnorm import fold_width, normalize

ROOT = Path(__file__).resolve().parents[1]


def load_split(d: Path, limit: int | None) -> tuple[list[np.ndarray], list[list[str]]]:
    """Images and label rows of a split."""
    rows = [ln.split("\t") for ln in (d / "labels.tsv").read_text(encoding="utf-8").splitlines() if ln][:limit]
    return [cv2.imread(str(d / r[0]), cv2.IMREAD_GRAYSCALE) for r in rows], rows


def run(engine: str, data: Path, limit: int | None, out_dir: Path) -> dict:
    """Evaluate one engine."""
    rec = build_recognizer(engine)
    model_chars = {fold_width(c) for c in rec.chars[1:]}
    res: dict = {"engine": engine}
    # calibration: fit on val
    imgs, rows = load_split(data / "val", limit)
    decs = rec.recognize(imgs)
    feats = [d.features for d in decs]
    correct = [normalize(d.text) == normalize(r[1]) for d, r in zip(decs, rows)]
    cal = Calibrator().fit(feats, correct)
    cal.save(ROOT / "models" / f"calibrator_{engine}.json")
    for split in ("test_clean", "test_hard"):
        imgs, rows = load_split(data / split, limit)
        t0 = time.perf_counter()
        decs = rec.recognize(imgs)
        dt = time.perf_counter() - t0
        sc = TextScorer()
        failures = []
        for d, r in zip(decs, rows):
            sc.add(d.text, r[1], [a for a, _ in d.candidates], group=r[2])
            if normalize(d.text) != normalize(r[1]):
                failures.append({"image": r[0], "gt": r[1], "pred": d.text, "alts": [a for a, _ in d.candidates[:3]],
                                 "source": r[2]})
        correct = [normalize(d.text) == normalize(r[1]) for d, r in zip(decs, rows)]
        raw = [min(1.0, d.raw_conf) for d in decs]
        calp = cal.predict([d.features for d in decs]).tolist()
        oov = oov_report((normalize(r[1]) for r in rows), model_chars)
        res[split] = {**sc.summary(), "lines_per_s": len(imgs) / dt,
                      "ece_raw": ece(raw, correct), "ece_calibrated": ece(calp, correct),
                      "reliability_calibrated": reliability(calp, correct),
                      "oov_chars": dict(oov.most_common(15)),
                      "lines_with_oov": sum(any(c in oov for c in normalize(r[1])) for r in rows),
                      "failures_sample": failures[:40]}
        print(f"{engine:15s} {split:10s} CER={res[split]['cer']:.4f} exact={res[split]['exact']:.3f} "
              f"top5={res[split]['topk_recall']:.3f} ECE raw={res[split]['ece_raw']:.3f} -> "
              f"cal={res[split]['ece_calibrated']:.3f}  {res[split]['lines_per_s']:.1f} lines/s", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"rec_{engine}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines", default="ppocr_v6,ppocr_japan_v4")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "rec")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    args = ap.parse_args()
    for e in args.engines.split(","):
        run(e, args.data, args.limit, args.out)


if __name__ == "__main__":
    main()

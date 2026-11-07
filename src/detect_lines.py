"""Text-line detection inside crops: DBNet (PP-OCRv6 det, ONNX).

Post-processing follows DBNet (threshold -> contours -> box score -> unclip) but the unclip
is done on the min-area rectangle (offset = area * ratio / perimeter), so no pyclipper needed.
Returned polygons are 4 points, clockwise from the top-left, with the long side horizontal.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DET_MODEL = ROOT / "models" / "ppocr" / "det_v6_small.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def order_box(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL (by sums/diffs), long side horizontal."""
    pts = np.asarray(pts, np.float64)
    s, d = pts.sum(1), np.diff(pts, axis=1)[:, 0]
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


class DBNetDetector:
    """DBNet text detector."""

    # thresholds tuned on a det/train subset with eval/tune_det.py (PP-OCR defaults: unclip 1.5, box 0.6)
    def __init__(self, path: Path = DET_MODEL, limit_side: int = 1280, thresh: float = 0.3,
                 box_thresh: float = 0.6, unclip: float = 0.6, min_size: int = 3) -> None:
        import onnxruntime as ort

        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python scripts/download_models.py`")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.limit, self.thresh, self.box_thresh, self.unclip, self.min_size = limit_side, thresh, box_thresh, unclip, min_size

    def _resize(self, img: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = img.shape[:2]
        s = min(1.0, self.limit / max(h, w))
        if max(h, w) * s < 64:            # tiny crops: upsample so DB has enough pixels
            s = 64 / max(h, w)
        nh, nw = max(32, int(round(h * s / 32)) * 32), max(32, int(round(w * s / 32)) * 32)
        return cv2.resize(img, (nw, nh)), nh / h, nw / w

    PAD = 16  # DB struggles with text touching the border

    def prob_map(self, gray: np.ndarray) -> tuple[np.ndarray, float, float]:
        """DB probability map of the padded, resized crop and the resize factors."""
        g = cv2.copyMakeBorder(gray, self.PAD, self.PAD, self.PAD, self.PAD, cv2.BORDER_CONSTANT, value=255)
        img, sy, sx = self._resize(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
        x = ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        return self.sess.run(None, {self.inp: x})[0][0, 0], sy, sx

    def detect(self, gray: np.ndarray) -> list[np.ndarray]:
        """Line polygons (4x2, crop pixel coords)."""
        return self.postprocess(*self.prob_map(gray), gray.shape)

    def postprocess(self, prob: np.ndarray, sy: float, sx: float, shape: tuple[int, ...]) -> list[np.ndarray]:
        """Threshold -> contours -> scored, unclipped boxes in crop coordinates."""
        pad = self.PAD
        bitmap = (prob > self.thresh).astype(np.uint8)
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            if len(c) < 4:
                continue
            rect = cv2.minAreaRect(c)
            if min(rect[1]) < self.min_size:
                continue
            mask = np.zeros_like(bitmap)
            cv2.fillPoly(mask, [c], 1)
            if prob[mask.astype(bool)].mean() < self.box_thresh:
                continue
            (cx, cy), (w, h), ang = rect
            dist = (w * h) * self.unclip / max(1e-6, 2 * (w + h))
            box = cv2.boxPoints(((cx, cy), (w + 2 * dist, h + 2 * dist), ang))
            box[:, 0] = box[:, 0] / sx - pad
            box[:, 1] = box[:, 1] / sy - pad
            box[:, 0] = np.clip(box[:, 0], 0, shape[1] - 1)
            box[:, 1] = np.clip(box[:, 1], 0, shape[0] - 1)
            # crops are already orientation-corrected, so keep the upright TL,TR,BR,BL order
            out.append(order_box(box))
        return out

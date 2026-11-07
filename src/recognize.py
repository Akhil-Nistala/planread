"""Line recognisers behind one interface.

* ``ppocr_v6``       -- PP-OCRv6 small, multilingual (ja/en/zh), pretrained, ONNX (default)
* ``ppocr_japan_v4`` -- PP-OCRv4 mobile Japanese, pretrained, ONNX
* ``crnn``           -- CRNN trained in this repo on synthetic drawing text (src/crnn.py)

All return CTC probabilities that go through the same decoder (:mod:`src.ctc`), so top-k
alternatives and calibration features are computed identically for every engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from .ctc import Decoded, decode

ROOT = Path(__file__).resolve().parents[1]
PPOCR_MODELS = {
    "ppocr_v6": ROOT / "models" / "ppocr" / "rec_v6_small.onnx",
    "ppocr_japan_v4": ROOT / "models" / "ppocr" / "rec_japan_v4_mobile.onnx",
}


class Recognizer(Protocol):
    """Recognises a batch of single-line grayscale crops."""

    name: str
    chars: list[str]

    def recognize(self, crops: Sequence[np.ndarray]) -> list[Decoded]:
        """One :class:`Decoded` per crop, in input order."""
        ...


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def trim(gray: np.ndarray, ink_thresh: int = 170, margin: int = 3) -> np.ndarray:
    """Trim white margins (keeps a few px)."""
    ys, xs = np.where(gray < ink_thresh)
    if len(xs) == 0:
        return gray
    y0, y1 = max(0, ys.min() - margin), min(gray.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(gray.shape[1], xs.max() + margin + 1)
    return gray[y0:y1, x0:x1]


class PPOCRRecognizer:
    """PP-OCR CTC recogniser via onnxruntime (charset read from the model metadata)."""

    def __init__(self, name: str, beam: int = 10, top_k: int = 5, max_w: int = 1280, batch: int = 16) -> None:
        import onnxruntime as ort

        path = PPOCR_MODELS[name]
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run `python scripts/download_models.py`")
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map["character"].splitlines()
        self.name, self.chars = name, ["<blank>"] + meta + [" "]
        self.inp = self.sess.get_inputs()[0].name
        self.h, self.max_w, self.beam, self.top_k, self.batch = 48, max_w, beam, top_k, batch

    def _prep(self, gray: np.ndarray, w: int) -> np.ndarray:
        H, W = gray.shape
        nw = int(min(w, max(8, np.ceil(self.h * W / max(H, 1)))))
        img = cv2.resize(gray, (nw, self.h), interpolation=cv2.INTER_AREA if H > self.h else cv2.INTER_CUBIC)
        x = np.zeros((3, self.h, w), np.float32)  # PP-OCR pads with 0 in normalised space
        x[:, :, :nw] = (img.astype(np.float32) / 255.0 - 0.5) / 0.5
        return x

    def probs(self, crops: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Per-crop T x C probability matrices (padding time-steps removed)."""
        crops = [trim(c) for c in crops]
        ratios = [c.shape[1] / max(1, c.shape[0]) for c in crops]
        order = np.argsort(ratios)
        out: list[np.ndarray | None] = [None] * len(crops)
        for s in range(0, len(crops), self.batch):
            idx = order[s:s + self.batch]
            w = int(min(self.max_w, max(np.ceil(self.h * ratios[i]) for i in idx)))
            w = max(32, int(np.ceil(w / 8) * 8))
            x = np.stack([self._prep(crops[i], w) for i in idx])
            y = self.sess.run(None, {self.inp: x})[0]
            if not np.allclose(y[0].sum(-1)[:3], 1.0, atol=1e-3):
                y = _softmax(y)
            for j, i in enumerate(idx):
                valid = int(np.ceil(min(w, self.h * ratios[i]) / w * y.shape[1]))
                out[i] = y[j, :max(1, valid)]
        return out  # type: ignore[return-value]

    def recognize(self, crops: Sequence[np.ndarray]) -> list[Decoded]:
        """Decode a batch of crops."""
        return [decode(p, self.chars, self.beam, self.top_k) for p in self.probs(crops)] if crops else []


class CRNNRecognizer:
    """In-repo CRNN (PyTorch)."""

    def __init__(self, path: str | Path, beam: int = 10, top_k: int = 5) -> None:
        import torch

        from .crnn import CRNN

        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.chars = ck["classes"]
        self.model = CRNN(len(self.chars))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.name, self.beam, self.top_k = "crnn", beam, top_k

    def recognize(self, crops: Sequence[np.ndarray]) -> list[Decoded]:
        """Decode crops one by one (variable width, no padding artefacts)."""
        import torch

        from .crnn import preprocess

        out = []
        with torch.no_grad():
            for c in crops:
                x = torch.from_numpy(preprocess(trim(c)))[None, None]
                p = torch.softmax(self.model(x), -1)[0].numpy()
                out.append(decode(p, self.chars, self.beam, self.top_k))
        return out


def build_recognizer(engine: str, beam: int = 10, top_k: int = 5, crnn_model: str | Path | None = None) -> Recognizer:
    """Factory keyed by ``configs/ocr.yaml::recognition.engine``."""
    if engine in PPOCR_MODELS:
        return PPOCRRecognizer(engine, beam, top_k)
    if engine == "crnn":
        return CRNNRecognizer(crnn_model or ROOT / "models" / "crnn.pt", beam, top_k)
    raise ValueError(f"unknown recognition engine {engine!r}")

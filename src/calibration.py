"""Confidence calibration: P(line read exactly right | decoding features).

A small L2-regularised logistic regression (fitted with Newton/IRLS in numpy) over features
from :func:`src.ctc.decode`. Fitted on the synthetic *validation* split, evaluated on test
with Expected Calibration Error (ECE). Identity fallback (= raw score) when no model is fitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

FEATURES = ["mean_char_prob", "min_char_prob", "beam_top1", "beam_margin", "seq_logp_per_t", "length"]


def _design(feats: Sequence[dict[str, float]]) -> np.ndarray:
    X = np.array([[f[k] for k in FEATURES] for f in feats], np.float64)
    X[:, FEATURES.index("length")] = np.log1p(X[:, FEATURES.index("length")])
    eps = 1e-4
    for k in ("mean_char_prob", "min_char_prob", "beam_top1"):  # logit of probability-like features
        i = FEATURES.index(k)
        p = np.clip(X[:, i], eps, 1 - eps)
        X[:, i] = np.log(p / (1 - p))
    return np.hstack([X, np.ones((len(X), 1))])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class Calibrator:
    """Logistic calibrator; ``w is None`` means identity (returns the raw mean char prob)."""

    w: list[float] | None = None
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    def fit(self, feats: Sequence[dict[str, float]], correct: Sequence[bool], l2: float = 1.0,
            iters: int = 50) -> "Calibrator":
        """Fit with IRLS; standardises features (except the bias column)."""
        X = _design(feats)
        mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-9
        X[:, :-1] = (X[:, :-1] - mu) / sd
        y = np.asarray(correct, np.float64)
        w = np.zeros(X.shape[1])
        R = l2 * np.eye(X.shape[1])
        R[-1, -1] = 0.0
        for _ in range(iters):
            p = _sigmoid(X @ w)
            g = X.T @ (p - y) + R @ w
            H = (X * (p * (1 - p))[:, None]).T @ X + R + 1e-9 * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-7:
                break
        self.w, self.mean, self.std = w.tolist(), mu.tolist(), sd.tolist()
        return self

    def predict(self, feats: Sequence[dict[str, float]]) -> np.ndarray:
        """Calibrated probabilities."""
        if self.w is None:
            return np.array([f["mean_char_prob"] for f in feats])
        X = _design(feats)
        X[:, :-1] = (X[:, :-1] - np.array(self.mean)) / np.array(self.std)
        return _sigmoid(X @ np.array(self.w))

    def save(self, path: str | Path) -> None:
        """Write JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"features": FEATURES, "w": self.w, "mean": self.mean, "std": self.std}))

    @classmethod
    def load(cls, path: str | Path | None) -> "Calibrator":
        """Load JSON, or identity calibrator if the file does not exist."""
        if not path or not Path(path).exists():
            return cls()
        d = json.loads(Path(path).read_text())
        if d["features"] != FEATURES:
            raise ValueError("calibrator was fitted on a different feature set")
        return cls(d["w"], d["mean"], d["std"])


def ece(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum(abs(c[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def reliability(conf: Sequence[float], correct: Sequence[bool], bins: int = 10) -> list[dict[str, float]]:
    """Per-bin (mean confidence, accuracy, count) for reliability diagrams."""
    c, y = np.asarray(conf, float), np.asarray(correct, float)
    idx = np.minimum((c * bins).astype(int), bins - 1)
    return [{"bin": b / bins, "conf": float(c[idx == b].mean()), "acc": float(y[idx == b].mean()),
             "n": int((idx == b).sum())} for b in range(bins) if (idx == b).any()]

"""Shared OCR metrics."""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import cv2
import numpy as np

from src.textnorm import normalize


def script_of(ch: str) -> str:
    """Coarse script class of a character."""
    o = ord(ch)
    if 0x4E00 <= o <= 0x9FFF or ch in "々〆〇":
        return "kanji"
    if 0x3040 <= o <= 0x30FF:
        return "kana"
    if ch.isascii() and ch.isalnum():
        return "latin_digit"
    if ch.isspace():
        return "space"
    return "symbol"


def align_errors(pred: str, gt: str) -> tuple[int, Counter]:
    """Edit distance and errors attributed to GT-char script (subs + deletions; insertions -> 'insert')."""
    n, m = len(gt), len(pred)
    D = np.zeros((n + 1, m + 1), np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]))
    errs: Counter = Counter()
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]):
            if gt[i - 1] != pred[j - 1]:
                errs[script_of(gt[i - 1])] += 1
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            errs[script_of(gt[i - 1])] += 1
            i -= 1
        else:
            errs["insert"] += 1
            j -= 1
    return int(D[n, m]), errs


class TextScorer:
    """Accumulates CER / exact match / top-k recall, overall and per group and per script."""

    def __init__(self) -> None:
        self.n = 0
        self.edits = 0
        self.chars = 0
        self.exact = 0
        self.topk = 0
        self.script_err: Counter = Counter()
        self.script_tot: Counter = Counter()
        self.groups: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # n, edits, chars, exact

    def add(self, pred: str, gt: str, alts: Iterable[str] = (), group: str | None = None) -> None:
        """Score one prediction (both sides normalised)."""
        p, g = normalize(pred), normalize(gt)
        d, errs = align_errors(p, g)
        self.n += 1
        self.edits += d
        self.chars += max(1, len(g))
        self.exact += p == g
        self.topk += p == g or any(normalize(a) == g for a in alts)
        self.script_err.update(errs)
        self.script_tot.update(script_of(c) for c in g)
        if group:
            s = self.groups[group]
            s[0] += 1
            s[1] += d
            s[2] += max(1, len(g))
            s[3] += p == g

    def summary(self) -> dict:
        """Metrics dict."""
        return {
            "n": self.n, "cer": self.edits / max(1, self.chars), "exact": self.exact / max(1, self.n),
            "topk_recall": self.topk / max(1, self.n),
            "cer_by_script": {k: self.script_err[k] / v for k, v in sorted(self.script_tot.items()) if k != "space"},
            "insertions_per_line": self.script_err["insert"] / max(1, self.n),
            "by_group": {k: {"n": v[0], "cer": v[1] / max(1, v[2]), "exact": v[3] / max(1, v[0])}
                         for k, v in sorted(self.groups.items())},
        }


def poly_iou(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """IoU of two convex quadrilaterals."""
    pa, pb = np.asarray(a, np.float32), np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    ua = cv2.contourArea(pa) + cv2.contourArea(pb) - inter
    return float(inter / ua) if ua > 0 else 0.0


def det_prf(preds: list[list], gts: list[list], thr: float = 0.5) -> tuple[int, int, int]:
    """Greedy one-to-one matching at IoU >= thr -> (tp, n_pred, n_gt)."""
    used = set()
    tp = 0
    for p in preds:
        best, bi = 0.0, -1
        for i, g in enumerate(gts):
            if i in used:
                continue
            iou = poly_iou(p, g)
            if iou > best:
                best, bi = iou, i
        if best >= thr:
            used.add(bi)
            tp += 1
    return tp, len(preds), len(gts)


def char_category_counts(texts: Iterable[str]) -> Counter:
    """Script histogram (for dataset statistics)."""
    return Counter(script_of(c) for t in texts for c in normalize(t) if not c.isspace())


def is_cjk(ch: str) -> bool:
    """CJK ideograph or kana."""
    return script_of(ch) in ("kanji", "kana") or unicodedata.east_asian_width(ch) in "WF"

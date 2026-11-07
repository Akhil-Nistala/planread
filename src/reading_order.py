"""Reading order for text-line boxes inside an (already rectified, upright) crop."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _yspan(poly: np.ndarray) -> tuple[float, float]:
    return float(poly[:, 1].min()), float(poly[:, 1].max())


def sort_lines(polys: Sequence[np.ndarray], y_overlap: float = 0.5) -> list[list[int]]:
    """Group boxes into visual rows, rows top-to-bottom, boxes left-to-right.

    Two boxes are in the same row if their vertical overlap is at least ``y_overlap`` of the
    shorter box's height. Returns rows as lists of indices into ``polys``.
    """
    order = sorted(range(len(polys)), key=lambda i: (_yspan(polys[i])[0] + _yspan(polys[i])[1]) / 2)
    rows: list[list[int]] = []
    spans: list[tuple[float, float]] = []
    for i in order:
        y0, y1 = _yspan(polys[i])
        placed = False
        for r, (ry0, ry1) in enumerate(spans):
            inter = min(y1, ry1) - max(y0, ry0)
            if inter >= y_overlap * min(y1 - y0, ry1 - ry0):
                rows[r].append(i)
                spans[r] = (min(ry0, y0), max(ry1, y1))
                placed = True
                break
        if not placed:
            rows.append([i])
            spans.append((y0, y1))
    rows_sorted = sorted(zip(spans, rows), key=lambda sr: sr[0][0])
    return [sorted(r, key=lambda i: float(polys[i][:, 0].min())) for _, r in rows_sorted]

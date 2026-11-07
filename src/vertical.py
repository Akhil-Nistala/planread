"""縦書き (vertical Japanese) support: split a vertical crop into glyph cells and re-flow them
horizontally so a horizontal line recogniser can read it.

Approach: Japanese glyphs are roughly square, so a vertical column is cut at the gaps of the
row-ink profile; long ink runs (touching glyphs) are split at multiples of the column width.

Known limitations (documented in README): multi-column vertical blocks must be split into
columns first (we only split by large vertical gaps between columns); vertical-form
punctuation (︵ ︶ etc.) and rotated long-vowel marks (｜ for ー) are not re-shaped.
"""

from __future__ import annotations

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def split_columns(gray: np.ndarray, ink_thresh: int = 160, min_gap_frac: float = 0.25) -> list[tuple[int, int]]:
    """Columns of a (possibly multi-column) vertical block, right-to-left (Japanese reading order)."""
    ink = (gray < ink_thresh).sum(0) > 0
    runs = _runs(ink)
    if not runs:
        return []
    cols = [list(runs[0])]
    for s, e in runs[1:]:
        gap = s - cols[-1][1]
        if gap < min_gap_frac * max(1, cols[-1][1] - cols[-1][0]):
            cols[-1][1] = e
        else:
            cols.append([s, e])
    return [tuple(c) for c in reversed(cols)]


def split_glyphs(column: np.ndarray, ink_thresh: int = 160) -> list[tuple[int, int]]:
    """Row ranges of glyphs in a single vertical column."""
    h, w = column.shape
    rows = (column < ink_thresh).sum(1) > 0
    glyph = max(4, w)  # glyph height ~ column width
    out: list[tuple[int, int]] = []
    for s, e in _runs(rows):
        if e - s < 0.15 * glyph and out and s - out[-1][1] < 0.3 * glyph:
            out[-1] = (out[-1][0], e)  # tiny fragment (e.g. dakuten, part of a radical) -> merge
            continue
        n = max(1, int(round((e - s) / glyph)))
        if n == 1:
            out.append((s, e))
        else:  # touching glyphs: split evenly
            edges = np.linspace(s, e, n + 1).round().astype(int)
            out.extend(zip(edges[:-1], edges[1:]))
    # merge consecutive fragments that are much shorter than a glyph (split radicals like 旦, 二)
    merged: list[tuple[int, int]] = []
    for s, e in out:
        if merged and (e - merged[-1][0]) <= 1.05 * glyph and (merged[-1][1] - merged[-1][0]) < 0.6 * glyph:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def reflow_vertical(gray: np.ndarray, ink_thresh: int = 160) -> tuple[np.ndarray, int]:
    """Return a horizontal strip of the glyphs (all columns, reading order) and the glyph count."""
    pieces: list[np.ndarray] = []
    for c0, c1 in split_columns(gray, ink_thresh) or [(0, gray.shape[1])]:
        col = gray[:, c0:c1]
        w = col.shape[1]
        for s, e in split_glyphs(col, ink_thresh):
            g = col[s:e]
            side = max(w, e - s)
            cell = np.full((side, side), 255, np.uint8)
            y, x = (side - (e - s)) // 2, (side - w) // 2
            cell[y:y + e - s, x:x + w] = g
            pieces.append(cell)
            pieces.append(np.full((side, max(1, side // 8)), 255, np.uint8))  # small inter-glyph gap
    if not pieces:
        return gray, 0
    H = max(p.shape[0] for p in pieces)
    strip = np.hstack([np.pad(p, ((0, H - p.shape[0]), (0, 0)), constant_values=255) for p in pieces])
    pad = max(2, H // 6)
    return np.pad(strip, pad, constant_values=255), len(pieces) // 2

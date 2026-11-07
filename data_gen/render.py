"""Text-line rendering (horizontal + 縦書き vertical) and drawing-style augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class FontFace:
    """One face of a font file (TTC files hold several)."""

    path: str
    index: int
    cmap: frozenset[int]

    def covers(self, text: str) -> bool:
        """True if every non-space character has a glyph."""
        return all(ord(c) in self.cmap for c in text if not c.isspace())

    def pil(self, size: int) -> ImageFont.FreeTypeFont:
        """PIL font object at ``size`` px."""
        return _pil_font(self.path, self.index, size)


@lru_cache(maxsize=256)
def _pil_font(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=index)


def load_faces(paths: list[str]) -> list[FontFace]:
    """Load all faces of the existing font files in ``paths``."""
    faces: list[FontFace] = []
    for p in paths:
        if not Path(p).exists():
            continue
        if p.lower().endswith(".ttc"):
            coll = TTCollection(p, lazy=True)
            fonts = list(coll.fonts)
        else:
            fonts = [TTFont(p, lazy=True)]
        for i, f in enumerate(fonts):
            faces.append(FontFace(p, i, frozenset(f.getBestCmap().keys())))
    return faces


class FontPool:
    """Chooses a face that covers a string (Latin faces only for pure ASCII)."""

    def __init__(self, ja_paths: list[str], latin_paths: list[str]) -> None:
        self.ja = load_faces(ja_paths)
        self.latin = load_faces(latin_paths)
        if not self.ja:
            raise RuntimeError("No Japanese font found; install Noto Sans CJK or edit configs/synth.yaml")

    def pick(self, text: str, rng: np.random.Generator) -> FontFace | None:
        """Random face covering ``text`` or None."""
        pool = list(self.ja)
        if text.isascii() and self.latin:
            pool += self.latin * 2
        rng.shuffle(pool)
        return next((f for f in pool if f.covers(text)), None)


def render_line(text: str, face: FontFace, size: int, rng: np.random.Generator) -> np.ndarray:
    """Render a horizontal line: uint8 grayscale, dark ink on white, with random padding."""
    font = face.pil(size)
    x0, y0, x1, y1 = font.getbbox(text)
    pad = [int(rng.integers(2, max(3, size // 3))) for _ in range(4)]
    w, h = max(1, x1 - x0) + pad[0] + pad[2], max(1, y1 - y0) + pad[1] + pad[3]
    img = Image.new("L", (w, h), 255)
    ink = int(rng.integers(0, 70))
    ImageDraw.Draw(img).text((pad[0] - x0, pad[1] - y0), text, font=font, fill=ink)
    return np.asarray(img).copy()


def render_vertical(text: str, face: FontFace, size: int, rng: np.random.Generator) -> np.ndarray:
    """Render 縦書き: glyphs upright, stacked top-to-bottom (punctuation is not re-shaped)."""
    font = face.pil(size)
    gap = int(size * rng.uniform(0.05, 0.25))
    cells = [c for c in text if not c.isspace()] or ["ア"]
    W = int(size * 1.3)
    H = len(cells) * (size + gap) + gap
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    ink = int(rng.integers(0, 70))
    for i, c in enumerate(cells):
        bx0, by0, bx1, by1 = font.getbbox(c)
        cx = (W - (bx1 - bx0)) / 2 - bx0
        cy = gap + i * (size + gap) + (size - (by1 - by0)) / 2 - by0
        d.text((cx, cy), c, font=font, fill=ink)
    return np.asarray(img).copy()


# --------------------------------------------------------------------------- augmentations
def _line(img: np.ndarray, p0, p1, rng, thick: int | None = None, dashed: bool = False) -> None:
    color = int(rng.integers(0, 110))
    t = thick or int(rng.integers(1, 3))
    if not dashed:
        cv2.line(img, tuple(map(int, p0)), tuple(map(int, p1)), color, t, cv2.LINE_AA)
        return
    p0, p1 = np.array(p0, float), np.array(p1, float)
    L = np.linalg.norm(p1 - p0)
    if L < 1:
        return
    d = (p1 - p0) / L
    s = 0.0
    while s < L:
        a, b = p0 + d * s, p0 + d * min(L, s + 8)
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), color, t, cv2.LINE_AA)
        s += 12


def cross_line(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw a dimension line (with ticks), leader line or dashed grid line across the text."""
    h, w = img.shape
    kind = int(rng.integers(4))
    if kind == 0:  # horizontal dimension line through the text body
        y = int(rng.uniform(0.25, 0.8) * h)
        _line(img, (0, y), (w, y), rng, thick=1)
        for x in (int(rng.uniform(0, 0.2) * w), int(rng.uniform(0.8, 1) * w)):
            _line(img, (x - 4, y + 4), (x + 4, y - 4), rng, thick=1)
    elif kind == 1:  # diagonal leader line
        _line(img, (rng.uniform(0, w), 0), (rng.uniform(0, w), h), rng)
    elif kind == 2:  # dashed grid line (vertical or horizontal)
        if rng.random() < 0.5:
            x = rng.uniform(0.1, 0.9) * w
            _line(img, (x, 0), (x, h), rng, thick=1, dashed=True)
        else:
            y = rng.uniform(0.2, 0.8) * h
            _line(img, (0, y), (w, y), rng, thick=1, dashed=True)
    else:  # wall-like thick line clipping the top or bottom
        y = int(rng.choice([rng.uniform(0, 0.15), rng.uniform(0.85, 1.0)]) * h)
        _line(img, (0, y), (w, y), rng, thick=int(rng.integers(2, 4)))
    return img


def hatch(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Diagonal hatching over a random span of the line."""
    h, w = img.shape
    x0 = int(rng.uniform(0, 0.6) * w)
    x1 = min(w, x0 + int(rng.uniform(0.2, 0.6) * w))
    step = int(rng.integers(5, 12))
    color = int(rng.integers(120, 200))
    for x in range(x0 - h, x1, step):
        cv2.line(img, (x, h), (x + h, 0), color, 1, cv2.LINE_AA)
    return img


def cell_border(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Partial table-cell border (some of top/bottom/left/right)."""
    h, w = img.shape
    for side in range(4):
        if rng.random() < 0.5:
            continue
        o = int(rng.integers(0, 2))
        pts = [((0, o), (w, o)), ((0, h - 1 - o), (w, h - 1 - o)), ((o, 0), (o, h)), ((w - 1 - o, 0), (w - 1 - o, h))]
        _line(img, *pts[side], rng, thick=1)
    return img


def augment(img: np.ndarray, rng: np.random.Generator, cfg: dict[str, Any], hard: bool = False) -> np.ndarray:
    """Apply drawing + scan degradations. ``hard`` forces at least one crossing line and blur/noise."""
    a = cfg
    img = img.copy()
    if hard or rng.random() < a["p_cross_line"]:
        img = cross_line(img, rng)
        if hard and rng.random() < 0.4:
            img = cross_line(img, rng)
    if rng.random() < a["p_hatch"]:
        img = hatch(img, rng)
    if rng.random() < a["p_cell_border"]:
        img = cell_border(img, rng)
    h, w = img.shape
    ang = rng.uniform(-a["max_rotation_deg"], a["max_rotation_deg"])
    if abs(ang) > 0.2:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)
    if rng.random() < a["p_perspective"]:
        d = 0.04 * min(w, h * 4)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + rng.uniform(-d, d, src.shape).astype(np.float32) * np.float32([[1, 0.2]])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=255)
    if rng.random() < 0.25:  # stroke weight
        k = np.ones((2, 2), np.uint8)
        img = cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)
    if rng.random() < a["p_lowres"] or hard and rng.random() < 0.5:
        f = rng.uniform(0.35, 0.7)
        small = cv2.resize(img, (max(4, int(w * f)), max(4, int(h * f))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < a["p_blur"] or hard:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.1))
    f = img.astype(np.float32)
    if rng.random() < a["p_fade"]:
        lo = rng.uniform(40, 120)
        f = lo + f * (255 - lo) / 255
    f = f * rng.uniform(0.85, 1.0) + rng.uniform(0, 25)  # paper tone
    if rng.random() < a["p_noise"] or hard:
        f += rng.normal(0, rng.uniform(3, 14), f.shape)
    img = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < a["p_jpeg"] or hard:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 80))])
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return img

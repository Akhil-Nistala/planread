"""Synthetic *scanned* drawing sheets + PlanScope-format JSON (oracle regions) + ground truth.

PlanScope's trained detector is not available yet, so end-to-end evaluation uses oracle
layout: we know where every region/cell is because we drew it. Each sheet produces

  sheet_XXX.png        300-DPI scan-like page image (raster route)
  sheet_XXX.layout.json     what PlanScope emits for a raster page: regions/cells with
                       text=null, text_source="needs_ocr" (schema v1.0)
  sheet_XXX.gt.json    true text for every region and cell + a category tag

Usage::

    python -m data_gen.make_docs --out data/docs --n 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from data_gen.render import FontPool, render_line, render_vertical
from data_gen.text_sources import TextSampler, load_yaml
from src.charset import load_charset

ROOT = Path(__file__).resolve().parents[1]
DPI = 300
A3_PX = (int(420 / 25.4 * DPI), int(297 / 25.4 * DPI))  # 4960 x 3507
COLS = ["room", "floor", "baseboard", "wall", "ceiling"]


def box_poly(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """Axis-aligned box as a clockwise polygon from the top-left."""
    return [[float(x0), float(y0)], [float(x1), float(y0)], [float(x1), float(y1)], [float(x0), float(y1)]]


class Sheet:
    """Draws on a grayscale page and records regions + GT."""

    def __init__(self, rng: np.random.Generator, fonts: FontPool, sampler: TextSampler, vocab: dict) -> None:
        self.rng, self.fonts, self.ts, self.vocab = rng, fonts, sampler, vocab
        self.W, self.H = A3_PX
        self.img = np.full((self.H, self.W), 255, np.uint8)
        self.regions: list[dict[str, Any]] = []
        self.gt: dict[str, Any] = {"regions": {}}

    # ---------------------------------------------------------------- primitives
    def paste(self, patch: np.ndarray, x: int, y: int) -> tuple[int, int, int, int]:
        h, w = patch.shape
        x, y = int(np.clip(x, 0, self.W - w)), int(np.clip(y, 0, self.H - h))
        self.img[y:y + h, x:x + w] = np.minimum(self.img[y:y + h, x:x + w], patch)
        return x, y, x + w, y + h

    def text_patch(self, text: str, size: int) -> np.ndarray:
        face = self.fonts.pick(text, self.rng)
        assert face is not None, text
        p = render_line(text, face, size, self.rng)
        return p

    def line(self, p0, p1, t: int = 2, color: int = 0) -> None:
        cv2.line(self.img, tuple(map(int, p0)), tuple(map(int, p1)), color, t, cv2.LINE_AA)

    def covered(self, text: str) -> str:
        for _ in range(30):
            if self.fonts.pick(text, self.rng):
                return text
            text, _ = self.ts.sample()
        return "ア"

    def add_region(self, cls: str, poly: list[list[float]], gt_text: str, category: str,
                   table: dict | None = None) -> str:
        rid = f"r{len(self.regions) + 1}"
        self.regions.append({"id": rid, "class": cls, "conf": 1.0, "polygon": poly, "text": None,
                             "text_source": "needs_ocr", "table": table})
        self.gt["regions"][rid] = {"text": gt_text, "category": category}
        return rid

    # ---------------------------------------------------------------- content
    def plan_clutter(self) -> None:
        x1 = int(self.W * 0.58)
        xs = np.linspace(200, x1 - 80, 5)
        ys = np.linspace(200, self.H - 250, 3)
        for x in xs:
            for yy in range(150, self.H - 150, 60):
                self.line((x, yy), (x, yy + 35), 1, 120)
        for y in ys:
            for xx in range(150, x1, 60):
                self.line((xx, y), (xx + 35, y), 1, 120)
        for i in range(len(xs) - 1):
            for j in range(len(ys) - 1):
                cv2.rectangle(self.img, (int(xs[i]) + 10, int(ys[j]) + 10), (int(xs[i + 1]) - 10, int(ys[j + 1]) - 10), 0, 6)
                cv2.rectangle(self.img, (int(xs[i]) + 26, int(ys[j]) + 26), (int(xs[i + 1]) - 26, int(ys[j + 1]) - 26), 60, 2)
        yd = self.H - 170
        self.line((xs[0], yd), (xs[-1], yd), 2)
        for x in xs:
            self.line((x - 14, yd + 14), (x + 14, yd - 14), 3)

    def schedule(self, lang: str) -> None:
        rng = self.rng
        n_rows = int(rng.integers(5, 10))
        size = int(rng.integers(26, 34))
        h = self.vocab["headers"][lang]
        rooms = list(rng.permutation(self.vocab["rooms"][lang]))[:n_rows]
        rows = [[h[c] for c in COLS]]
        for room in rooms:
            row = [room]
            for c in COLS[1:]:
                v = str(rng.choice(self.vocab["materials"][c][lang]))
                if rng.random() < 0.15:
                    v = v + " " + self.ts.dimension()
                row.append(self.covered(v))
            rows.append(row)
        patches = [[self.text_patch(t, size) for t in r] for r in rows]
        pad = 18
        widths = [max(patches[r][c].shape[1] for r in range(len(rows))) + 2 * pad for c in range(len(COLS))]
        row_h = max(p.shape[0] for r in patches for p in r) + 2 * pad
        x0, y0 = int(self.W * 0.61), int(rng.integers(220, 400))
        x0 = min(x0, self.W - 150 - sum(widths))
        title = self.text_patch("仕上表" if lang == "ja" else "FINISH SCHEDULE", int(size * 1.4))
        self.paste(title, x0, y0 - title.shape[0] - 12)
        X = np.cumsum([x0] + widths)
        Y = [y0 + r * row_h for r in range(len(rows) + 1)]
        cv2.rectangle(self.img, (x0, y0), (int(X[-1]), y0 + row_h), 225, -1)
        cells_out, gt_cells = [], []
        for r in range(len(rows)):
            row_cells, row_gt = [], []
            for c in range(len(COLS)):
                p = patches[r][c]
                self.paste(p, X[c] + pad - 4, Y[r] + (row_h - p.shape[0]) // 2)
                row_cells.append({"text": None, "bbox": [float(X[c]), float(Y[r]), float(X[c + 1]), float(Y[r + 1])],
                                  "text_source": "needs_ocr", "is_header": r == 0})
                row_gt.append(rows[r][c])
            cells_out.append({"cells": row_cells})
            gt_cells.append(row_gt)
        for y in Y:
            self.line((x0, y), (X[-1], y), 4 if y in (Y[0], Y[1], Y[-1]) else 2)
        for x in X:
            self.line((x, y0), (x, Y[-1]), 4 if x in (X[0], X[-1]) else 2)
        table = {"headers": [None] * len(COLS), "canonical_headers": [None] * len(COLS), "rows": cells_out}
        rid = self.add_region("schedule_table", box_poly(x0, y0, X[-1], Y[-1]),
                              "\n".join(" ".join(r) for r in rows), "table", table)
        self.gt["regions"][rid]["cells"] = gt_cells
        self.gt["regions"][rid]["canonical_headers"] = COLS
        self.table_bottom = Y[-1]

    def notes(self, lang: str) -> None:
        rng = self.rng
        size = int(rng.integers(26, 32))
        notes = list(rng.permutation(self.vocab["notes"][lang]))[: int(rng.integers(3, 6))]
        lines = [("特記仕様書" if lang == "ja" else "GENERAL NOTES")] + [f"{i + 1}. {n}" for i, n in enumerate(notes)]
        x0, y = int(self.W * 0.61), int(self.table_bottom + rng.integers(180, 320))
        top, right = y, x0
        for i, t in enumerate(lines):
            p = self.text_patch(t, int(size * (1.3 if i == 0 else 1)))
            bx = self.paste(p, x0, y)
            right = max(right, bx[2])
            y = bx[3] + int(rng.integers(6, 20))
        self.add_region("notes_block", box_poly(x0 - 10, top - 10, right + 10, y + 5), "\n".join(lines), "notes")
        self.notes_bottom = y

    def callouts(self) -> None:
        rng = self.rng
        n = int(rng.integers(5, 9))
        kinds = ["h"] * 3 + ["rot90", "rot270", "vertical"]
        for i in range(n):
            kind = kinds[i % len(kinds)] if i < len(kinds) else str(rng.choice(["h", "h", "rot90"]))
            src = rng.choice(["material", "dimension", "product", "room"])
            text = self.covered(getattr(self.ts, str(src))())
            size = int(rng.integers(26, 40))
            if kind == "vertical":
                text = "".join(c for c in text if not c.isspace())[:8]
                while len(text) < 2 or text.isascii():
                    text = self.covered(self.ts.room() + self.ts.material())[:6].replace(" ", "")
                patch = render_vertical(text, self.fonts.pick(text, rng), size, rng)
            else:
                patch = self.text_patch(text, size)
                if kind == "rot90":
                    patch = np.ascontiguousarray(np.rot90(patch, 1))
                elif kind == "rot270":
                    patch = np.ascontiguousarray(np.rot90(patch, 3))
            x = int(rng.integers(260, int(self.W * 0.55) - patch.shape[1] - 10))
            y = int(rng.integers(260, self.H - 400 - patch.shape[0]))
            bx = self.paste(patch, x, y)
            # leader line from the callout to a point in the plan (sometimes crossing the text)
            ex, ey = bx[0] - int(rng.integers(60, 200)), bx[3] + int(rng.integers(40, 200))
            if rng.random() < 0.35:
                sx, sy = bx[2], (bx[1] + bx[3]) // 2   # crosses through the text
            else:
                sx, sy = bx[0], bx[3]
            self.line((sx, sy), (ex, ey), 2)
            cv2.circle(self.img, (ex, ey), 6, 0, -1)
            cat = {"h": "callout", "rot90": "callout_rot90", "rot270": "callout_rot270", "vertical": "callout_vertical"}[kind]
            self.add_region("text_callout", box_poly(bx[0] - 6, bx[1] - 6, bx[2] + 6, bx[3] + 6), text, cat)

    def title_block(self, lang: str) -> None:
        rng = self.rng
        size = int(rng.integers(28, 34))
        rows = (["工事名称 " + self.covered(self.ts._pseudo_kata() + "ビル改修工事"), "図面名称 仕上表・1階平面図",
                 f"縮尺 1/{rng.choice([50, 100, 200])}", f"図面番号 A-{rng.integers(100, 400)}",
                 "設計 " + self.covered(self.ts.manufacturer())]
                if lang == "ja" else
                [f"PROJECT {self.ts._pseudo_kata()} Refurbishment".replace("PROJECT ", "PROJECT "),
                 "TITLE Finish Schedule", f"SCALE 1:{rng.choice([50, 100, 200])}", f"DWG No. A-{rng.integers(100, 400)}"])
        rows = [self.covered(r) for r in rows]
        patches = [self.text_patch(r, size) for r in rows]
        bw = max(p.shape[1] for p in patches) + 60
        rh = max(p.shape[0] for p in patches) + 24
        x0, y0 = self.W - 120 - bw, self.H - 120 - rh * len(rows)
        cv2.rectangle(self.img, (x0, y0), (x0 + bw, y0 + rh * len(rows)), 0, 4)
        for i, p in enumerate(patches):
            if i:
                self.line((x0, y0 + i * rh), (x0 + bw, y0 + i * rh), 2)
            self.paste(p, x0 + 24, y0 + i * rh + (rh - p.shape[0]) // 2)
        self.add_region("title_block", box_poly(x0, y0, x0 + bw, y0 + rh * len(rows)), "\n".join(rows), "title_block")

    def scan(self) -> np.ndarray:
        rng = self.rng
        g = self.img.astype(np.float32)
        g = cv2.GaussianBlur(g, (3, 3), rng.uniform(0.6, 1.2))
        g = g * rng.uniform(0.75, 0.9) + 255 * rng.uniform(0.08, 0.2)
        g += rng.normal(0, rng.uniform(5, 12), g.shape)
        g = np.clip(g, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 75))])
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def make_sheet(i: int, seed: int, fonts: FontPool, vocab: dict, synth: dict, charset: list[str],
               out: Path) -> None:
    """Generate one sheet (png + A json + GT json)."""
    rng = np.random.default_rng([seed, 99, i])
    lang = "ja" if i % 4 != 3 else "en"
    s = Sheet(rng, fonts, TextSampler(vocab, synth, charset, rng), vocab)
    cv2.rectangle(s.img, (60, 60), (s.W - 60, s.H - 60), 0, 5)
    s.plan_clutter()
    s.schedule(lang)
    s.notes(lang)
    s.callouts()
    s.title_block(lang)
    page = s.scan()
    name = f"sheet_{i:03d}"
    Image.fromarray(page).save(out / f"{name}.png", dpi=(DPI, DPI))
    layout_doc = {"schema_version": "1.0", "document": f"{name}.png", "pages": [{
        "page": 1, "route": "raster", "dpi": DPI, "width": s.W, "height": s.H, "needs_ocr": True,
        "routing": None, "words": [], "regions": s.regions}]}
    (out / f"{name}.layout.json").write_text(json.dumps(layout_doc, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{name}.gt.json").write_text(json.dumps(s.gt, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "docs")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    synth = load_yaml(ROOT / "configs" / "synth.yaml")
    vocab = load_yaml(ROOT / "configs" / "vocab.yaml")
    charset = load_charset(ROOT / "charset" / "charset.txt")
    fonts = FontPool(synth["fonts"]["ja"], synth["fonts"]["latin"])
    args.out.mkdir(parents=True, exist_ok=True)
    n = args.n or synth["sizes"]["docs"]
    for i in range(n):
        make_sheet(i, synth["seed"], fonts, vocab, synth, charset, args.out)
    print(f"wrote {n} sheets to {args.out}")


if __name__ == "__main__":
    main()

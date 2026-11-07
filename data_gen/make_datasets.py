"""Generate synthetic OCR datasets (reproducible from the seed).

Outputs under ``--out`` (default data/):
  rec/{train,val,test_clean,test_hard}/  line images + labels.tsv   (path, text, source, font)
  orient/{train,test}/                   crops + labels.tsv          (path, label in 0|90|180|270|vertical)
  det/{train,test}/                      multi-line blocks + PaddleOCR det labels (label.txt)

Usage::

    python -m data_gen.make_datasets --what rec,orient,det --scale 1.0
    python -m data_gen.make_datasets --what rec --scale 0.1     # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from data_gen.render import FontPool, augment, cross_line, render_line, render_vertical
from data_gen.text_sources import TextSampler, load_yaml
from src.charset import load_charset

ROOT = Path(__file__).resolve().parents[1]
SPLIT_IDS = {"train": 0, "val": 1, "test_clean": 2, "test_hard": 3, "test": 4}
ORIENT_LABELS = ["0", "90", "180", "270", "vertical"]


class Generator:
    """Holds shared resources (fonts, sampler config) for all dataset kinds."""

    def __init__(self, seed: int, synth_path: Path, vocab_path: Path, charset_path: Path) -> None:
        self.seed = seed
        self.cfg = load_yaml(synth_path)
        self.vocab = load_yaml(vocab_path)
        self.charset = load_charset(charset_path)
        self.fonts = FontPool(self.cfg["fonts"]["ja"], self.cfg["fonts"]["latin"])

    def rng(self, *key: int) -> np.random.Generator:
        """Per-sample generator -> any sample is reproducible independently."""
        return np.random.default_rng([self.seed, *key])

    def sampler(self, rng: np.random.Generator) -> TextSampler:
        """Text sampler bound to ``rng``."""
        return TextSampler(self.vocab, self.cfg, self.charset, rng)

    def line(self, rng: np.random.Generator, hard: bool = False, clean: bool = False,
             size_range: tuple[int, int] = (14, 40)) -> tuple[np.ndarray, str, str, str]:
        """One augmented line image + text, source and font name."""
        ts = self.sampler(rng)
        for _ in range(20):
            text, src = ts.sample()
            face = self.fonts.pick(text, rng)
            if face is not None:
                break
        else:
            raise RuntimeError("could not find a font covering sampled text")
        img = render_line(text, face, int(rng.integers(*size_range)), rng)
        if not clean:
            img = augment(img, rng, self.cfg["augment"], hard=hard)
        return img, text, src, f"{Path(face.path).stem}#{face.index}"

    # ------------------------------------------------------------------ rec
    def make_rec(self, out: Path, split: str, n: int) -> None:
        """Recognition lines. ``test_clean`` = no augmentation, ``test_hard`` = forced crossing lines."""
        d = out / "rec" / split
        (d / "images").mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(n):
            rng = self.rng(SPLIT_IDS[split], i)
            img, text, src, font = self.line(rng, hard=split == "test_hard", clean=split == "test_clean")
            name = f"images/{i:06d}.png"
            cv2.imwrite(str(d / name), img)
            rows.append(f"{name}\t{text}\t{src}\t{font}")
        (d / "labels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # --------------------------------------------------------------- orient
    def make_orient(self, out: Path, split: str, n: int) -> None:
        """5-class orientation crops: horizontal text rotated 0/90/180/270, or upright 縦書き."""
        d = out / "orient" / split
        (d / "images").mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(n):
            rng = self.rng(10 + SPLIT_IDS[split], i)
            label = ORIENT_LABELS[i % 5]
            if label == "vertical":
                ts = self.sampler(rng)
                for _ in range(20):
                    text, _ = ts.sample()
                    text = "".join(c for c in text if not c.isspace())[:12]
                    face = self.fonts.pick(text, rng)
                    if face and len(text) >= 2:
                        break
                img = augment(render_vertical(text, face, int(rng.integers(16, 36)), rng), rng,
                              self.cfg["augment"])
            else:
                img, *_ = self.line(rng, size_range=(16, 36))
                img = np.ascontiguousarray(np.rot90(img, ORIENT_LABELS.index(label)))  # CCW by label
            name = f"images/{i:06d}.png"
            cv2.imwrite(str(d / name), img)
            rows.append(f"{name}\t{label}")
        (d / "labels.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ det
    def make_det(self, out: Path, split: str, n: int) -> None:
        """Multi-line text blocks (notes/callouts/cells) with lines crossing text.

        Labels in PaddleOCR det format: ``path\\t[{"transcription": str, "points": [[x,y]*4]}]``.
        """
        d = out / "det" / split
        (d / "images").mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(n):
            rng = self.rng(20 + SPLIT_IDS[split], i)
            n_lines = int(rng.integers(1, 7))
            size = int(rng.integers(14, 32))
            lines: list[tuple[np.ndarray, str]] = []
            ts = self.sampler(rng)
            while len(lines) < n_lines:
                t, _ = ts.sample()
                f = self.fonts.pick(t, rng)
                if f:
                    lines.append((render_line(t, f, size, rng), t))
            W = max(l[0].shape[1] for l in lines) + int(rng.integers(10, 80))
            gaps = [int(rng.integers(0, size // 2 + 1)) for _ in lines]
            H = sum(l[0].shape[0] for l in lines) + sum(gaps) + 20
            canvas = np.full((H, W), 255, np.uint8)
            y = 10
            anns = []
            for (im, t), g in zip(lines, gaps):
                h, w = im.shape
                x = int(rng.integers(5, max(6, W - w - 5)))
                canvas[y:y + h, x:x + w] = np.minimum(canvas[y:y + h, x:x + w], im)
                # tight box from ink (padding in render_line is not part of the text)
                ys, xs = np.where(im < 160)
                if len(xs):
                    x0, x1, y0, y1 = x + xs.min(), x + xs.max(), y + ys.min(), y + ys.max()
                    anns.append({"transcription": t, "points": [[int(x0), int(y0)], [int(x1), int(y0)],
                                                                 [int(x1), int(y1)], [int(x0), int(y1)]]})
                y += h + g
            for _ in range(int(rng.integers(1, 4))):
                canvas = cross_line(canvas, rng)
            aug = dict(self.cfg["augment"], max_rotation_deg=0.0, p_perspective=0.0, p_cross_line=0.0)
            canvas = augment(canvas, rng, aug)
            name = f"images/{i:06d}.png"
            cv2.imwrite(str(d / name), canvas)
            rows.append(f"{name}\t{json.dumps(anns, ensure_ascii=False)}")
        (d / "label.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", default="rec,orient,det")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all dataset sizes")
    ap.add_argument("--synth", type=Path, default=ROOT / "configs" / "synth.yaml")
    ap.add_argument("--vocab", type=Path, default=ROOT / "configs" / "vocab.yaml")
    ap.add_argument("--charset", type=Path, default=ROOT / "charset" / "charset.txt")
    args = ap.parse_args()

    cfg = load_yaml(args.synth)
    g = Generator(cfg["seed"], args.synth, args.vocab, args.charset)
    sz = {k: max(10, int(v * args.scale)) for k, v in cfg["sizes"].items()}
    jobs = {
        "rec": [("train", sz["rec_train"]), ("val", sz["rec_val"]), ("test_clean", sz["rec_test"]),
                ("test_hard", sz["rec_test"])],
        "orient": [("train", sz["orient_train"]), ("test", sz["orient_test"])],
        "det": [("train", sz["det_train"]), ("test", sz["det_test"])],
    }
    for what in args.what.split(","):
        for split, n in jobs[what]:
            t0 = time.perf_counter()
            getattr(g, f"make_{what}")(args.out, split, n)
            print(f"{what:6s} {split:10s} n={n:6d}  {time.perf_counter() - t0:6.1f}s")


if __name__ == "__main__":
    main()

"""Text orientation classifier: 0 / 90 / 180 / 270 (CCW rotation of horizontal text) / vertical (縦書き).

Tiny CNN on a 128x128 letterboxed crop (aspect ratio is preserved, so "tall" is visible to the
model). Trains in a few minutes on CPU.

Usage::

    python -m src.orientation train --data data/orient --out models/orientation.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["0", "90", "180", "270", "vertical"]


def letterbox(gray: np.ndarray, size: int = 128) -> np.ndarray:
    """Fit into size x size on white, keep aspect; returns float32 with ink ~ +1."""
    h, w = gray.shape[:2]
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    img = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 255, np.uint8)
    y, x = (size - nh) // 2, (size - nw) // 2
    canvas[y:y + nh, x:x + nw] = img
    return 1.0 - canvas.astype(np.float32) / 127.5


class OrientNet(nn.Module):
    """5-way orientation CNN (~0.3M params)."""

    def __init__(self, n: int = len(LABELS)) -> None:
        super().__init__()

        def blk(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.MaxPool2d(2))

        self.f = nn.Sequential(blk(1, 24), blk(24, 48), blk(48, 96), blk(96, 128), blk(128, 160))  # 4x4
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(160 * 16, n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: B x 1 x 128 x 128 -> logits."""
        return self.head(self.f(x))


class OrientationClassifier:
    """Inference wrapper. If no model file exists, falls back to an aspect-ratio heuristic."""

    def __init__(self, path: str | Path | None, size: int = 128) -> None:
        self.size = size
        self.model: OrientNet | None = None
        if path and Path(path).exists():
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.model = OrientNet()
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval()

    def predict(self, gray: np.ndarray) -> np.ndarray:
        """Probabilities over LABELS."""
        if self.model is None:
            h, w = gray.shape[:2]
            p = np.full(len(LABELS), 0.02)
            p[LABELS.index("90" if h > 1.6 * w else "0")] = 0.92
            return p / p.sum()
        x = torch.from_numpy(letterbox(gray, self.size))[None, None]
        with torch.no_grad():
            return torch.softmax(self.model(x), -1)[0].numpy()


class OrientDataset(Dataset):
    """labels.tsv rows: path, label. Train-time random flips are NOT used (they would change labels)."""

    def __init__(self, root: Path, size: int = 128) -> None:
        self.root, self.size = root, size
        self.items = [ln.split("\t") for ln in (root / "labels.tsv").read_text(encoding="utf-8").splitlines() if ln]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        p, lab = self.items[i]
        img = letterbox(cv2.imread(str(self.root / p), cv2.IMREAD_GRAYSCALE), self.size)
        return torch.from_numpy(img)[None], LABELS.index(lab)


def evaluate(model: nn.Module, dl: DataLoader) -> tuple[float, np.ndarray]:
    """Accuracy and confusion matrix (rows = true)."""
    model.eval()
    cm = np.zeros((len(LABELS), len(LABELS)), int)
    with torch.no_grad():
        for x, y in dl:
            for t, p in zip(y.tolist(), model(x).argmax(-1).tolist()):
                cm[t, p] += 1
    return float(np.trace(cm) / cm.sum()), cm


def train(args: argparse.Namespace) -> None:
    """Train + report test accuracy."""
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tr, te = OrientDataset(args.data / "train"), OrientDataset(args.data / "test")
    dl = DataLoader(tr, batch_size=64, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    dt = DataLoader(te, batch_size=128)
    model = OrientNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.epochs * len(dl))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = -1.0
    for ep in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        for x, y in dl:
            loss = lossf(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        acc, cm = evaluate(model, dt)
        print(f"epoch {ep + 1}/{args.epochs} test_acc={acc:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "labels": LABELS, "test_acc": acc,
                        "confusion": cm.tolist()}, args.out)
    print(f"best test acc {best:.4f} -> {args.out}")


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=ROOT / "data" / "orient")
    t.add_argument("--out", type=Path, default=ROOT / "models" / "orientation.pt")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--threads", type=int, default=torch.get_num_threads())
    t.add_argument("--seed", type=int, default=42)
    train(ap.parse_args())


if __name__ == "__main__":
    main()

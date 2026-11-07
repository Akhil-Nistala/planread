"""CRNN (CNN + BiLSTM + CTC) recogniser trained in this repo on synthetic drawing text.

This is the *trainable* recogniser path. It is deliberately small so a short run is possible
on CPU; the full-size run is meant for a GPU (see README "Training on Colab").

Usage::

    python -m src.crnn train --data data/rec --out models/crnn.pt --epochs 8 --device cpu
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .charset import load_charset, oov_report
from .textnorm import fold_width, normalize

ROOT = Path(__file__).resolve().parents[1]
IMG_H = 32
MAX_W = 480


def model_charset(charset: Sequence[str]) -> list[str]:
    """Width-folded, de-duplicated charset used as CTC classes (index 0 = blank)."""
    seen: dict[str, None] = {}
    for c in charset:
        for f in fold_width(c):
            seen.setdefault(f, None)
    return ["<blank>"] + list(seen)


def preprocess(gray: np.ndarray, h: int = IMG_H, max_w: int = MAX_W) -> np.ndarray:
    """Resize to height ``h`` keeping aspect; float32 in [-1, 1], ink bright (1 = ink)."""
    H, W = gray.shape[:2]
    w = int(min(max_w, max(8, round(W * h / max(H, 1)))))
    img = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA if H > h else cv2.INTER_CUBIC)
    return (1.0 - img.astype(np.float32) / 127.5)  # white paper -> -1, black ink -> +1


class CRNN(nn.Module):
    """Compact CRNN: 32xW gray -> (W/4) x C log-probs."""

    def __init__(self, n_classes: int, hidden: int = 192) -> None:
        super().__init__()

        def blk(i: int, o: int) -> list[nn.Module]:
            return [nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        self.cnn = nn.Sequential(
            *blk(1, 32), nn.MaxPool2d(2, 2),                 # 16 x W/2
            *blk(32, 64), nn.MaxPool2d(2, 2),                # 8 x W/4
            *blk(64, 128), *blk(128, 128), nn.MaxPool2d((2, 1), (2, 1)),   # 4 x W/4
            *blk(128, 256), *blk(256, 256), nn.MaxPool2d((2, 1), (2, 1)),  # 2 x W/4
            nn.Conv2d(256, 256, (2, 1), bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),  # 1 x W/4
        )
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: B x 1 x 32 x W -> B x T x C logits."""
        f = self.cnn(x).squeeze(2).transpose(1, 2)  # B x T x 256
        return self.fc(self.rnn(f)[0])


# ----------------------------------------------------------------------------- data
class LineDataset(Dataset):
    """Reads ``labels.tsv`` written by data_gen.make_datasets."""

    def __init__(self, root: Path, classes: list[str], limit: int | None = None) -> None:
        self.root = root
        rows = [ln.split("\t") for ln in (root / "labels.tsv").read_text(encoding="utf-8").splitlines() if ln]
        self.items = rows[:limit] if limit else rows
        self.idx = {c: i for i, c in enumerate(classes)}
        self.oov = oov_report((normalize(r[1]) for r in self.items), classes)
        self.widths = []
        for r in self.items:  # cheap width estimate for bucketing (header read only)
            im = cv2.imread(str(root / r[0]), cv2.IMREAD_GRAYSCALE)
            self.widths.append(min(MAX_W, int(im.shape[1] * IMG_H / im.shape[0])))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[np.ndarray, list[int], str]:
        path, text = self.items[i][0], normalize(self.items[i][1])
        img = preprocess(cv2.imread(str(self.root / path), cv2.IMREAD_GRAYSCALE))
        return img, [self.idx[c] for c in text if c in self.idx], text


class WidthBucketSampler(Sampler[list[int]]):
    """Batches of similar width (less padding -> faster CPU training)."""

    def __init__(self, widths: list[int], batch: int, shuffle: bool, seed: int = 0) -> None:
        self.widths, self.batch, self.shuffle, self.seed, self.epoch = widths, batch, shuffle, seed, 0

    def __iter__(self):
        order = np.argsort(self.widths, kind="stable")
        batches = [order[i:i + self.batch].tolist() for i in range(0, len(order), self.batch)]
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
            self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return math.ceil(len(self.widths) / self.batch)


def collate(batch):
    """Pad images to the batch max width; flatten targets for CTC."""
    w = max(b[0].shape[1] for b in batch)
    x = np.full((len(batch), 1, IMG_H, w), -1.0, np.float32)
    for i, (img, _, _) in enumerate(batch):
        x[i, 0, :, :img.shape[1]] = img
    targets = torch.tensor([t for b in batch for t in b[1]], dtype=torch.long)
    lengths = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    return torch.from_numpy(x), targets, lengths, [b[2] for b in batch]


def cer(pred: str, gt: str) -> float:
    """Character error rate = Levenshtein(pred, gt) / len(gt)."""
    return levenshtein(pred, gt) / max(1, len(gt))


def levenshtein(a: str, b: str) -> int:
    """Edit distance."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def greedy_batch(logits: torch.Tensor, classes: list[str]) -> list[str]:
    """Greedy CTC decode for a batch."""
    out = []
    for seq in logits.argmax(-1).tolist():
        s, prev = [], 0
        for k in seq:
            if k and k != prev:
                s.append(classes[k])
            prev = k
        out.append("".join(s))
    return out


def train(args: argparse.Namespace) -> None:
    """Train loop with OneCycle LR; keeps the checkpoint with the best val CER."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    classes = model_charset(load_charset(args.charset))
    tr = LineDataset(args.data / "train", classes, args.limit)
    va = LineDataset(args.data / "val", classes, args.val_limit)
    if tr.oov:
        print(f"[warn] {sum(tr.oov.values())} out-of-charset chars in train (dropped): {tr.oov.most_common(10)}")
    dl = DataLoader(tr, batch_sampler=WidthBucketSampler(tr.widths, args.batch, True, args.seed),
                    collate_fn=collate, num_workers=args.workers)
    dv = DataLoader(va, batch_sampler=WidthBucketSampler(va.widths, 128, False), collate_fn=collate)
    dev = torch.device(args.device)
    model = CRNN(len(classes)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.15)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    best = math.inf
    log = []
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for x, tg, ln, _ in dl:
            x = x.to(dev)
            lp = F.log_softmax(model(x), -1).transpose(0, 1)  # T x B x C
            il = torch.full((x.shape[0],), lp.shape[0], dtype=torch.long)
            loss = ctc(lp, tg, il, ln)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += loss.item() * x.shape[0]
            n += x.shape[0]
        model.eval()
        errs, chars, exact, m = 0, 0, 0, 0
        with torch.no_grad():
            for x, _, _, texts in dv:
                for p, g in zip(greedy_batch(model(x.to(dev)), classes), texts):
                    errs += levenshtein(p, g)
                    chars += max(1, len(g))
                    exact += p == g
                    m += 1
        v_cer = errs / chars
        log.append({"epoch": ep + 1, "loss": tot / n, "val_cer": v_cer, "val_exact": exact / m,
                    "sec": time.perf_counter() - t0})
        print(f"epoch {ep + 1}/{args.epochs} loss={tot / n:.3f} val_CER={v_cer:.4f} "
              f"val_exact={exact / m:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        if v_cer < best:
            best = v_cer
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "classes": classes, "log": log}, args.out)
    print(f"best val CER {best:.4f} -> {args.out}")


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", type=Path, default=ROOT / "data" / "rec")
    t.add_argument("--charset", type=Path, default=ROOT / "charset" / "charset.txt")
    t.add_argument("--out", type=Path, default=ROOT / "models" / "crnn.pt")
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    t.add_argument("--threads", type=int, default=max(1, (torch.get_num_threads())))
    t.add_argument("--workers", type=int, default=0)
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--val-limit", type=int, default=1000)
    t.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()

"""CTC decoding: greedy best path + prefix beam search with top-k candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

NEG_INF = -np.inf


@dataclass
class Decoded:
    """Decoding result for one line."""

    text: str
    candidates: list[tuple[str, float]]   # (text, relative probability), sorted, sums to <= 1
    features: dict[str, float]            # inputs for confidence calibration

    @property
    def raw_conf(self) -> float:
        """Uncalibrated score: mean max-probability of emitted characters (PP-OCR convention)."""
        return self.features["mean_char_prob"]


def greedy(probs: np.ndarray, chars: Sequence[str], blank: int = 0) -> tuple[str, list[float]]:
    """Best-path decoding; returns text and the max-probabilities of emitted characters."""
    idx = probs.argmax(1)
    out, ps, prev = [], [], -1
    for t, k in enumerate(idx):
        if k != blank and k != prev:
            out.append(chars[k])
            ps.append(float(probs[t, k]))
        prev = k
    return "".join(out), ps


def prefix_beam_search(probs: np.ndarray, chars: Sequence[str], beam: int = 10, top_k: int = 5,
                       blank: int = 0, prune: float = 1e-3, cutoff: int = 12) -> list[tuple[str, float]]:
    """CTC prefix beam search. Returns up to ``top_k`` (text, log-prob) pairs, best first.

    ``probs``: T x C softmax outputs; ``chars[k]`` is the string for class k (``chars[blank]`` unused).
    """
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}  # prefix -> (log p_blank, log p_nonblank)
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    for t in range(probs.shape[0]):
        row = probs[t]
        cand = np.argsort(-row)[:cutoff]
        cand = [int(k) for k in cand if row[k] >= prune] or [int(cand[0])]
        nxt: dict[tuple[int, ...], list[float]] = {}

        def add(prefix: tuple[int, ...], pb: float, pnb: float) -> None:
            cur = nxt.setdefault(prefix, [NEG_INF, NEG_INF])
            cur[0] = np.logaddexp(cur[0], pb)
            cur[1] = np.logaddexp(cur[1], pnb)

        for prefix, (pb, pnb) in beams.items():
            total = np.logaddexp(pb, pnb)
            for k in cand:
                lp = logp[t, k]
                if k == blank:
                    add(prefix, total + lp, NEG_INF)
                    continue
                last = prefix[-1] if prefix else None
                ext = prefix + (k,)
                if k == last:
                    add(ext, NEG_INF, pb + lp)          # repeated char needs a blank in between
                    add(prefix, NEG_INF, pnb + lp)      # collapse repeat
                else:
                    add(ext, NEG_INF, total + lp)
        beams = dict(sorted(((p, (v[0], v[1])) for p, v in nxt.items()),
                            key=lambda kv: -np.logaddexp(*kv[1]))[:beam])
    ranked = sorted(((p, float(np.logaddexp(*v))) for p, v in beams.items()), key=lambda kv: -kv[1])
    merged: dict[str, float] = {}
    for p, lp in ranked:  # different label paths can map to the same string (e.g. space handling)
        s = "".join(chars[k] for k in p)
        merged[s] = float(np.logaddexp(merged.get(s, NEG_INF), lp))
    return sorted(merged.items(), key=lambda kv: -kv[1])[:top_k]


def decode(probs: np.ndarray, chars: Sequence[str], beam: int = 10, top_k: int = 5, blank: int = 0) -> Decoded:
    """Greedy + beam decoding with calibration features."""
    g_text, g_ps = greedy(probs, chars, blank)
    cands = prefix_beam_search(probs, chars, beam=beam, top_k=top_k, blank=blank)
    lps = np.array([lp for _, lp in cands])
    rel = np.exp(lps - np.logaddexp.reduce(lps)) if len(lps) else np.array([1.0])
    # relative probs among the returned beams, scaled by how much mass the best beam has in absolute terms
    candidates = [(s, float(r)) for (s, _), r in zip(cands, rel)]
    best = candidates[0][0] if candidates else g_text
    feats = {
        "mean_char_prob": float(np.mean(g_ps)) if g_ps else 0.0,
        "min_char_prob": float(np.min(g_ps)) if g_ps else 0.0,
        "beam_top1": float(rel[0]) if len(rel) else 1.0,
        "beam_margin": float(rel[0] - rel[1]) if len(rel) > 1 else 1.0,
        "seq_logp_per_t": float(lps[0] / max(1, probs.shape[0])) if len(lps) else 0.0,
        "length": float(len(best)),
    }
    return Decoded(best, candidates, feats)

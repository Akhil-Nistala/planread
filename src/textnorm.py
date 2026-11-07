"""Text normalisation shared by training targets, evaluation and output.

Full-width ASCII (Ｔ１２．５) is folded to half-width (T12.5) and whitespace is collapsed.
Unlike full NFKC we keep drawing symbols intact (NFKC would turn ㎡ into "m2", ① into "1",
㈱ into "(株)"), because they carry meaning for downstream matching.
"""

from __future__ import annotations

_FW = {c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)}
_FW[0x3000] = 0x20  # ideographic space
_EXTRA = {ord("－"): "-", ord("〜"): "～", ord("―"): "-", ord("‐"): "-"}


def fold_width(s: str) -> str:
    """Full-width ASCII -> half-width; unify a few dash/tilde look-alikes."""
    return s.translate(_FW).translate(_EXTRA)


def normalize(s: str) -> str:
    """Comparison form: width-folded, whitespace collapsed, stripped."""
    return " ".join(fold_width(s).split())

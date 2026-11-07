"""Character set for recognition: build, load, and out-of-charset (OOV) reporting.

Sources (in this order, de-duplicated):
  (a) every character of the material / room / notes vocabulary (configs/vocab.yaml)
  (b) common kanji: JIS X 0208 Level-1 (2,965 kanji, derived offline from the EUC-JP codec;
      covers almost all Jōyō kanji -- swap in an explicit Jōyō list via ``--extra`` if preferred)
  (c) hiragana, katakana, full-width and half-width ASCII, CJK punctuation
  (d) drawing symbols (φ × @ ± ㎡ ℃ ...)

Usage::

    python -m src.charset build --out charset/charset.txt
"""

from __future__ import annotations

import argparse
import logging
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

DRAWING_SYMBOLS = (
    "φΦø∅×±@㎡㎜㎝㎏㎥°℃～〜※→←↑↓■□●○◎△▲▽▼◇◆☆★・―‐－＝≒≦≧≤≥∠⊥Σμ"
    "①②③④⑤⑥⑦⑧⑨⑩⑪⑫ⅠⅡⅢⅣⅤⅥ㈱㈲№"
)
CJK_PUNCT = "、。，．・：；？！゛゜´｀¨＾￣＿ヽヾゝゞ〃仝々〆〇ー―‐／＼～∥｜…‥‘’“”（）〔〕［］｛｝〈〉《》「」『』【】＋－±×÷＝≠＜＞％＃＆＊＠§☆"


def jis_level1_kanji() -> list[str]:
    """JIS X 0208 Level-1 kanji: rows 16-47 (EUC-JP bytes 0xB0A1..0xCFD3)."""
    out = []
    for row in range(16, 48):
        for cell in range(1, 95):
            try:
                ch = bytes([0xA0 + row, 0xA0 + cell]).decode("euc_jp")
            except UnicodeDecodeError:
                continue
            if unicodedata.name(ch, "").startswith("CJK UNIFIED IDEOGRAPH"):
                out.append(ch)
    return out


def kana() -> list[str]:
    """Hiragana U+3041-3096, katakana U+30A1-30FA, long-vowel mark."""
    return [chr(c) for c in range(0x3041, 0x3097)] + [chr(c) for c in range(0x30A1, 0x30FB)] + ["ー"]


def ascii_sets() -> list[str]:
    """Half-width printable ASCII (incl. space) and full-width forms U+FF01-FF5E."""
    return [chr(c) for c in range(0x20, 0x7F)] + [chr(c) for c in range(0xFF01, 0xFF5F)]


def vocab_chars(vocab_path: Path) -> list[str]:
    """All characters appearing in the vocabulary YAML values."""
    data = yaml.safe_load(vocab_path.read_text(encoding="utf-8"))
    chars: list[str] = []

    def walk(x: object) -> None:
        if isinstance(x, str):
            chars.extend(x)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(data)
    return chars


def build_charset(vocab_path: Path, extra: Iterable[str] = ()) -> list[str]:
    """Ordered, de-duplicated charset (space kept; other whitespace/control chars dropped)."""
    seen: dict[str, None] = {}
    for group in (vocab_chars(vocab_path), jis_level1_kanji(), kana(), ascii_sets(),
                  list(CJK_PUNCT), list(DRAWING_SYMBOLS), list(extra)):
        for ch in group:
            if ch == " " or (not ch.isspace() and unicodedata.category(ch)[0] != "C"):
                seen.setdefault(ch, None)
    return list(seen)


def load_charset(path: str | Path) -> list[str]:
    """Load a charset file (one character per line; the literal line '<space>' encodes ' ')."""
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    return [" " if ln == "<space>" else ln for ln in lines if ln]


def save_charset(chars: list[str], path: str | Path) -> None:
    """Save a charset, one char per line."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join("<space>" if c == " " else c for c in chars) + "\n", encoding="utf-8")


def oov_report(texts: Iterable[str], charset: Iterable[str]) -> Counter:
    """Count characters in ``texts`` that are not in ``charset`` (logged as warnings by callers)."""
    cs = set(charset)
    return Counter(ch for t in texts for ch in t if ch not in cs)


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--vocab", type=Path, default=ROOT / "configs" / "vocab.yaml")
    b.add_argument("--out", type=Path, default=ROOT / "charset" / "charset.txt")
    b.add_argument("--extra", type=Path, default=None, help="optional file of extra characters")
    args = ap.parse_args()
    extra = args.extra.read_text(encoding="utf-8") if args.extra else ""
    chars = build_charset(args.vocab, extra)
    save_charset(chars, args.out)
    print(f"charset: {len(chars)} chars -> {args.out}")


if __name__ == "__main__":
    main()

"""Text samplers for synthetic OCR data (all vocab-driven; no real brand names)."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def to_fullwidth(s: str) -> str:
    """Convert printable ASCII to full-width forms (common in Japanese drawings)."""
    return "".join(chr(ord(c) + 0xFEE0) if 0x21 <= ord(c) <= 0x7E else ("　" if c == " " else c) for c in s)


class TextSampler:
    """Samples drawing-like strings from configurable sources."""

    def __init__(self, vocab: dict[str, Any], synth_cfg: dict[str, Any], charset: list[str],
                 rng: np.random.Generator) -> None:
        self.v, self.cfg, self.rng = vocab, synth_cfg, rng
        self.kanji_kana = [c for c in charset if ord(c) >= 0x3040 and unicodedata.category(c).startswith("L")]
        self.symbols = [c for c in charset if unicodedata.category(c)[0] in "SP" and ord(c) > 0x7F]
        mix = synth_cfg["text_mix"]
        self.sources: dict[str, Callable[[], str]] = {
            "material": self.material, "room": self.room, "product": self.product,
            "manufacturer": self.manufacturer, "dimension": self.dimension,
            "note_fragment": self.note_fragment, "random_charset": self.random_charset,
        }
        self.names = list(mix)
        w = np.array([mix[n] for n in self.names], float)
        self.weights = w / w.sum()

    # -- helpers -------------------------------------------------------------
    def _c(self, seq: list) -> Any:
        return seq[int(self.rng.integers(len(seq)))]

    def _lang(self) -> str:
        return "ja" if self.rng.random() < 0.7 else "en"

    def _num(self) -> str:
        return self._c(["6", "8", "9.5", "12", "12.5", "15", "18", "21", "25", "50", "60", "100"])

    # -- sources ---------------------------------------------------------------
    def material(self) -> str:
        cat = self._c(list(self.v["materials"]))
        s = self._c(self.v["materials"][cat][self._lang()])
        if self.rng.random() < 0.25:
            s += self._c([f" t{self._num()}", f" t={self._num()}", f"（{self._num()}mm）", " 下地共", " EP塗装"])
        return s

    def room(self) -> str:
        return self._c(self.v["rooms"][self._lang()])

    def _pseudo_kata(self, lo: int = 3, hi: int = 6) -> str:
        syl = self.cfg["pseudo_words"]["kata_syllables"]
        s = "".join(self._c(syl) for _ in range(int(self.rng.integers(lo, hi + 1))))
        return s.lstrip("ー") or "ア"

    def product(self) -> str:
        pw = self.cfg["pseudo_words"]
        series = f"{self._c(pw['product_series'])}-{int(self.rng.integers(10, 9999))}"
        form = int(self.rng.integers(3))
        if form == 0:
            return f"{self._pseudo_kata()} {series}"
        if form == 1:
            return f"{self._pseudo_kata()}{self._c(['ボード', 'タイル', 'シート', 'パネル'])} {series}"
        return series + self._c(["", "R", "S", " 不燃", " 準不燃"])

    def manufacturer(self) -> str:
        pw = self.cfg["pseudo_words"]
        if self.rng.random() < 0.7:
            name = self._pseudo_kata(2, 4) + self._c(pw["company_suffix"])
            return ("㈱" + name) if self.rng.random() < 0.2 and not name.endswith("㈱") else name
        cap = "".join(chr(int(self.rng.integers(65, 91))) for _ in range(int(self.rng.integers(3, 7))))
        return f"{cap.capitalize()} {self._c(pw['latin_company'])}"

    def dimension(self) -> str:
        n = lambda a, b: int(self.rng.integers(a, b))  # noqa: E731
        t = [f"t={self._num()}", f"t{self._num()}", f"φ{n(10, 300)}", f"{n(3, 13) * 100}×{n(3, 13) * 100}",
             f"@{self._c(['303', '455', '600', '900'])}", f"±{n(0, 50)}", f"H={n(20, 40) * 100}",
             f"CH={n(22, 30) * 100}", f"{n(1, 300)}mm", f"{n(5, 99)}.{n(0, 9)}㎡", f"FL+{n(0, 500)}",
             f"W{n(3, 20) * 100}×D{n(3, 10) * 100}×H{n(5, 25) * 100}", f"{n(3, 9) * 100}×{n(3, 9) * 100}×t{self._num()}",
             f"{n(1, 9)}-φ{n(10, 50)}@{n(100, 300)}", f"{n(10, 40)}℃"]
        s = self._c(t)
        return to_fullwidth(s) if self.rng.random() < 0.15 else s

    def note_fragment(self) -> str:
        note = self._c(self.v["notes"][self._lang()])
        L = int(self.rng.integers(6, 22))
        if len(note) <= L:
            return note
        i = int(self.rng.integers(0, len(note) - L))
        return note[i:i + L].strip() or note[:L]

    def random_charset(self) -> str:
        n = int(self.rng.integers(3, 11))
        s = "".join(self._c(self.kanji_kana) for _ in range(n))
        if self.rng.random() < 0.3:
            s += self._c(self.symbols)
        return s

    def sample(self) -> tuple[str, str]:
        """Return ``(text, source_name)``."""
        name = self.names[int(self.rng.choice(len(self.names), p=self.weights))]
        text = self.sources[name]().strip()
        return (text or "ア"), name


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file as dict."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))

"""Unit tests: crops, CTC decoding, calibration, reading order, charset, text normalisation, headers."""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration import Calibrator, ece
from src.charset import jis_level1_kanji, kana, load_charset, oov_report, save_charset
from src.crops import crop_bbox, rectify, rotate_crop
from src.ctc import decode, greedy, prefix_beam_search
from src.pipeline import canonical_header
from src.reading_order import sort_lines
from src.textnorm import fold_width, normalize


# ---------------------------------------------------------------------- crops
def _page_with_block() -> np.ndarray:
    page = np.full((200, 300), 255, np.uint8)
    page[50:70, 100:200] = 0    # a dark bar (the "text")
    page[50:55, 100:110] = 128  # marker at the bar's top-left
    return page


def test_rectify_axis_aligned_is_plain_crop() -> None:
    page = _page_with_block()
    c = rectify(page, [[100, 50], [200, 50], [200, 70], [100, 70]])
    assert c.image.shape == (20, 100)
    assert c.image.mean() < 40
    np.testing.assert_allclose(c.map_points(np.array([[0, 0], [100, 20]])), [[100, 50], [200, 70]], atol=1e-6)


def test_rectify_respects_polygon_order() -> None:
    """A polygon starting at the bottom-left corner yields a 90-degree rotated crop."""
    page = _page_with_block()
    c = rectify(page, [[100, 70], [100, 50], [200, 50], [200, 70]])  # p0->p1 goes up
    assert c.image.shape == (100, 20)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rotate_crop_keeps_page_mapping(k: int) -> None:
    page = _page_with_block()
    c = crop_bbox(page, [90, 40, 210, 80])
    r = rotate_crop(c, k)
    h, w = r.image.shape
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    mapped = r.map_points(corners)
    # rotated corners must map onto the original crop's corners (in some order)
    expected = {(90, 40), (209, 40), (209, 79), (90, 79)}
    assert {tuple(np.round(p).astype(int)) for p in mapped} == expected
    # and pixel content must agree: dark pixel in rotated crop maps to a dark page pixel
    ys, xs = np.where(r.image < 10)
    px = r.map_points(np.stack([xs[:50], ys[:50]], 1)).round().astype(int)
    assert (page[px[:, 1], px[:, 0]] < 10).all()


def test_crop_bbox_clips_to_image() -> None:
    c = crop_bbox(np.zeros((10, 10), np.uint8), [-5, -5, 20, 20])
    assert c.image.shape == (10, 10)


# ---------------------------------------------------------------------- ctc
CHARS = ["<b>", "a", "b"]


def _probs(seq: list[int], p: float = 0.9) -> np.ndarray:
    m = np.full((len(seq), 3), (1 - p) / 2)
    for t, k in enumerate(seq):
        m[t, k] = p
    return m


def test_greedy_collapses_repeats_and_blanks() -> None:
    assert greedy(_probs([1, 1, 0, 2, 2, 0]), CHARS)[0] == "ab"
    assert greedy(_probs([1, 0, 1]), CHARS)[0] == "aa"


def test_beam_agrees_with_greedy_on_peaked_input() -> None:
    probs = _probs([1, 1, 0, 2, 0, 1])
    best = prefix_beam_search(probs, CHARS, beam=8, top_k=3)
    assert best[0][0] == greedy(probs, CHARS)[0] == "aba"
    assert all(best[i][1] >= best[i + 1][1] for i in range(len(best) - 1))


def test_beam_sums_paths() -> None:
    """'a' is the most likely string even though no single path for it beats blank-only paths."""
    probs = np.array([[0.4, 0.6, 0.0], [0.4, 0.6, 0.0]]) + 1e-9
    best = prefix_beam_search(probs, CHARS, beam=4, top_k=2)
    # P('a') = .6*.6 + .6*.4 + .4*.6 = 0.84 ; P('') = .16
    assert best[0][0] == "a"
    assert np.isclose(np.exp(best[0][1]), 0.84, atol=1e-6)


def test_decode_candidates_are_relative_probs() -> None:
    d = decode(_probs([1, 0, 2], 0.7), CHARS, beam=10, top_k=5)
    assert d.text == "ab" and len(d.candidates) <= 5
    assert 0.99 <= sum(s for _, s in d.candidates) <= 1.0 + 1e-6
    assert set(d.features) >= {"mean_char_prob", "beam_margin"}


# ---------------------------------------------------------------- calibration
def test_calibrator_reduces_ece_on_overconfident_scores() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    raw = rng.uniform(0.7, 1.0, n)               # model always says >= 0.7 ...
    correct = rng.random(n) < (raw - 0.7) / 0.3  # ... but is right far less often
    feats = [{"mean_char_prob": r, "min_char_prob": r, "beam_top1": r, "beam_margin": r,
              "seq_logp_per_t": np.log(r), "length": 5.0} for r in raw]
    cal = Calibrator().fit(feats[:2000], correct[:2000])
    p = cal.predict(feats[2000:])
    assert ece(p, correct[2000:]) < 0.5 * ece(raw[2000:], correct[2000:])
    order = np.argsort(raw[2000:])
    assert np.all(np.diff(p[order]) >= -1e-9)  # monotone in the raw score here


def test_calibrator_roundtrip_and_identity(tmp_path) -> None:
    feats = [{"mean_char_prob": 0.9, "min_char_prob": 0.8, "beam_top1": 0.9, "beam_margin": 0.5,
              "seq_logp_per_t": -0.1, "length": 4.0}] * 10 + [{"mean_char_prob": 0.3, "min_char_prob": 0.1,
              "beam_top1": 0.4, "beam_margin": 0.1, "seq_logp_per_t": -2.0, "length": 4.0}] * 10
    cal = Calibrator().fit(feats, [True] * 10 + [False] * 10)
    cal.save(tmp_path / "c.json")
    np.testing.assert_allclose(Calibrator.load(tmp_path / "c.json").predict(feats), cal.predict(feats))
    assert Calibrator.load(tmp_path / "missing.json").predict(feats[:1])[0] == pytest.approx(0.9)


# --------------------------------------------------------------- reading order
def _box(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def test_reading_order_rows_then_columns() -> None:
    boxes = [_box(200, 0, 300, 20), _box(0, 40, 100, 60), _box(0, 2, 100, 22), _box(120, 41, 180, 61)]
    assert sort_lines(boxes) == [[2, 0], [1, 3]]


# --------------------------------------------------------------- charset / text
def test_charset_sources() -> None:
    assert len(jis_level1_kanji()) == 2965
    assert "亜" in jis_level1_kanji() and "腕" in jis_level1_kanji()
    k = kana()
    assert "ぁ" in k and "ヴ" in k and "ー" in k


def test_charset_roundtrip_and_oov(tmp_path) -> None:
    save_charset(["a", " ", "床"], tmp_path / "cs.txt")
    assert load_charset(tmp_path / "cs.txt") == ["a", " ", "床"]
    assert oov_report(["床a㎡", "㎡"], ["a", "床"]) == {"㎡": 2}


def test_width_folding_keeps_symbols() -> None:
    assert fold_width("ｔ＝１２．５") == "t=12.5"
    assert normalize("  ケイ酸カルシウム板　t8 ") == "ケイ酸カルシウム板 t8"
    assert normalize("18.6㎡ φ100 ±0 @455") == "18.6㎡ φ100 ±0 @455"


@pytest.mark.parametrize("text,field", [("室名", "room"), ("床", "floor"), ("巾木", "baseboard"), ("幅木", "baseboard"),
                                         ("天井仕上", "ceiling"), ("Ceiling", "ceiling"), ("Baseboard", "baseboard"),
                                         ("天丼", "ceiling"),  # 丼/井 is a classic OCR confusion
                                         ("Celling", "ceiling"), ("", None), ("備考", "remarks"),
                                         ("構造", None), ("x", None)])
def test_canonical_headers(text: str, field: str | None) -> None:
    assert canonical_header(text) == field

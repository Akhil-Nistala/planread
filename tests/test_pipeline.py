"""Vertical-text reflow and the end-to-end pipeline on one generated sheet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data_gen.render import FontPool, render_vertical
from data_gen.text_sources import load_yaml
from src.recognize import PPOCR_MODELS
from src.vertical import reflow_vertical, split_columns

ROOT = Path(__file__).resolve().parents[1]
SYNTH = load_yaml(ROOT / "configs" / "synth.yaml")


@pytest.fixture(scope="module")
def fonts() -> FontPool:
    try:
        return FontPool(SYNTH["fonts"]["ja"], SYNTH["fonts"]["latin"])
    except RuntimeError:
        pytest.skip("no Japanese font installed")


@pytest.mark.parametrize("text", ["石膏ボード", "岩綿吸音板", "天井", "ケイ酸カルシウム板"])
def test_vertical_reflow_recovers_glyph_count(fonts: FontPool, text: str) -> None:
    rng = np.random.default_rng(0)
    face = fonts.pick(text, rng)
    img = render_vertical(text, face, 32, rng)
    strip, n = reflow_vertical(img)
    assert n == len(text)
    assert strip.shape[1] > strip.shape[0]  # now horizontal


def test_split_columns_right_to_left(fonts: FontPool) -> None:
    rng = np.random.default_rng(1)
    face = fonts.pick("天井床", rng)
    a = render_vertical("天井", face, 32, rng)
    b = render_vertical("床壁", face, 32, rng)
    h = max(a.shape[0], b.shape[0])
    gap = np.full((h, 40), 255, np.uint8)
    block = np.hstack([np.pad(b, ((0, h - b.shape[0]), (0, 0)), constant_values=255), gap,
                       np.pad(a, ((0, h - a.shape[0]), (0, 0)), constant_values=255)])
    cols = split_columns(block)
    assert len(cols) == 2 and cols[0][0] > cols[1][0]  # first column read is the right-most


needs_models = pytest.mark.skipif(not PPOCR_MODELS["ppocr_v6"].exists(),
                                  reason="run scripts/download_models.py")


@needs_models
def test_end_to_end_sheet(tmp_path: Path, fonts: FontPool) -> None:
    from data_gen.make_docs import make_sheet
    from src.charset import load_charset
    from src.pipeline import OCRPipeline, load_config, load_page_images
    from src.schema import validate_output

    vocab = load_yaml(ROOT / "configs" / "vocab.yaml")
    make_sheet(0, 7, fonts, vocab, SYNTH, load_charset(ROOT / "charset" / "charset.txt"), tmp_path)
    layout_doc = json.loads((tmp_path / "sheet_000.layout.json").read_text(encoding="utf-8"))
    validate_output(layout_doc)  # PlanScope v1.0 input is valid under v1.1
    out = OCRPipeline(load_config()).process(layout_doc, load_page_images(tmp_path / "sheet_000.png", layout_doc))
    validate_output(out)
    assert out["schema_version"] == "1.1"
    page = out["pages"][0]
    assert all(r["text_source"] == "ocr" for r in page["regions"])
    cells = [c for r in page["regions"] if r["table"] for row in r["table"]["rows"] for c in row["cells"]]
    assert cells and all(c["text_source"] == "ocr" and c["text"] is not None for c in cells)
    table = next(r["table"] for r in page["regions"] if r["table"])
    assert sum(h is not None for h in table["canonical_headers"]) >= 3
    # untouched PlanScope fields
    for a, b in zip(layout_doc["pages"][0]["regions"], page["regions"]):
        assert (a["id"], a["class"], a["polygon"]) == (b["id"], b["class"], b["polygon"])

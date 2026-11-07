"""Demo site: PlanScope JSON + page image -> OCR-filled JSON (rotated, bilingual drawing text).

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from src.crops import crop_bbox, rectify
from src.pipeline import OCRPipeline, load_config, load_page_images
from src.recognize import PPOCR_MODELS
from src.textnorm import normalize

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "data" / "demo"
FONT_CANDIDATES = ["C:/Windows/Fonts/YuGothM.ttc", "C:/Windows/Fonts/msgothic.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]

st.set_page_config(page_title="Drawing OCR", layout="wide")


def ensure_demo() -> None:
    """Models + 3 demo sheets on first run."""
    if not all(p.exists() for p in PPOCR_MODELS.values()):
        with st.spinner("Downloading OCR models (first run only)…"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / "download_models.py")], check=True)
    if not (DEMO / "sheet_002.layout.json").exists():
        with st.spinner("Generating demo sheets (first run only)…"):
            subprocess.run([sys.executable, "-m", "data_gen.make_docs", "--out", str(DEMO), "--n", "3"],
                           check=True, cwd=ROOT)


@st.cache_resource(show_spinner=False)
def pipeline(engine: str) -> OCRPipeline:
    """Cached pipeline per engine."""
    return OCRPipeline(load_config(), engine)


@st.cache_data(show_spinner=False)
def run(layout_json: str, src: str, engine: str) -> dict:
    """OCR a document (cached)."""
    layout_doc = json.loads(Path(layout_json).read_text(encoding="utf-8"))
    return pipeline(engine).process(layout_doc, load_page_images(Path(src), layout_doc))


def conf_color(c: float) -> tuple[int, int, int]:
    """Green (confident) -> orange -> red (needs review)."""
    return (30, 160, 60) if c >= 0.8 else (230, 140, 0) if c >= 0.5 else (210, 40, 40)


def jp_font(size: int) -> ImageFont.ImageFont:
    """A font that can draw Japanese labels."""
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def overlay(gray: np.ndarray, page: dict, max_w: int = 2000) -> Image.Image:
    """Regions/cells coloured by calibrated confidence, with the read text."""
    s = min(1.0, max_w / gray.shape[1])
    img = Image.fromarray(cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)).convert("RGB")
    d = ImageDraw.Draw(img)
    f = jp_font(max(11, int(26 * s * 1.6)))
    for r in page["regions"]:
        if r["table"]:
            for row in r["table"]["rows"]:
                for c in row["cells"]:
                    if c["text_source"] == "ocr":
                        b = [v * s for v in c["bbox"]]
                        d.rectangle(b, outline=conf_color(c["ocr_conf"]), width=2)
            continue
        pts = [(x * s, y * s) for x, y in r["polygon"]]
        col = conf_color(r.get("ocr_conf", 1.0))
        d.polygon(pts, outline=col, width=3)
        if r.get("text"):
            label = r["text"].split("\n")[0][:24]
            x, y = pts[0]
            d.rectangle(d.textbbox((x, y - f.size - 4), label, font=f), fill=(255, 255, 255))
            d.text((x, y - f.size - 4), label, fill=col, font=f)
    return img


ensure_demo()
st.title("PlanRead — rotated, bilingual OCR for building drawings")
st.caption("Stage 2 of the asbestos-material screening pipeline: reads every `needs_ocr` region and table cell from "
           "PlanScope's JSON (oriented crop → orientation 0/90/180/270/縦書き → line detection → CTC recognition "
           "with top-5 alternatives → calibrated confidence) and writes the same JSON contract back (v1.1). "
           "Demo sheets are **synthetic**; PlanScope's layout is taken as given (oracle regions).")

with st.sidebar:
    st.header("Input")
    sheets = sorted(p.name[: -len(".layout.json")] for p in DEMO.glob("*.layout.json"))
    choice = st.radio("Demo sheet", sheets)
    st.caption("…or upload PlanScope output + its page image / PDF")
    up_json = st.file_uploader("PlanScope JSON", type=["json"])
    up_src = st.file_uploader("Page image or PDF", type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"])
    st.header("Recogniser")
    engines = [e for e, p in PPOCR_MODELS.items() if p.exists()] + (["crnn"] if (ROOT / "models" / "crnn.pt").exists() else [])
    engine = st.selectbox("Engine", engines, index=0)

gt = None
if up_json is not None and up_src is not None:
    tmp = Path(tempfile.mkdtemp())
    layout_json, src = tmp / up_json.name, tmp / up_src.name
    layout_json.write_bytes(up_json.getvalue())
    src.write_bytes(up_src.getvalue())
else:
    layout_json, src = DEMO / f"{choice}.layout.json", DEMO / f"{choice}.png"
    gt_path = DEMO / f"{choice}.gt.json"
    gt = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else None

with st.spinner(f"Running OCR with {engine}…"):
    try:
        doc = run(str(layout_json), str(src), engine)
    except Exception as e:  # show errors in the UI
        st.error(f"Failed: {e}")
        st.stop()

page = doc["pages"][0]
gray = load_page_images(Path(src), json.loads(Path(layout_json).read_text(encoding="utf-8")))[page["page"]]
regions = [r for r in page["regions"] if r["text_source"] == "ocr"]
cells = [c for r in regions if r["table"] for row in r["table"]["rows"] for c in row["cells"] if c["text_source"] == "ocr"]
items = [r for r in regions if not r["table"]] + cells
low = sum(1 for x in items if x["ocr_conf"] < 0.5)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Regions OCR'd", len(regions))
c2.metric("Table cells OCR'd", len(cells))
c3.metric("Low confidence (<0.5) → review", low)
c4.metric("OCR time", f"{page.get('timings', {}).get('ocr_s', 0):.1f} s")
if gt:
    hits, n = 0, 0
    for r in page["regions"]:
        g = gt["regions"][r["id"]]
        if r["table"]:
            for i, row in enumerate(r["table"]["rows"]):
                for j, c in enumerate(row["cells"]):
                    hits += normalize(c["text"] or "") == normalize(g["cells"][i][j])
                    n += 1
        else:
            hits += normalize(r["text"] or "") == normalize(g["text"])
            n += 1
    st.info(f"Against ground truth: **{hits}/{n}** items exactly right ({hits / max(1, n):.0%}). "
            "Green ≥ 0.8 calibrated confidence, orange 0.5–0.8, red < 0.5.")

st.image(overlay(gray, page), width="stretch")

tab_table, tab_regions, tab_json = st.tabs(["Finish schedule (仕上表)", "Regions", "JSON"])
with tab_table:
    for r in regions:
        if not r["table"]:
            continue
        t = r["table"]
        st.write("**Canonical headers:**", " · ".join(f"{h} → `{c}`" for h, c in zip(t["headers"], t["canonical_headers"])))
        body = [row["cells"] for row in t["rows"] if not any(c.get("is_header") for c in row["cells"])]
        st.dataframe([{(t["canonical_headers"][j] or f"col{j}"): f"{c['text']}  ({c['ocr_conf']:.2f})"
                       for j, c in enumerate(row)} for row in body], width="stretch")
with tab_regions:
    for r in regions:
        if r["table"]:
            continue
        a, b = st.columns([1, 2])
        crop = rectify(gray, r["polygon"], 4).image
        a.image(crop, caption=f"{r['id']} · {r['class']}", width="stretch")
        ok = ""
        if gt:
            g = gt["regions"][r["id"]]["text"]
            ok = " ✅" if normalize(r["text"] or "") == normalize(g) else f" ❌ (GT: `{g}`)"
        b.markdown(f"**{r['text'] or '∅'}**{ok}  \nconf **{r['ocr_conf']:.2f}** · orientation `{r.get('ocr_orientation')}`"
                   f" · flags {r.get('ocr_flags')}")
        if len(r["ocr_lines"]) == 1:
            b.caption("Top-5: " + " | ".join(f"{x['text']} ({x['score']:.2f})" for x in r["ocr_lines"][0]["alternatives"]))
with tab_json:
    js = json.dumps(doc, ensure_ascii=False, indent=1)
    st.download_button("Download JSON (contract v1.1)", js, file_name=f"{Path(doc['document']).stem}.ocr.json")
    st.code(js[:50_000], language="json")

"""PlanRead pipeline: PlanScope JSON (+ page images) -> same JSON with OCR text filled in.

For every region / table cell with ``text_source == "needs_ocr"``:

  crop + rectify  ->  orientation (0/90/180/270/vertical)  ->  line detection (DBNet) +
  reading order   ->  recognition (CTC, beam top-k)       ->  calibrated confidence

The output keeps PlanScope's contract (v1.0 fields untouched) and adds the OCR fields of
contract v1.1 (see src/schema.py). Vector-text regions are passed through unchanged.

Usage::

    python -m src.pipeline --layout-json doc.json --source doc.pdf --out doc.ocr.json
    python -m src.pipeline --layout-json data/docs/sheet_000.layout.json --source data/docs/sheet_000.png --out out.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .calibration import Calibrator
from .crops import Crop, crop_bbox, rectify, rotate_crop
from .ctc import Decoded
from .orientation import LABELS, OrientationClassifier
from .reading_order import sort_lines
from .recognize import Recognizer, build_recognizer
from .schema import SCHEMA_VERSION, validate_output
from .textnorm import normalize
from .vertical import reflow_vertical

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger(__name__)

# header text (normalised, spaces removed) -> canonical field
CANONICAL_HEADERS = {
    "室名": "room", "部屋名": "room", "室": "room", "room": "room", "roomname": "room",
    "床": "floor", "床仕上": "floor", "床仕上げ": "floor", "floor": "floor",
    "幅木": "baseboard", "巾木": "baseboard", "baseboard": "baseboard", "skirting": "baseboard",
    "壁": "wall", "壁仕上": "wall", "壁仕上げ": "wall", "wall": "wall", "walls": "wall",
    "天井": "ceiling", "天井仕上": "ceiling", "天井仕上げ": "ceiling", "ceiling": "ceiling",
    "備考": "remarks", "remarks": "remarks", "notes": "remarks",
}


def canonical_header(text: str | None) -> str | None:
    """Map a header cell to a canonical field (exact, then 1-edit tolerant for OCR noise)."""
    if not text:
        return None
    key = normalize(text).replace(" ", "").lower()
    if key in CANONICAL_HEADERS:
        return CANONICAL_HEADERS[key]
    from .crnn import levenshtein

    best = min(CANONICAL_HEADERS, key=lambda k: levenshtein(k, key))
    return CANONICAL_HEADERS[best] if len(key) >= 2 and levenshtein(best, key) <= max(1, len(best) // 4) else None


@dataclass
class LineResult:
    """A recognised line in page coordinates."""

    decoded: Decoded
    polygon: list[list[float]]
    conf: float
    row: int


class OCRPipeline:
    """Holds models; ``process`` fills one PlanScope document."""

    def __init__(self, cfg: dict[str, Any], engine: str | None = None) -> None:
        self.cfg = cfg
        rc = cfg["recognition"]
        self.engine = engine or rc["engine"]
        self.rec: Recognizer = build_recognizer(self.engine, rc["beam_width"], rc["top_k"],
                                                ROOT / rc["crnn_model"])
        self.cal = Calibrator.load(ROOT / rc["calibrator"].format(engine=self.engine))
        self.orient = OrientationClassifier(ROOT / cfg["orientation"]["model"], cfg["orientation"]["input_size"])
        self._det = None

    @property
    def det(self):
        """Lazy DBNet (not needed for single-line crops)."""
        if self._det is None:
            from .detect_lines import DBNetDetector

            self._det = DBNetDetector()
        return self._det

    # ------------------------------------------------------------------ pieces
    def _recognize(self, crops: list[np.ndarray]) -> tuple[list[Decoded], np.ndarray]:
        decs = self.rec.recognize(crops)
        conf = self.cal.predict([d.features for d in decs]) if decs else np.zeros(0)
        return decs, conf

    def _lines(self, crop: Crop, single_line: bool) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """(line image, line polygon in crop coords, row index) in reading order."""
        h, w = crop.image.shape
        whole = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
        if single_line:
            return [(crop.image, whole, 0)]
        polys = self.det.detect(crop.image)
        if not polys:
            return [(crop.image, whole, 0)]
        rows = sort_lines(polys, self.cfg["detection"]["line_merge_y_overlap"])
        out = []
        for r, idxs in enumerate(rows):
            for i in idxs:
                out.append((rectify(crop.image, polys[i], pad=3).image, polys[i], r))
        return out

    def _orient(self, crop: Crop, is_cell: bool) -> tuple[str, Crop, list[str]]:
        """Pick the orientation and return the corrected crop."""
        oc = self.cfg["orientation"]
        probs = self.orient.predict(crop.image)
        order = list(np.argsort(-probs))
        flags: list[str] = []
        label = LABELS[order[0]]
        h, w = crop.image.shape
        if label == "vertical" and h / max(w, 1) < oc["vertical"]["min_aspect"]:
            label = LABELS[order[1]]  # vertical needs a tall crop
        if probs[order[0]] < oc["low_conf"]:
            # ambiguous: let the recogniser decide between the top-2 candidates
            cands = [LABELS[i] for i in order[:2] if LABELS[i] != "vertical"] or ["0"]
            scored = []
            for lab in cands:
                img = rotate_crop(crop, -LABELS.index(lab)).image
                _, conf = self._recognize([img])
                scored.append((float(conf[0]) if len(conf) else 0.0, lab))
            label = max(scored)[1]
            flags.append("orientation_rescored")
        if label == "vertical":
            return label, crop, flags
        return label, rotate_crop(crop, -LABELS.index(label)), flags

    def ocr_item(self, page_img: np.ndarray, polygon=None, bbox=None, is_cell: bool = False) -> dict[str, Any]:
        """OCR one region (polygon) or cell (bbox); returns the fields to merge into the JSON item."""
        pad = self.cfg["crops"]["pad_px"]
        crop = crop_bbox(page_img, bbox, pad=0) if bbox is not None else rectify(page_img, polygon, pad=pad)
        if min(crop.image.shape) < self.cfg["crops"]["min_side_px"] or (crop.image < 160).sum() < 5:
            return {"text": "", "text_source": "ocr", "ocr_conf": 0.0, "ocr_orientation": "0",
                    "ocr_lines": [], "ocr_flags": ["empty"]}
        if is_cell:  # strip table rules so they are not read as characters
            crop = Crop(_remove_borders(crop.image), crop.to_page)
        label, crop, flags = self._orient(crop, is_cell)
        if label == "vertical":
            if self.cfg["orientation"]["vertical"]["handling"] == "flag":
                return {"text": "", "text_source": "ocr", "ocr_conf": 0.0, "ocr_orientation": "vertical",
                        "ocr_lines": [], "ocr_flags": flags + ["vertical_unsupported"]}
            strip, _ = reflow_vertical(crop.image)
            h, w = crop.image.shape
            lines = [(strip, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float), 0)]
            flags.append("vertical_split")
        else:
            h, w = crop.image.shape
            single = is_cell or h / max(w, 1) < self.cfg["detection"]["single_line_max_aspect"] and h < 90
            lines = self._lines(crop, single)
        decs, confs = self._recognize([l[0] for l in lines])
        results = [LineResult(d, crop.map_points(p).round(2).tolist(), float(c), r)
                   for d, c, (_, p, r) in zip(decs, confs, lines) if normalize(d.text)]
        if not results:
            return {"text": "", "text_source": "ocr", "ocr_conf": 0.0, "ocr_orientation": label,
                    "ocr_lines": [], "ocr_flags": flags + ["empty"]}
        rows: dict[int, list[str]] = {}
        for lr in results:
            rows.setdefault(lr.row, []).append(normalize(lr.decoded.text))
        text = "\n".join(" ".join(v) for _, v in sorted(rows.items()))
        conf = min(lr.conf for lr in results)
        if conf < 0.5:
            flags.append("low_conf")
        return {
            "text": text, "text_source": "ocr", "ocr_conf": round(conf, 4), "ocr_orientation": label,
            "ocr_lines": [{
                "text": normalize(lr.decoded.text), "polygon": lr.polygon, "conf": round(lr.conf, 4),
                "raw_conf": round(min(1.0, max(0.0, lr.decoded.raw_conf)), 4),
                "alternatives": [{"text": normalize(t), "score": round(min(1.0, s), 4)}
                                 for t, s in lr.decoded.candidates] or [{"text": normalize(lr.decoded.text), "score": 1.0}],
            } for lr in results],
            "ocr_flags": flags,
        }

    # ------------------------------------------------------------------ document
    def process(self, layout_doc: dict[str, Any], page_images: dict[int, np.ndarray]) -> dict[str, Any]:
        """Return a new document with every needs_ocr item filled (input is not modified)."""
        doc = copy.deepcopy(layout_doc)
        doc["schema_version"] = SCHEMA_VERSION
        for page in doc["pages"]:
            todo = [r for r in page["regions"] if r["text_source"] == "needs_ocr" or _table_needs_ocr(r)]
            if not todo:
                continue
            img = page_images[page["page"]]
            if img.shape[0] != page["height"] or img.shape[1] != page["width"]:
                raise ValueError(f"page {page['page']}: image {img.shape[:2]} != JSON {page['height']}x{page['width']}")
            t0 = time.perf_counter()
            for r in todo:
                if r["table"] is not None:
                    self._fill_table(img, r)
                elif r["text_source"] == "needs_ocr":
                    r.update(self.ocr_item(img, polygon=r["polygon"]))
            page.setdefault("timings", {})["ocr_s"] = round(time.perf_counter() - t0, 4)
        validate_output(doc)
        return doc

    def _fill_table(self, img: np.ndarray, region: dict[str, Any]) -> None:
        t = region["table"]
        confs = []
        for row in t["rows"]:
            for c in row["cells"]:
                if c["text_source"] == "needs_ocr":
                    c.update(self.ocr_item(img, bbox=c["bbox"], is_cell=True))
                if c["text_source"] == "ocr":
                    confs.append(c["ocr_conf"])
        header_row = next((row for row in t["rows"] if any(c.get("is_header") for c in row["cells"])), None)
        if header_row is not None and all(h is None for h in t["headers"]):
            t["headers"] = [c["text"] for c in header_row["cells"]]
            t["canonical_headers"] = [canonical_header(h) for h in t["headers"]]
        elif any(h is not None for h in t["headers"]) and all(c is None for c in t["canonical_headers"]):
            t["canonical_headers"] = [canonical_header(h) for h in t["headers"]]
        if region["text_source"] == "needs_ocr":
            region.update({
                "text": "\n".join(" ".join(c["text"] or "" for c in row["cells"]) for row in t["rows"]),
                "text_source": "ocr", "ocr_conf": round(min(confs), 4) if confs else 0.0,
                "ocr_lines": [], "ocr_flags": ["text_from_cells"],
            })


def _table_needs_ocr(region: dict[str, Any]) -> bool:
    t = region.get("table")
    return bool(t) and any(c["text_source"] == "needs_ocr" for row in t["rows"] for c in row["cells"])


def _remove_borders(gray: np.ndarray) -> np.ndarray:
    """Whiten long horizontal/vertical rules (table borders) inside a cell crop."""
    import cv2

    g = gray.copy()
    inv = (g < 150).astype(np.uint8) * 255
    h, w = g.shape
    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * 0.6)), 1)))
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(h * 0.8)))))
    mask = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    g[mask > 0] = 255
    return g


def load_page_images(source: Path, layout_doc: dict[str, Any]) -> dict[int, np.ndarray]:
    """Page images at the JSON's DPI: render PDFs with PyMuPDF (same as PlanScope) or load an image."""
    import cv2

    if source.suffix.lower() == ".pdf":
        import pymupdf

        out = {}
        with pymupdf.open(source) as pdf:
            for p in layout_doc["pages"]:
                pix = pdf[p["page"] - 1].get_pixmap(dpi=p["dpi"], colorspace=pymupdf.csGRAY, alpha=False)
                out[p["page"]] = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
        return out
    img = cv2.imdecode(np.fromfile(str(source), np.uint8), cv2.IMREAD_GRAYSCALE)
    return {1: img}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load configs/ocr.yaml."""
    return yaml.safe_load((path or ROOT / "configs" / "ocr.yaml").read_text(encoding="utf-8"))


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout-json", required=True, type=Path, help="PlanScope output JSON")
    ap.add_argument("--source", required=True, type=Path, help="the PDF or page image the JSON refers to")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--engine", default=None, help="override recognition.engine")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    layout_doc = json.loads(args.layout_json.read_text(encoding="utf-8"))
    pipe = OCRPipeline(cfg, args.engine)
    t0 = time.perf_counter()
    out = pipe.process(layout_doc, load_page_images(args.source, layout_doc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    n = sum(1 for p in out["pages"] for r in p["regions"] if r["text_source"] == "ocr")
    print(f"{args.layout_json.name}: {n} regions OCR'd in {time.perf_counter() - t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()

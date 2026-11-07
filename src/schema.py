"""Output contract v1.1 = PlanScope contract v1.0 + OCR fields (strictly additive).

The v1.1 schema is *derived* from the vendored PlanScope schema
(``contracts/planscope_schema_v1.0.json``, copied from planscope@fc8cd0a) so
nothing else can drift:

* ``text_source`` enum gains ``"ocr"``
* regions and cells gain optional ``ocr_conf`` (calibrated, 0-1), ``ocr_orientation`` and
  ``ocr_lines`` (per line: text, polygon in page px, calibrated conf, top-k alternatives)
* ``schema_version`` may be ``"1.0"`` (untouched PlanScope output) or ``"1.1"``

Every v1.0 document is a valid v1.1 document. ``python -m src.schema`` regenerates
``contracts/schema_v1.1.json`` (a test checks the committed file is up to date).
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
LAYOUT_SCHEMA = ROOT / "contracts" / "planscope_schema_v1.0.json"
OCR_SCHEMA = ROOT / "contracts" / "schema_v1.1.json"
SCHEMA_VERSION = "1.1"
ORIENTATIONS = ["0", "90", "180", "270", "vertical"]


class OutputValidationError(ValueError):
    """Raised when an output violates the contract."""


def build_schema() -> dict[str, Any]:
    """Derive the v1.1 schema from PlanScope's v1.0 schema."""
    s = copy.deepcopy(json.loads(LAYOUT_SCHEMA.read_text(encoding="utf-8")))
    s["$id"] = s["$id"].replace("drawing-layout.schema.json", "drawing-layout-ocr.schema.json")
    s["title"] = "PlanScope v1.0 layout + PlanRead OCR fields"
    s["properties"]["schema_version"] = {"type": "string", "enum": ["1.0", SCHEMA_VERSION]}
    d = s["$defs"]
    d["text_source"]["enum"] = ["vector", "needs_ocr", "ocr"]
    d["alternative"] = {
        "type": "object", "additionalProperties": False, "required": ["text", "score"],
        "properties": {"text": {"type": "string"},
                       "score": {"type": "number", "minimum": 0, "maximum": 1,
                                 "description": "Relative beam probability (alternatives of a line sum to <= 1)."}},
    }
    d["ocr_line"] = {
        "type": "object", "additionalProperties": False,
        "required": ["text", "polygon", "conf", "raw_conf", "alternatives"],
        "properties": {
            "text": {"type": "string"},
            "polygon": {"$ref": "#/$defs/polygon"},
            "conf": {"type": "number", "minimum": 0, "maximum": 1, "description": "Calibrated P(line exactly correct)."},
            "raw_conf": {"type": "number", "minimum": 0, "maximum": 1, "description": "Uncalibrated model score."},
            "alternatives": {"type": "array", "items": {"$ref": "#/$defs/alternative"}, "minItems": 1},
        },
    }
    ocr_props = {
        "ocr_conf": {"type": "number", "minimum": 0, "maximum": 1,
                     "description": "Calibrated confidence of the whole region/cell (min over its lines; 0 if empty)."},
        "ocr_orientation": {"type": "string", "enum": ORIENTATIONS,
                            "description": "Text orientation found before recognition (CCW degrees, or vertical 縦書き)."},
        "ocr_lines": {"type": "array", "items": {"$ref": "#/$defs/ocr_line"}},
        "ocr_flags": {"type": "array", "items": {"type": "string"},
                      "description": "e.g. 'vertical_split', 'empty', 'low_conf'."},
    }
    d["cell"]["properties"].update(copy.deepcopy(ocr_props))
    d["region"]["properties"].update(copy.deepcopy(ocr_props))
    return s


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(OCR_SCHEMA.read_text(encoding="utf-8")) if OCR_SCHEMA.exists() else build_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _check_item(obj: dict, where: str, errors: list[str]) -> None:
    src = obj["text_source"]
    if (src == "needs_ocr") != (obj["text"] is None):
        errors.append(f"{where}: text must be null iff text_source == 'needs_ocr'")
    has_ocr = any(k in obj for k in ("ocr_conf", "ocr_lines"))
    if src == "ocr" and not ("ocr_conf" in obj and "ocr_lines" in obj):
        errors.append(f"{where}: text_source 'ocr' requires ocr_conf and ocr_lines")
    if src != "ocr" and has_ocr:
        errors.append(f"{where}: ocr_* fields only allowed when text_source == 'ocr'")


def semantic_errors(doc: dict[str, Any]) -> list[str]:
    """Invariants beyond JSON Schema (PlanScope's plus the OCR ones)."""
    errors: list[str] = []
    for p in doc["pages"]:
        tag = f"page {p['page']}"
        if p["needs_ocr"] != (p["route"] == "raster"):
            errors.append(f"{tag}: needs_ocr must equal (route == 'raster')")
        if p["route"] == "raster" and p["words"]:
            errors.append(f"{tag}: raster pages must not carry text-layer words")
        for r in p["regions"]:
            rtag = f"{tag} region {r['id']}"
            _check_item(r, rtag, errors)
            t = r["table"]
            if t is None:
                continue
            if r["class"] != "schedule_table":
                errors.append(f"{rtag}: only schedule_table regions may carry a table")
            if len(t["headers"]) != len(t["canonical_headers"]):
                errors.append(f"{rtag}: headers and canonical_headers differ in length")
            for i, row in enumerate(t["rows"]):
                for j, c in enumerate(row["cells"]):
                    _check_item(c, f"{rtag} cell ({i},{j})", errors)
    return errors


def validate_output(doc: dict[str, Any]) -> None:
    """Validate against the v1.1 schema + semantic invariants; raise on failure."""
    errs = sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path))
    msgs = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errs]
    if not msgs:
        msgs = semantic_errors(doc)
    if msgs:
        raise OutputValidationError("; ".join(msgs[:20]))


if __name__ == "__main__":
    OCR_SCHEMA.write_text(json.dumps(build_schema(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OCR_SCHEMA}")

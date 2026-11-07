"""Output contract: v1.1 is a strict, additive extension of PlanScope's v1.0."""

from __future__ import annotations

import copy
import json

import pytest

from src.schema import LAYOUT_SCHEMA, OCR_SCHEMA, OutputValidationError, build_schema, validate_output


def _layout_doc() -> dict:
    cell = {"text": None, "bbox": [0, 0, 10, 10], "text_source": "needs_ocr", "is_header": True}
    return {"schema_version": "1.0", "document": "x.png", "pages": [{
        "page": 1, "route": "raster", "dpi": 300, "width": 100, "height": 100, "needs_ocr": True,
        "routing": None, "words": [],
        "regions": [
            {"id": "r1", "class": "text_callout", "conf": 0.9, "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]],
             "text": None, "text_source": "needs_ocr", "table": None},
            {"id": "r2", "class": "schedule_table", "conf": 0.9, "polygon": [[0, 0], [50, 0], [50, 50], [0, 50]],
             "text": None, "text_source": "needs_ocr",
             "table": {"headers": [None], "canonical_headers": [None], "rows": [{"cells": [cell]}]}},
        ]}]}


def _ocr_fields(text: str = "石膏ボード") -> dict:
    return {"text": text, "text_source": "ocr", "ocr_conf": 0.8, "ocr_orientation": "90", "ocr_flags": [],
            "ocr_lines": [{"text": text, "polygon": [[0, 0], [10, 0], [10, 5], [0, 5]], "conf": 0.8, "raw_conf": 0.95,
                           "alternatives": [{"text": text, "score": 0.9}, {"text": "石膏ボ一ド", "score": 0.1}]}]}


def test_committed_schema_is_up_to_date() -> None:
    assert json.loads(OCR_SCHEMA.read_text(encoding="utf-8")) == build_schema(), "run `python -m src.schema`"


def test_v1_0_documents_remain_valid() -> None:
    validate_output(_layout_doc())


def test_extension_is_additive() -> None:
    """Every PlanScope definition is preserved except the explicitly extended ones."""
    a = json.loads(LAYOUT_SCHEMA.read_text(encoding="utf-8"))
    b = build_schema()
    for name, d in a["$defs"].items():
        if name in ("text_source", "cell", "region"):
            continue
        assert b["$defs"][name] == d, name
    assert set(a["$defs"]["text_source"]["enum"]) < set(b["$defs"]["text_source"]["enum"])
    for name in ("cell", "region"):
        assert b["$defs"][name]["required"] == a["$defs"][name]["required"]
        for k, v in a["$defs"][name]["properties"].items():
            if k != "text_source":
                assert b["$defs"][name]["properties"][k] == v


def test_filled_document_valid() -> None:
    d = _layout_doc()
    d["schema_version"] = "1.1"
    d["pages"][0]["regions"][0].update(_ocr_fields())
    d["pages"][0]["regions"][1].update(_ocr_fields("室名"))
    d["pages"][0]["regions"][1]["table"]["rows"][0]["cells"][0].update(_ocr_fields("室名"))
    validate_output(d)


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r.update({"text_source": "ocr", "text": "x"}), "requires ocr_conf"),
    (lambda r: r.update({"ocr_conf": 0.5}), "only allowed"),
    (lambda r: r.update(_ocr_fields()) or r.update({"ocr_conf": 1.5}), "ocr_conf"),
    (lambda r: r.update(_ocr_fields()) or r["ocr_lines"][0].update({"alternatives": []}), "alternatives"),
    (lambda r: r.update(_ocr_fields()) or r.update({"ocr_orientation": "45"}), "ocr_orientation"),
])
def test_contract_violations(mutate, msg) -> None:
    d = copy.deepcopy(_layout_doc())
    mutate(d["pages"][0]["regions"][0])
    with pytest.raises(OutputValidationError, match=msg):
        validate_output(d)

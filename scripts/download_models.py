"""Download pretrained PP-OCR ONNX models (pinned to RapidOCR model zoo v3.10.0, SHA256-checked).

Usage::

    python scripts/download_models.py            # -> models/ppocr/
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx"
MODELS = {
    # text-line detection (DBNet)
    "det_v6_small.onnx": (f"{BASE}/PP-OCRv6/det/PP-OCRv6_det_small.onnx",
                          "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    # recognition: multilingual PP-OCRv6 (ja+en+zh) and Japanese-only PP-OCRv4
    "rec_v6_small.onnx": (f"{BASE}/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
                          "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
    "rec_japan_v4_mobile.onnx": (f"{BASE}/PP-OCRv4/rec/japan_PP-OCRv4_rec_mobile.onnx",
                                 "e1075a67dba758ecfc7ebc78a10ae61c95ac8fb66a9c86fab5541e33f085cb7a"),
}


def sha256(p: Path) -> str:
    """File SHA256."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Download missing models and verify checksums."""
    out = ROOT / "models" / "ppocr"
    out.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in MODELS.items():
        dst = out / name
        if not dst.exists() or sha256(dst) != digest:
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(url, dst)
        if sha256(dst) != digest:
            sys.exit(f"checksum mismatch for {name}")
        print(f"ok  {name}  ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

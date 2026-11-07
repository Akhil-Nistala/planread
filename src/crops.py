"""Crop extraction + rectification of PlanScope regions / cells.

Every crop carries ``to_page``: a 3x3 homography mapping crop pixels back to page pixels,
so line polygons found inside a crop can be reported in page coordinates (PlanScope's
coordinate convention: pixels of the rendered page at ``pages[].dpi``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass
class Crop:
    """An axis-aligned, rectified crop and its mapping back to the page."""

    image: np.ndarray       # H x W (grayscale uint8)
    to_page: np.ndarray     # 3x3 homography crop -> page

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        """Map Nx2 crop points to page coordinates."""
        p = np.hstack([np.asarray(pts, np.float64), np.ones((len(pts), 1))]) @ self.to_page.T
        return p[:, :2] / p[:, 2:3]


def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB/BGR/gray -> gray uint8."""
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def expand_polygon(poly: np.ndarray, pad: float) -> np.ndarray:
    """Grow a 4-point (possibly rotated) box by ``pad`` px along its own axes."""
    p = np.asarray(poly, np.float64)
    ux = p[1] - p[0]
    uy = p[3] - p[0]
    ux /= max(np.linalg.norm(ux), 1e-6)
    uy /= max(np.linalg.norm(uy), 1e-6)
    d = [(-ux - uy), (ux - uy), (ux + uy), (-ux + uy)]
    return p + pad * np.array(d)


def rectify(image: np.ndarray, polygon: Sequence[Sequence[float]], pad: float = 0.0) -> Crop:
    """Perspective-rectify a 4-point polygon (clockwise from its own top-left) to an upright crop.

    The polygon's point order defines the output orientation: edge p0->p1 becomes the top edge.
    """
    src = expand_polygon(np.asarray(polygon, np.float64), pad) if pad else np.asarray(polygon, np.float64)
    w = int(round(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3]))))
    h = int(round(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1]))))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64)
    M = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    out = cv2.warpPerspective(to_gray(image), M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return Crop(out, np.linalg.inv(M))


def crop_bbox(image: np.ndarray, bbox: Sequence[float], pad: float = 0.0) -> Crop:
    """Axis-aligned crop [x0, y0, x1, y1] (clipped to the image)."""
    H, W = image.shape[:2]
    x0, y0 = max(0, int(np.floor(bbox[0] - pad))), max(0, int(np.floor(bbox[1] - pad)))
    x1, y1 = min(W, int(np.ceil(bbox[2] + pad))), min(H, int(np.ceil(bbox[3] + pad)))
    out = to_gray(image)[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].copy()
    T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
    return Crop(out, T)


def rotate_crop(crop: Crop, k: int) -> Crop:
    """Rotate a crop by k*90 degrees counter-clockwise, keeping the mapping to the page."""
    k %= 4
    if k == 0:
        return crop
    h, w = crop.image.shape[:2]
    img = np.ascontiguousarray(np.rot90(crop.image, k))
    # point in rotated image -> point in original crop
    if k == 1:    # CCW 90: (x', y') = (y, w-1-x)  =>  x = w-1-y', y = x'
        R = np.array([[0, -1, w - 1], [1, 0, 0], [0, 0, 1]], np.float64)
    elif k == 2:
        R = np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], np.float64)
    else:         # CW 90: (x', y') = (h-1-y, x)  =>  x = y', y = h-1-x'
        R = np.array([[0, 1, 0], [-1, 0, h - 1], [0, 0, 1]], np.float64)
    return Crop(img, crop.to_page @ R)

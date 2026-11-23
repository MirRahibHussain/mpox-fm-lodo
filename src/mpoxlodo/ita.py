"""Individual Typology Angle (ITA) from lesion-free skin pixels.

ITA = arctan((L* - 50) / b*) * 180 / pi, computed on CIELAB pixels that look
like skin (YCrCb rule) and are not lesion-like (most red / darkest pixels
removed). The median over remaining pixels is the image's ITA.
Bands (Del Bino et al.): >55 very light, 41-55 light, 28-41 intermediate,
10-28 tan, -30-10 brown, < -30 dark. We collapse them into three groups that
approximate Fitzpatrick I-II (>41), III-IV (10..41) and V-VI (<10).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    r, g, b = [rgb[..., i].astype(np.float64) for i in range(3)]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cr = (r - y) * 0.713 + 128
    cb = (b - y) * 0.564 + 128
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127) & (y > 30)


def ita_of_image(path: str, max_side: int = 256, lesion_quantile: float = 0.25) -> float:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        rgb = np.asarray(im)
    mask = skin_mask(rgb)
    if mask.sum() < 200:
        mask = np.ones(rgb.shape[:2], dtype=bool)
    lab = rgb2lab(rgb / 255.0)
    L, a, b = lab[..., 0][mask], lab[..., 1][mask], lab[..., 2][mask]
    # drop lesion-like pixels: the reddest and the darkest quarter
    keep = (a <= np.quantile(a, 1 - lesion_quantile)) & (L >= np.quantile(L, lesion_quantile))
    if keep.sum() < 50:
        keep = np.ones_like(L, dtype=bool)
    ita = np.degrees(np.arctan2(L[keep] - 50.0, b[keep]))
    return float(np.median(ita))


def ita_band(ita: float) -> str:
    if np.isnan(ita):
        return "unknown"
    if ita > 41:
        return "I-II"
    if ita >= 10:
        return "III-IV"
    return "V-VI"


def fitzpatrick_to_band(fst: int) -> str:
    if fst in (1, 2):
        return "I-II"
    if fst in (3, 4):
        return "III-IV"
    if fst in (5, 6):
        return "V-VI"
    return "unknown"

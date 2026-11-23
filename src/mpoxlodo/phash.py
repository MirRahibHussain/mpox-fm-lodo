"""Perceptual (pHash) and difference (dHash) hashes as 64-bit integers.

Implemented with NumPy/SciPy so no extra package is needed.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fft import dctn


def _gray(path: str, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L").resize(size, Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.float64)


def _bits_to_int(bits: np.ndarray) -> int:
    v = 0
    for b in bits.ravel():
        v = (v << 1) | int(b)
    return v


def phash(path: str) -> int:
    a = _gray(path, (32, 32))
    d = dctn(a, norm="ortho")[:8, :8]
    med = np.median(d.ravel()[1:])  # ignore the DC term
    return _bits_to_int(d > med)


def dhash(path: str) -> int:
    a = _gray(path, (9, 8))
    return _bits_to_int(a[:, 1:] > a[:, :-1])


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = x.view(np.uint8).reshape(x.shape + (8,))
    return _POP8[b].sum(-1)


def hamming_pairs(hashes: np.ndarray, max_dist: int, block: int = 2048):
    """Yield (i, j, dist) for all i<j with Hamming distance <= max_dist."""
    h = hashes.astype(np.uint64)
    n = len(h)
    for s in range(0, n, block):
        e = min(n, s + block)
        x = np.bitwise_xor(h[s:e, None], h[None, :])
        d = _popcount64(x)
        ii, jj = np.nonzero(d <= max_dist)
        keep = jj > (ii + s)
        for i, j in zip(ii[keep] + s, jj[keep]):
            yield int(i), int(j), int(d[i - s, j])

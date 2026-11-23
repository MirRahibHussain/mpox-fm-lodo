"""Extract and cache frozen embeddings for every indexed image, once per model."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import ImageTable, build_transforms
from .models import REGISTRY, load_backbone


def _stale(fn: str, key: str) -> bool:
    """True when the cached features were made by an older loader revision."""
    want = REGISTRY[key].rev if key in REGISTRY else 1
    try:
        z = np.load(fn, allow_pickle=True)
        have = int(z["rev"]) if "rev" in z.files else 1
    except Exception:
        return True
    return have != want


def invalidate(out_dir: str, key: str) -> None:
    """Remove cached features and every frozen-feature prediction of a model."""
    import shutil
    fn = os.path.join(out_dir, f"{key}.npz")
    if os.path.exists(fn):
        os.remove(fn)
    preds = os.path.join(os.path.dirname(os.path.abspath(out_dir)), "preds")
    for regime in ("lp", "zs"):
        d = os.path.join(preds, regime, key)
        if os.path.isdir(d):
            shutil.rmtree(d)
    print(f"[{key}] cached features were from an older loader: removed features and lp/zs predictions")


def extract(index: pd.DataFrame, key: str, out_dir: str, batch_size: int = 64,
            num_workers: int = 2, device: str = "cuda", limit: int = 0) -> str:
    import torch
    from torch.utils.data import DataLoader

    os.makedirs(out_dir, exist_ok=True)
    fn = os.path.join(out_dir, f"{key}.npz")
    if os.path.exists(fn) and not limit and _stale(fn, key):
        invalidate(out_dir, key)
    if os.path.exists(fn):
        return fn
    model, spec, _ = load_backbone(key)
    model = model.to(device).eval()
    df = index if not limit else index.head(limit)
    ds = ImageTable(df["path"], np.zeros(len(df)), build_transforms(spec.img_size, spec.mean, spec.std, False))
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    feats = []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
        for x, _ in dl:
            feats.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    emb = np.concatenate(feats).astype(np.float32)
    np.savez_compressed(fn, ids=np.array(df["image_id"]), emb=emb, rev=np.array(spec.rev))
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return fn


def load(out_dir: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(os.path.join(out_dir, f"{key}.npz"), allow_pickle=True)
    return z["ids"], z["emb"]

"""Parameters, latency (GPU and CPU, batch 1) and peak GPU memory per model."""
from __future__ import annotations

import time

import pandas as pd

from .models import REGISTRY, load_backbone, param_count


def measure(key: str, n_warm: int = 5, n_iter: int = 30) -> dict:
    import torch
    model, spec, _ = load_backbone(key)
    x = torch.randn(1, 3, spec.img_size, spec.img_size)
    res = {"model": key, "family": spec.family, "params_M": param_count(model) / 1e6,
           "img_size": spec.img_size}
    model.eval()
    with torch.no_grad():
        for _ in range(n_warm):
            model(x)
        t = time.perf_counter()
        for _ in range(min(n_iter, 10)):
            model(x)
        res["cpu_ms"] = (time.perf_counter() - t) / min(n_iter, 10) * 1000
        if torch.cuda.is_available():
            m, xg = model.cuda(), x.cuda()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(n_warm):
                m(xg)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n_iter):
                m(xg)
            torch.cuda.synchronize()
            res["gpu_ms"] = (time.perf_counter() - t) / n_iter * 1000
            res["gpu_peak_MB"] = torch.cuda.max_memory_allocated() / 2**20
    del model
    return res


def measure_all(keys=None) -> pd.DataFrame:
    rows = []
    for k in keys or REGISTRY:
        try:
            rows.append(measure(k))
        except Exception as e:
            rows.append({"model": k, "error": str(e)[:200]})
    return pd.DataFrame(rows)

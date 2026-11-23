"""Metrics matching the manuscript (binary and four-class)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


def ece(y: np.ndarray, p_pos: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error for the predicted class confidence (binary)."""
    pred = (p_pos >= 0.5).astype(int)
    conf = np.where(pred == 1, p_pos, 1 - p_pos)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total, err = len(y), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            err += m.sum() / total * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(err)


def threshold_for_sensitivity(y_val: np.ndarray, p_val: np.ndarray, target: float = 0.90) -> float:
    """Highest threshold whose validation sensitivity is >= target."""
    pos = np.sort(p_val[y_val == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])


def _safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def binary_metrics(df: pd.DataFrame, labels: pd.Series | None = None,
                   val: pd.DataFrame | None = None) -> dict:
    """df has columns y (0/1) and p1. labels: canonical label per row (for look-alike rates)."""
    y = df["y"].to_numpy().astype(int)
    p = df["p1"].to_numpy()
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    res = {
        "n": len(y), "n_pos": int(y.sum()),
        "bal_acc": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan"),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "auroc": _safe_auc(y, p),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "ece": ece(y, p),
    }
    if val is not None and len(val) and "p1" in val:
        thr = threshold_for_sensitivity(val["y"].to_numpy(), val["p1"].to_numpy())
        pr = (p >= thr).astype(int)
        res["thr90"] = thr
        res["spec_at_90sens"] = float(((pr == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
        res["sens_at_thr90"] = float(((pr == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
    if labels is not None:
        lab = labels.to_numpy()
        healthy = lab == "healthy"
        disease = (y == 0) & ~healthy
        res["auroc_vs_healthy"] = _safe_auc(y[(y == 1) | healthy], p[(y == 1) | healthy])
        res["auroc_vs_disease"] = _safe_auc(y[(y == 1) | disease], p[(y == 1) | disease])
        res["auroc_gap"] = res["auroc_vs_healthy"] - res["auroc_vs_disease"]
        for c in sorted(set(lab[y == 0])):
            m = lab == c
            res[f"called_mpox__{c}"] = float(pred[m].mean())
    return res


def multiclass_metrics(df: pd.DataFrame, n_classes: int) -> dict:
    y = df["y"].to_numpy().astype(int)
    P = df[[f"p{k}" for k in range(n_classes)]].to_numpy()
    pred = P.argmax(1)
    res = {"n": len(y), "bal_acc": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}
    try:
        res["macro_auroc"] = float(roc_auc_score(y, P, multi_class="ovr", average="macro",
                                                 labels=list(range(n_classes))))
    except ValueError:
        res["macro_auroc"] = float("nan")
    return res


def bootstrap_ci(df: pd.DataFrame, metric: str = "bal_acc", n: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = df["y"].to_numpy()
    idx_pos, idx_neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if metric == "bal_acc":
        # fast vectorised path: stratified resampling of per-class correctness
        if not len(idx_pos) or not len(idx_neg):
            return float("nan"), float("nan")
        pred = (df["p1"].to_numpy() >= 0.5).astype(np.float64)
        hit_pos, hit_neg = pred[idx_pos], 1 - pred[idx_neg]
        sens = hit_pos[rng.integers(0, len(hit_pos), (n, len(hit_pos)))].mean(1)
        spec = hit_neg[rng.integers(0, len(hit_neg), (n, len(hit_neg)))].mean(1)
        vals = (sens + spec) / 2
        return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
    vals = []
    for _ in range(n):
        # stratified resampling keeps both classes present
        s = np.concatenate([rng.choice(idx_pos, len(idx_pos)) if len(idx_pos) else idx_pos,
                            rng.choice(idx_neg, len(idx_neg)) if len(idx_neg) else idx_neg])
        v = binary_metrics(df.iloc[s])[metric]
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))

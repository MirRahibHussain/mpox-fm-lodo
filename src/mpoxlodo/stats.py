"""Friedman + Nemenyi (critical difference), Wilcoxon signed-rank with Holm."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Studentized range q_alpha / sqrt(2) for alpha = 0.05 (Demsar 2006, Table 5a), k = 2..20
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102,
        10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426,
        17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}


def friedman(scores: pd.DataFrame) -> dict:
    """scores: rows = held-out sources (blocks), columns = models. Higher is better."""
    s = scores.dropna(axis=0, how="any")
    chi2, p = stats.friedmanchisquare(*[s[c].to_numpy() for c in s.columns])
    ranks = s.rank(axis=1, ascending=False).mean().sort_values()
    k, n = s.shape[1], s.shape[0]
    cd = _Q05.get(k, np.nan) * np.sqrt(k * (k + 1) / (6.0 * n))
    return {"chi2": float(chi2), "p": float(p), "n_blocks": n, "k": k,
            "mean_ranks": ranks.to_dict(), "critical_difference": float(cd)}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def paired_wilcoxon(a: pd.Series, b: pd.Series) -> dict:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 2 or np.allclose(d.iloc[:, 0], d.iloc[:, 1]):
        return {"n": len(d), "stat": float("nan"), "p": float("nan"), "median_diff": 0.0}
    r = stats.wilcoxon(d.iloc[:, 0], d.iloc[:, 1])
    return {"n": len(d), "stat": float(r.statistic), "p": float(r.pvalue),
            "median_diff": float((d.iloc[:, 0] - d.iloc[:, 1]).median())}


def cd_diagram(result: dict, path: str, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranks = result["mean_ranks"]
    k, cd = result["k"], result["critical_difference"]
    names, vals = list(ranks.keys()), np.array(list(ranks.values()))
    fig, ax = plt.subplots(figsize=(7, 0.35 * k + 1.4))
    ax.set_xlim(1, k)
    ax.set_ylim(0, k + 1)
    ax.invert_xaxis()
    ax.get_yaxis().set_visible(False)
    for i, (n, v) in enumerate(zip(names, vals)):
        y = k - i
        ax.plot([v, v], [0.4, y], color="0.6", lw=0.8)
        ax.text(v, y, f" {n} ({v:.2f})", va="center", fontsize=8)
    ax.plot([1, 1 + cd], [0.2, 0.2], color="k", lw=2)
    ax.text(1 + cd / 2, 0.45, f"CD = {cd:.2f}", ha="center", fontsize=8)
    ax.set_xlabel("mean rank (lower is better)")
    ax.set_title(title or f"Friedman p = {result['p']:.3g}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

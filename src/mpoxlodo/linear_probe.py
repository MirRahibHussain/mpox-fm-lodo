"""Linear probe on cached embeddings (CPU, minutes for all splits)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def run_split(split: dict, ids: np.ndarray, emb: np.ndarray, index: pd.DataFrame,
              label_col: str, seed: int = 0) -> pd.DataFrame:
    pos = {i: k for k, i in enumerate(ids)}
    lab = index.set_index("image_id")[label_col]

    def xy(lst):
        lst = [i for i in lst if i in pos]
        return emb[[pos[i] for i in lst]], lab.loc[lst].to_numpy(), lst

    Xtr, ytr, _ = xy(split["train"])
    Xva, yva, _ = xy(split["val"])
    Xte, yte, te_ids = xy(split["test"])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best, best_c = -1.0, C_GRID[0]
    for c in C_GRID:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, random_state=seed)
        clf.fit(Xtr, ytr)
        s = balanced_accuracy_score(yva, clf.predict(Xva)) if len(yva) else 0.0
        if s > best:
            best, best_c = s, c
    # final model on train only: validation stays reserved for threshold choice
    clf = LogisticRegression(C=best_c, class_weight="balanced", max_iter=3000, random_state=seed)
    clf.fit(Xtr, ytr)
    out = pd.DataFrame({"image_id": te_ids, "y": yte})
    prob = clf.predict_proba(Xte)
    val_prob = clf.predict_proba(Xva) if len(Xva) else np.zeros((0, prob.shape[1]))
    for k in range(prob.shape[1]):
        out[f"p{k}"] = prob[:, k]
    out.attrs["best_C"] = best_c
    out.attrs["val"] = pd.DataFrame({"y": yva, **{f"p{k}": val_prob[:, k] for k in range(val_prob.shape[1])}})
    return out

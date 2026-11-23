"""Stage 10 (CPU): sensitivity of the main conclusions to the duplicate rule.

The duplicate groups (and hence the LODO families, the leakage removal and the
pooled grouped folds) are rebuilt under a strict rule, the originally registered rule,
the main rule and the loosest possible rule (every audit candidate).
For each rule a linear probe (fixed C, so the rules differ only in their
splits) is trained for every model with cached features, and we report:

- number of duplicate pairs, groups spanning sources and LODO families;
- mean pooled (group-aware) and mean LODO balanced accuracy per model family;
- the generalization gap (pooled minus LODO);
- Spearman rho between each rule's LODO model ranking and the main rule's.

Writes results/sensitivity_runs.csv (one row per rule x model x split) and
results/sensitivity_summary.csv (one row per rule x model).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from mpoxlodo import dedup, features, io, splits
from mpoxlodo.models import FAMILY_GROUP, REGISTRY

RULES = [("strict", 2, 0.99),
         ("original_registered", *dedup.ORIGINAL_RULE),
         ("main", dedup.RULE_HAMMING, dedup.RULE_COSINE),
         ("all_candidates", 10, 0.95)]


def probe(split, ids, emb, lab, C):
    pos = {i: k for k, i in enumerate(ids)}
    tr = [i for i in split["train"] if i in pos]
    te = [i for i in split["test"] if i in pos]
    if not tr or not te:
        return None
    sc = StandardScaler().fit(emb[[pos[i] for i in tr]])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit(sc.transform(emb[[pos[i] for i in tr]]), lab.loc[tr].to_numpy())
    pred = clf.predict(sc.transform(emb[[pos[i] for i in te]]))
    y = lab.loc[te].to_numpy()
    if len(set(y)) < 2:   # specificity-only fold
        return {"bal_acc": np.nan, "specificity": float((pred == 0).mean()), "n_test": len(te)}
    return {"bal_acc": float(balanced_accuracy_score(y, pred)),
            "specificity": float((pred[y == 0] == 0).mean()), "n_test": len(te)}


def run_model(k, rule_splits, index, C):
    ids, emb = features.load(io.p("features"), k)
    lab = index.set_index("image_id")["y_bin"]
    rows = []
    for rule, sp in rule_splits.items():
        for s in sp:
            r = probe(s, ids, emb, lab, C)
            if r:
                rows.append({"rule": rule, "model": k, "split": s["name"], "protocol": s["protocol"], **r})
    print(f"[{k}] sensitivity done", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--family-threshold", type=float, default=0.50)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    index = io.load_index()
    cands = pd.read_csv(io.p("audit", "candidates_for_review.csv"), keep_default_na=False)
    for col in ("phash_dist", "cosine"):
        cands[col] = pd.to_numeric(cands[col], errors="coerce").fillna(64 if col == "phash_dist" else 0)
    cands["auto_duplicate"] = cands["auto_duplicate"].astype(str).str.lower().eq("true")

    rule_splits, meta = {}, {}
    for name, ham, cos in RULES:
        pairs = cands[dedup.rule_duplicate(cands, ham, cos)]
        idx = index.copy()
        idx["group"] = dedup.groups_from_pairs(idx, pairs).to_numpy()
        fams = dedup.families_from_groups(idx, a.family_threshold)
        sp = (splits.pooled_folds(idx, "binary", seed=a.seed)
              + splits.lodo_folds(idx, fams, "binary", seed=a.seed))
        for s in sp:
            splits.assert_no_leakage(s, idx)
            s["name"] = f"{name}:{s['name']}"
        rule_splits[name] = sp
        dev = idx[idx["role"] == "dev"]
        meta[name] = {"phash_max": ham, "cosine_min": cos, "pairs": int(len(pairs)),
                      "cross_source_pairs": int((pairs["dataset_a"] != pairs["dataset_b"]).sum()),
                      "dev_images_in_cross_source_groups":
                          int(dev.groupby("group")["dataset"].transform("nunique").gt(1).sum()),
                      "n_families": len(set(fams.values())),
                      "families": {f: sorted(d for d, g in fams.items() if g == f) for f in set(fams.values())}}
        print(name, json.dumps({k: v for k, v in meta[name].items() if k != "families"}), flush=True)

    models = [k for k in REGISTRY if os.path.exists(os.path.join(io.p("features"), f"{k}.npz"))]
    out = Parallel(n_jobs=a.jobs)(delayed(run_model)(k, rule_splits, index, a.C) for k in models)
    runs = pd.DataFrame([r for rows in out for r in rows])
    os.makedirs(io.p("results"), exist_ok=True)
    runs.to_csv(io.p("results", "sensitivity_runs.csv"), index=False)

    summ = []
    for (rule, k), g in runs.groupby(["rule", "model"]):
        pooled = g[g["protocol"] == "pooled"]["bal_acc"].mean()
        lodo = g[g["protocol"] == "lodo"]["bal_acc"].mean()
        spec = g[g["protocol"] == "lodo_specificity"]["specificity"].mean()
        summ.append({"rule": rule, "model": k, "family": FAMILY_GROUP.get(REGISTRY[k].family, ""),
                     "pooled_bal_acc": pooled, "lodo_bal_acc": lodo, "gap": pooled - lodo,
                     "lodo_specificity_negative_only": spec})
    summ = pd.DataFrame(summ)
    main_rank = summ[summ["rule"] == "main"].set_index("model")["lodo_bal_acc"]
    for rule in summ["rule"].unique():
        r = summ[summ["rule"] == rule].set_index("model")["lodo_bal_acc"].reindex(main_rank.index)
        rho = spearmanr(main_rank, r, nan_policy="omit").correlation if len(main_rank) > 2 else np.nan
        meta[rule]["spearman_lodo_ranking_vs_main"] = float(rho)
        best = summ[summ["rule"] == rule].sort_values("lodo_bal_acc", ascending=False).iloc[0]
        meta[rule]["best_model_lodo"] = f"{best['model']} ({best['family']})"
    summ.to_csv(io.p("results", "sensitivity_summary.csv"), index=False)
    with open(io.p("results", "sensitivity_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    by_fam = summ.groupby(["rule", "family"])[["pooled_bal_acc", "lodo_bal_acc", "gap"]].mean().round(4)
    print(by_fam.to_string())
    print(json.dumps({r: {k: v for k, v in m.items() if k != "families"} for r, m in meta.items()}, indent=2))

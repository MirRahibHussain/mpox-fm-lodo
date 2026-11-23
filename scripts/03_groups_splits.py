"""Stage 3 (CPU): duplicate groups, families, ITA and all splits."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mpoxlodo import dedup, io, splits
from mpoxlodo.ita import ita_band, ita_of_image

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Two sources are held out together when more than half of either one's images
    # duplicate images of the other (i.e. one is largely derived from the other).
    # Smaller overlaps are handled by removing the shared groups from training.
    ap.add_argument("--family-threshold", type=float, default=0.50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ita", action="store_true")
    a = ap.parse_args()

    index = io.load_index()
    reviewed_fn = io.p("audit", "candidates_reviewed.csv")
    cand_fn = reviewed_fn if os.path.exists(reviewed_fn) else io.p("audit", "candidates_for_review.csv")
    cands = pd.read_csv(cand_fn, keep_default_na=False)
    for col in ("phash_dist", "cosine"):
        cands[col] = pd.to_numeric(cands[col], errors="coerce").fillna(64 if col == "phash_dist" else 0)
    cands["auto_duplicate"] = cands["auto_duplicate"].astype(str).str.lower().eq("true")
    pairs, provisional = dedup.confirmed_pairs(cands)
    kappa = dedup.cohen_kappa(cands["reviewer_1"], cands["reviewer_2"])
    validation = None
    smp = None
    key_fn = io.p("audit", "review_key.csv")
    if provisional and os.path.exists(key_fn):
        repo_reviews = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reviews")
        smp = dedup.load_reviewer_forms(pd.read_csv(key_fn, keep_default_na=False),
                                        [io.p("audit"), repo_reviews, "/kaggle/input"])
    smp_fn = io.p("audit", "review_sample_reviewed.csv")
    if smp is None and provisional and os.path.exists(smp_fn):
        smp = pd.read_csv(smp_fn, keep_default_na=False)
    if smp is not None:
        smp.to_csv(io.p("audit", "review_sample_reviewed_merged.csv"), index=False)
        validation = dedup.rule_validation(smp)
        kappa = validation["kappa"]
        provisional = validation["pairs_reviewed"] < 50
        print("automatic duplicate rule validated on the reviewed sample:", json.dumps(validation))
    if provisional:
        print("WARNING: the automatic duplicate rule has not been validated by the two reviewers yet "
              "(review_form_R1.csv / review_form_R2.csv). Results are PROVISIONAL.")

    index["group"] = dedup.groups_from_pairs(index, pairs).to_numpy()
    om = dedup.overlap_matrix(index)
    om.to_csv(io.p("audit", "overlap_matrix.csv"))
    fams = dedup.families_from_groups(index, a.family_threshold)
    big = pd.Series(fams).value_counts()
    if (big > 3).any():
        print(f"WARNING: a family absorbed {int(big.max())} sources; consider a higher --family-threshold")

    dev_groups = set(index.loc[index["role"] == "dev", "group"])
    summary = {
        "provisional": provisional, "kappa": kappa, "rule_validation": validation,
        "confirmed_pairs": int(len(pairs)),
        "cross_dataset_pairs": int((pairs["dataset_a"] != pairs["dataset_b"]).sum()),
        "dev_images": int((index["role"] == "dev").sum()),
        "dev_images_in_cross_source_groups": int(index[index["role"] == "dev"].groupby("group")["dataset"]
                                                  .transform("nunique").gt(1).sum()),
        "external_images_matching_dev": {d: int(g["group"].isin(dev_groups).sum())
                                         for d, g in index[index["role"] == "external"].groupby("dataset")},
        "families": fams,
    }
    with open(io.p("audit", "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))

    if not a.no_ita and "ita" not in index:
        vals = []
        for pth in index["path"]:
            try:
                vals.append(ita_of_image(pth))
            except Exception:
                vals.append(np.nan)
        index["ita"] = vals
        index["ita_band"] = index["ita"].map(ita_band)
    from mpoxlodo.config import FOUR_CLASS
    index["y4"] = index["label"].map({c: i for i, c in enumerate(FOUR_CLASS)}).fillna(-1).astype(int)
    index.to_csv(io.index_path(), index=False)

    all_splits = []
    for task in ("binary", "four_class"):
        all_splits += splits.pooled_folds(index, task, seed=a.seed)
        all_splits += splits.lodo_folds(index, fams, task, seed=a.seed)
        all_splits += splits.external_split(index, task, seed=a.seed)
    all_splits += splits.lodo_folds(index, fams, "binary", seed=a.seed,
                                    exclude_from_train=splits.NEGATIVE_ONLY)
    all_splits += splits.pooled_naive_folds(index, seed=a.seed)
    for s in all_splits:
        splits.assert_no_leakage(s, index)
    # if the splits changed since the predictions were made (e.g. a new duplicate rule),
    # move every cached prediction aside so all stages recompute on the new splits
    import hashlib, shutil
    digest = hashlib.sha256(json.dumps([[s["name"], sorted(s["train"]), sorted(s["test"])]
                                        for s in sorted(all_splits, key=lambda x: x["name"])]).encode()).hexdigest()[:12]
    hash_fn = io.p("splits", "split_hash.txt")
    old = open(hash_fn).read().strip() if os.path.exists(hash_fn) else "unknown"
    preds_dir = os.path.join(os.path.dirname(io.p("splits")), "preds")
    if old != digest and os.path.isdir(preds_dir) and os.listdir(preds_dir):
        dest = preds_dir + f"_superseded_{old}"
        shutil.move(preds_dir, dest)
        print(f"splits changed ({old} -> {digest}): previous predictions moved to {dest}")
    splits.save_splits(all_splits, io.p("splits"))
    with open(hash_fn, "w") as f:
        f.write(digest)
    summ = splits.summarize(all_splits, index)
    summ.to_csv(io.p("splits", "split_summary.csv"), index=False)
    print(summ.to_string(index=False))
    print(f"\n{len(all_splits)} splits written; all passed the leakage check.")

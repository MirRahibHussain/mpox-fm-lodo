"""Stage 5 (CPU): linear probe for every model on every split."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mpoxlodo import features, io, linear_probe, splits
from mpoxlodo.models import REGISTRY

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(REGISTRY))
    a = ap.parse_args()
    index = io.load_index()
    all_splits = splits.load_splits(io.p("splits"))
    for k in a.models:
        fn = os.path.join(io.p("features"), f"{k}.npz")
        if not os.path.exists(fn):
            print(f"[{k}] no features, skipped")
            continue
        ids, emb = features.load(io.p("features"), k)
        for s in all_splits:
            if io.has_pred("lp", k, s["name"], 0) or not s["test"]:
                continue
            label_col = "y_bin" if s["task"] == "binary" else "y4"
            if label_col == "y4" and "y4" not in index:
                from mpoxlodo.config import FOUR_CLASS
                index["y4"] = index["label"].map({c: i for i, c in enumerate(FOUR_CLASS)}).fillna(-1).astype(int)
            df = linear_probe.run_split(s, ids, emb, index, label_col)
            io.save_pred(df, "lp", k, s["name"], 0)
        print(f"[{k}] linear probe done")

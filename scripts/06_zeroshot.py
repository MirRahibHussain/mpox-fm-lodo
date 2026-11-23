"""Stage 6 (GPU): zero-shot for CLIP-type models on LODO and external test sets."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mpoxlodo import io, splits, zeroshot
from mpoxlodo.config import FOUR_CLASS
from mpoxlodo.models import REGISTRY

if __name__ == "__main__":
    index = io.load_index()
    index["y4"] = index["label"].map({c: i for i, c in enumerate(FOUR_CLASS)}).fillna(-1).astype(int)
    keep = [s for s in splits.load_splits(io.p("splits"))
            if s["protocol"] in ("lodo", "lodo_specificity", "external", "pooled") and s["test"]]
    for k, spec in REGISTRY.items():
        if not spec.zero_shot:
            continue
        for s in keep:
            if io.has_pred("zs", k, s["name"], 0):
                continue
            try:
                df = zeroshot.run_split(s, index, k, "y_bin" if s["task"] == "binary" else "y4", s["task"])
                io.save_pred(df, "zs", k, s["name"], 0)
            except Exception as e:
                print(f"[{k}] {s['name']} failed: {e}")
                break
        print(f"[{k}] zero-shot done")

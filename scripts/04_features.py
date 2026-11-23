"""Stage 4 (GPU): frozen embeddings for every model (cached; safe to re-run)."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mpoxlodo import features, io
from mpoxlodo.models import REGISTRY

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(REGISTRY))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--smoke", action="store_true", help="only 64 images, to test loading")
    a = ap.parse_args()
    index = io.load_index()
    out_dir = io.p("features_smoke" if a.smoke else "features")
    for k in a.models:
        t = time.time()
        try:
            fn = features.extract(index, k, out_dir, batch_size=a.batch_size, limit=64 if a.smoke else 0)
            print(f"[{k}] ok -> {fn} ({(time.time() - t) / 60:.1f} min)")
        except Exception as e:
            print(f"[{k}] FAILED: {e}")

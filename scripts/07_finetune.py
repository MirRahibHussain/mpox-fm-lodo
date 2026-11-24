"""Stage 7 (GPU): fine-tuning (FT) and LP-then-FT for the selected models.

By default the best model of each family is selected by POOLED linear-probe
balanced accuracy (never by LODO or external results), as fixed in the
pre-registration. Override with --models. Resumable: finished runs are skipped,
so the stage can be split across several Kaggle sessions.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mpoxlodo import finetune, io, metrics, splits
from mpoxlodo.config import TrainConfig
from mpoxlodo.models import FAMILY_GROUP, REGISTRY


def select_models() -> list[str]:
    rows = []
    for k, spec in REGISTRY.items():
        for s in splits.load_splits(io.p("splits"), protocol="pooled", task="binary"):
            fn = io.pred_path("lp", k, s["name"], 0)
            if os.path.exists(fn):
                rows.append({"model": k, "group": FAMILY_GROUP[spec.family],
                             "bal_acc": metrics.binary_metrics(pd.read_csv(fn))["bal_acc"]})
    if not rows:
        raise SystemExit("Run stage 5 (linear probe) first, or pass --models.")
    df = pd.DataFrame(rows).groupby(["group", "model"])["bal_acc"].mean().reset_index()
    best = df.sort_values("bal_acc", ascending=False).groupby("group").head(1)
    print("Selected by pooled linear-probe balanced accuracy:\n", best.to_string(index=False))
    best.to_csv(io.p("results", "ft_model_selection.csv"), index=False)
    return list(best["model"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--regimes", nargs="*", default=["ft", "lpft"])
    ap.add_argument("--protocols", nargs="*", default=["pooled", "lodo", "lodo_specificity", "external"])
    ap.add_argument("--tasks", nargs="*", default=["binary"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    # Plan A (development folds): 3 seeds on LODO folds, 1 seed on pooled folds, and LP-FT only
    # for the selected foundation model. External splits follow the pre-registration
    # (FT and LP-FT for every selected model, 3 seeds).
    ap.add_argument("--pooled-seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--lpft-groups", nargs="*", default=["Foundation model"],
                    help="model groups that get LP-FT on development folds (external: all)")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--dermnet-cap", type=int, default=TrainConfig.dermnet_cap)
    ap.add_argument("--max-runs", type=int, default=0, help="stop after N runs (Kaggle session budget)")
    ap.add_argument("--max-hours", type=float, default=0, help="do not start a run after this many hours")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    # run_budget_hours.txt (repository root) caps the session, e.g. when little GPU quota is left this week
    cap_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run_budget_hours.txt")
    if os.path.exists(cap_fn):
        cap = float(open(cap_fn).read().split()[0])
        if not a.max_hours or cap < a.max_hours:
            print(f"time budget capped at {cap} h by run_budget_hours.txt")
            a.max_hours = cap

    index = io.load_index()
    cfg = TrainConfig(epochs=a.epochs, dermnet_cap=a.dermnet_cap)
    models = a.models or select_models()
    group_of = {k: FAMILY_GROUP[REGISTRY[k].family] for k in models}
    all_s = [s for s in splits.load_splits(io.p("splits"))
             if s["task"] in a.tasks and s["test"]]
    todo = [s for s in all_s if s["protocol"] in a.protocols and s["protocol"] != "external"]
    # all external splits of a task share one training set: train once, predict on every external image
    if "external" in a.protocols:
        for task in a.tasks:
            ext = [s for s in all_s if s["protocol"] in ("external", "external_withdup") and s["task"] == task]
            if ext:
                union = sorted({i for s in ext for i in s["test"]})
                todo.append({**ext[0], "name": f"external_{task}_ALL", "protocol": "external_all",
                             "test": union, "_parts": ext})

    import time
    t0, last_min, done = time.time(), 0.0, 0
    # seed-major order: seed 0 of every model and regime first, then seed 1, ...
    for seed in a.seeds:
        for regime in a.regimes:
            for k in models:
                for s in todo:
                    is_ext = s["protocol"] == "external_all"
                    if s["protocol"] == "pooled" and seed not in a.pooled_seeds:
                        continue
                    if regime == "lpft" and not is_ext and group_of[k] not in a.lpft_groups:
                        continue
                    tag = regime + ("_smoke" if a.smoke else "")
                    if io.has_pred(tag, k, s["name"], seed):
                        continue
                    label_col = "y_bin" if s["task"] == "binary" else "y4"
                    n_cls = 2 if s["task"] == "binary" else 4
                    if a.max_hours and (time.time() - t0) / 3600 + last_min / 60 > a.max_hours:
                        print(f"time budget of {a.max_hours} h reached; re-run to continue.")
                        sys.exit(0)
                    print(f"[{k}|{regime}|seed {seed}] {s['name']} ...", flush=True)
                    df = finetune.run(s, index, k, label_col, n_cls, seed, cfg, regime, smoke=a.smoke)
                    io.save_pred(df, tag, k, s["name"], seed)
                    for part in s.get("_parts", []):
                        sub = df[df["image_id"].isin(set(part["test"]))].copy()
                        sub.attrs["val"] = df.attrs.get("val")
                        io.save_pred(sub, tag, k, part["name"], seed)
                    print(f"    val bal.acc {df.attrs['best_val_bacc']:.3f}, {df.attrs['minutes']:.1f} min")
                    done += 1
                    last_min = df.attrs["minutes"]
                    if a.max_runs and done >= a.max_runs:
                        print("max runs reached; re-run to continue.")
                        sys.exit(0)

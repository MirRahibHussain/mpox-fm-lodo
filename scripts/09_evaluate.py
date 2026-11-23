"""Stage 9 (CPU): metrics, statistics, tables and figures for the manuscript.

Outputs in <WORK_DIR>/results/:
  results_long.csv          every (regime, model, split, seed) with all metrics
  table5_dedup_inflation    pooled balanced accuracy, naive vs grouped folds
  table6_main               pooled vs LODO per model and regime, with gap
  table7_external           external balanced accuracy (clean and with duplicates)
  table8_skin_tone          LODO sensitivity by ITA band
  lookalike_and_auroc       AUROC vs healthy / vs disease, look-alikes called Mpox
  four_class                macro-F1 pooled vs LODO
  stats.json                Friedman/Nemenyi and Holm-corrected Wilcoxon tests
  fig2_overlap.png, fig3_lodo_heatmap_<regime>.png, fig4_cd_<regime>.png
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mpoxlodo import io, metrics, splits, stats
from mpoxlodo.models import FAMILY_GROUP, REGISTRY

FILE_RE = re.compile(r"(?P<split>.+)__s(?P<seed>\d+)\.csv$")


def collect(index: pd.DataFrame) -> pd.DataFrame:
    meta = {s["name"]: s for s in splits.load_splits(io.p("splits"))}
    lab = index.set_index("image_id")["label"]
    rows = []
    for fn in glob.glob(os.path.join(io.p("preds"), "*", "*", "*.csv")):
        if fn.endswith("__val.csv"):
            continue
        regime, model = fn.split(os.sep)[-3], fn.split(os.sep)[-2]
        if regime.endswith("_smoke"):
            continue
        m = FILE_RE.search(os.path.basename(fn))
        if not m or m["split"] not in meta:
            continue
        s = meta[m["split"]]
        df = pd.read_csv(fn)
        if not len(df):
            continue
        vfn = fn[:-4] + "__val.csv"
        val = pd.read_csv(vfn) if os.path.exists(vfn) else None
        if s["task"] == "binary":
            res = metrics.binary_metrics(df, labels=df["image_id"].map(lab), val=val)
            lo, hi = metrics.bootstrap_ci(df, n=500) if df["y"].nunique() == 2 else (np.nan, np.nan)
            res["bal_acc_lo"], res["bal_acc_hi"] = lo, hi
        else:
            res = metrics.multiclass_metrics(df, 4)
        spec = REGISTRY.get(model)
        rows.append({"regime": regime, "model": model,
                     "family": spec.family if spec else "", "group": FAMILY_GROUP.get(spec.family, "") if spec else "",
                     "split": s["name"], "protocol": s["protocol"], "task": s["task"],
                     "held_out": s["held_out"], "seed": int(m["seed"]), **res})
    return pd.DataFrame(rows)


def mean_sd(x: pd.Series) -> str:
    x = x.dropna()
    if not len(x):
        return ""
    return f"{100 * x.mean():.1f} ± {100 * x.std(ddof=1):.1f}" if len(x) > 1 else f"{100 * x.mean():.1f}"


def per_seed_protocol_mean(r: pd.DataFrame, protocol: str, metric: str = "bal_acc") -> pd.DataFrame:
    sub = r[(r["protocol"] == protocol) & (r["task"] == "binary")]
    return sub.groupby(["regime", "model", "seed"])[metric].mean().reset_index()


def main():
    index = io.load_index()
    out = io.p("results")
    r = collect(index)
    if r.empty:
        raise SystemExit("No predictions found. Run stages 5-7 first.")
    r.to_csv(os.path.join(out, "results_long.csv"), index=False)
    regimes = sorted(r["regime"].unique())

    # ---- Table 5: dedup inflation (pooled naive vs pooled grouped), per family group
    naive = per_seed_protocol_mean(r, "pooled_naive")
    grouped = per_seed_protocol_mean(r, "pooled")
    t5 = naive.merge(grouped, on=["regime", "model", "seed"], suffixes=("_naive", "_grouped"))
    t5["group"] = t5["model"].map(lambda k: FAMILY_GROUP[REGISTRY[k].family] if k in REGISTRY else "")
    t5 = t5.groupby(["regime", "group"]).agg(before=("bal_acc_naive", "mean"),
                                             after=("bal_acc_grouped", "mean")).reset_index()
    t5["difference"] = t5["after"] - t5["before"]
    t5.to_csv(os.path.join(out, "table5_dedup_inflation.csv"), index=False)

    # ---- Table 6: pooled vs LODO per model and regime
    pooled = per_seed_protocol_mean(r, "pooled")
    lodo = per_seed_protocol_mean(r, "lodo")
    t6 = pooled.merge(lodo, on=["regime", "model", "seed"], how="outer", suffixes=("_pooled", "_lodo"))
    t6["gap"] = t6["bal_acc_pooled"] - t6["bal_acc_lodo"]
    t6_fmt = (t6.groupby(["model", "regime"])
              .agg(pooled=("bal_acc_pooled", mean_sd), lodo=("bal_acc_lodo", mean_sd), gap=("gap", mean_sd))
              .reset_index())
    t6_fmt["group"] = t6_fmt["model"].map(lambda k: FAMILY_GROUP[REGISTRY[k].family] if k in REGISTRY else "")
    t6_wide = t6_fmt.pivot(index=["group", "model"], columns="regime", values=["pooled", "lodo", "gap"])
    t6_wide.columns = [f"{a}_{b}" for a, b in t6_wide.columns]
    t6_wide.reset_index().to_csv(os.path.join(out, "table6_main.csv"), index=False)
    fam_gap = t6.assign(group=t6["model"].map(lambda k: FAMILY_GROUP[REGISTRY[k].family]))\
        .groupby(["regime", "group"])["gap"].mean().reset_index()
    fam_gap.to_csv(os.path.join(out, "table6_family_gap.csv"), index=False)

    # weighted LODO mean (by test size) as a sensitivity analysis
    lw = r[(r["protocol"] == "lodo") & (r["task"] == "binary")]
    lw = lw.groupby(["regime", "model", "seed"]).apply(
        lambda g: np.average(g["bal_acc"], weights=g["n"]), include_groups=False).rename("lodo_weighted").reset_index()
    lw.to_csv(os.path.join(out, "lodo_weighted.csv"), index=False)

    # specificity-only folds
    spec_only = r[(r["protocol"] == "lodo_specificity")].groupby(["regime", "model", "held_out"])[
        "specificity"].mean().reset_index()
    spec_only["false_positive_rate"] = 1 - spec_only["specificity"]
    spec_only.to_csv(os.path.join(out, "lodo_specificity_only.csv"), index=False)

    # ---- Table 7: external
    ext = r[r["protocol"].isin(["external", "external_withdup"]) & (r["task"] == "binary")]
    if len(ext):
        t7 = ext.groupby(["regime", "model", "protocol", "held_out"])[
            ["bal_acc", "sensitivity", "specificity", "auroc"]].mean().reset_index()
        t7.to_csv(os.path.join(out, "table7_external.csv"), index=False)

    # ---- look-alikes, AUROC vs healthy / disease, specificity at 90% sensitivity
    cols = [c for c in r.columns if c.startswith("called_mpox__")] + \
           ["auroc_vs_healthy", "auroc_vs_disease", "auroc_gap", "spec_at_90sens"]
    cols = [c for c in cols if c in r]
    la = r[(r["protocol"].isin(["lodo", "lodo_noneg"])) & (r["task"] == "binary")]
    la.groupby(["regime", "model", "protocol"])[cols].mean().reset_index()\
        .to_csv(os.path.join(out, "lookalike_and_auroc.csv"), index=False)

    # ---- Table 8: skin tone (LODO sensitivity by ITA band), pooled over folds and seeds
    if "ita_band" in index:
        band = index.set_index("image_id")["ita_band"]
        rows = []
        for fn in glob.glob(os.path.join(io.p("preds"), "*", "*", "lodo_binary_*__s*.csv")):
            if fn.endswith("__val.csv") or "_noneg" in fn:
                continue
            regime, model = fn.split(os.sep)[-3], fn.split(os.sep)[-2]
            df = pd.read_csv(fn)
            df = df[df["y"] == 1]
            df["band"] = df["image_id"].map(band)
            for b, g in df.groupby("band"):
                rows.append({"regime": regime, "model": model, "band": b, "n": len(g),
                             "tp": int((g["p1"] >= 0.5).sum())})
        if rows:
            t8 = pd.DataFrame(rows).groupby(["regime", "model", "band"])[["n", "tp"]].sum().reset_index()
            t8["sensitivity"] = t8["tp"] / t8["n"]
            t8.pivot_table(index=["regime", "model"], columns="band", values="sensitivity")\
                .to_csv(os.path.join(out, "table8_skin_tone.csv"))

    # ---- four-class
    fc = r[r["task"] == "four_class"]
    if len(fc):
        fc.groupby(["regime", "model", "protocol"])[["macro_f1", "bal_acc", "macro_auroc"]].mean()\
            .reset_index().to_csv(os.path.join(out, "four_class.csv"), index=False)

    # ---- statistics
    st = {}
    lodo_rows = r[(r["protocol"] == "lodo") & (r["task"] == "binary")]
    for regime in regimes:
        sub = lodo_rows[lodo_rows["regime"] == regime]
        if sub["model"].nunique() < 3:
            continue
        mat = sub.groupby(["held_out", "model"])["bal_acc"].mean().unstack()
        try:
            fr = stats.friedman(mat)
            st[f"friedman_{regime}"] = fr
            stats.cd_diagram(fr, os.path.join(out, f"fig4_cd_{regime}.png"))
        except Exception as e:
            st[f"friedman_{regime}"] = {"error": str(e)}
        # family-level paired Wilcoxon over held-out sources (mean of models in each family)
        fam = sub.assign(group=sub["model"].map(lambda k: FAMILY_GROUP[REGISTRY[k].family]))\
            .groupby(["held_out", "group"])["bal_acc"].mean().unstack()
        pairs = [("Foundation model", "CNN"), ("Foundation model", "ImageNet ViT"), ("ImageNet ViT", "CNN")]
        tests = {f"{a} vs {b}": stats.paired_wilcoxon(fam[a], fam[b]) for a, b in pairs
                 if a in fam and b in fam}
        adj = stats.holm({k: v["p"] for k, v in tests.items() if not np.isnan(v["p"])})
        for k in tests:
            tests[k]["p_holm"] = adj.get(k, float("nan"))
        st[f"wilcoxon_family_{regime}"] = tests
        # heatmap: models x held-out sources
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(1 + 0.8 * mat.shape[0], 0.45 * mat.shape[1] + 1.5))
            im = ax.imshow(mat.T.to_numpy(), vmin=0.4, vmax=1.0, cmap="viridis", aspect="auto")
            ax.set_xticks(range(mat.shape[0]), mat.index, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(mat.shape[1]), mat.columns, fontsize=7)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.iloc[i, j]
                    if not np.isnan(v):
                        ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w")
            fig.colorbar(im, ax=ax, label="balanced accuracy")
            ax.set_title(f"LODO balanced accuracy ({regime})", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(out, f"fig3_lodo_heatmap_{regime}.png"), dpi=200)
            plt.close(fig)
        except Exception as e:
            print("heatmap failed:", e)
    # ranking agreement LODO vs external
    if len(ext):
        a_ = per_seed_protocol_mean(r, "lodo").groupby(["regime", "model"])["bal_acc"].mean()
        b_ = r[r["protocol"] == "external"].groupby(["regime", "model"])["bal_acc"].mean()
        j = pd.concat([a_, b_], axis=1, keys=["lodo", "external"]).dropna()
        if len(j) >= 3:
            from scipy.stats import spearmanr
            rho, p = spearmanr(j["lodo"], j["external"])
            st["spearman_lodo_vs_external"] = {"rho": float(rho), "p": float(p), "n": len(j)}
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(st, f, indent=2, default=float)

    # ---- Fig 2: overlap matrix
    om_fn = io.p("audit", "overlap_matrix.csv")
    if os.path.exists(om_fn):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        om = pd.read_csv(om_fn, index_col=0)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(om.to_numpy() * 100, cmap="magma_r", vmin=0)
        ax.set_xticks(range(len(om)), om.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(om)), om.index, fontsize=7)
        for i in range(len(om)):
            for j in range(len(om)):
                v = om.iloc[i, j] * 100
                if v >= 0.5:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6)
        fig.colorbar(im, ax=ax, label="% of row source's images duplicated in column source")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig2_overlap.png"), dpi=200)
        plt.close(fig)
    print("Results written to", out)
    print(open(os.path.join(out, "stats.json")).read()[:3000])


if __name__ == "__main__":
    main()

"""Near-duplicate audit: candidate pairs, manual review, groups and families.

Workflow
1. hash_candidates()      pHash/dHash pairs with Hamming distance <= threshold
2. embedding_candidates() DINOv2 cosine-similarity pairs >= threshold (GPU step)
3. write_review_sheet()   CSV + contact-sheet PNGs for two annotators
4. groups_from_pairs()    union-find over confirmed pairs -> duplicate groups
5. families_from_groups() merge sources when > 50% of either one duplicates the other
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from .phash import dhash, hamming_pairs, phash

AUTO_ACCEPT_HAMMING = 0     # identical perceptual hash -> duplicate without review
PHASH_THRESHOLD = 10        # as in Aboshosha et al. (2026)
EMBED_THRESHOLD = 0.95


def compute_hashes(index: pd.DataFrame) -> pd.DataFrame:
    out = index.copy()
    ph, dh, ok = [], [], []
    for p in out["path"]:
        try:
            ph.append(phash(p))
            dh.append(dhash(p))
            ok.append(True)
        except Exception:
            ph.append(0)
            dh.append(0)
            ok.append(False)
    out["phash"] = pd.Series(np.array(ph, dtype=np.uint64), index=out.index)
    out["dhash"] = pd.Series(np.array(dh, dtype=np.uint64), index=out.index)
    out["hash_ok"] = ok
    return out


def hash_candidates(index: pd.DataFrame, max_dist: int = PHASH_THRESHOLD) -> pd.DataFrame:
    ok = index[index["hash_ok"]].reset_index(drop=True)
    hashes = ok["phash"].to_numpy(dtype=np.uint64)
    recs = []
    for i, j, d in hamming_pairs(hashes, max_dist):
        recs.append({"a": ok.at[i, "image_id"], "b": ok.at[j, "image_id"],
                     "phash_dist": d, "source": "phash"})
    # identical files across datasets (md5) are always duplicates
    if "md5" in index and index["md5"].astype(bool).any():
        for _, g in index[index["md5"] != ""].groupby("md5"):
            ids = list(g["image_id"])
            for k in range(1, len(ids)):
                recs.append({"a": ids[0], "b": ids[k], "phash_dist": 0, "source": "md5"})
    df = pd.DataFrame(recs, columns=["a", "b", "phash_dist", "source"])
    return df.drop_duplicates(subset=["a", "b"])


def embedding_candidates(ids: list[str], emb: np.ndarray, thr: float = EMBED_THRESHOLD,
                         block: int = 2048) -> pd.DataFrame:
    x = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    recs = []
    for s in range(0, len(x), block):
        sim = x[s:s + block] @ x.T
        ii, jj = np.nonzero(sim >= thr)
        keep = jj > (ii + s)
        for i, j in zip(ii[keep] + s, jj[keep]):
            recs.append({"a": ids[i], "b": ids[j], "cosine": float(sim[i - s, j]),
                         "source": "embedding"})
    return pd.DataFrame(recs, columns=["a", "b", "cosine", "source"])


def merge_candidates(hash_df: pd.DataFrame, emb_df: Optional[pd.DataFrame],
                     index: pd.DataFrame) -> pd.DataFrame:
    frames = [hash_df.assign(cosine=np.nan)]
    if emb_df is not None and len(emb_df):
        frames.append(emb_df.assign(phash_dist=np.nan))
    c = pd.concat(frames, ignore_index=True)
    # order each pair so (a, b) is canonical, then aggregate evidence
    swap = c["a"] > c["b"]
    c.loc[swap, ["a", "b"]] = c.loc[swap, ["b", "a"]].to_numpy()
    agg = c.groupby(["a", "b"], as_index=False).agg(
        phash_dist=("phash_dist", "min"),
        cosine=("cosine", "max"),
        sources=("source", lambda s: "+".join(sorted(set(s)))))
    agg["phash_dist"] = agg["phash_dist"].fillna(64)
    agg["cosine"] = agg["cosine"].fillna(0.0)
    meta = index.set_index("image_id")
    agg["dataset_a"] = agg["a"].map(meta["dataset"])
    agg["dataset_b"] = agg["b"].map(meta["dataset"])
    agg["label_a"] = agg["a"].map(meta["label"])
    agg["label_b"] = agg["b"].map(meta["label"])
    agg["cross_dataset"] = agg["dataset_a"] != agg["dataset_b"]
    agg["auto_duplicate"] = (agg["phash_dist"] <= AUTO_ACCEPT_HAMMING) | agg["sources"].str.contains("md5")
    # reviewers fill these two columns with: duplicate / near_duplicate / distinct
    agg["reviewer_1"] = np.where(agg["auto_duplicate"], "duplicate", "")
    agg["reviewer_2"] = np.where(agg["auto_duplicate"], "duplicate", "")
    return agg


# Automatic duplicate rule (Amendment 2): pHash distance <= 8 or DINOv2 cosine >= 0.95
# (or byte-identical / identical hash). Chosen after the blind two-reviewer validation, in
# which the originally registered rule (<= 4, >= 0.98) had precision 1.00 but recall 0.64
# within the candidate set, and this rule precision 0.95 and recall 0.99.
RULE_HAMMING = 8
RULE_COSINE = 0.95
ORIGINAL_RULE = (4, 0.98)   # the pre-registered rule; kept as a sensitivity analysis


def rule_duplicate(cands: pd.DataFrame, ham: int = RULE_HAMMING, cos: float = RULE_COSINE) -> pd.Series:
    return (cands["auto_duplicate"].astype(bool) | (cands["phash_dist"] <= ham) | (cands["cosine"] >= cos))


def review_sample(cands: pd.DataFrame, n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Stratified random sample of candidate pairs for the two-reviewer validation.

    Half the pairs are drawn from pairs the automatic rule calls duplicate (to
    estimate its precision) and half from candidates it rejects (to estimate how
    many true duplicates it misses), each balanced between within- and
    cross-dataset pairs. Reviewers are blind to the rule's decision.
    """
    c = cands.copy()
    c["rule_duplicate"] = rule_duplicate(c)
    parts = []
    for rule in (True, False):
        for cross in (True, False):
            sub = c[(c["rule_duplicate"] == rule) & (c["cross_dataset"] == cross)]
            parts.append(sub.sample(min(len(sub), n // 4), random_state=seed))
    smp = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    smp.insert(0, "pair_no", range(len(smp)))
    smp["reviewer_1"] = ""
    smp["reviewer_2"] = ""
    keep = ["pair_no", "a", "b", "dataset_a", "dataset_b", "label_a", "label_b", "cross_dataset",
            "phash_dist", "cosine", "rule_duplicate", "reviewer_1", "reviewer_2"]
    return smp[keep]


REVIEW_CHOICES = ("duplicate", "near_duplicate", "distinct")


def review_forms(sample: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the sample into a hidden key (with the rule's decision) and a blind
    form for the reviewers (pair number only; images are on the sheets)."""
    key = sample.drop(columns=["reviewer_1", "reviewer_2"])
    form = pd.DataFrame({"pair_no": sample["pair_no"], "decision": ""})
    return key, form


def load_reviewer_forms(key: pd.DataFrame, search_dirs: list[str]) -> Optional[pd.DataFrame]:
    """Find review_form_R1*.csv and review_form_R2*.csv (filled by the two
    reviewers) in any of `search_dirs` (recursively) and join them to the key."""
    import glob
    def n_filled(path):
        try:
            f = pd.read_csv(path, keep_default_na=False)
            return int(f["decision"].astype(str).str.strip().ne("").sum())
        except Exception:
            return -1

    # The audit writes blank forms next to the key, so take the copy with the
    # most answers rather than the first one found.
    found = {}
    for r in ("R1", "R2"):
        best, best_n = None, 0
        for d in search_dirs:
            for h in sorted(glob.glob(os.path.join(d, "**", f"review_form_{r}*.csv"), recursive=True)):
                n = n_filled(h)
                if n > best_n:
                    best, best_n = h, n
        if best:
            found[r] = best
    if len(found) < 2:
        return None
    out = key.copy()
    for r, col in (("R1", "reviewer_1"), ("R2", "reviewer_2")):
        f = pd.read_csv(found[r], keep_default_na=False)
        dec = f.set_index("pair_no")["decision"].astype(str).str.strip().str.lower()
        dec = dec.str.replace("-", "_").str.replace(" ", "_")
        dec = dec.replace({"d": "duplicate", "dup": "duplicate", "n": "near_duplicate", "near": "near_duplicate",
                           "x": "distinct", "no": "distinct", "different": "distinct"})
        bad = sorted(set(dec[dec != ""]) - set(REVIEW_CHOICES))
        if bad:
            raise ValueError(f"{found[r]}: unknown decisions {bad}; use {REVIEW_CHOICES}")
        out[col] = out["pair_no"].map(dec).fillna("")
        print(f"reviewer form {r}: {found[r]} ({int((out[col] != '').sum())} pairs filled)")
    return out


def rule_validation(sample: pd.DataFrame) -> dict:
    """Agreement of the automatic rule with the two reviewers' consensus."""
    dup = {"duplicate", "near_duplicate"}
    r1, r2 = sample["reviewer_1"].astype(str).str.strip(), sample["reviewer_2"].astype(str).str.strip()
    done = (r1 != "") & (r2 != "")
    s = sample[done]
    agree = (r1[done].isin(dup) == r2[done].isin(dup))
    cons = r1[done].isin(dup) & agree
    rule = s["rule_duplicate"].astype(str).str.lower().eq("true")
    tp = int((rule & cons).sum()); fp = int((rule & ~r1[done].isin(dup) & agree).sum())
    fn = int((~rule & cons).sum()); tn = int((~rule & ~r1[done].isin(dup) & agree).sum())
    return {"pairs_reviewed": int(done.sum()), "reviewer_agreement": float(agree.mean()) if len(s) else float("nan"),
            "kappa": cohen_kappa(r1.where(done, ""), r2.where(done, "")),
            "kappa_binary": cohen_kappa(r1[done].isin(dup).map({True: "dup", False: "distinct"}),
                                        r2[done].isin(dup).map({True: "dup", False: "distinct"})),
            "rule_precision": tp / (tp + fp) if tp + fp else float("nan"),
            "rule_miss_rate_among_rejected": fn / (fn + tn) if fn + tn else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "disagreements": int((~agree).sum())}


def write_review_sheet(cands: pd.DataFrame, index: pd.DataFrame, out_dir: str,
                       per_page: int = 20) -> list[str]:
    """Save side-by-side thumbnails of the pairs to review (all pairs without
    auto_duplicate=True, or every row when the column is absent)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    todo = (cands[~cands["auto_duplicate"]] if "auto_duplicate" in cands else cands).reset_index(drop=True)
    paths = index.set_index("image_id")["path"]
    files = []
    for page in range(0, len(todo), per_page):
        chunk = todo.iloc[page:page + per_page]
        fig, axes = plt.subplots(len(chunk), 2, figsize=(6, 2.6 * len(chunk)))
        axes = np.atleast_2d(axes)
        for r, (_, row) in enumerate(chunk.iterrows()):
            for c, key in enumerate(["a", "b"]):
                ax = axes[r, c]
                try:
                    ax.imshow(Image.open(paths[row[key]]).convert("RGB"))
                except Exception:
                    pass
                if "pair_no" in row:   # blind sheet: pair number and side only
                    ax.set_title(f"Pair {row['pair_no']}  -  {'A' if c == 0 else 'B'}", fontsize=9)
                else:
                    ax.set_title(f"#{page + r} {row[key]}"[:60], fontsize=6)
                ax.axis("off")
        fig.tight_layout()
        first = chunk["pair_no"].iloc[0] if "pair_no" in chunk else page
        last = chunk["pair_no"].iloc[-1] if "pair_no" in chunk else page + len(chunk) - 1
        fn = os.path.join(out_dir, f"pairs_{first:03d}-{last:03d}.png")
        fig.savefig(fn, dpi=80)
        plt.close(fig)
        files.append(fn)
    return files


def cohen_kappa(r1: pd.Series, r2: pd.Series) -> float:
    from sklearn.metrics import cohen_kappa_score
    m = (r1 != "") & (r2 != "")
    if m.sum() < 2:
        return float("nan")
    return float(cohen_kappa_score(r1[m], r2[m]))


def confirmed_pairs(reviewed: pd.DataFrame, provisional_cos: float = RULE_COSINE,
                    provisional_ham: int = RULE_HAMMING) -> tuple[pd.DataFrame, bool]:
    """Pairs judged duplicate by consensus.

    A pair counts when both reviewers say duplicate/near_duplicate, or when a
    'final' column (written after discussion) says so. If no manual review has
    been filled in yet, a provisional automatic rule is used and the second
    return value is True so results can be marked provisional.
    """
    dup = {"duplicate", "near_duplicate"}
    has_review = ((reviewed["reviewer_1"] != "") & ~reviewed["auto_duplicate"]).any()
    if "final" in reviewed and reviewed["final"].astype(str).str.len().gt(0).any():
        keep = reviewed["final"].isin(dup) | reviewed["auto_duplicate"]
        return reviewed[keep], False
    if has_review:
        keep = (reviewed["reviewer_1"].isin(dup) & reviewed["reviewer_2"].isin(dup)) | reviewed["auto_duplicate"]
        return reviewed[keep], False
    keep = rule_duplicate(reviewed, provisional_ham, provisional_cos)
    return reviewed[keep], True


class _UF:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def groups_from_pairs(index: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    """Duplicate-group id per image (images without duplicates get their own group).

    Patients also join groups: all images of one patient share a group, so
    group-aware splitting keeps a patient on one side.
    """
    uf = _UF(index["image_id"])
    for a, b in zip(pairs["a"], pairs["b"]):
        if a in uf.p and b in uf.p:
            uf.union(a, b)
    for _, g in index.groupby("patient"):
        ids = list(g["image_id"])
        for k in range(1, len(ids)):
            uf.union(ids[0], ids[k])
    roots = index["image_id"].map(uf.find)
    codes = {r: i for i, r in enumerate(sorted(set(roots)))}
    return roots.map(codes).rename("group")


def overlap_matrix(index: pd.DataFrame) -> pd.DataFrame:
    """Share of each source's images (rows) that sit in a group with another source (cols)."""
    ds = sorted(index["dataset"].unique())
    m = pd.DataFrame(0.0, index=ds, columns=ds)
    by_group = index.groupby("group")["dataset"].apply(set)
    gmap = index["group"].map(by_group)
    for d in ds:
        sub = gmap[index["dataset"] == d]
        n = len(sub)
        if n == 0:
            continue
        for e in ds:
            if e != d:
                m.loc[d, e] = sum(e in s for s in sub) / n
    return m


def families_from_groups(index: pd.DataFrame, threshold: float = 0.50,
                         dev_only: bool = True) -> dict[str, str]:
    """Merge development sources whose overlap exceeds `threshold` into families."""
    sub = index[index["role"] == "dev"] if dev_only else index
    m = overlap_matrix(sub)
    uf = _UF(list(m.index))
    # predefined families from config (e.g. MSLD + MSLD v2.0)
    for fam, g in sub.groupby("family"):
        names = sorted(g["dataset"].unique())
        for k in range(1, len(names)):
            uf.union(names[0], names[k])
    for d in m.index:
        for e in m.columns:
            if d != e and max(m.loc[d, e], m.loc[e, d]) > threshold:
                uf.union(d, e)
    return {d: uf.find(d) for d in m.index}

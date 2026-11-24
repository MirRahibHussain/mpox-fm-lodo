"""Leakage-free splits: pooled grouped CV, LODO, external test, ablations.

Every split is a dict with keys: name, protocol, task, held_out, train, val,
test (lists of image_id) and is checked by `assert_no_leakage`.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Optional

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .config import FOUR_CLASS

NEGATIVE_ONLY = ("dermnet", "skin_disease")


def _val_split(df: pd.DataFrame, frac: float, seed: int, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grouped, roughly stratified train/val split (by class, then group)."""
    tr_parts, va_parts = [], []
    for _, sub in df.groupby(label_col):
        if sub["group"].nunique() < 2:
            tr_parts.append(sub)
            continue
        gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        tr, va = next(gss.split(sub, groups=sub["group"]))
        tr_parts.append(sub.iloc[tr])
        va_parts.append(sub.iloc[va])
    if not tr_parts:
        return df.iloc[:0], df.iloc[:0]
    tr = pd.concat(tr_parts)
    va = pd.concat(va_parts) if va_parts else tr.iloc[:0]
    # a group split by class stratification must not straddle train/val
    clash = set(tr["group"]) & set(va["group"])
    if clash:
        va = va[~va["group"].isin(clash)]
    return tr, va


def _pack(name, protocol, task, held_out, tr, va, te, notes=""):
    return {"name": name, "protocol": protocol, "task": task, "held_out": held_out,
            "train": list(tr["image_id"]), "val": list(va["image_id"]),
            "test": list(te["image_id"]), "notes": notes}


def task_frame(index: pd.DataFrame, task: str) -> tuple[pd.DataFrame, str]:
    if task == "binary":
        return index, "y_bin"
    if task == "four_class":
        sub = index[index["four_class"] & index["label"].isin(FOUR_CLASS)].copy()
        sub["y4"] = sub["label"].map({c: i for i, c in enumerate(FOUR_CLASS)})
        return sub, "y4"
    raise ValueError(task)


def pooled_folds(index: pd.DataFrame, task: str = "binary", k: int = 5, seed: int = 0,
                 val_frac: float = 0.10) -> list[dict]:
    dev, ycol = task_frame(index[index["role"] == "dev"], task)
    dev = dev.reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    out = []
    for f, (tr_idx, te_idx) in enumerate(sgkf.split(dev, dev[ycol], groups=dev["group"])):
        tr, va = _val_split(dev.iloc[tr_idx], val_frac, seed + f, ycol)
        out.append(_pack(f"pooled_{task}_f{f}", "pooled", task, f"fold{f}", tr, va, dev.iloc[te_idx]))
    return out


def pooled_naive_folds(index: pd.DataFrame, k: int = 5, seed: int = 0,
                       val_frac: float = 0.10) -> list[dict]:
    """Conventional image-level folds that ignore duplicate groups (binary task).

    Used only to measure how much leakage inflates pooled accuracy (Table 5);
    never used for model selection.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split
    dev = index[index["role"] == "dev"].reset_index(drop=True)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    out = []
    for f, (tr_idx, te_idx) in enumerate(skf.split(dev, dev["y_bin"])):
        tr = dev.iloc[tr_idx]
        tr, va = train_test_split(tr, test_size=val_frac, stratify=tr["y_bin"], random_state=seed + f)
        out.append(_pack(f"pooled_naive_binary_f{f}", "pooled_naive", "binary", f"fold{f}", tr, va,
                         dev.iloc[te_idx], notes="image-level split, duplicates NOT grouped"))
    return out


def lodo_folds(index: pd.DataFrame, families: dict[str, str], task: str = "binary",
               seed: int = 0, val_frac: float = 0.10,
               exclude_from_train: Iterable[str] = ()) -> list[dict]:
    """One fold per family. Training images sharing a duplicate group with any
    held-out image are removed from that fold's training set."""
    dev, ycol = task_frame(index[index["role"] == "dev"], task)
    dev = dev.copy()
    dev["lodo_family"] = dev["dataset"].map(families)
    out = []
    for fam in sorted(dev["lodo_family"].unique()):
        te = dev[dev["lodo_family"] == fam]
        members = sorted(te["dataset"].unique())
        if exclude_from_train and set(members) <= set(exclude_from_train):
            continue
        has_pos = (te["y_bin"] == 1).any()
        if task == "four_class" and te[ycol].nunique() < 2:
            continue
        pool = dev[(dev["lodo_family"] != fam) & ~dev["dataset"].isin(list(exclude_from_train))]
        leak = set(te["group"])
        n_before = len(pool)
        pool = pool[~pool["group"].isin(leak)]
        if pool.empty or pool[ycol].nunique() < 2:
            print(f"skip LODO fold {task}/{fam}: training pool has <2 classes after holding it out")
            continue
        tr, va = _val_split(pool, val_frac, seed, ycol)
        kind = "lodo" if has_pos else "lodo_specificity"
        suffix = "_noneg" if exclude_from_train else ""
        out.append(_pack(f"{kind}_{task}_{fam}{suffix}", kind + suffix, task, "+".join(members), tr, va, te,
                         notes=f"removed {n_before - len(pool)} training images duplicated in held-out family"))
    return out


def external_split(index: pd.DataFrame, task: str = "binary", seed: int = 0,
                   val_frac: float = 0.10) -> list[dict]:
    """Train on the whole development pool; test once on each external dataset,
    both after removing images that duplicate development images (primary)
    and with them kept (to measure inflation)."""
    dev, ycol = task_frame(index[index["role"] == "dev"], task)
    ext_all = index[index["role"] == "external"]
    if task == "four_class":
        ext_all = ext_all[ext_all["label"].isin(FOUR_CLASS)]
    tr, va = _val_split(dev, val_frac, seed, ycol)
    dev_groups = set(dev["group"])
    out = []
    for name, te in ext_all.groupby("dataset"):
        clean = te[~te["group"].isin(dev_groups)]
        out.append(_pack(f"external_{task}_{name}", "external", task, name, tr, va, clean,
                         notes=f"{len(te) - len(clean)} of {len(te)} images removed as duplicates of development data"))
        out.append(_pack(f"external_{task}_{name}_withdup", "external_withdup", task, name, tr, va, te,
                         notes="duplicates kept (inflation analysis only)"))
    return out


def assert_no_leakage(split: dict, index: pd.DataFrame) -> None:
    if split["protocol"] in ("external_withdup", "pooled_naive"):
        return
    meta = index.set_index("image_id")
    tr = set(split["train"]) | set(split["val"])
    te = set(split["test"])
    assert not (tr & te), f"{split['name']}: image in both train and test"
    g_tr = set(meta.loc[list(tr), "group"])
    g_te = set(meta.loc[list(te), "group"])
    both = g_tr & g_te
    assert not both, f"{split['name']}: {len(both)} duplicate/patient groups cross train and test"
    g_va = set(meta.loc[split["val"], "group"]) if split["val"] else set()
    g_tr_only = set(meta.loc[split["train"], "group"])
    assert not (g_va & g_tr_only), f"{split['name']}: groups cross train and validation"


def save_splits(splits: list[dict], out_dir: str) -> list[str]:
    """Write the splits and move split files left over from an earlier run (for
    example under another duplicate rule) to <out_dir>_superseded, so later
    stages never pick them up. Returns the names that were moved."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    keep = {s["name"] for s in splits}
    stale = [fn[:-5] for fn in os.listdir(out_dir) if fn.endswith(".json") and fn[:-5] not in keep]
    if stale:
        dest = out_dir.rstrip("/") + "_superseded"
        os.makedirs(dest, exist_ok=True)
        for name in stale:
            shutil.move(os.path.join(out_dir, name + ".json"), os.path.join(dest, name + ".json"))
    for s in splits:
        with open(os.path.join(out_dir, s["name"] + ".json"), "w") as f:
            json.dump(s, f)
    return stale


def load_splits(out_dir: str, protocol: Optional[str] = None, task: Optional[str] = None) -> list[dict]:
    res = []
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(out_dir, fn)) as f:
                s = json.load(f)
            if (protocol is None or s["protocol"] == protocol) and (task is None or s["task"] == task):
                res.append(s)
    return res


def summarize(splits: list[dict], index: pd.DataFrame) -> pd.DataFrame:
    meta = index.set_index("image_id")
    rows = []
    for s in splits:
        te = meta.loc[s["test"]] if s["test"] else meta.iloc[:0]
        rows.append({"split": s["name"], "protocol": s["protocol"], "task": s["task"],
                     "held_out": s["held_out"], "n_train": len(s["train"]), "n_val": len(s["val"]),
                     "n_test": len(s["test"]), "test_mpox": int((te["y_bin"] == 1).sum()) if len(te) else 0,
                     "notes": s.get("notes", "")})
    return pd.DataFrame(rows)

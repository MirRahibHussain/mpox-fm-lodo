"""Build one index row per original image across all datasets."""
from __future__ import annotations

import hashlib
import os
import re
from typing import Iterable, Optional

import pandas as pd

from .config import (DATASETS, DATA_ROOT, IMAGE_EXTS, SKIP_SUBSTRINGS, SKIP_TOKENS,
                     DatasetSpec, binary_label, canonical_label)


def should_skip(path: str) -> bool:
    for part in re.split(r"[\\/]", path.lower())[:-1]:
        if any(s in part for s in SKIP_SUBSTRINGS):
            return True
        tokens = re.split(r"[\s_\-.]+", part)
        if tokens and tokens[0] in SKIP_TOKENS:
            return True
    return False


def find_root(spec: DatasetSpec, data_root: str = DATA_ROOT) -> Optional[str]:
    """Locate a dataset folder under any of the ':'-separated data roots.

    Top-level matches in every root are tried before any nested search, and a
    nested search only accepts Kaggle's /datasets/<owner>/<slug> layout, so a
    class folder inside another dataset (e.g. ".../Monkeypox") is never taken
    for a dataset of the same name.
    """
    roots = [r for r in data_root.split(":") if r and os.path.isdir(r)]
    want = spec.folder.lower()
    for root in roots:
        exact = os.path.join(root, spec.folder)
        if os.path.isdir(exact):
            return exact
        for name in os.listdir(root):
            if name.lower() == want and os.path.isdir(os.path.join(root, name)):
                return os.path.join(root, name)
    for root in roots:
        nest = os.path.join(root, "datasets")
        if not os.path.isdir(nest):
            continue
        for owner in sorted(os.listdir(nest)):
            od = os.path.join(nest, owner)
            if not os.path.isdir(od):
                continue
            for name in os.listdir(od):
                if name.lower() == want and os.path.isdir(os.path.join(od, name)):
                    return os.path.join(od, name)
    return None


def file_md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_images(root: str) -> Iterable[str]:
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                yield os.path.join(dirpath, fn)


def build_index(specs: list[DatasetSpec] = DATASETS, data_root: str = DATA_ROOT,
                compute_md5: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (index, report). index has one row per unique original image file.

    Exact byte-identical files inside one dataset (e.g. the same image in a
    train folder and a fold folder) are collapsed to one row.
    """
    rows, report = [], []
    for spec in specs:
        root = find_root(spec, data_root)
        if root is None:
            report.append({"dataset": spec.name, "found": False, "images": 0,
                           "skipped_aug": 0, "unlabelled": 0, "root": "", "unlabelled_dirs": ""})
            continue
        n_skip = n_unl = 0
        unl_dirs: dict[str, int] = {}
        seen_md5 = set()
        for p in iter_images(root):
            rel = os.path.relpath(p, root)
            if should_skip(rel):
                n_skip += 1
                continue
            label = canonical_label(rel, spec)
            if label is None:
                n_unl += 1
                d = os.path.dirname(rel) or "."
                unl_dirs[d] = unl_dirs.get(d, 0) + 1
                continue
            md5 = file_md5(p) if compute_md5 else ""
            if compute_md5:
                if md5 in seen_md5:
                    continue
                seen_md5.add(md5)
            stem = os.path.splitext(os.path.basename(p))[0]
            patient = None
            if spec.patient_regex:
                m = re.match(spec.patient_regex, stem)
                if m:
                    patient = f"{spec.name}:{m.group(1).upper()}"
            rows.append({
                "image_id": f"{spec.name}/{rel}",
                "path": p,
                "dataset": spec.name,
                "role": spec.role,
                "family": spec.family or spec.name,
                "label": label,
                "y_bin": binary_label(label),
                "four_class": spec.four_class,
                "patient": patient or f"{spec.name}:{rel}",
                "md5": md5,
            })
        report.append({"dataset": spec.name, "found": True, "root": root,
                       "images": sum(r["dataset"] == spec.name for r in rows),
                       "skipped_aug": n_skip, "unlabelled": n_unl,
                       "unlabelled_dirs": "; ".join(f"{k} ({v})" for k, v in
                                                    sorted(unl_dirs.items(), key=lambda kv: -kv[1])[:6])})
    return pd.DataFrame(rows), pd.DataFrame(report)

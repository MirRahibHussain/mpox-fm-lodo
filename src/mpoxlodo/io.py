"""Paths and prediction files."""
from __future__ import annotations

import os

import pandas as pd

from .config import WORK_DIR


def p(*parts: str) -> str:
    path = os.path.join(WORK_DIR, *parts)
    os.makedirs(os.path.dirname(path) if os.path.splitext(path)[1] else path, exist_ok=True)
    return path


def index_path() -> str:
    return p("index", "index.csv")


def load_index() -> pd.DataFrame:
    return pd.read_csv(index_path(), keep_default_na=False,
                       dtype={"patient": str, "md5": str, "image_id": str})


def pred_path(regime: str, model: str, split: str, seed: int, val: bool = False) -> str:
    suffix = "__val" if val else ""
    return p("preds", regime, model, f"{split}__s{seed}{suffix}.csv")


def save_pred(df: pd.DataFrame, regime: str, model: str, split: str, seed: int) -> None:
    df.to_csv(pred_path(regime, model, split, seed), index=False)
    val = df.attrs.get("val")
    if val is not None and len(val):
        val.to_csv(pred_path(regime, model, split, seed, val=True), index=False)


def has_pred(regime: str, model: str, split: str, seed: int) -> bool:
    return os.path.exists(pred_path(regime, model, split, seed))

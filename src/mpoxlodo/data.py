"""Torch dataset and transforms."""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image


def build_transforms(img_size: int, mean, std, train: bool):
    from torchvision import transforms as T
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            # brightness/contrast only: hue and saturation are left alone so skin tone is not altered
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


class ImageTable:
    """Minimal map-style dataset over (path, label) pairs."""

    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.t = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.t(im.convert("RGB"))
        return x, self.labels[i]


def subset(index: pd.DataFrame, ids, label_col: str) -> tuple[list, np.ndarray]:
    meta = index.set_index("image_id").loc[list(ids)]
    return list(meta["path"]), meta[label_col].to_numpy()


def cap_dataset(ids, index: pd.DataFrame, dataset: str, cap: int, seed: int):
    """Randomly keep at most `cap` images of one large source (e.g. Dermnet) in a training list."""
    if cap <= 0:
        return list(ids)
    meta = index.set_index("image_id")
    ids = list(ids)
    big = [i for i in ids if meta.at[i, "dataset"] == dataset]
    if len(big) <= cap:
        return ids
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(big, size=cap, replace=False))
    return [i for i in ids if meta.at[i, "dataset"] != dataset or i in keep]

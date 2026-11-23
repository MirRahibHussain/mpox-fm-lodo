"""Zero-shot classification with CLIP, BiomedCLIP and MedSigLIP text prompts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ImageTable, build_transforms, subset
from .models import load_backbone

TEMPLATES = [
    "a clinical photograph of {}.",
    "a close-up photo of skin with {}.",
    "a dermatology image showing {}.",
    "a photo of a skin rash caused by {}.",
    "an image of {} on human skin.",
]
BINARY_CLASSES = {0: ["healthy skin or a skin disease other than mpox", "a non-mpox skin condition"],
                  1: ["mpox skin lesions", "monkeypox lesions"]}
FOUR_CLASSES = {0: ["mpox skin lesions", "monkeypox lesions"], 1: ["chickenpox lesions"],
                2: ["measles rash"], 3: ["healthy skin with no lesions"]}


def class_text_embeddings(encode_text, classes: dict, device) -> "torch.Tensor":
    import torch
    rows = []
    for k in sorted(classes):
        texts = [t.format(c) for c in classes[k] for t in TEMPLATES]
        with torch.no_grad():
            e = encode_text(texts, device).float()
        e = e / e.norm(dim=-1, keepdim=True)
        m = e.mean(0)
        rows.append(m / m.norm())
    return torch.stack(rows)


def run_split(split: dict, index: pd.DataFrame, key: str, label_col: str, task: str,
              device: str = "cuda", batch_size: int = 64) -> pd.DataFrame:
    import torch
    from torch.utils.data import DataLoader

    model, spec, extras = load_backbone(key)
    if "encode_text" not in extras:
        raise ValueError(f"{key} has no text encoder")
    model = model.to(device).eval()
    classes = BINARY_CLASSES if task == "binary" else FOUR_CLASSES
    T = class_text_embeddings(extras["encode_text"], classes, device)
    paths, ys = subset(index, split["test"], label_col)
    dl = DataLoader(ImageTable(paths, ys, build_transforms(spec.img_size, spec.mean, spec.std, False)),
                    batch_size=batch_size, shuffle=False, num_workers=2)
    probs = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            f = extras["image_head"](x, device) if "image_head" in extras else model(x)
            f = f.float()
            f = f / f.norm(dim=-1, keepdim=True)
            logits = extras.get("logit_scale", 100.0) * f @ T.T
            probs.append(torch.softmax(logits, 1).cpu().numpy())
    p = np.concatenate(probs) if probs else np.zeros((0, len(classes)))
    out = pd.DataFrame({"image_id": split["test"], "y": ys})
    for k in range(p.shape[1]):
        out[f"p{k}"] = p[:, k]
    out.attrs["val"] = pd.DataFrame({"y": []})
    return out

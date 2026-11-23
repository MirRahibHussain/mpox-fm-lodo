"""Full fine-tuning and LP-then-FT for one split and one seed."""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .config import TrainConfig
from .data import ImageTable, build_transforms, cap_dataset, subset
from .models import load_backbone


def set_seed(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loaders(split, index, label_col, spec, cfg: TrainConfig, seed: int, smoke: bool):
    from torch.utils.data import DataLoader
    tr_ids = cap_dataset(split["train"], index, "dermnet", cfg.dermnet_cap, seed)
    va_ids, te_ids = split["val"], split["test"]
    if smoke:
        tr_ids, va_ids, te_ids = tr_ids[:64], va_ids[:32], te_ids[:32]
    ttr = build_transforms(spec.img_size, spec.mean, spec.std, True)
    tev = build_transforms(spec.img_size, spec.mean, spec.std, False)
    p_tr, y_tr = subset(index, tr_ids, label_col)
    p_va, y_va = subset(index, va_ids, label_col)
    p_te, y_te = subset(index, te_ids, label_col)
    g = __import__("torch").Generator().manual_seed(seed)
    mk = lambda p, y, t, sh: DataLoader(ImageTable(p, y, t), batch_size=cfg.batch_size, shuffle=sh,
                                         num_workers=cfg.num_workers, pin_memory=True, generator=g if sh else None,
                                         drop_last=sh and len(p) > cfg.batch_size)
    return (mk(p_tr, y_tr, ttr, True), mk(p_va, y_va, tev, False), mk(p_te, y_te, tev, False),
            y_tr, y_va, y_te, list(te_ids))


def _predict(model, head, dl, device, amp):
    import torch
    model.eval()
    head.eval()
    probs, ys = [], []
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=amp):
        for x, y in dl:
            logits = head(model(x.to(device, non_blocking=True)).float())
            probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
            ys.append(y.numpy())
    if not probs:
        return np.zeros((0, head.out_features)), np.zeros(0)
    return np.concatenate(probs), np.concatenate(ys)


def run(split: dict, index: pd.DataFrame, key: str, label_col: str, n_classes: int, seed: int,
        cfg: TrainConfig = TrainConfig(), regime: str = "ft", device: str = "cuda",
        smoke: bool = False) -> pd.DataFrame:
    """regime: 'ft' (full fine-tuning) or 'lpft' (head first, then full)."""
    import torch
    import torch.nn as nn

    set_seed(seed)
    model, spec, _ = load_backbone(key)
    model = model.to(device)
    head = nn.Linear(model.feat_dim, n_classes).to(device)
    dl_tr, dl_va, dl_te, y_tr, y_va, y_te, te_ids = _loaders(split, index, label_col, spec, cfg, seed, smoke)

    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    base_lr = cfg.lr_transformer if spec.transformer else cfg.lr_cnn
    epochs = 1 if smoke else cfg.epochs
    amp = cfg.amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def train_phase(params, lr, n_epochs, freeze_backbone):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
        steps = max(1, n_epochs * len(dl_tr))
        warm = max(1, len(dl_tr)) if n_epochs > 1 else 1
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        best, best_state, bad = -1.0, None, 0
        for ep in range(n_epochs):
            model.train(not freeze_backbone)
            head.train()
            for x, y in dl_tr:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=amp):
                    if freeze_backbone:
                        with torch.no_grad():
                            f = model(x)
                    else:
                        f = model(x)
                    loss = crit(head(f.float()), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            pv, yv = _predict(model, head, dl_va, device, amp)
            score = balanced_accuracy_score(yv, pv.argmax(1)) if len(yv) else 0.0
            if score > best:
                best, bad = score, 0
                best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state[0])
            head.load_state_dict(best_state[1])
        return best

    t0 = time.time()
    if regime == "lpft":
        for p in model.parameters():
            p.requires_grad_(False)
        train_phase(head.parameters(), 1e-3, 1 if smoke else 3, True)
        for p in model.parameters():
            p.requires_grad_(True)
        best = train_phase(list(model.parameters()) + list(head.parameters()), base_lr * 0.5, epochs, False)
    else:
        best = train_phase(list(model.parameters()) + list(head.parameters()), base_lr, epochs, False)

    pv, yv = _predict(model, head, dl_va, device, amp)
    pt, yt = _predict(model, head, dl_te, device, amp)
    out = pd.DataFrame({"image_id": te_ids, "y": yt})
    for k in range(n_classes):
        out[f"p{k}"] = pt[:, k]
    out.attrs["val"] = pd.DataFrame({"y": yv, **{f"p{k}": pv[:, k] for k in range(n_classes)}})
    out.attrs["best_val_bacc"] = best
    out.attrs["minutes"] = (time.time() - t0) / 60
    del model, head
    torch.cuda.empty_cache()
    return out

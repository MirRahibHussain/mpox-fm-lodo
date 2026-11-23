"""Create a tiny synthetic copy of the dataset layout to test the CPU stages.

Images are random textures; some are copied (resized / flipped) across
datasets to plant known duplicates, plus augmented folders that must be skipped.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mpoxlodo.config import DATASETS

rng = np.random.default_rng(0)


def tex(seed, size=96):
    r = np.random.default_rng(seed)
    base = r.integers(0, 255, (8, 8, 3)).astype(np.uint8)
    im = Image.fromarray(base).resize((size, size), Image.Resampling.BICUBIC)
    a = np.asarray(im).astype(np.int16) + r.integers(-10, 10, (size, size, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def main(root):
    planted = []
    seed = 1000
    class_dirs = {
        "default": ["Monkeypox", "Chickenpox", "Measles", "Normal"],
        "msld": ["Monkeypox", "Others"],
        "msld_v2": ["Mpox", "Chickenpox", "Measles", "Healthy", "Cowpox", "HFMD"],
        "mpox_vision": ["Monkeypox", "Chickenpox", "Measles", "Acne"],
        "mpox2025": ["Mpox", "Normal"],
        "mcsi": ["mpox", "chickenpox", "acne", "healthy"],
    }
    shared = {}
    for spec in DATASETS:
        droot = os.path.join(root, spec.folder)
        if spec.all_negative_label:
            dirs = ["Eczema", "Psoriasis", "Acne and Rosacea"]
        else:
            dirs = class_dirs.get(spec.name, class_dirs["default"])
        for c in dirs:
            d = os.path.join(droot, "Original Images", c)
            os.makedirs(d, exist_ok=True)
            n = 6
            for k in range(n):
                seed += 1
                if spec.name == "msld_v2":
                    prefix = {"Mpox": "MKP", "Chickenpox": "CHP", "Measles": "MSL", "Healthy": "HEALTHY",
                              "Cowpox": "CWP", "HFMD": "HFMD"}[c]
                    fn = f"{prefix}_{k // 2:02d}_{k % 2 + 1:02d}.jpg"  # 2 images per patient
                else:
                    fn = f"{c[:3]}_{k}.jpg"
                tex(seed).save(os.path.join(d, fn))
                if c.lower() in ("monkeypox", "mpox") and k == 0:
                    shared.setdefault("mpox", []).append(seed)
            # augmented copies that must be skipped
            ad = os.path.join(droot, "Augmented Images", c)
            os.makedirs(ad, exist_ok=True)
            tex(seed).rotate(90).save(os.path.join(ad, f"aug_{c}.jpg"))
    # plant cross-dataset duplicates: copy MSID's first mpox image (resized) into MID and MCSI
    src = os.path.join(root, "monkeypoxskinimagedataset", "Original Images", "Monkeypox", "Mon_0.jpg")
    im = Image.open(src)
    for tgt in [("monkeypox-images-dataset", "Monkeypox"), ("mcsi", "mpox"), ("mpox-vision", "Monkeypox")]:
        d = os.path.join(root, tgt[0], "Original Images", tgt[1])
        im.resize((128, 128)).save(os.path.join(d, "planted_dup.jpg"))
        planted.append(tgt[0])
    print("synthetic data at", root, "planted duplicates in", planted)


if __name__ == "__main__":
    main(sys.argv[1])

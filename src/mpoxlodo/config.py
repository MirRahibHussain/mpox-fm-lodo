"""Dataset registry and label vocabulary.

Every dataset is described once here. Paths are resolved at run time by
looking for the dataset's folder under DATA_ROOT (default /kaggle/input).
Edit `folder` if your Kaggle input folder has a different name.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

# Several roots may be given, separated by ":" (Kaggle inputs + datasets fetched by kaggle_setup.sh)
DATA_ROOT = os.environ.get("MPOX_DATA_ROOT", "/kaggle/input:/kaggle/working/extra_data")
WORK_DIR = os.environ.get("MPOX_WORK_DIR", "/kaggle/working/mpox_out")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Folder names that mark augmented or derived copies. A path component is
# skipped when it contains one of SKIP_SUBSTRINGS or equals / starts with one
# of SKIP_TOKENS (case-insensitive). Only original images are used.
SKIP_SUBSTRINGS = ("augment", "synthetic", "generated")
SKIP_TOKENS = ("aug", "gan", "ddpm")

# Canonical labels. "other_disease" = any skin condition that is not one of
# the named look-alikes (used for Dermnet, Skin-Disease-Dataset, "Others").
CANONICAL = [
    "mpox", "chickenpox", "measles", "healthy", "cowpox", "hfmd",
    "smallpox", "acne", "roseola", "scarlet", "other_disease",
]

# Folder-name synonyms -> canonical label. Matched on lower-cased path parts.
SYNONYMS = {
    "mpox": ["monkeypox", "monkey pox", "monkey_pox", "mpox", "mkp", "monkeypox_",
             "monkeypox images", "monkeypox_original"],
    "chickenpox": ["chickenpox", "chicken pox", "chicken_pox", "chp", "varicella"],
    "measles": ["measles", "msl"],
    "healthy": ["healthy", "normal", "normal skin", "normal_skin", "healthy skin"],
    "cowpox": ["cowpox", "cow pox", "cwp"],
    "hfmd": ["hfmd", "hand foot mouth", "hand-foot-mouth", "hand, foot and mouth"],
    "smallpox": ["smallpox", "small pox"],
    "acne": ["acne"],
    "roseola": ["roseola"],
    "scarlet": ["scarlet", "scarlet fever"],
    "other_disease": ["others", "other", "non-monkeypox", "non_monkeypox", "nonmonkeypox"],
}

FOUR_CLASS = ["mpox", "chickenpox", "measles", "healthy"]


@dataclass
class DatasetSpec:
    name: str                 # short id used everywhere
    folder: str               # folder name under DATA_ROOT
    role: str                 # "dev" (11 development datasets) or "external"
    family: Optional[str] = None   # sources sharing a family are held out together
    all_negative_label: Optional[str] = None  # e.g. Dermnet: every image -> other_disease
    patient_regex: Optional[str] = None       # regex on file stem; group(1) = patient key
    four_class: bool = False  # contains all four classes of the 4-class task
    source_url: str = ""
    licence: str = ""
    notes: str = ""


DATASETS: list[DatasetSpec] = [
    # ---- development pool: the 11 datasets identified by Andreadis et al. (2026)
    DatasetSpec("dermnet", "dermnet", "dev", all_negative_label="other_disease",
                source_url="https://www.kaggle.com/datasets/shubhamgoel27/dermnet"),
    DatasetSpec("msid", "monkeypoxskinimagedataset", "dev", four_class=True,
                source_url="https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset"),
    DatasetSpec("mid", "monkeypox-images-dataset", "dev", four_class=True,
                source_url="https://www.kaggle.com/datasets/sachinkumar413/monkeypox-images-dataset"),
    DatasetSpec("msld", "monkeypox-skin-lesion-dataset", "dev", family="msld_family",
                patient_regex=r"^([A-Za-z]+\d+)_\d+",
                source_url="https://www.kaggle.com/datasets/nafin59/monkeypox-skin-lesion-dataset"),
    DatasetSpec("data_monkeypox", "data-monkeypox", "dev",
                source_url="https://www.kaggle.com/datasets/ahmadnasayrah/data-monkeypox"),
    DatasetSpec("mpx2022_remastered", "monkeypox-2022-remastered", "dev", four_class=True,
                source_url="https://www.kaggle.com/datasets/maxmelichov/monkeypox-2022-remastered"),
    DatasetSpec("msld_v2", "mpox-skin-lesion-dataset-version-20-msld-v20", "dev",
                family="msld_family", four_class=True,
                patient_regex=r"^([A-Za-z]+_\d+)_\d+",
                source_url="https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20"),
    DatasetSpec("monkeypox_dataset", "monkeypox-dataset", "dev",
                source_url="https://www.kaggle.com/datasets/piyush19/monkeypox-dataset"),
    DatasetSpec("skin_disease", "skin-disease-dataset", "dev", all_negative_label="other_disease",
                source_url="https://www.kaggle.com/datasets/subirbiswas19/skin-disease-dataset"),
    DatasetSpec("mpx_dataset_2022", "monkeypox-dataset-2022", "dev", four_class=True,
                source_url="https://github.com/mahsan2/Monkeypox-dataset-2022",
                notes="GitHub only: clone it in the notebook or upload as a private Kaggle dataset"),
    DatasetSpec("monkeypox_gh", "MonkeyPox", "dev",
                source_url="https://github.com/BihanBanerjee/MonkeyPox",
                notes="GitHub only: clone it in the notebook or upload as a private Kaggle dataset"),
    # ---- external post-review test sets (never used for training or selection)
    DatasetSpec("mpox_vision", "mpox-vision", "external",
                source_url="https://doi.org/10.17632/hytkgj86gj.2", licence="CC BY 4.0",
                notes="Download from Mendeley Data and upload as a private Kaggle dataset"),
    DatasetSpec("mpox2025", "mpox2025", "external",
                source_url="https://github.com/AAM-Mustahid/MicroMPoxNet",
                licence="research and academic use only",
                notes="Google Drive file 1P3ZZ01TSJS4v9uhBy0USzfbMXWbO8QIh (gdown)"),
    DatasetSpec("mcsi", "mcsi", "external",
                source_url="https://doi.org/10.5281/zenodo.8360076", licence="CC BY 4.0",
                notes="https://zenodo.org/records/8360076/files/MCSI.zip"),
]

BY_NAME = {d.name: d for d in DATASETS}


def canonical_label(path: str, spec: DatasetSpec) -> Optional[str]:
    """Map an image path to a canonical label using folder-name synonyms.

    The deepest matching path part wins (class folders are usually the
    parent of the image). Returns None when no label can be inferred.
    """
    if spec.all_negative_label:
        return spec.all_negative_label
    parts = [p.lower() for p in re.split(r"[\\/]", path)[:-1]]
    for part in reversed(parts):
        clean = part.strip().replace("-", " ").replace("_", " ")
        for label, words in SYNONYMS.items():
            for w in words:
                w2 = w.replace("-", " ").replace("_", " ")
                if clean == w2 or clean.startswith(w2 + " ") or clean.endswith(" " + w2):
                    return label
    # file-name prefixes used by MSLD v2.0 (MKP_, CHP_, MSL_, CWP_, HFMD_, HEALTHY_)
    stem = os.path.basename(path).upper()
    prefixes = {"MKP": "mpox", "CHP": "chickenpox", "MSL": "measles", "CWP": "cowpox",
                "HFMD": "hfmd", "HEALTHY": "healthy", "NORMAL": "healthy"}
    for pre, label in prefixes.items():
        if stem.startswith(pre + "_"):
            return label
    # file names that start with a class word (e.g. "mpox_012.jpg", "Chickenpox (3).png")
    low = os.path.splitext(os.path.basename(path))[0].lower()
    for label, words in SYNONYMS.items():
        if label == "other_disease":
            continue
        for w in words:
            w2 = w.replace("_", " ").strip()
            if len(w2) >= 4 and re.match(re.escape(w2) + r"(?![a-z])", low.replace("_", " ").replace("-", " ")):
                return label
    return None


def binary_label(label: str) -> int:
    return 1 if label == "mpox" else 0


@dataclass
class TrainConfig:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    lr_cnn: float = 1e-4
    lr_transformer: float = 5e-5
    weight_decay: float = 0.05
    patience: int = 4
    num_workers: int = 2
    seeds: tuple = (0, 1, 2)
    dermnet_cap: int = 4000   # max Dermnet images per training fold (compute budget)
    val_fraction: float = 0.10
    amp: bool = True
    extra: dict = field(default_factory=dict)

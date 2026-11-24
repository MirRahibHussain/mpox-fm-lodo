#!/usr/bin/env bash
# One-time setup inside a Kaggle notebook (Settings: GPU on, Internet on).
# Installs missing packages and fetches the datasets that are not on Kaggle
# into /kaggle/working/extra_data, which is then searched as a second data root.
set -euo pipefail

pip install -q open_clip_torch gdown 2>/dev/null || true

# Resume: if a previous run's output (this notebook's output, or a dataset made from it) is
# attached as an input, restore its mpox_out folder so finished stages are skipped.
WORK="${MPOX_WORK_DIR:-/kaggle/working/mpox_out}"
if [ ! -f "$WORK/index/index.csv" ] && [ -d /kaggle/input ]; then
  prev=$(find /kaggle/input -maxdepth 7 -path "*mpox_out/index/index.csv" 2>/dev/null | head -1)
  if [ -n "$prev" ]; then
    src=$(dirname "$(dirname "$prev")")
    mkdir -p "$WORK" && cp -r "$src/." "$WORK/" && echo "restored previous outputs from $src"
  fi
fi

EXTRA=/kaggle/working/extra_data
mkdir -p "$EXTRA"
cd "$EXTRA"

# GitHub-only development datasets
[ -d monkeypox-dataset-2022 ] || git clone -q --depth 1 https://github.com/mahsan2/Monkeypox-dataset-2022 monkeypox-dataset-2022 || echo "clone failed: Monkeypox-dataset-2022"
[ -d MonkeyPox ] || git clone -q --depth 1 https://github.com/BihanBanerjee/MonkeyPox MonkeyPox || echo "clone failed: MonkeyPox"

# External test sets (never used for training)
if [ ! -d mcsi ]; then
  wget -q -O MCSI.zip "https://zenodo.org/records/8360076/files/MCSI.zip?download=1" && mkdir -p mcsi && unzip -q MCSI.zip -d mcsi && rm MCSI.zip || echo "download failed: MCSI"
fi
if [ ! -d mpox2025 ]; then
  gdown -q 1P3ZZ01TSJS4v9uhBy0USzfbMXWbO8QIh -O mpox2025.zip && mkdir -p mpox2025 && unzip -q mpox2025.zip -d mpox2025 && rm mpox2025.zip || echo "download failed: Mpox2025 (download it manually from the authors' GitHub link)"
fi
# Check the Hugging Face token without printing it (MedSigLIP is gated)
python - <<'PY' || true
import os
t = (os.environ.get("HF_TOKEN") or "").strip()
print(f"HF token: length {len(t)}, starts with hf_: {t.startswith('hf_')}")
if t:
    try:
        from huggingface_hub import whoami, auth_check
        print("HF user:", whoami(token=t).get("name"))
        auth_check("google/medsiglip-448", token=t)
        print("HF access to google/medsiglip-448: ok")
    except Exception as e:
        print("HF check failed:", type(e).__name__, str(e).splitlines()[0][:200])
PY
# PanDerm ViT-L/16 weights (CC BY-NC-ND 4.0, academic use) from the authors' Google Drive link
if [ ! -f /tmp/panderm/panderm_ll_data6_checkpoint-499.pth ]; then
  mkdir -p /tmp/panderm
  gdown -q 1SwEzaOlFV_gBKf2UzeowMC8z9UH7AQbE -O /tmp/panderm/panderm_ll_data6_checkpoint-499.pth \
    || { rm -f /tmp/panderm/panderm_ll_data6_checkpoint-499.pth; echo "download failed: PanDerm weights (PanDerm will be skipped)"; }
fi
# MPox-Vision (Mendeley Data, CC BY 4.0): public "Download All" archive of version 2
if [ ! -d mpox-vision ]; then
  ok=0
  for u in "https://data.mendeley.com/public-api/zip/hytkgj86gj/download/2" \
           "https://data.mendeley.com/public-api/zip/hytkgj86gj/download"; do
    if wget -q -O mpoxvision.zip "$u" && unzip -tq mpoxvision.zip >/dev/null 2>&1; then ok=1; break; fi
  done
  if [ "$ok" = 1 ]; then
    mkdir -p mpox-vision && unzip -q mpoxvision.zip -d mpox-vision && rm mpoxvision.zip
    # the archive may hold further zip files
    find mpox-vision -name "*.zip" | while read z; do unzip -q -o "$z" -d "${z%.zip}" && rm "$z"; done
    echo "MPox-Vision downloaded: $(find mpox-vision -type f | wc -l) files"
  else
    rm -f mpoxvision.zip
    echo "download failed: MPox-Vision (upload it as a private Kaggle dataset named 'mpox-vision')"
  fi
fi
ls -1 "$EXTRA"

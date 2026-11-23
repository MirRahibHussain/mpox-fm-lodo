#!/usr/bin/env bash
# Run the pipeline stage by stage on Kaggle. Each stage is resumable.
#   bash run_all.sh setup       # packages + GitHub/Zenodo/Drive datasets
#   bash run_all.sh audit       # stages 1-2 (index + duplicate candidates, GPU for embeddings)
#   -> two reviewers fill audit/candidates_reviewed.csv (see README)
#   bash run_all.sh splits      # stage 3 (groups, families, ITA, splits + leakage check)
#   bash run_all.sh features    # stage 4 (GPU, ~1-2 h for 12 models)
#   bash run_all.sh probe       # stages 5-6 (linear probe on CPU, zero-shot on GPU)
#   -> commit PREREGISTRATION_EXTERNAL.md to GitHub BEFORE the next step
#   bash run_all.sh finetune    # stage 7 (GPU, many sessions; re-run until done)
#   bash run_all.sh report      # stages 8-9 (efficiency + all tables/figures)
#   bash run_all.sh smoke       # quick end-to-end test on a few images
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export MPOX_WORK_DIR="${MPOX_WORK_DIR:-/kaggle/working/mpox_out}"
PY="python -u"

case "${1:-}" in
  setup)    bash "$HERE/kaggle_setup.sh" ;;
  audit)    $PY "$HERE/scripts/01_build_index.py" && $PY "$HERE/scripts/02_audit.py" --embed ;;
  splits)   $PY "$HERE/scripts/03_groups_splits.py" ;;
  features) $PY "$HERE/scripts/04_features.py" ;;
  probe)    $PY "$HERE/scripts/05_linear_probe.py" && $PY "$HERE/scripts/06_zeroshot.py" ;;
  finetune) $PY "$HERE/scripts/07_finetune.py" --max-runs "${MAX_RUNS:-12}" ;;
  report)   $PY "$HERE/scripts/08_efficiency.py" && $PY "$HERE/scripts/09_evaluate.py" && $PY "$HERE/scripts/10_sensitivity.py" ;;
  smoke)
    $PY "$HERE/scripts/01_build_index.py"
    $PY "$HERE/scripts/02_audit.py" --no-sheets
    $PY "$HERE/scripts/03_groups_splits.py" --no-ita
    $PY "$HERE/scripts/04_features.py" --smoke --models resnet50 dinov2_b14 clip_b16
    $PY "$HERE/scripts/07_finetune.py" --smoke --models resnet50 --regimes ft --seeds 0 --protocols lodo --max-runs 1
    ;;
  *) sed -n '2,13p' "$0" ;;
esac

#!/usr/bin/env bash
# CPU-only end-to-end test of stages 1, 2, 3, 5 and 9 on synthetic images.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
export MPOX_DATA_ROOT=/tmp/mpox_syn MPOX_WORK_DIR=/tmp/mpox_syn_out
rm -rf "$MPOX_DATA_ROOT" "$MPOX_WORK_DIR"
python "$HERE/tests/make_synthetic.py" "$MPOX_DATA_ROOT"
python "$HERE/scripts/01_build_index.py" > /dev/null
python "$HERE/scripts/02_audit.py"
ls "$MPOX_WORK_DIR/audit/for_reviewers/sheets" | head -2
# simulate the two reviewers filling their forms
python - <<PY
import pandas as pd
d = "$MPOX_WORK_DIR/audit/for_reviewers/"
for r, v in (("R1", "d"), ("R2", "duplicate")):
    f = pd.read_csv(d + f"review_form_{r}.csv"); f["decision"] = v; f.to_csv(d + f"review_form_{r}.csv", index=False)
PY
python "$HERE/scripts/03_groups_splits.py" | tail -1
python - <<PY
import sys, os, numpy as np
sys.path.insert(0, "$HERE/src")
from mpoxlodo import io
idx = io.load_index(); rng = np.random.default_rng(0)
for k in ["resnet50", "dinov2_b14", "clip_b16"]:
    e = rng.normal(size=(len(idx), 16)).astype("float32"); e[:, 0] += 1.5 * idx["y_bin"].to_numpy()
    np.savez_compressed(os.path.join(io.p("features"), k + ".npz"), ids=np.array(idx["image_id"]), emb=e)
PY
python "$HERE/scripts/05_linear_probe.py" --models resnet50 dinov2_b14 clip_b16 2>/dev/null
python "$HERE/scripts/09_evaluate.py" 2>/dev/null | head -3
python "$HERE/scripts/10_sensitivity.py" --jobs 1 2>/dev/null | tail -4
test -f "$MPOX_WORK_DIR/results/sensitivity_summary.csv"
grep -q '"rule_validation": {' "$MPOX_WORK_DIR/audit/audit_summary.json"
test -f "$MPOX_WORK_DIR/results/table6_main.csv" && echo "CPU TESTS PASSED"

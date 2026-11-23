"""Stage 1 (CPU): index every original image of the 14 datasets."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mpoxlodo import io
from mpoxlodo.config import DATA_ROOT
from mpoxlodo.index import build_index

if __name__ == "__main__":
    if os.path.exists(io.index_path()) and "--force" not in sys.argv:
        print("index exists (resumed run): kept, so duplicate groups and the review sample stay identical. "
              "Use --force to rebuild.")
        sys.exit(0)
    index, report = build_index(data_root=os.environ.get("MPOX_DATA_ROOT", DATA_ROOT))
    index.to_csv(io.index_path(), index=False)
    report.to_csv(io.p("index", "dataset_report.csv"), index=False)
    print(report.to_string(index=False))
    if len(index):
        print("\nImages per dataset and label:")
        print(index.pivot_table(index="dataset", columns="label", values="image_id",
                                aggfunc="count", fill_value=0).to_string())
    missing = report[~report["found"]]["dataset"].tolist()
    if missing:
        print(f"\nWARNING: not found under {DATA_ROOT}: {missing}. Add them as inputs or fix `folder` in config.py")

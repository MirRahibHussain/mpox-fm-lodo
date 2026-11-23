"""Stage 8 (GPU): parameters, latency and memory for every model (Table S8)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mpoxlodo import efficiency, io

if __name__ == "__main__":
    df = efficiency.measure_all()
    df.to_csv(io.p("results", "table_S8_efficiency.csv"), index=False)
    print(df.to_string(index=False))

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root on sys.path for module imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.core.recs_shared import recompute_edges_and_recommendations
from nhl_betting.utils.io import PROC_DIR


def main(date: str) -> int:
    recs = recompute_edges_and_recommendations(date, min_ev=0.0)
    recs_path = PROC_DIR / f"recommendations_{date}.csv"
    edges_path = PROC_DIR / f"edges_{date}.csv"
    print({
        "date": date,
        "recs_count": len(recs),
        "recs_csv_exists": recs_path.exists(),
        "edges_csv_exists": edges_path.exists(),
        "recs_csv": str(recs_path),
        "edges_csv": str(edges_path),
    })
    try:
        import pandas as pd
        if recs_path.exists():
            df = pd.read_csv(recs_path)
            print({"recs_csv_rows": len(df), "recs_csv_cols": list(df.columns)})
    except Exception as e:
        print({"warn": "read_recs_csv_failed", "error": str(e)})
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/quick_recs_check.py YYYY-MM-DD")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))

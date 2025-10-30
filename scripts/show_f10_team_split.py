from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root on sys.path
ROOT = Path(__file__).resolve().parent.parent
import sys as _sys
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

import os
import pandas as pd
from nhl_betting.core.recs_shared import recompute_edges_and_recommendations
from nhl_betting.utils.io import PROC_DIR

def main(date: str) -> int:
    # Allow overriding early factor via env for experiments
    factor = os.getenv("F10_EARLY_FACTOR")
    print({"date": date, "F10_EARLY_FACTOR": factor})
    recompute_edges_and_recommendations(date, min_ev=0.0)
    df = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
    cols = [
        "p_f10_home_scores",
        "p_f10_away_scores",
        "p_f10_home_allows",
        "p_f10_away_allows",
        "p_f10_yes",
        "p_f10_no",
    ]
    present = [c for c in cols if c in df.columns]
    print({"columns": present})
    out = df[["home","away"] + present].to_dict(orient="records")
    for row in out:
        print(row)
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/show_f10_team_split.py YYYY-MM-DD")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))

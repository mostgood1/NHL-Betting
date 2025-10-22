"""Report PBP coverage and quality metrics from games_with_periods.csv.

Outputs a console summary and writes per-season coverage to data/processed/pbp_coverage.csv
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.utils.io import RAW_DIR, PROC_DIR


def _is_int_like_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.notna() & ((s - s.round()).abs() < 1e-9)


def main():
    path = RAW_DIR / "games_with_periods.csv"
    if not path.exists():
        print(f"[error] {path} not found")
        sys.exit(1)
    df = pd.read_csv(path)
    # Normalize date
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    except Exception:
        df["date"] = df["date"].astype(str)

    total = len(df)
    pbp_mask = df.get("period_source", "").astype(str).str.lower().eq("pbp")
    pbp_rows = int(pbp_mask.sum())
    coverage = (pbp_rows / total * 100) if total else 0.0

    # PBP date range
    df_pbp = df[pbp_mask].copy()
    date_min = df_pbp["date"].min() if not df_pbp.empty else None
    date_max = df_pbp["date"].max() if not df_pbp.empty else None

    # Integer-like quality check on period splits and first-10
    int_p1h = _is_int_like_series(df_pbp.get("period1_home_goals")) if not df_pbp.empty else pd.Series(dtype=bool)
    int_p1a = _is_int_like_series(df_pbp.get("period1_away_goals")) if not df_pbp.empty else pd.Series(dtype=bool)
    int_first10 = _is_int_like_series(df_pbp.get("goals_first_10min")) if not df_pbp.empty else pd.Series(dtype=bool)
    int_p1_rate = float((int_p1h & int_p1a).mean()) if not df_pbp.empty else 0.0
    int_first10_rate = float(int_first10.mean()) if not df_pbp.empty else 0.0

    # Per-season coverage
    def _season_str(x):
        try:
            return str(int(x))
        except Exception:
            return str(x)
    df["season_str"] = df.get("season").map(_season_str)
    seasons = sorted(df["season_str"].dropna().unique())
    rows = []
    for s in seasons:
        df_s = df[df["season_str"] == s]
        total_s = len(df_s)
        pbp_s = int(df_s["period_source"].astype(str).str.lower().eq("pbp").sum())
        rows.append({
            "season": s,
            "total": total_s,
            "pbp": pbp_s,
            "coverage_pct": round((pbp_s / total_s * 100) if total_s else 0.0, 2),
        })
    out_df = pd.DataFrame(rows)
    out_path = PROC_DIR / "pbp_coverage.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("\n=== PBP Coverage Report ===")
    print(f"Total games: {total}")
    print(f"PBP-tagged games: {pbp_rows} ({coverage:.2f}%)")
    print(f"PBP date range: {date_min} .. {date_max}")
    print(f"Integer-like P1 split rate: {int_p1_rate:.2%}")
    print(f"Integer-like first-10 rate: {int_first10_rate:.2%}")
    if not out_df.empty:
        print("\nPer-season:")
        print(out_df.to_string(index=False))
    print(f"\nSaved per-season CSV to {out_path}")


if __name__ == "__main__":
    main()

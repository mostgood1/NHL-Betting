import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def daterange(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d"); e = datetime.strptime(end, "%Y-%m-%d")
    if e < s: s, e = e, s
    out = []
    d = s
    while d <= e:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def build_for_date(date: str, lookback_days: int = 5) -> Optional[pd.DataFrame]:
    try:
        d0 = datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        return None
    # Aggregate per team: average of max SAVES lambda over last N days
    vals: Dict[str, List[float]] = {}
    for k in range(1, lookback_days + 1):
        dd = (d0 - timedelta(days=k)).strftime("%Y-%m-%d")
        p = PROC_DIR / f"props_projections_all_{dd}.csv"
        if not (p.exists() and getattr(p.stat(), "st_size", 0) > 0):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        mcol = "market"; tcol = "team"
        lcol = next((c for c in ["proj_lambda", "lambda"] if c in df.columns), None)
        if not lcol or mcol not in df.columns or tcol not in df.columns:
            continue
        ss = df[df[mcol] == "SAVES"]
        if ss.empty:
            continue
        for team, sub in ss.groupby(ss[tcol].str.upper()):
            try:
                mx = float(pd.to_numeric(sub[lcol], errors="coerce").max())
            except Exception:
                continue
            if pd.isna(mx):
                continue
            vals.setdefault(str(team).upper(), []).append(mx)
    if not vals:
        return None
    rows = []
    for team, arr in vals.items():
        if not arr:
            continue
        rows.append({"team": team, "saves_recent": float(pd.Series(arr).mean())})
    if not rows:
        return None
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Build per-date goalie recent form from props saves")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--lookback-days", type=int, default=5)
    args = ap.parse_args()
    for day in daterange(args.start, args.end):
        df = build_for_date(day, lookback_days=int(args.lookback_days))
        if df is None or df.empty:
            print(f"[gform] no data for {day}")
            continue
        out_path = PROC_DIR / f"goalie_form_{day}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[gform] wrote {out_path} ({len(df)} teams)")


if __name__ == "__main__":
    main()

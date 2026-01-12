import argparse
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def moneypuck_skaters_gbg(season: int) -> Optional[pd.DataFrame]:
    """Fetch MoneyPuck skaters game-by-game CSV (regular + playoffs) for given season year (e.g., 2025 for 2025-26).
    Returns concatenated DataFrame or None.
    """
    urls = [
        f"https://moneypuck.com/moneypuck/playerData/gameByGame/{season}/regular/skaters.csv",
        f"https://moneypuck.com/moneypuck/playerData/gameByGame/{season}/playoffs/skaters.csv",
    ]
    dfs: List[pd.DataFrame] = []
    for url in urls:
        try:
            r = requests.get(url, timeout=25)
            if r.status_code != 200 or not r.text or len(r.text) < 100:
                continue
            df = pd.read_csv(StringIO(r.text))
            if not df.empty:
                df["season_src"] = url
                dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def infer_season_for_date(d: str) -> int:
    dt = datetime.strptime(d, "%Y-%m-%d")
    return dt.year if dt.month >= 7 else dt.year - 1


def build_team_xg(df_gbg: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Compute season-to-date team xGF/60 from MoneyPuck skaters game-by-game.
    Approximation: xGF/60 ~= average expected goals per team per game (regulation length).
    Requires columns: team, xGoals.
    """
    df = df_gbg.copy()
    if "team" not in df.columns or "xGoals" not in df.columns:
        return None
    # Ensure numeric
    df["xGoals"] = pd.to_numeric(df["xGoals"], errors="coerce")
    df = df.dropna(subset=["xGoals"]).copy()
    if df.empty:
        return None
    # Aggregate per team: total xGoals and games count
    team_games = df.groupby("team").size().reset_index(name="games")
    team_xg = df.groupby("team")["xGoals"].sum().reset_index(name="xg_total")
    out = team_games.merge(team_xg, on="team", how="inner")
    out["xgf60"] = out.apply(lambda r: float(r["xg_total"]) / max(1, float(r["games"])), axis=1)
    out = out.rename(columns={"team": "abbr"})
    out = out[["abbr", "xgf60"]]
    # Normalize team abbr uppercase
    out["abbr"] = out["abbr"].map(lambda s: str(s).upper())
    return out


def main():
    ap = argparse.ArgumentParser(description="Build team xGF/60 season-to-date from MoneyPuck skaters game-by-game")
    ap.add_argument("--date", type=str, default=datetime.utcnow().strftime("%Y-%m-%d"), help="Reference date to infer season")
    args = ap.parse_args()
    season = infer_season_for_date(args.date)

    df = moneypuck_skaters_gbg(season)
    if df is None or df.empty:
        print(f"[team-xg] No MoneyPuck skaters data for season {season}")
        return
    out = build_team_xg(df)
    if out is None or out.empty:
        print("[team-xg] Failed to compute team xGF/60")
        return
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    season_code = f"{season}-{season+1}"
    path_season = PROC_DIR / f"team_xg_{season_code}.csv"
    path_latest = PROC_DIR / "team_xg_latest.csv"
    out.to_csv(path_season, index=False)
    out.to_csv(path_latest, index=False)
    print(f"[team-xg] wrote {path_season} and {path_latest} ({len(out)} teams)")


if __name__ == "__main__":
    main()

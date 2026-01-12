import argparse
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

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


def moneypuck_goalies_gbg(season: int) -> Optional[pd.DataFrame]:
    """Fetch MoneyPuck goalie game-by-game CSV for given season year (e.g., 2025 for 2025-26)."""
    urls = [
        f"https://moneypuck.com/moneypuck/playerData/gameByGame/{season}/regular/goalies.csv",
        f"https://moneypuck.com/moneypuck/playerData/gameByGame/{season}/playoffs/goalies.csv",
    ]
    dfs: List[pd.DataFrame] = []
    for url in urls:
        try:
            r = requests.get(url, timeout=20)
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


def infer_seasons_for_range(start: str, end: str) -> List[int]:
    s = datetime.strptime(start, "%Y-%m-%d"); e = datetime.strptime(end, "%Y-%m-%d")
    years = set()
    d = s
    while d <= e:
        # MoneyPuck season key aligns to fall year (Oct..Jun)
        season_year = d.year if d.month >= 7 else d.year - 1
        years.add(season_year)
        d += timedelta(days=7)
    return sorted(years)


def build_team_goalie_form(df_gbg: pd.DataFrame, day: str, lookback_games: int = 5) -> Optional[pd.DataFrame]:
    # filter to games strictly before the target day
    try:
        cutoff = datetime.strptime(day, "%Y-%m-%d")
    except Exception:
        return None
    # MoneyPuck columns: 'date','team','shotsAgainst','goalsAgainst','saves','savePercentage'
    # Ensure parsed date
    df = df_gbg.copy()
    if "date" not in df.columns or "team" not in df.columns:
        return None
    try:
        df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    except Exception:
        return None
    df = df[df["_date"] < pd.Timestamp(cutoff)]
    if df.empty:
        return None
    # Ensure numeric
    for col in ("savePercentage", "saves", "shotsAgainst"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # compute rolling average save% per team across all goalies
    # Sort by date
    df = df.sort_values(["team", "_date"])  # team is already MP abbr
    rows = []
    for team, sub in df.groupby("team"):
        # Keep last N games per team
        sub = sub.dropna(subset=["savePercentage"]) if "savePercentage" in sub.columns else sub
        if sub.empty:
            continue
        lastn = sub.tail(lookback_games)
        if lastn.empty:
            continue
        sv = None
        if "savePercentage" in lastn.columns:
            sv = float(pd.to_numeric(lastn["savePercentage"], errors="coerce").dropna().mean())
        elif "saves" in lastn.columns and "shotsAgainst" in lastn.columns:
            shots = float(pd.to_numeric(lastn["shotsAgainst"], errors="coerce").sum())
            svs = float(pd.to_numeric(lastn["saves"], errors="coerce").sum())
            sv = (svs / shots) if shots > 0 else None
        if sv is None or pd.isna(sv):
            continue
        rows.append({"team": str(team).upper(), "sv_pct_l10": float(sv)})
    if not rows:
        return None
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Build per-date team goalie recent form from MoneyPuck game-by-game")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--lookback-games", type=int, default=5)
    args = ap.parse_args()

    seasons = infer_seasons_for_range(args.start, args.end)
    gbg_list: List[pd.DataFrame] = []
    for yr in seasons:
        df = moneypuck_goalies_gbg(yr)
        if df is not None and not df.empty:
            gbg_list.append(df)
        else:
            print(f"[gform-mp] no MoneyPuck data fetched for season {yr}")
    if not gbg_list:
        print("[gform-mp] no MoneyPuck goalie game-by-game data available; aborting")
        return
    gbg = pd.concat(gbg_list, ignore_index=True)

    for day in daterange(args.start, args.end):
        df = build_team_goalie_form(gbg, day, lookback_games=int(args.lookback_games))
        if df is None or df.empty:
            print(f"[gform-mp] no data for {day}")
            continue
        out_path = PROC_DIR / f"goalie_form_{day}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[gform-mp] wrote {out_path} ({len(df)} teams)")


if __name__ == "__main__":
    main()

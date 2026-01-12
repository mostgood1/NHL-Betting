import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def nhl_get(url: str, timeout: float = 20.0) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_schedule(date: str) -> List[dict]:
    data = nhl_get(f"https://statsapi.web.nhl.com/api/v1/schedule?date={date}&expand=schedule.teams,schedule.linescore")
    games = []
    try:
        for d in (data.get("dates") or []):
            for g in (d.get("games") or []):
                games.append(g)
    except Exception:
        pass
    return games


def get_live(game_pk: int) -> Optional[dict]:
    return nhl_get(f"https://statsapi.web.nhl.com/api/v1/game/{game_pk}/feed/live")


def extract_goalie_team_stats(live: dict) -> List[Tuple[str, str, float, float]]:
    """Return rows of (date, TEAM_ABBR, saves, shotsAgainst) from liveData.boxscore."""
    rows: List[Tuple[str, str, float, float]] = []
    try:
        game_date = (((live or {}).get("gameData") or {}).get("datetime") or {}).get("dateTime")
        if game_date:
            # convert to date string ET-equivalent (use UTC date as approximation)
            dt = pd.to_datetime(game_date, utc=True)
            day = dt.tz_convert("America/New_York").strftime("%Y-%m-%d")
        else:
            day = None
    except Exception:
        day = None
    try:
        box = (((live or {}).get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        for side in ("home", "away"):
            side_obj = (box.get(side) or {})
            team_abbr = (((side_obj.get("team") or {}).get("triCode")) or (side_obj.get("team") or {}).get("abbreviation") or (side_obj.get("team") or {}).get("name"))
            if not team_abbr:
                continue
            team_abbr = str(team_abbr).upper()
            players = (side_obj.get("players") or {})
            # Go through players to find goalies with goalieStats
            for pid, p in players.items():
                stats = (p.get("stats") or {}).get("goalieStats") or None
                if not stats:
                    continue
                saves = float(stats.get("saves", 0) or 0)
                shots = float(stats.get("shots", 0) or (saves + float(stats.get("goalsAgainst", 0) or 0)))
                rows.append((day, team_abbr, saves, shots))
    except Exception:
        pass
    return rows


def build_history(start: str, end: str, pad_days: int = 60) -> pd.DataFrame:
    # Build goalie game rows for a padded window before start and up to end-1
    s = datetime.strptime(start, "%Y-%m-%d"); e = datetime.strptime(end, "%Y-%m-%d")
    hist_start = (s - timedelta(days=pad_days)).strftime("%Y-%m-%d")
    hist_end = (e - timedelta(days=1)).strftime("%Y-%m-%d")
    rows: List[Tuple[str, str, float, float]] = []
    for day in daterange(hist_start, hist_end):
        games = get_schedule(day)
        if not games:
            continue
        for g in games:
            try:
                game_pk = int(g.get("gamePk"))
                live = get_live(game_pk)
                if not live:
                    continue
                rows.extend(extract_goalie_team_stats(live))
            except Exception:
                continue
    if not rows:
        return pd.DataFrame(columns=["date", "team", "saves", "shots"])
    df = pd.DataFrame(rows, columns=["date", "team", "saves", "shots"])
    df = df.dropna(subset=["date", "team"]).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # type: ignore
    return df


def build_for_day(hist: pd.DataFrame, day: str, lookback_games: int = 5) -> Optional[pd.DataFrame]:
    if hist.empty:
        return None
    cutoff = pd.to_datetime(day)
    rows = []
    for team, sub in hist.groupby("team"):
        sub = sub[sub["date"] < cutoff].sort_values("date")
        if sub.empty:
            continue
        lastn = sub.tail(lookback_games)
        shots = float(pd.to_numeric(lastn["shots"], errors="coerce").sum())
        saves = float(pd.to_numeric(lastn["saves"], errors="coerce").sum())
        sv = (saves / shots) if shots > 0 else None
        if sv is None:
            continue
        rows.append({"team": str(team).upper(), "sv_pct_l10": float(sv)})
    if not rows:
        return None
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Build per-date team goalie recent form (sv% last N team goalie games) from NHL StatsAPI")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--lookback-games", type=int, default=5)
    args = ap.parse_args()

    hist = build_history(args.start, args.end, pad_days=max(21, int(args.lookback_games) * 5))
    if hist.empty:
        print("[gform-nhl] no historical goalie stats built; aborting")
        return
    for day in daterange(args.start, args.end):
        df = build_for_day(hist, day, lookback_games=int(args.lookback_games))
        if df is None or df.empty:
            print(f"[gform-nhl] no data for {day}")
            continue
        out_path = PROC_DIR / f"goalie_form_{day}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"[gform-nhl] wrote {out_path} ({len(df)} teams)")


if __name__ == "__main__":
    main()

"""Fetch NHL PBP from NHL Web API and populate period/first-10 into games_with_periods.csv.

This expands PBP coverage without relying on R packages. For each completed game in the
target date range, we call:
  https://api-web.nhle.com/v1/gamecenter/{gamePk}/play-by-play
and aggregate:
  - period1/2/3 home/away goals
  - goals_first_10min (number of goals in first 10:00 of P1)

When PBP aggregation succeeds, we update the row and set period_source='pbp'.

Usage (PowerShell):
  .\.venv\Scripts\Activate.ps1; python scripts/fetch_nhl_pbp_to_periods.py --start 2023-10-01 --end 2024-07-01
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, Optional, Tuple, List, Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.utils.io import RAW_DIR, save_df


def fetch_game_pbp_json(gamePk: int, retries: int = 3, timeout: int = 20) -> Optional[Dict]:
    url = f"https://api-web.nhle.com/v1/gamecenter/{gamePk}/play-by-play"
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r.json()
        except Exception:
            pass
        time.sleep(0.5 * (2 ** i))
    return None


def fetch_game_boxscore_json(gamePk: int, retries: int = 3, timeout: int = 20) -> Optional[Dict]:
    url = f"https://api-web.nhle.com/v1/gamecenter/{gamePk}/boxscore"
    for i in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200 and r.text:
                return r.json()
        except Exception:
            pass
        time.sleep(0.5 * (2 ** i))
    return None


def _extract_possible_plays(pbp: Dict) -> List[Dict[str, Any]]:
    """Return a list of play dicts from various potential structures.

    The NHL Web API has returned different shapes across time. Look for:
      - top-level 'plays' (list)
      - nested {'plays': {'plays': [...]}}
      - alternative keys like 'allPlays' or 'playByPlay'
      - if all else fails, scan all top-level list values and pick those with typeDescKey
    """
    if not isinstance(pbp, dict):
        return []
    plays = pbp.get("plays")
    if isinstance(plays, list):
        return plays
    if isinstance(plays, dict):
        inner = plays.get("plays") or plays.get("allPlays")
        if isinstance(inner, list):
            return inner
    # Alternatives sometimes seen
    for key in ("allPlays", "playByPlay"):
        alt = pbp.get(key)
        if isinstance(alt, list):
            return alt
        if isinstance(alt, dict):
            inner = alt.get("plays") or alt.get("allPlays")
            if isinstance(inner, list):
                return inner
    # Fallback: find any list value containing dicts with typeDescKey
    for k, v in pbp.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and any("typeDescKey" in d for d in v if isinstance(d, dict)):
            return v  # assume this is plays
    return []


def _aggregate_from_boxscore_json(box: Dict) -> Optional[Dict]:
    """Aggregate period splits and first-10 from boxscore JSON structure.

    Uses linescore.byPeriod for period splits and summary.scoring for goal times.
    """
    try:
        periods = {1: {"home": 0, "away": 0}, 2: {"home": 0, "away": 0}, 3: {"home": 0, "away": 0}}
        # Period splits
        ls = (box or {}).get("linescore", {})
        byp = ls.get("byPeriod") or []
        for p in byp:
            pd = (p or {}).get("periodDescriptor") or {}
            num = int(pd.get("number") or pd.get("period") or 0)
            if num in periods:
                try:
                    periods[num]["home"] = int(p.get("home") or 0)
                    periods[num]["away"] = int(p.get("away") or 0)
                except Exception:
                    pass
        # First 10 minutes goals from summary.scoring
        first10 = 0
        scoring = (box or {}).get("summary", {}).get("scoring", []) or []
        for per in scoring:
            pd = (per or {}).get("periodDescriptor") or {}
            num = int(pd.get("number") or 0)
            if num != 1:
                continue
            goals = (per or {}).get("goals", []) or []
            for g in goals:
                tip = str(g.get("timeInPeriod", "20:00"))
                parts = tip.split(":")
                if len(parts) >= 2:
                    try:
                        mm = int(parts[0])
                        if mm < 10:
                            first10 += 1
                    except Exception:
                        pass
        return {
            "period1_home_goals": periods[1]["home"],
            "period1_away_goals": periods[1]["away"],
            "period2_home_goals": periods[2]["home"],
            "period2_away_goals": periods[2]["away"],
            "period3_home_goals": periods[3]["home"],
            "period3_away_goals": periods[3]["away"],
            "goals_first_10min": first10,
        }
    except Exception:
        return None


def aggregate_from_pbp_json(pbp: Dict, home_abbr: str, away_abbr: str, gamePk: Optional[int] = None) -> Optional[Dict]:
    try:
        periods = {1: {"home": 0, "away": 0}, 2: {"home": 0, "away": 0}, 3: {"home": 0, "away": 0}}
        first10 = 0
        plays = _extract_possible_plays(pbp)
        # Map team IDs from root JSON (more reliable than abbrev in plays)
        home_team_id = None
        away_team_id = None
        try:
            if isinstance(pbp, dict):
                home_team_id = ((pbp.get("homeTeam") or {}).get("id"))
                away_team_id = ((pbp.get("awayTeam") or {}).get("id"))
        except Exception:
            home_team_id = None
            away_team_id = None
        for play in plays:
            if (play.get("typeDescKey") or "").lower() != "goal":
                continue
            pd = play.get("periodDescriptor", {}) or {}
            num = int(pd.get("number") or pd.get("period") or 0)
            if num not in periods:
                continue
            # Determine scoring side
            side = None
            # 1) Prefer eventOwnerTeamId from details
            details = play.get("details") or {}
            event_owner_id = details.get("eventOwnerTeamId") if isinstance(details, dict) else None
            if event_owner_id is not None and (home_team_id is not None or away_team_id is not None):
                try:
                    if home_team_id is not None and int(event_owner_id) == int(home_team_id):
                        side = "home"
                    elif away_team_id is not None and int(event_owner_id) == int(away_team_id):
                        side = "away"
                except Exception:
                    pass
            # 2) Fallback to explicit homeAway/side if present
            if side is None:
                homeAway = (play.get("side") or play.get("homeAway") or "").lower()
                if homeAway in ("home", "away"):
                    side = homeAway
            # 3) Fallback to teamAbbrev mapping
            if side is None:
                teamObj = play.get("teamAbbrev") or play.get("team") or {}
                if isinstance(teamObj, dict):
                    team_abbr = (teamObj.get("default") or teamObj.get("DEFAULT") or teamObj.get("abbrev") or "").upper()
                else:
                    team_abbr = str(teamObj or "").upper()
                if team_abbr == str(home_abbr).upper():
                    side = "home"
                elif team_abbr == str(away_abbr).upper():
                    side = "away"
            # If still unknown, skip to avoid corrupt counts
            if side not in ("home", "away"):
                continue
            periods[num][side] += 1

            # First 10 min of P1
            tip = str(play.get("timeInPeriod", "20:00"))
            parts = tip.split(":")
            if len(parts) >= 2:
                try:
                    mm = int(parts[0])
                    if num == 1 and mm < 10:
                        first10 += 1
                except Exception:
                    pass
        agg = {
            "period1_home_goals": periods[1]["home"],
            "period1_away_goals": periods[1]["away"],
            "period2_home_goals": periods[2]["home"],
            "period2_away_goals": periods[2]["away"],
            "period3_home_goals": periods[3]["home"],
            "period3_away_goals": periods[3]["away"],
            "goals_first_10min": first10,
        }
        # If no goals were detected from PBP but the game has goals, try boxscore fallback
        if (agg["period1_home_goals"] + agg["period1_away_goals"] + agg["period2_home_goals"] + agg["period2_away_goals"] + agg["period3_home_goals"] + agg["period3_away_goals"]) == 0 and gamePk is not None:
            box = fetch_game_boxscore_json(gamePk)
            alt = _aggregate_from_boxscore_json(box) if box else None
            if alt:
                # Minimal runtime debug: indicate fallback used when aggregation from PBP yielded zero
                print(f"[pbp] used boxscore fallback for gamePk={gamePk} -> periods: P1 {alt['period1_home_goals']}-{alt['period1_away_goals']}, P2 {alt['period2_home_goals']}-{alt['period2_away_goals']}, P3 {alt['period3_home_goals']}-{alt['period3_away_goals']}")
                return alt
        return agg
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2023-10-01")
    ap.add_argument("--end", type=str, default="2025-10-17")
    ap.add_argument("--max", type=int, default=100000, help="Max games to process")
    ap.add_argument("--debug", type=int, default=0, help="Print diagnostics for first N rejections")
    args = ap.parse_args()

    games_path = RAW_DIR / "games_with_periods.csv"
    if not games_path.exists():
        print(f"[error] {games_path} not found")
        return
    df = pd.read_csv(games_path)
    # Normalize date for filter
    try:
        df["date_only"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    except Exception:
        df["date_only"] = df["date"].astype(str).str.slice(0,10)

    # Target: completed games in window that are not yet PBP-tagged
    mask = (df["date_only"] >= args.start) & (df["date_only"] <= args.end)
    mask &= df.get("period_source", "").astype(str).str.lower().ne("pbp")
    # also ensure final scores look valid (non-null)
    mask &= df["home_goals"].notna() & df["away_goals"].notna()

    todo = df[mask].copy()
    print(f"[pbp] Candidates: {len(todo)} (from {args.start} to {args.end})")

    updates = 0
    no_json = 0
    no_agg = 0
    rejected = 0
    debug_left = int(getattr(args, "debug", 0) or 0)
    for i, row in todo.head(args.max).iterrows():
        gpk = int(row.get("gamePk"))
        home = str(row.get("home")).upper()
        away = str(row.get("away")).upper()
        pbp = fetch_game_pbp_json(gpk)
        if not pbp:
            no_json += 1
            continue
        # Extract plays count for diagnostics
        plays = _extract_possible_plays(pbp)
        agg = aggregate_from_pbp_json(pbp, home, away, gamePk=gpk)
        if not agg:
            no_agg += 1
            continue
        # Basic sanity: totals should not exceed final
        p1 = agg["period1_home_goals"] + agg["period1_away_goals"]
        p2 = agg["period2_home_goals"] + agg["period2_away_goals"]
        p3 = agg["period3_home_goals"] + agg["period3_away_goals"]
        total_period = p1 + p2 + p3
        final_total = int(row.get("home_goals", 0)) + int(row.get("away_goals", 0))
        # Accept 0 when final is 0; otherwise require at least one period goal and never exceed final
        if (final_total > 0 and total_period == 0) or total_period > final_total:
            if debug_left > 0:
                debug_left -= 1
                # Print a compact diagnostic line
                sample_play = {}
                try:
                    if isinstance(plays, list) and plays:
                        p0 = plays[0]
                        sample_play = {k: p0.get(k) for k in ("typeDescKey","periodDescriptor","teamAbbrev","timeInPeriod","homeAway","side")}
                except Exception:
                    sample_play = {}
                print(f"[debug] reject gamePk={gpk} {home}-{away} final={final_total} p1={p1} p2={p2} p3={p3} plays={len(plays) if isinstance(plays,list) else 'n/a'} sample={sample_play}")
            rejected += 1
            continue
        # Apply updates
        for k, v in agg.items():
            df.at[i, k] = int(v)
        df.at[i, "period_source"] = "pbp"
        updates += 1
        if updates % 50 == 0:
            print(f"[pbp] Updated {updates} rows...")
        time.sleep(0.15)

    print(f"[pbp] Summary: updates={updates}, no_json={no_json}, no_agg={no_agg}, rejected={rejected}")
    if updates:
        # Drop helper
        df = df.drop(columns=["date_only"], errors="ignore")
        save_df(df, games_path)
        print(f"[pbp] Wrote updates: {updates} -> {games_path}")
    else:
        print("[pbp] No updates applied")


if __name__ == "__main__":
    main()

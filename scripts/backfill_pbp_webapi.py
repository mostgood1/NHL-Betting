"""Backfill PBP-derived period counts from NHL Web API for games in a date range.

Reads data/raw/games_with_periods.csv, fetches play-by-play JSON for target games,
computes period-by-period goal counts and first-10-min goals, and merges results.

Usage:
  python scripts/backfill_pbp_webapi.py --start 2024-10-01 --end 2025-10-27
  python scripts/backfill_pbp_webapi.py --season 2024
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Dict
import time
import requests
import pandas as pd

# Repo root and IO helpers
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from nhl_betting.utils.io import RAW_DIR, save_df

API_BASE = "https://api-web.nhle.com/v1"


def fetch_game_pbp(game_id: int, retries: int = 3) -> Optional[Dict]:
    url = f"{API_BASE}/gamecenter/{game_id}/play-by-play"
    last = None
    for a in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            last = RuntimeError(f"status {r.status_code}")
        except Exception as e:
            last = e
        time.sleep(0.4 * (2 ** a))
    if last:
        print(f"[warn] PBP fetch failed for {game_id}: {last}")
    return None


def extract_from_pbp(pbp: Dict) -> Optional[Dict]:
    """Compute period-by-period goals (home/away) and first-10-min goals from PBP JSON."""
    try:
        periods = {1: {"home": 0, "away": 0}, 2: {"home": 0, "away": 0}, 3: {"home": 0, "away": 0}}
        first_10 = 0
        plays = pbp.get("plays", [])
        for play in plays:
            if play.get("typeDescKey") != "goal":
                continue
            pdsc = play.get("periodDescriptor", {})
            num = int(pdsc.get("number", 0) or 0)
            if num not in periods:
                continue
            side = play.get("side") or play.get("homeAway", "")
            if side == "home":
                periods[num]["home"] += 1
            elif side == "away":
                periods[num]["away"] += 1
            else:
                # If side missing, count toward total as home for determinism
                periods[num]["home"] += 1
            # First 10 mins only for P1
            time_str = str(play.get("timeInPeriod", "20:00"))
            parts = time_str.split(":")
            if len(parts) == 2:
                try:
                    minutes = int(parts[0])
                    if minutes < 10 and num == 1:
                        first_10 += 1
                except Exception:
                    pass
        return {
            "period1_home_goals": periods[1]["home"],
            "period1_away_goals": periods[1]["away"],
            "period2_home_goals": periods[2]["home"],
            "period2_away_goals": periods[2]["away"],
            "period3_home_goals": periods[3]["home"],
            "period3_away_goals": periods[3]["away"],
            "goals_first_10min": first_10,
        }
    except Exception:
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    ap.add_argument("--season", type=int, default=None, help="Season start year (e.g., 2024)")
    ap.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between API calls")
    args = ap.parse_args()

    path = RAW_DIR / "games_with_periods.csv"
    if not path.exists():
        print(f"[error] {path} not found")
        sys.exit(1)
    base = pd.read_csv(path)
    # Normalize date to YYYY-MM-DD
    try:
        base["date_only"] = pd.to_datetime(base["date"]).dt.strftime("%Y-%m-%d")
    except Exception:
        base["date_only"] = base["date"].astype(str).str.slice(0, 10)

    # Filter window
    if args.season and (not args.start and not args.end):
        start = f"{args.season}-07-01"
        end = f"{args.season+1}-07-01"
    else:
        start = args.start
        end = args.end
    if start:
        base = base[base["date_only"] >= start]
    if end:
        base = base[base["date_only"] <= end]

    # Target games: not already pbp-tagged or obviously approximate splits
    mask_non_pbp = base.get("period_source", "").astype(str).str.lower().ne("pbp")
    # Optional: prioritize completed games with goals
    mask_has_final = base[["home_goals", "away_goals"]].notna().all(axis=1)
    target = base[mask_non_pbp & mask_has_final].copy()

    if target.empty:
        print("[info] No target games to backfill in range.")
        return

    print(f"[backfill] Attempting PBP for {len(target)} games from {start or target['date_only'].min()} to {end or target['date_only'].max()}")

    updates = []
    for i, row in target.iterrows():
        gid = int(row.get("gamePk") or 0)
        if gid <= 0:
            continue
        pbp = fetch_game_pbp(gid)
        if pbp is None:
            continue
        vals = extract_from_pbp(pbp)
        if not vals:
            continue
        vals["gamePk"] = gid
        vals["period_source"] = "pbp"
        updates.append(vals)
        # Rate limit
        time.sleep(max(0.0, float(args.sleep)))
        if len(updates) % 100 == 0:
            print(f"  ..{len(updates)} updates")

    if not updates:
        print("[warn] No PBP updates collected.")
        return

    upd = pd.DataFrame(updates)
    # Merge onto full dataset
    full = pd.read_csv(path)
    merged = full.merge(upd, on="gamePk", how="left", suffixes=("", "_pbp"))
    for c in [
        "period1_home_goals","period1_away_goals",
        "period2_home_goals","period2_away_goals",
        "period3_home_goals","period3_away_goals",
        "goals_first_10min",
        "period_source",
    ]:
        pbp_col = c + ("_pbp")
        if pbp_col in merged.columns:
            merged[c] = merged[pbp_col].fillna(merged.get(c))
    # Drop helper pbp columns
    drop_cols = [c for c in merged.columns if c.endswith("_pbp")] + ["date_only"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    save_df(merged, path)
    print(f"[backfill] Wrote PBP updates for {len(upd)} games -> {path}")


if __name__ == "__main__":
    main()

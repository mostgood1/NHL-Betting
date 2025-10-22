"""Collect historical NHL game data with period-by-period scoring for model training.

This script fetches games from the NHL Web API for multiple seasons,
extracting detailed period goals and first 10 minute scoring data.

Usage:
    python scripts/collect_historical_games.py --seasons 2023 2024 2025
    python scripts/collect_historical_games.py --start 2023-10-01 --end 2025-10-17
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
import time
from typing import Optional, List, Dict, Tuple

from nhl_betting.utils.io import RAW_DIR, save_df


def fetch_schedule(start_date: str, end_date: str, verbose: bool = True) -> List[Dict]:
    """Fetch game schedule from NHL Web API."""
    base_url = "https://api-web.nhle.com/v1/schedule"
    games = []
    
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Fetch in weekly chunks to avoid timeouts
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=6), end)
        
        url = f"{base_url}/{current.strftime('%Y-%m-%d')}"
        if verbose:
            print(f"[fetch] {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract games from response
            if "gameWeek" in data:
                for day in data["gameWeek"]:
                    if "games" in day:
                        games.extend(day["games"])
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            print(f"[error] Failed to fetch {current}: {e}")
        
        current = chunk_end + timedelta(days=1)
    
    if verbose:
        print(f"[fetch] Collected {len(games)} games")
    
    return games


def fetch_game_details(game_id: int, retries: int = 3) -> Optional[Dict]:
    """Fetch detailed game data including period-by-period scoring."""
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore"
    
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 * (2 ** attempt))  # Exponential backoff
            else:
                print(f"[error] Failed to fetch game {game_id}: {e}")
                return None


def fetch_game_pbp(game_id: int, retries: int = 3) -> Optional[Dict]:
    """Fetch play-by-play data to derive period goals and first 10-min goals.
    Uses NHL Web API play-by-play endpoint.
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(1 * (2 ** attempt))
            else:
                return None


def extract_period_goals(boxscore: Dict) -> Dict:
    """Extract period-by-period goals and first 10 min data from boxscore."""
    result = {
        "period1_home_goals": 0,
        "period1_away_goals": 0,
        "period2_home_goals": 0,
        "period2_away_goals": 0,
        "period3_home_goals": 0,
        "period3_away_goals": 0,
        "goals_first_10min": 0,
    }
    
    try:
        # Get linescore for period goals
        linescore = boxscore.get("linescore", {})
        
        if "byPeriod" in linescore:
            for period_data in linescore["byPeriod"]:
                period_num = period_data.get("periodDescriptor", {}).get("number", 0)
                
                if period_num == 1:
                    result["period1_home_goals"] = period_data.get("home", 0)
                    result["period1_away_goals"] = period_data.get("away", 0)
                elif period_num == 2:
                    result["period2_home_goals"] = period_data.get("home", 0)
                    result["period2_away_goals"] = period_data.get("away", 0)
                elif period_num == 3:
                    result["period3_home_goals"] = period_data.get("home", 0)
                    result["period3_away_goals"] = period_data.get("away", 0)
        
        # Try to get first 10 min goals from plays
        # This requires looking at goal times in period 1
        plays = boxscore.get("summary", {}).get("scoring", [])
        first_10_goals = 0
        
        for period_plays in plays:
            period_num = period_plays.get("periodDescriptor", {}).get("number", 0)
            if period_num == 1:
                goals = period_plays.get("goals", [])
                for goal in goals:
                    time_str = goal.get("timeInPeriod", "20:00")
                    try:
                        # Parse MM:SS format
                        parts = time_str.split(":")
                        if len(parts) == 2:
                            minutes = int(parts[0])
                            if minutes < 10:
                                first_10_goals += 1
                    except Exception:
                        pass
        
        result["goals_first_10min"] = first_10_goals
        
    except Exception as e:
        print(f"[warn] Error extracting period data: {e}")
    
    return result


def extract_from_pbp(pbp: Dict) -> Optional[Dict]:
    """Compute period-by-period goals and first 10-min goals from PBP JSON."""
    try:
        periods = {1: {"home": 0, "away": 0}, 2: {"home": 0, "away": 0}, 3: {"home": 0, "away": 0}}
        first_10 = 0
        plays = pbp.get("plays", [])
        for play in plays:
            if play.get("typeDescKey") != "goal":
                continue
            pd = play.get("periodDescriptor", {})
            num = int(pd.get("number", 0) or 0)
            if num not in periods:
                continue
            team = (play.get("teamAbbrev", {}) or play.get("team", {}))
            if isinstance(team, dict):
                team_abbrev = team.get("default", "")
            else:
                team_abbrev = str(team)
            # We don't know home/away from play alone; count as total and split by strength? Instead, use side if present
            side = play.get("side") or play.get("homeAway", "")
            if side == "home":
                periods[num]["home"] += 1
            elif side == "away":
                periods[num]["away"] += 1
            else:
                # If side not present, just increment home for determinism (rare)
                periods[num]["home"] += 1
            # First 10 minutes check
            time_str = play.get("timeInPeriod", "20:00")
            parts = str(time_str).split(":")
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


def collect_games(start_date: str, end_date: str, verbose: bool = True) -> pd.DataFrame:
    """Collect historical game data with period details."""
    
    # Fetch schedule
    schedule_games = fetch_schedule(start_date, end_date, verbose=verbose)
    
    if not schedule_games:
        print("[error] No games found in schedule")
        return pd.DataFrame()
    
    # Extract game data
    records = []
    total = len(schedule_games)
    
    for idx, game in enumerate(schedule_games, 1):
        if verbose and idx % 50 == 0:
            print(f"[progress] {idx}/{total} games processed")
        
        try:
            game_id = game.get("id")
            game_state = game.get("gameState", "")
            
            # Only process completed games
            if game_state not in ["OFF", "FINAL"]:
                continue
            
            # Basic info
            record = {
                "gamePk": game_id,
                "date": game.get("startTimeUTC", ""),
                "season": game.get("season", 0),
                "type": game.get("gameType", 2),  # 2 = regular season
                "home": game.get("homeTeam", {}).get("abbrev", ""),
                "away": game.get("awayTeam", {}).get("abbrev", ""),
                "home_goals": game.get("homeTeam", {}).get("score", 0),
                "away_goals": game.get("awayTeam", {}).get("score", 0),
            }
            
            # Try to fetch real boxscore details for period/first-10 goals
            filled_from_api = False
            details = fetch_game_details(game_id)
            if details:
                try:
                    period = extract_period_goals(details)
                    record.update(period)
                    # Sanity check: if API extraction yields zero P1 goals but total goals > 0, try PBP
                    total_goals_chk = (record["home_goals"] + record["away_goals"]) or 0
                    p1_total_chk = float(record.get("period1_home_goals", 0)) + float(record.get("period1_away_goals", 0))
                    if total_goals_chk > 0 and p1_total_chk == 0:
                        pbp = fetch_game_pbp(game_id)
                        if pbp:
                            alt = extract_from_pbp(pbp)
                            if alt:
                                record.update(alt)
                                p1_total_chk = float(record.get("period1_home_goals", 0)) + float(record.get("period1_away_goals", 0))
                    filled_from_api = total_goals_chk == 0 or p1_total_chk > 0
                    if filled_from_api:
                        record["period_source"] = "api" if p1_total_chk > 0 else "none"
                except Exception as e:
                    if verbose:
                        print(f"[warn] Using fallback period estimates for {game_id}: {e}")
            
            if not filled_from_api:
                # Fallback: Estimate period goals based on NHL scoring patterns
                total_home = record["home_goals"]
                total_away = record["away_goals"]
                total_goals = total_home + total_away
                
                import random
                random.seed(game_id)  # Consistent per game
                
                # P1: 28-34% (avg 31%) | P2: 31-37% (avg 34%) | P3: remainder
                p1_pct = 0.28 + random.random() * 0.06
                p2_pct = 0.31 + random.random() * 0.06
                p3_pct = max(0.0, 1.0 - p1_pct - p2_pct)
                
                home_ratio = total_home / total_goals if total_goals > 0 else 0.5
                away_ratio = total_away / total_goals if total_goals > 0 else 0.5
                
                record["period1_home_goals"] = round(total_goals * p1_pct * home_ratio, 2)
                record["period1_away_goals"] = round(total_goals * p1_pct * away_ratio, 2)
                record["period2_home_goals"] = round(total_goals * p2_pct * home_ratio, 2)
                record["period2_away_goals"] = round(total_goals * p2_pct * away_ratio, 2)
                record["period3_home_goals"] = round(total_goals * p3_pct * home_ratio, 2)
                record["period3_away_goals"] = round(total_goals * p3_pct * away_ratio, 2)
                
                # Estimate first 10 min as ~15-20% of P1 goals
                first_10_pct = 0.15 + random.random() * 0.05
                p1_goals = (record["period1_home_goals"] + record["period1_away_goals"]) if total_goals > 0 else 0
                record["goals_first_10min"] = round(p1_goals * first_10_pct, 2)
                record["period_source"] = "fallback"
            
            records.append(record)
            
            # Rate limiting between games
            time.sleep(0.3)
            
        except Exception as e:
            print(f"[error] Failed to process game {game.get('id')}: {e}")
            continue
    
    df = pd.DataFrame(records)
    
    if verbose:
        print(f"\n[summary] Collected {len(df)} completed games")
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Total goals: {df['home_goals'].sum() + df['away_goals'].sum()}")
        print(f"  Period 1 goals: {df['period1_home_goals'].sum() + df['period1_away_goals'].sum()}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect historical NHL game data")
    parser.add_argument("--start", help="Start date YYYY-MM-DD", default="2023-10-01")
    parser.add_argument("--end", help="End date YYYY-MM-DD", default="2025-10-17")
    parser.add_argument("--output", help="Output CSV path", default=None)
    parser.add_argument("--verbose", action="store_true", default=True)
    
    args = parser.parse_args()
    
    print(f"[collect] Fetching games from {args.start} to {args.end}")
    print(f"[collect] This may take 10-20 minutes for 2+ seasons...\n")
    
    df = collect_games(args.start, args.end, verbose=args.verbose)
    
    if not df.empty:
        output_path = Path(args.output) if args.output else (RAW_DIR / "games_with_periods.csv")
        save_df(df, output_path)
        print(f"\n[done] Saved {len(df)} games to {output_path}")
    else:
        print("\n[error] No games collected")
        sys.exit(1)

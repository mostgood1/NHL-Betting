"""Prepare game-level features for neural network training.

This module computes comprehensive features for each game including:
- Team Elo ratings at game time
- Recent form (last N games stats)
- Head-to-head history
- Rest days
- Time-of-season effects
- Period-by-period goals
- First 10 minute goals
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from nhl_betting.utils.io import RAW_DIR, MODEL_DIR


def load_games() -> pd.DataFrame:
    """Load raw games data."""
    # Try games_with_periods.csv first (larger dataset), fallback to games.csv
    games_with_periods = RAW_DIR / "games_with_periods.csv"
    games_csv = RAW_DIR / "games.csv"
    
    if games_with_periods.exists():
        print(f"[load] Using {games_with_periods.name}")
        df = pd.read_csv(games_with_periods)
    elif games_csv.exists():
        print(f"[load] Using {games_csv.name}")
        df = pd.read_csv(games_csv)
    else:
        raise FileNotFoundError(f"Games data not found: {games_csv} or {games_with_periods}")
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df


def compute_elo_history(games_df: pd.DataFrame, k: float = 20, initial: float = 1500) -> pd.DataFrame:
    """Compute Elo ratings for each game (as they were at game time).
    
    Returns DataFrame with columns: date, home, away, home_elo, away_elo
    """
    ratings = {}
    game_records = []
    
    for _, game in games_df.iterrows():
        home = game["home"]
        away = game["away"]
        date = game["date"]
        
        # Get current ratings
        home_elo = ratings.get(home, initial)
        away_elo = ratings.get(away, initial)
        
        # Record game with pre-game ratings
        game_records.append({
            "gamePk": game.get("gamePk", 0),
            "date": date,
            "home": home,
            "away": away,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_goals": game.get("home_goals", 0),
            "away_goals": game.get("away_goals", 0),
        })
        
        # Update ratings after game
        home_score = 1.0 if game["home_goals"] > game["away_goals"] else 0.0
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        
        ratings[home] = home_elo + k * (home_score - expected_home)
        ratings[away] = away_elo + k * ((1 - home_score) - (1 - expected_home))
    
    return pd.DataFrame(game_records)


def compute_recent_form(
    games_df: pd.DataFrame,
    window: int = 10
) -> pd.DataFrame:
    """Compute recent form statistics for each team at each game.
    
    For each game, computes stats from the last N games before that date.
    
    Returns DataFrame with columns for each team's recent performance.
    """
    records = []
    
    for idx, game in games_df.iterrows():
        home = game["home"]
        away = game["away"]
        date = game["date"]
        
        # Get recent games before this date
        recent_games = games_df[games_df["date"] < date].tail(1000)
        
        # Home team recent stats
        home_recent = recent_games[
            (recent_games["home"] == home) | (recent_games["away"] == home)
        ].tail(window)
        
        home_goals_for = []
        home_goals_against = []
        home_wins = 0
        
        for _, g in home_recent.iterrows():
            if g["home"] == home:
                home_goals_for.append(g["home_goals"])
                home_goals_against.append(g["away_goals"])
                if g["home_goals"] > g["away_goals"]:
                    home_wins += 1
            else:
                home_goals_for.append(g["away_goals"])
                home_goals_against.append(g["home_goals"])
                if g["away_goals"] > g["home_goals"]:
                    home_wins += 1
        
        # Away team recent stats
        away_recent = recent_games[
            (recent_games["home"] == away) | (recent_games["away"] == away)
        ].tail(window)
        
        away_goals_for = []
        away_goals_against = []
        away_wins = 0
        
        for _, g in away_recent.iterrows():
            if g["home"] == away:
                away_goals_for.append(g["home_goals"])
                away_goals_against.append(g["away_goals"])
                if g["home_goals"] > g["away_goals"]:
                    away_wins += 1
            else:
                away_goals_for.append(g["away_goals"])
                away_goals_against.append(g["home_goals"])
                if g["away_goals"] > g["home_goals"]:
                    away_wins += 1
        
        records.append({
            "gamePk": game.get("gamePk", 0),
            "date": date,
            "home": home,
            "away": away,
            "home_goals_last10": np.mean(home_goals_for) if home_goals_for else 0,
            "home_goals_against_last10": np.mean(home_goals_against) if home_goals_against else 0,
            "home_wins_last10": home_wins,
            "away_goals_last10": np.mean(away_goals_for) if away_goals_for else 0,
            "away_goals_against_last10": np.mean(away_goals_against) if away_goals_against else 0,
            "away_wins_last10": away_wins,
        })
    
    return pd.DataFrame(records)


def compute_rest_days(games_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rest days between games for each team."""
    records = []
    last_game = {}  # team -> date
    
    for _, game in games_df.iterrows():
        home = game["home"]
        away = game["away"]
        date = game["date"]
        
        # Calculate rest days
        home_rest = (date - last_game[home]).days if home in last_game else 3
        away_rest = (date - last_game[away]).days if away in last_game else 3
        
        records.append({
            "gamePk": game.get("gamePk", 0),
            "date": date,
            "home": home,
            "away": away,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
        })
        
        # Update last game dates
        last_game[home] = date
        last_game[away] = date
    
    return pd.DataFrame(records)


def extract_period_goals(games_df: pd.DataFrame) -> pd.DataFrame:
    """Extract period-by-period goals if available in data.
    
    If not available, will need to fetch from NHL API or use dummy values.
    """
    records = []
    
    for _, game in games_df.iterrows():
        record = {
            "gamePk": game.get("gamePk", 0),
            "date": game["date"],
            "home": game["home"],
            "away": game["away"],
        }
        
        # Try to get period goals if columns exist
        for period in [1, 2, 3]:
            home_col = f"period{period}_home_goals"
            away_col = f"period{period}_away_goals"
            
            if home_col in game and away_col in game:
                record[home_col] = game[home_col]
                record[away_col] = game[away_col]
            else:
                # Estimate based on total (for now, split evenly)
                total_home = game.get("home_goals", 0)
                total_away = game.get("away_goals", 0)
                record[home_col] = total_home / 3.0
                record[away_col] = total_away / 3.0
        
        # First 10 min goals (if available, otherwise estimate)
        if "goals_first_10min" in game:
            record["goals_first_10min"] = game["goals_first_10min"]
        else:
            # Rough estimate: ~20% of first period goals
            record["goals_first_10min"] = (
                record.get("period1_home_goals", 0) + 
                record.get("period1_away_goals", 0)
            ) * 0.2
        
        records.append(record)
    
    return pd.DataFrame(records)


def compute_season_progress(games_df: pd.DataFrame) -> pd.DataFrame:
    """Compute how far into the season each team is at game time."""
    records = []
    games_played = {}  # team -> count
    
    for _, game in games_df.iterrows():
        home = game["home"]
        away = game["away"]
        
        home_gp = games_played.get(home, 0)
        away_gp = games_played.get(away, 0)
        
        records.append({
            "gamePk": game.get("gamePk", 0),
            "date": game["date"],
            "home": home,
            "away": away,
            "home_games_played": home_gp,
            "away_games_played": away_gp,
            "games_played_season": (home_gp + away_gp) / 2,
        })
        
        games_played[home] = home_gp + 1
        games_played[away] = away_gp + 1
    
    return pd.DataFrame(records)


def prepare_game_features(output_path: Optional[Path] = None) -> pd.DataFrame:
    """Prepare comprehensive game features for neural network training.
    
    Combines:
    - Base game results
    - Elo ratings
    - Recent form
    - Rest days
    - Period goals
    - Season progress
    
    Returns:
        DataFrame with all features ready for training
    """
    print("[prepare] Loading games data...")
    games_df = load_games()
    print(f"[prepare] Loaded {len(games_df)} games")
    
    print("[prepare] Computing Elo ratings...")
    elo_df = compute_elo_history(games_df)
    
    print("[prepare] Computing recent form...")
    form_df = compute_recent_form(games_df)
    
    print("[prepare] Computing rest days...")
    rest_df = compute_rest_days(games_df)
    
    print("[prepare] Extracting period goals...")
    period_df = extract_period_goals(games_df)
    
    print("[prepare] Computing season progress...")
    season_df = compute_season_progress(games_df)
    
    print("[prepare] Merging all features...")
    # Merge all dataframes
    result = games_df.copy()
    
    # Add Elo
    result = result.merge(
        elo_df[["gamePk", "home_elo", "away_elo"]],
        on="gamePk",
        how="left"
    )
    
    # Add form
    result = result.merge(
        form_df[[
            "gamePk", 
            "home_goals_last10", "home_goals_against_last10", "home_wins_last10",
            "away_goals_last10", "away_goals_against_last10", "away_wins_last10"
        ]],
        on="gamePk",
        how="left"
    )
    
    # Add rest
    result = result.merge(
        rest_df[["gamePk", "home_rest_days", "away_rest_days"]],
        on="gamePk",
        how="left"
    )
    
    # Add period goals (only if not already in result)
    period_cols = [
        "period1_home_goals", "period1_away_goals",
        "period2_home_goals", "period2_away_goals",
        "period3_home_goals", "period3_away_goals",
        "goals_first_10min"
    ]
    
    # Check if period columns already exist in result
    if not all(col in result.columns for col in period_cols):
        result = result.merge(
            period_df[["gamePk"] + period_cols],
            on="gamePk",
            how="left"
        )
    
    # Add season progress
    result = result.merge(
        season_df[["gamePk", "games_played_season"]],
        on="gamePk",
        how="left"
    )
    
    # Rename final goals columns if needed
    if "final_home_goals" not in result.columns and "home_goals" in result.columns:
        result["final_home_goals"] = result["home_goals"]
        result["final_away_goals"] = result["away_goals"]
    
    # Fill NaN values
    result = result.fillna({
        "home_elo": 1500,
        "away_elo": 1500,
        "home_goals_last10": 0,
        "home_goals_against_last10": 0,
        "home_wins_last10": 0,
        "away_goals_last10": 0,
        "away_goals_against_last10": 0,
        "away_wins_last10": 0,
        "home_rest_days": 1,
        "away_rest_days": 1,
        "games_played_season": 0,
    })
    
    print(f"[prepare] Final feature matrix: {result.shape}")
    print(f"[prepare] Columns: {list(result.columns)}")
    
    # Save
    if output_path is None:
        output_path = RAW_DIR / "games_with_features.csv"
    
    result.to_csv(output_path, index=False)
    print(f"[prepare] Saved to {output_path}")
    
    return result


if __name__ == "__main__":
    prepare_game_features()

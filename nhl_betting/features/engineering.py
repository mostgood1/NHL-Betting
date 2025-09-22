from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass
class TeamFeatureRow:
    date: str
    team: str
    opponent: str
    is_home: int
    goals_for: float
    goals_against: float
    rest_days: int
    b2b: int


def make_team_game_features(games_df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: date, home, away, home_goals, away_goals
    rows: List[TeamFeatureRow] = []
    for _, g in games_df.iterrows():
        date = g["date"]
        # Home row
        rows.append(
            TeamFeatureRow(
                date=date,
                team=g["home"],
                opponent=g["away"],
                is_home=1,
                goals_for=g.get("home_goals", None) if pd.notna(g.get("home_goals", None)) else None,
                goals_against=g.get("away_goals", None) if pd.notna(g.get("away_goals", None)) else None,
                rest_days=0,
                b2b=0,
            )
        )
        # Away row
        rows.append(
            TeamFeatureRow(
                date=date,
                team=g["away"],
                opponent=g["home"],
                is_home=0,
                goals_for=g.get("away_goals", None) if pd.notna(g.get("away_goals", None)) else None,
                goals_against=g.get("home_goals", None) if pd.notna(g.get("home_goals", None)) else None,
                rest_days=0,
                b2b=0,
            )
        )
    df = pd.DataFrame([r.__dict__ for r in rows])

    # Compute rest and b2b flags per team
    # Parse to UTC (handles both naive and aware); then drop tz for simplicity
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(None)
    df = df.sort_values(["team", "date"])  # type: ignore
    df["prev_date"] = df.groupby("team")["date"].shift(1)
    df["rest_days"] = (df["date"] - df["prev_date"]).dt.days.fillna(7).astype(int)
    df["b2b"] = (df["rest_days"] == 1).astype(int)
    df = df.drop(columns=["prev_date"]).reset_index(drop=True)

    # Rolling means of goals for/against (exclude current game)
    for col in ["goals_for", "goals_against"]:
        roll = df.groupby("team")[col].rolling(window=10, min_periods=3).mean().shift(1)
        df[f"{col}_roll10"] = roll.reset_index(level=0, drop=True)
    return df

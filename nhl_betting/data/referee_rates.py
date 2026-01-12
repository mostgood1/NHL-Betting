from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _norm_team(v: str) -> str:
    try:
        return str(v).strip().upper()
    except Exception:
        return str(v).upper()


def load_referee_rates(date: str) -> Optional[Tuple[Dict[str, float], float]]:
    """Load per-game referee penalty/calls rate for a date, if available.

    Expected file (optional): data/processed/ref_assignments_YYYY-MM-DD.csv
    Columns (flexible):
      - home, away (team abbreviations or names)
      - one of: calls_per60, penalties_per60, pen_per60, rate

    Returns:
      (per_game_map, base_rate)
        per_game_map key: "HOME_ABBR|AWAY_ABBR" → float rate
        base_rate: float league-average rate in the file

    If the file is missing or unusable, returns None.
    """
    path = PROC_DIR / f"ref_assignments_{date}.csv"
    if not (path.exists() and getattr(path.stat(), "st_size", 0) > 0):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # Identify columns
    home_col = next((c for c in df.columns if c.lower() in ("home", "home_abbr", "home_team")), None)
    away_col = next((c for c in df.columns if c.lower() in ("away", "away_abbr", "away_team")), None)
    rate_col = next(
        (c for c in df.columns if c.lower() in ("calls_per60", "penalties_per60", "pen_per60", "rate")),
        None,
    )
    if not home_col or not away_col or not rate_col:
        return None
    try:
        df = df[[home_col, away_col, rate_col]].dropna()
        df.columns = ["home", "away", "rate"]
        df["home"] = df["home"].map(_norm_team)
        df["away"] = df["away"].map(_norm_team)
        df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
        df = df.dropna(subset=["rate"])  # type: ignore
        if df.empty:
            return None
        base_rate = float(df["rate"].mean())
        per_game: Dict[str, float] = {}
        for _, r in df.iterrows():
            key = f"{r['home']}|{r['away']}"
            try:
                per_game[key] = float(r["rate"])  # type: ignore
            except Exception:
                continue
        return per_game, base_rate
    except Exception:
        return None

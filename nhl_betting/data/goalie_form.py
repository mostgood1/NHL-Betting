from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _norm_team(v: str) -> str:
    try:
        return str(v).strip().upper()
    except Exception:
        return str(v).upper()


def load_goalie_form(date: str) -> Optional[Dict[str, float]]:
    """Load optional team-level goalie recent form metric for a date.

    Expected file (optional): data/processed/goalie_form_YYYY-MM-DD.csv
    Columns (flexible):
      - team (abbr or name)
      - one of: sv_pct_l10, svpct_l10, gsaa_l10, gsaa

    Returns a dict TEAM_ABBR -> float(form_metric). If file missing/unusable, returns None.
    """
    path = PROC_DIR / f"goalie_form_{date}.csv"
    if not (path.exists() and getattr(path.stat(), "st_size", 0) > 0):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    team_col = next((c for c in df.columns if c.lower() in ("team", "team_abbr", "abbr", "name")), None)
    form_col = next((c for c in df.columns if c.lower() in ("sv_pct_l10", "svpct_l10", "gsaa_l10", "gsaa", "saves_recent")), None)
    if not team_col or not form_col:
        return None
    try:
        df = df[[team_col, form_col]].dropna()
        df.columns = ["team", "form"]
        df["team"] = df["team"].map(_norm_team)
        df["form"] = pd.to_numeric(df["form"], errors="coerce")
        df = df.dropna(subset=["form"])  # type: ignore
        if df.empty:
            return None
        out: Dict[str, float] = {}
        for _, r in df.iterrows():
            try:
                out[str(r["team"])]= float(r["form"])  # type: ignore
            except Exception:
                continue
        return out
    except Exception:
        return None

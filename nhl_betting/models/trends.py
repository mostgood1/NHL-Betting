from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from nhl_betting.utils.io import MODEL_DIR


TRENDS_PATH = MODEL_DIR / "trend_adjustments.json"


@dataclass
class TrendAdjustments:
    """Holds subgroup adjustments for moneyline and scoring.

    All adjustments are small deltas applied additively:
    - ml_home: added to P(home ML) before clamping to [0.01, 0.99]
    - goals: added to team-specific lambda (scoring rate)
    Keys use namespaces: team:ABR, div:Division, conf:Conference
    """

    ml_home: Dict[str, float]
    goals: Dict[str, float]

    @staticmethod
    def load() -> "TrendAdjustments":
        if TRENDS_PATH.exists():
            with open(TRENDS_PATH, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    return TrendAdjustments(ml_home=data.get("ml_home", {}), goals=data.get("goals", {}))
                except Exception:
                    pass
        return TrendAdjustments(ml_home={}, goals={})

    def save(self) -> None:
        TRENDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TRENDS_PATH, "w", encoding="utf-8") as f:
            json.dump({"ml_home": self.ml_home, "goals": self.goals}, f, indent=2)


def team_keys(team_name: str) -> Tuple[str, str, str]:
    """Return (team_key, div_key, conf_key) for the given display name.

    Falls back to empty string for unknowns.
    """
    try:
        from nhl_betting.web.teams import get_team_assets as _assets
        abbr = (_assets(str(team_name)).get("abbr") or "").upper()
        division = _assets(str(team_name)).get("division") or ""
        conference = _assets(str(team_name)).get("conference") or ""
    except Exception:
        abbr = ""
        division = ""
        conference = ""
    tkey = f"team:{abbr}" if abbr else ""
    dkey = f"div:{division}" if division else ""
    ckey = f"conf:{conference}" if conference else ""
    return tkey, dkey, ckey


def get_adjustment(adj: Dict[str, float], keys: Tuple[str, str, str]) -> float:
    """Sum of all provided keys present in adj dict.

    Order: team + division + conference. Missing keys contribute 0.
    """
    total = 0.0
    for k in keys:
        if k and k in adj:
            try:
                total += float(adj[k])
            except Exception:
                continue
    return total

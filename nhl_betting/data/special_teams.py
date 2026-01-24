from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from ..utils.io import PROC_DIR


def _read_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_team_specials() -> Dict[str, Dict[str, float]]:
    """Load team PP/PK percentages from processed JSON; fallback to generic file.

    Returns: { 'TEAM': { 'pp_pct': float, 'pk_pct': float }, ... }
    """
    # Prefer season-specific if available
    generic = PROC_DIR / "team_special_teams.json"
    obj = _read_json(generic)
    teams = obj.get("teams") or {}
    # Normalize keys to upper
    return {str(k).upper(): {"pp_pct": float(v.get("pp_pct", 0.20)), "pk_pct": float(v.get("pk_pct", 0.80))} for k, v in teams.items()} if isinstance(teams, dict) else {}


def load_team_penalty_rates() -> Dict[str, Dict[str, float]]:
    """Load team penalty rates (committed/drawn per game) from processed JSON.

    Returns: { 'TEAM': { 'committed_per60': float, 'drawn_per60': float } }
    Note: values are per-game counts despite the name; treat as such.
    """
    generic = PROC_DIR / "team_penalty_rates.json"
    obj = _read_json(generic)
    # Normalize keys to upper
    return {str(k).upper(): {"committed_per60": float(v.get("committed_per60", 3.0)), "drawn_per60": float(v.get("drawn_per60", 3.0))} for k, v in obj.items()} if isinstance(obj, dict) else {}


__all__ = ["load_team_specials", "load_team_penalty_rates"]

from __future__ import annotations

"""Injury tracker schema and adapters (baseline).

This module defines a simple normalized injury status frame and a placeholder adapter.
Future adapters can integrate external sources; for now we allow manual entries and
lightweight placeholders.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd


@dataclass
class InjuryRecord:
    date: str  # ET YYYY-MM-DD
    player_id: int
    full_name: str
    team: str
    status: str  # OUT | GTD | IR | DTD | Probable | Healthy
    note: Optional[str] = None


def build_injury_snapshot(date: str, manual: Optional[List[Dict]] = None) -> pd.DataFrame:
    """Return a normalized injury snapshot for the given ET date.

    Parameters
    - date: target ET date (YYYY-MM-DD)
    - manual: optional list of dicts with keys matching InjuryRecord fields
    """
    rows: List[Dict] = []
    for rec in (manual or []):
        try:
            rows.append({
                "date": str(date),
                "player_id": int(rec.get("player_id")),
                "full_name": str(rec.get("full_name")),
                "team": str(rec.get("team")),
                "status": str(rec.get("status")),
                "note": rec.get("note"),
            })
        except Exception:
            continue
    return pd.DataFrame(rows, columns=["date","player_id","full_name","team","status","note"])


__all__ = ["InjuryRecord", "build_injury_snapshot"]

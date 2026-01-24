from __future__ import annotations

"""Lineup tracker: expected lines (EV), PP, PK with confidence scores (baseline).

Leverages existing TOI-based inference as a starting point; allows manual overrides
and confidence scoring per unit.
"""

from typing import Dict, List, Optional

import pandas as pd

from .rosters import build_roster_snapshot, infer_lines, project_toi


def build_lineup_snapshot(team_abbr: str, usage_df: Optional[pd.DataFrame] = None, overrides: Optional[Dict] = None) -> pd.DataFrame:
    """Return a lineup snapshot with columns: player_id, full_name, position, line_slot, pp_unit, pk_unit, proj_toi, confidence.

    - If usage_df is provided, use infer_lines/project_toi; else build a roster snapshot and project baseline TOI.
    - Overrides may set explicit slots or units for specific players.
    """
    if usage_df is None or usage_df.empty:
        roster = build_roster_snapshot(team_abbr)
        # Build minimal usage proxy and required columns for inference
        roster = roster.rename(columns={"full_name":"full_name","player_id":"player_id","position":"position"}).copy()
        roster["toi_avg"] = 15.0  # placeholder average
        roster["toi_pp_avg"] = 2.0
        roster["toi_sh_avg"] = 1.0
        df = project_toi(infer_lines(roster))
    else:
        df = usage_df.copy()
        for col in ("toi_avg","toi_pp_avg","toi_sh_avg"):
            if col not in df.columns:
                df[col] = 0.0
        df = project_toi(infer_lines(df))
    df["confidence"] = 0.5  # baseline
    overrides = overrides or {}
    # Apply overrides
    if overrides:
        df = df.copy()
        for pid, vals in overrides.items():
            mask = df["player_id"].eq(int(pid))
            for k, v in vals.items():
                if k in df.columns:
                    df.loc[mask, k] = v
    return df[["player_id","full_name","position","line_slot","pp_unit","pk_unit","proj_toi","confidence"]]


__all__ = ["build_lineup_snapshot"]

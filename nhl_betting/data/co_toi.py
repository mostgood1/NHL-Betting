from __future__ import annotations

"""Compute co-TOI (estimated shared even-strength minutes) from lineup snapshots.

Baseline heuristic: players on the same EV line share the line's EV minutes equally.
Future: refine with shift charts and on-ice detection.
"""

from typing import List, Dict

import pandas as pd


def build_co_toi_from_lineups(lineups: pd.DataFrame) -> pd.DataFrame:
    """Return pair-level co-TOI minutes estimated from lineup snapshot.

    Input columns expected: team, player_id, position, line_slot, proj_toi
    Output columns: team, player_id_a, player_id_b, co_toi_ev
    """
    req = {"team","player_id","position","line_slot","proj_toi"}
    if lineups is None or lineups.empty or not req.issubset(set(lineups.columns)):
        return pd.DataFrame(columns=["team","player_id_a","player_id_b","co_toi_ev"])
    rows: List[Dict] = []
    for team, g in lineups.groupby("team"):
        # Group by EV line_slot (e.g., L1/L2/L3/L4 and D1/D2/D3)
        for slot, grp in g.groupby("line_slot"):
            if slot is None:
                continue
            grp = grp.copy()
            # Estimate EV minutes as proj_toi excluding PP/PK; baseline uses proj_toi
            ev_minutes = float(grp["proj_toi"].astype(float).mean()) if len(grp) > 0 else 0.0
            pids = grp["player_id"].astype(int).tolist()
            n = len(pids)
            if n <= 1 or ev_minutes <= 0:
                continue
            # Assume equal share within the line for EV on-ice time overlap
            # Pairwise co-TOI: overlap approximated as ev_minutes * overlap_factor
            # Baseline overlap_factor=0.75 for forwards, 0.85 for defense pairs
            overlap_factor = 0.75
            if str(slot).startswith("D"):
                overlap_factor = 0.85
            co = ev_minutes * overlap_factor
            for i in range(n):
                for j in range(i+1, n):
                    rows.append({
                        "team": team,
                        "player_id_a": int(pids[i]),
                        "player_id_b": int(pids[j]),
                        "co_toi_ev": float(co),
                    })
    return pd.DataFrame(rows)


__all__ = ["build_co_toi_from_lineups"]

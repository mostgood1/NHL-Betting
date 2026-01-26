from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .state import GameState, Event


@dataclass
class PlayerPeriodStats:
    player_id: int
    team: str
    period: int
    shots: int = 0
    goals: int = 0
    assists: int = 0
    points: int = 0
    blocks: int = 0
    saves: int = 0
    toi: float = 0.0


def aggregate_events_to_boxscores(gs: GameState, events: List[Event]) -> pd.DataFrame:
    """Aggregate simulated events into per-player per-period boxscores.

    - shots/goals/assists/blocks: counted from events
    - points: goals + assists
    - saves: derived per period from opponent shots minus opponent goals, assigned to starter goalie
    - toi: accumulated from 'shift' events' meta['dur']
    """
    # Collect basic per-period stats from events
    per_key: Dict[tuple, PlayerPeriodStats] = {}

    def _get(player_id: int, team: str, period: int) -> PlayerPeriodStats:
        k = (player_id, team, period)
        s = per_key.get(k)
        if s is None:
            s = PlayerPeriodStats(player_id=player_id, team=team, period=period)
            per_key[k] = s
        return s

    # Track team-level shots/goals per period for saves attribution
    team_pd_counts: Dict[tuple, Dict[str, int]] = {}
    def _team_pd(team: str, period: int) -> Dict[str, int]:
        k = (team, period)
        d = team_pd_counts.get(k)
        if d is None:
            d = {"shots": 0, "goals": 0}
            team_pd_counts[k] = d
        return d

    for e in events:
        if e.period <= 0:
            continue
        if e.kind == "shot":
            d = _team_pd(e.team, e.period)
            d["shots"] += 1
            if e.player_id is not None:
                s = _get(int(e.player_id), e.team, e.period)
                s.shots += 1
        elif e.kind == "goal":
            d = _team_pd(e.team, e.period)
            d["goals"] += 1
            if e.player_id is not None:
                s = _get(int(e.player_id), e.team, e.period)
                s.goals += 1
                s.points += 1
        elif e.kind == "assist":
            if e.player_id is not None:
                s = _get(int(e.player_id), e.team, e.period)
                s.assists += 1
                s.points += 1
        elif e.kind == "block":
            if e.player_id is not None:
                s = _get(int(e.player_id), e.team, e.period)
                s.blocks += 1
        elif e.kind == "shift":
            if e.player_id is not None:
                s = _get(int(e.player_id), e.team, e.period)
                dur = float(e.meta.get("dur", 0.0))
                s.toi += max(0.0, dur)

    # Assign saves per period to starting goalies
    def _starter_goalie(team_name: str) -> Optional[int]:
        team = gs.home if team_name == gs.home.name else gs.away
        goalies = [p for p in team.players.values() if str(p.position) == "G"]
        if not goalies:
            return None
        # choose highest projected TOI as starter
        return int(max(goalies, key=lambda p: float(p.toi_proj or 0.0)).player_id)

    # Opponent saves: opponent shots - opponent goals in that period
    for team_name in (gs.home.name, gs.away.name):
        opp_name = gs.away.name if team_name == gs.home.name else gs.home.name
        goalie_id = _starter_goalie(team_name)
        if goalie_id is None:
            continue
        # periods observed
        periods = set([p for (_, t, p) in per_key.keys() if t == opp_name])
        for period in sorted(periods):
            d = _team_pd(opp_name, period)
            saves = max(0, int(d.get("shots", 0)) - int(d.get("goals", 0)))
            if saves > 0:
                s = _get(goalie_id, team_name, period)
                s.saves += saves

    # Convert to DataFrame and also include game totals per player
    rows = []
    # Per-period rows
    for s in per_key.values():
        rows.append({
            "team": s.team,
            "player_id": s.player_id,
            "period": int(s.period),
            "shots": int(s.shots),
            "goals": int(s.goals),
            "assists": int(s.assists),
            "points": int(s.points),
            "blocks": int(s.blocks),
            "saves": int(s.saves),
            "toi_sec": float(s.toi),
        })
    df = pd.DataFrame(rows)
    # Game-total rows (period=0)
    if not df.empty:
        agg = df.groupby(["team", "player_id"], as_index=False)[["shots","goals","assists","points","blocks","saves","toi_sec"]].sum()
        agg["period"] = 0
        df = pd.concat([df, agg], ignore_index=True)
    return df
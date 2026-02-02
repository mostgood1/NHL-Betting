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


SAVES_CAL: float = 0.50  # calibration factor to reduce simulated saves toward observed levels


def aggregate_events_to_boxscores(gs: GameState, events: List[Event], starter_goalies: Optional[Dict[str, int]] = None) -> pd.DataFrame:
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
        # Prefer provided starter map when available
        if starter_goalies and team_name in starter_goalies:
            return int(starter_goalies.get(team_name))
        team = gs.home if team_name == gs.home.name else gs.away
        goalies = [p for p in team.players.values() if str(getattr(p, "position", "")).strip().upper() == "G"]
        if not goalies:
            return None
        # choose highest projected TOI as starter
        return int(max(goalies, key=lambda p: float(p.toi_proj or 0.0)).player_id)

    # Opponent saves: attribute saves equal to opponent shots on goal that did NOT result in goals.
    # Engine emits separate 'shot' and 'goal' events; SOG consistency holds as saves + goals == shots.
    for team_name in (gs.home.name, gs.away.name):
        opp_name = gs.away.name if team_name == gs.home.name else gs.home.name
        goalie_id = _starter_goalie(team_name)
        if goalie_id is None:
            continue
        # periods observed
        periods = set([p for (_, t, p) in per_key.keys() if t == opp_name])
        for period in sorted(periods):
            d = _team_pd(opp_name, period)
            shots = int(d.get("shots", 0))
            goals = int(d.get("goals", 0))
            # saves = shots on goal that do not score
            saves = max(0, shots - goals)
            if saves > 0:
                s = _get(goalie_id, team_name, period)
                s.saves += saves
                # Ensure a minimal TOI presence when saves occur to avoid zero-TOI anomalies
                s.toi = max(float(s.toi), 60.0)


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
        # TOI sanity fallback: if total TOI is unrealistically low, fallback to projected TOI from GameState.
        # For goalies, we also apply a stronger sanity rule because saves can be attributed via
        # `starter_goalies` even when goalie shift events are missing (e.g., roster/goalie-id mismatch).
        try:
            # Build proj_toi seconds map from game state
            proj_map = {}
            for team_state in [gs.home, gs.away]:
                for pid, p in team_state.players.items():
                    try:
                        proj_map[int(pid)] = float(getattr(p, "toi_proj", 0.0) or 0.0) * 60.0
                    except Exception:
                        continue
            # Define a minimal threshold (e.g., 120 seconds); if below, set to projected
            thr_sec = 120.0
            goalie_full_game_sec = 60.0 * 60.0
            goalie_min_reasonable_sec = 40.0 * 60.0
            for i, r in agg.iterrows():
                try:
                    total_toi = float(r.get("toi_sec") or 0.0)
                    pid = int(r.get("player_id"))
                    team_name = str(r.get("team") or "")
                    # Identify goalies robustly using GameState and/or starter map
                    is_goalie = False
                    try:
                        team_state = gs.home if team_name == gs.home.name else gs.away
                        pstate = team_state.players.get(pid)
                        is_goalie = str(getattr(pstate, "position", "")).strip().upper() == "G"
                    except Exception:
                        is_goalie = False
                    if starter_goalies and team_name in starter_goalies:
                        try:
                            if int(starter_goalies.get(team_name)) == int(pid):
                                is_goalie = True
                        except Exception:
                            pass

                    # If goalie has saves but low TOI, treat as missing goalie shift events and
                    # set to a full-game value to restore realism.
                    if is_goalie and int(r.get("saves") or 0) > 0 and total_toi < goalie_min_reasonable_sec:
                        agg.at[i, "toi_sec"] = max(total_toi, goalie_full_game_sec)
                        continue

                    # If goalie is the designated starter, ensure non-trivial TOI even if saves
                    # happen to be 0 in this sim aggregate.
                    if is_goalie and total_toi < thr_sec and starter_goalies and team_name in starter_goalies:
                        agg.at[i, "toi_sec"] = max(total_toi, goalie_full_game_sec)
                        continue

                    if total_toi < thr_sec:
                        fallback = float(proj_map.get(pid, 0.0))
                        if fallback > 0:
                            agg.at[i, "toi_sec"] = fallback
                except Exception:
                    continue
        except Exception:
            pass
        agg["period"] = 0
        df = pd.concat([df, agg], ignore_index=True)
    return df
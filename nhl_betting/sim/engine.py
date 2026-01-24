from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .state import GameState, TeamState, PlayerState, Event
from .models import RateModels, TeamRates, PlayerRates


@dataclass
class SimConfig:
    periods: int = 3
    seconds_per_period: int = 20 * 60
    overtime_seconds: int = 5 * 60
    ot_enabled: bool = True
    seed: Optional[int] = None


class PossessionSimulator:
    """Placeholder for future possession-level simulation.

    Currently unused; will be extended to model discrete events (passes, shots, turnovers)
    conditioned on on-ice line combos and score effects.
    """
    def __init__(self, rates: TeamRates, rng: random.Random):
        self.rates = rates
        self.rng = rng


class PeriodSimulator:
    def __init__(self, cfg: SimConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng

    def _poisson(self, lam: float) -> int:
        # Use numpy for stable Poisson sampling
        return int(np.random.poisson(lam=max(lam, 1e-9)))

    def simulate_period(self, gs: GameState, rates: RateModels, period_idx: int) -> Tuple[int, int, List[Event]]:
        T = self.cfg.seconds_per_period
        # Convert per-60 rates to period totals (EV approximation)
        home_shots = self._poisson(rates.home.shots_per_60 * T / 3600.0)
        away_shots = self._poisson(rates.away.shots_per_60 * T / 3600.0)
        home_goals = self._poisson(rates.home.goals_per_60 * T / 3600.0)
        away_goals = self._poisson(rates.away.goals_per_60 * T / 3600.0)
        events: List[Event] = []
        # Distribute events uniformly in the period for now
        def _uniform_times(n: int) -> List[float]:
            return sorted([self.rng.random() * T for _ in range(n)])
        for t in _uniform_times(home_shots):
            events.append(Event(t=t + period_idx * T, period=period_idx + 1, team=gs.home.name, kind="shot"))
        for t in _uniform_times(away_shots):
            events.append(Event(t=t + period_idx * T, period=period_idx + 1, team=gs.away.name, kind="shot"))
        for t in _uniform_times(home_goals):
            events.append(Event(t=t + period_idx * T, period=period_idx + 1, team=gs.home.name, kind="goal"))
        for t in _uniform_times(away_goals):
            events.append(Event(t=t + period_idx * T, period=period_idx + 1, team=gs.away.name, kind="goal"))
        return home_goals, away_goals, events

    def simulate_period_with_lines(
        self,
        gs: GameState,
        rates: RateModels,
        period_idx: int,
        lineup_home: List[Dict],
        lineup_away: List[Dict],
        st_home: Optional[Dict[str, float]] = None,
        st_away: Optional[Dict[str, float]] = None,
        special_teams_cal: Optional[Dict[str, float]] = None,
    ) -> Tuple[int, int, List[Event]]:
        """Segment the period by EV lines and simulate events with simple score effects.

        Assumptions:
        - Rotate EV lines roughly equally across the period.
        - Score effects: trailing team shots +10%, leading team -10% (cap to [0.6, 1.4]).
        - Goals drawn proportionally to shots.
        """
        T = self.cfg.seconds_per_period
        segments = 8  # coarse rotation segments per period
        seg_len = T / segments
        events: List[Event] = []
        home_goals = 0
        away_goals = 0
        # Build ordered line groups (L1->L4, D1->D3); fallback to roster
        def _line_order(lineup: List[Dict], prefix: str) -> List[List[int]]:
            df = lineup
            slots = [f"{prefix}{i}" for i in ([1,2,3,4] if prefix == "L" else [1,2,3])]
            out = []
            for s in slots:
                pids = [int(r.get("player_id")) for r in df if str(r.get("line_slot")) == s]
                if pids:
                    out.append(pids)
            return out
        def _pp_units(lineup: List[Dict]) -> List[List[int]]:
            u1 = [int(r.get("player_id")) for r in lineup if r.get("pp_unit") == 1]
            u2 = [int(r.get("player_id")) for r in lineup if r.get("pp_unit") == 2]
            out = []
            if u1:
                out.append(u1)
            if u2:
                out.append(u2)
            return out
        def _pk_units(lineup: List[Dict]) -> List[List[int]]:
            u1 = [int(r.get("player_id")) for r in lineup if r.get("pk_unit") == 1]
            u2 = [int(r.get("player_id")) for r in lineup if r.get("pk_unit") == 2]
            out = []
            if u1:
                out.append(u1)
            if u2:
                out.append(u2)
            return out
        l_home = _line_order(lineup_home, "L") or [[pid for pid in gs.home.players.keys() if gs.home.players[pid].position in ("F","D")][:5]]
        d_home = _line_order(lineup_home, "D") or [[pid for pid in gs.home.players.keys() if gs.home.players[pid].position == "D"][:2]]
        l_away = _line_order(lineup_away, "L") or [[pid for pid in gs.away.players.keys() if gs.away.players[pid].position in ("F","D")][:5]]
        d_away = _line_order(lineup_away, "D") or [[pid for pid in gs.away.players.keys() if gs.away.players[pid].position == "D"][:2]]
        # Simple rotation indexes
        idx_lh = 0; idx_da = 0; idx_la = 0; idx_dh = 0
        # Special teams units
        pp_home = _pp_units(lineup_home)
        pk_home = _pk_units(lineup_home)
        pp_away = _pp_units(lineup_away)
        pk_away = _pk_units(lineup_away)
        idx_pp_h = 0; idx_pk_h = 0; idx_pp_a = 0; idx_pk_a = 0
        # Special teams parameters
        st_home = st_home or {"pp_pct": 0.2, "pk_pct": 0.8, "drawn_per_game": 3.0, "committed_per_game": 3.0}
        st_away = st_away or {"pp_pct": 0.2, "pk_pct": 0.8, "drawn_per_game": 3.0, "committed_per_game": 3.0}
        # Calibration multipliers (default to neutral 1.0)
        cal_pp_sh_mult = float((special_teams_cal or {}).get("pp_shot_multiplier", 1.0))
        cal_pk_sh_mult = float((special_teams_cal or {}).get("pk_shot_multiplier", 1.0))
        cal_pp_gl_mult = float((special_teams_cal or {}).get("pp_goal_multiplier", 1.0))
        cal_pk_gl_mult = float((special_teams_cal or {}).get("pk_goal_multiplier", 1.0))
        # Expected PP minutes per game approx = drawn_per_game * 2
        pp_frac_home = max(0.0, min(0.35, (float(st_home.get("drawn_per_game", 3.0)) * 2.0) / 60.0))
        pp_frac_away = max(0.0, min(0.35, (float(st_away.get("drawn_per_game", 3.0)) * 2.0) / 60.0))
        # Sample PP segments with these fractions
        for k in range(segments):
            # score effects factor
            diff = gs.home.score - gs.away.score
            home_factor = max(0.6, min(1.4, 1.0 + (-0.10 if diff > 0 else (0.10 if diff < 0 else 0.0))))
            away_factor = max(0.6, min(1.4, 1.0 + (-0.10 if diff < 0 else (0.10 if diff > 0 else 0.0))))
            # PP/PK effects
            seg_is_home_pp = (self.rng.random() < pp_frac_home) and not (self.rng.random() < pp_frac_away)
            seg_is_away_pp = (self.rng.random() < pp_frac_away) and not seg_is_home_pp
            # Base PP/PK shot factors further scaled by calibration
            pp_mult_shots = 1.4 * cal_pp_sh_mult
            pk_mult_shots = 0.7 * cal_pk_sh_mult
            if seg_is_home_pp:
                home_factor *= pp_mult_shots
                away_factor *= pk_mult_shots
            elif seg_is_away_pp:
                away_factor *= pp_mult_shots
                home_factor *= pk_mult_shots
            # Empty-net end-game effects: trailing team increases shot rate; leading team more likely to score into empty net
            # Simple heuristic: apply in final 3 minutes of P3 when down 1, and final 2 minutes when down >=2
            if period_idx == 2:  # third period (0-indexed)
                time_elapsed = k * seg_len
                time_remaining = T - time_elapsed
                empty_net_p = 0.18
                two_goal_scale = 0.30
                if time_remaining <= 180:  # last 3 minutes
                    if diff < 0:  # home trailing
                        home_factor *= 1.20
                        # away empty-net chance
                        p_empty_mult = 1.0 + empty_net_p + (two_goal_scale if diff <= -2 and time_remaining <= 120 else 0.0)
                    elif diff > 0:  # away trailing
                        away_factor *= 1.20
                        p_empty_mult = 1.0 + empty_net_p + (two_goal_scale if diff >= 2 and time_remaining <= 120 else 0.0)
                    else:
                        p_empty_mult = 1.0
                else:
                    p_empty_mult = 1.0
            else:
                p_empty_mult = 1.0
            # Expected shots in segment
            lam_h = max(1e-6, rates.home.shots_per_60 * seg_len / 3600.0 * home_factor)
            lam_a = max(1e-6, rates.away.shots_per_60 * seg_len / 3600.0 * away_factor)
            sh_h = int(np.random.poisson(lam_h))
            sh_a = int(np.random.poisson(lam_a))
            # Goals proportional to shots vs base conversion
            p_goal_home = (rates.home.goals_per_60 / max(rates.home.shots_per_60, 1e-3))
            p_goal_away = (rates.away.goals_per_60 / max(rates.away.shots_per_60, 1e-3))
            if seg_is_home_pp:
                p_goal_home = p_goal_home * (1.0 + 1.5 * float(st_home.get("pp_pct", 0.2))) * cal_pp_gl_mult
                p_goal_home = min(0.45, p_goal_home)
                p_goal_away = p_goal_away * (0.9 * float(st_away.get("pk_pct", 0.8))) * cal_pk_gl_mult
                p_goal_away = max(0.01, p_goal_away)
            elif seg_is_away_pp:
                p_goal_away = p_goal_away * (1.0 + 1.5 * float(st_away.get("pp_pct", 0.2))) * cal_pp_gl_mult
                p_goal_away = min(0.45, p_goal_away)
                p_goal_home = p_goal_home * (0.9 * float(st_home.get("pk_pct", 0.8))) * cal_pk_gl_mult
                p_goal_home = max(0.01, p_goal_home)
            # Apply empty-net multiplier to the leading team's conversion, if any
            if period_idx == 2 and p_empty_mult != 1.0:
                if diff < 0:  # away leading
                    p_goal_away = min(0.65, p_goal_away * p_empty_mult)
                elif diff > 0:  # home leading
                    p_goal_home = min(0.65, p_goal_home * p_empty_mult)
            g_h = sum(1 for _ in range(sh_h) if self.rng.random() < p_goal_home)
            g_a = sum(1 for _ in range(sh_a) if self.rng.random() < p_goal_away)
            home_goals += g_h; away_goals += g_a
            # Times in segment window
            t0 = k * seg_len + period_idx * T
            # Determine on-ice groups for attribution
            strength_h = "PP" if seg_is_home_pp else ("PK" if seg_is_away_pp else "EV")
            strength_a = "PP" if seg_is_away_pp else ("PK" if seg_is_home_pp else "EV")
            if seg_is_home_pp and pp_home:
                # Prefer PP unit 1 slightly (approx TOI share)
                if len(pp_home) > 1 and self.rng.random() < 0.65:
                    ice_h = pp_home[0]
                else:
                    ice_h = pp_home[idx_pp_h % len(pp_home)]
                    idx_pp_h = (idx_pp_h + 1) % max(1, len(pp_home))
                # Away team on PK if units available
                if pk_away:
                    if len(pk_away) > 1 and self.rng.random() < 0.60:
                        ice_a = pk_away[0]
                    else:
                        ice_a = pk_away[idx_pk_a % len(pk_away)]
                        idx_pk_a = (idx_pk_a + 1) % max(1, len(pk_away))
                else:
                    ice_a = (l_away[idx_la % len(l_away)] if l_away else []) + (d_away[idx_da % len(d_away)] if d_away else [])
            elif seg_is_away_pp and pp_away:
                # Prefer PP unit 1 slightly (approx TOI share)
                if len(pp_away) > 1 and self.rng.random() < 0.65:
                    ice_a = pp_away[0]
                else:
                    ice_a = pp_away[idx_pp_a % len(pp_away)]
                    idx_pp_a = (idx_pp_a + 1) % max(1, len(pp_away))
                # Home team on PK if units available
                if pk_home:
                    if len(pk_home) > 1 and self.rng.random() < 0.60:
                        ice_h = pk_home[0]
                    else:
                        ice_h = pk_home[idx_pk_h % len(pk_home)]
                        idx_pk_h = (idx_pk_h + 1) % max(1, len(pk_home))
                else:
                    ice_h = (l_home[idx_lh % len(l_home)] if l_home else []) + (d_home[idx_dh % len(d_home)] if d_home else [])
            else:
                ice_h = (l_home[idx_lh % len(l_home)] if l_home else []) + (d_home[idx_dh % len(d_home)] if d_home else [])
                ice_a = (l_away[idx_la % len(l_away)] if l_away else []) + (d_away[idx_da % len(d_away)] if d_away else [])
            # Attribute shots and goals to players on ice
            for _ in range(sh_h):
                pid = (self.rng.choice(ice_h) if ice_h else None)
                events.append(Event(t=t0 + self.rng.random() * seg_len, period=period_idx + 1, team=gs.home.name, kind="shot", player_id=pid, meta={"strength": strength_h}))
            for _ in range(sh_a):
                pid = (self.rng.choice(ice_a) if ice_a else None)
                events.append(Event(t=t0 + self.rng.random() * seg_len, period=period_idx + 1, team=gs.away.name, kind="shot", player_id=pid, meta={"strength": strength_a}))
            for _ in range(g_h):
                pid = (self.rng.choice(ice_h) if ice_h else None)
                events.append(Event(t=t0 + self.rng.random() * seg_len, period=period_idx + 1, team=gs.home.name, kind="goal", player_id=pid, meta={"strength": strength_h}))
            for _ in range(g_a):
                pid = (self.rng.choice(ice_a) if ice_a else None)
                events.append(Event(t=t0 + self.rng.random() * seg_len, period=period_idx + 1, team=gs.away.name, kind="goal", player_id=pid, meta={"strength": strength_a}))
            # rotate
            idx_lh = (idx_lh + 1) % max(1, len(l_home)); idx_dh = (idx_dh + 1) % max(1, len(d_home))
            idx_la = (idx_la + 1) % max(1, len(l_away)); idx_da = (idx_da + 1) % max(1, len(d_away))
        return home_goals, away_goals, events


class GameSimulator:
    def __init__(self, cfg: SimConfig, rates: RateModels):
        self.cfg = cfg
        self.rates = rates
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.period_sim = PeriodSimulator(cfg, self.rng)

    def _init_game_state(self, home_name: str, away_name: str, roster_home: List[Dict], roster_away: List[Dict]) -> GameState:
        home = TeamState(name=home_name, players={})
        away = TeamState(name=away_name, players={})
        for row in roster_home:
            pid = int(row.get("player_id"))
            p = PlayerState(player_id=pid, full_name=row.get("full_name"), position=row.get("position"), team=home_name, toi_proj=float(row.get("proj_toi", 0.0)))
            home.players[pid] = p
        for row in roster_away:
            pid = int(row.get("player_id"))
            p = PlayerState(player_id=pid, full_name=row.get("full_name"), position=row.get("position"), team=away_name, toi_proj=float(row.get("proj_toi", 0.0)))
            away.players[pid] = p
        return GameState(home=home, away=away, period=0, clock=self.cfg.seconds_per_period)

    def _allocate_player_stats(self, team: TeamState, kind: str, count: int) -> None:
        # Allocate team events to players proportional to projected TOI (simple baseline)
        skaters = [p for p in team.players.values() if p.position in ("F", "D")]
        goalies = [p for p in team.players.values() if p.position == "G"]
        pool = skaters if kind in ("shots", "goals", "blocks") else goalies
        total = sum(max(p.toi_proj, 1e-6) for p in pool) or 1.0
        weights = [(p, max(p.toi_proj, 1e-6) / total) for p in pool]
        # Multinomial draw
        probs = [w for (_, w) in weights]
        draws = np.random.multinomial(n=count, pvals=probs) if probs else np.zeros(len(weights), dtype=int)
        for (p, _), n in zip(weights, draws):
            p.stats[kind] = p.stats.get(kind, 0.0) + float(n)

    def simulate(self, home_name: str, away_name: str, roster_home: List[Dict], roster_away: List[Dict]) -> Tuple[GameState, List[Event]]:
        gs = self._init_game_state(home_name, away_name, roster_home, roster_away)
        events_all: List[Event] = []
        for pd in range(self.cfg.periods):
            hg, ag, ev = self.period_sim.simulate_period(gs, self.rates, pd)
            gs.home.score += int(hg)
            gs.away.score += int(ag)
            # Track shots separately: approximate shots = goals + misses; here use rates
            # Allocate player stats
            self._allocate_player_stats(gs.home, "shots", max(0, int(round(self.rates.home.shots_per_60 * self.cfg.seconds_per_period / 3600.0))))
            self._allocate_player_stats(gs.away, "shots", max(0, int(round(self.rates.away.shots_per_60 * self.cfg.seconds_per_period / 3600.0))))
            self._allocate_player_stats(gs.home, "goals", int(hg))
            self._allocate_player_stats(gs.away, "goals", int(ag))
            events_all.extend(ev)
        # Simple OT if tied
        if self.cfg.ot_enabled and gs.home.score == gs.away.score:
            ot_hg, ot_ag, _ = self.period_sim.simulate_period(gs, self.rates, self.cfg.periods)
            gs.home.score += int(ot_hg)
            gs.away.score += int(ot_ag)
        # Saves: opponent shots - opponent goals
        home_saves = max(0, int(round(self.rates.away.shots_per_60 * self.cfg.seconds_per_period / 3600.0 * self.cfg.periods)) - gs.away.score)
        away_saves = max(0, int(round(self.rates.home.shots_per_60 * self.cfg.seconds_per_period / 3600.0 * self.cfg.periods)) - gs.home.score)
        self._allocate_player_stats(gs.home, "saves", home_saves)
        self._allocate_player_stats(gs.away, "saves", away_saves)
        return gs, events_all

    def simulate_with_lineups(
        self,
        home_name: str,
        away_name: str,
        roster_home: List[Dict],
        roster_away: List[Dict],
        lineup_home: List[Dict],
        lineup_away: List[Dict],
        st_home: Optional[Dict[str, float]] = None,
        st_away: Optional[Dict[str, float]] = None,
        special_teams_cal: Optional[Dict[str, float]] = None,
    ) -> Tuple[GameState, List[Event]]:
        gs = self._init_game_state(home_name, away_name, roster_home, roster_away)
        events_all: List[Event] = []
        for pd in range(self.cfg.periods):
            hg, ag, ev = self.period_sim.simulate_period_with_lines(gs, self.rates, pd, lineup_home, lineup_away, st_home=st_home, st_away=st_away, special_teams_cal=special_teams_cal)
            gs.home.score += int(hg)
            gs.away.score += int(ag)
            # Attribute player events directly from simulated events (shots/goals)
            for e in ev:
                if e.kind in ("shot", "goal") and e.player_id is not None:
                    # Find player state and increment
                    team = gs.home if e.team == gs.home.name else gs.away
                    pstate = team.players.get(int(e.player_id))
                    if pstate:
                        key = "shots" if e.kind == "shot" else "goals"
                        pstate.stats[key] = pstate.stats.get(key, 0.0) + 1.0
            events_all.extend(ev)
        # Saves from events: opponent shots minus opponent goals
        shots_home = sum(1 for e in events_all if e.kind == "shot" and e.team == gs.home.name)
        shots_away = sum(1 for e in events_all if e.kind == "shot" and e.team == gs.away.name)
        goals_home = sum(1 for e in events_all if e.kind == "goal" and e.team == gs.home.name)
        goals_away = sum(1 for e in events_all if e.kind == "goal" and e.team == gs.away.name)
        home_saves = max(0, shots_away - goals_away)
        away_saves = max(0, shots_home - goals_home)
        # Assign saves to starting goalie if present; fallback to TOI allocation
        def _assign_saves(team: TeamState, count: int) -> None:
            goalies = [p for p in team.players.values() if p.position == "G"]
            if goalies:
                starter = max(goalies, key=lambda p: (p.toi_proj or 0.0))
                starter.stats["saves"] = starter.stats.get("saves", 0.0) + float(count)
            else:
                self._allocate_player_stats(team, "saves", count)
        _assign_saves(gs.home, home_saves)
        _assign_saves(gs.away, away_saves)
        return gs, events_all

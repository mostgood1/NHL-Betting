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

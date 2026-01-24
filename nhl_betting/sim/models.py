from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TeamRates:
    shots_per_60: float = 30.0
    goals_per_60: float = 2.9
    blocks_per_60: float = 12.0
    penalties_per_60: float = 3.0


@dataclass
class PlayerRates:
    shots_share: float = 0.05  # fraction of team shots
    goals_share: float = 0.05  # fraction of team goals
    blocks_share: float = 0.05  # fraction of team blocks
    saves_share: float = 1.0  # for goalies, fraction of opponent shots saved


@dataclass
class RateModels:
    home: TeamRates
    away: TeamRates
    player_rates: Dict[int, PlayerRates]

    @staticmethod
    def baseline(base_mu: float = 3.0) -> "RateModels":
        # Simple baseline: adjust goals per 60 around base_mu; shots approx 30/60
        home = TeamRates(shots_per_60=31.0, goals_per_60=base_mu, blocks_per_60=12.0, penalties_per_60=3.0)
        away = TeamRates(shots_per_60=30.0, goals_per_60=base_mu, blocks_per_60=12.0, penalties_per_60=3.0)
        return RateModels(home=home, away=away, player_rates={})

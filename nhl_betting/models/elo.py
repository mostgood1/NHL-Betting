from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import pandas as pd


@dataclass
class EloConfig:
    k: float = 20.0
    home_adv: float = 50.0  # Elo points


class Elo:
    def __init__(self, cfg: EloConfig | None = None):
        self.cfg = cfg or EloConfig()
        self.ratings: Dict[str, float] = {}

    def get(self, team: str) -> float:
        return self.ratings.get(team, 1500.0)

    def expected(self, team: str, opp: str, is_home: bool) -> float:
        ra = self.get(team) + (self.cfg.home_adv if is_home else 0.0)
        rb = self.get(opp)
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def update_game(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        # score: win=1, loss=0, ot/so affects? For baseline, win=1, loss=0
        home_score = 1.0 if home_goals > away_goals else 0.0
        away_score = 1.0 - home_score
        exp_home = self.expected(home, away, True)
        exp_away = 1.0 - exp_home
        k = self.cfg.k
        self.ratings[home] = self.get(home) + k * (home_score - exp_home)
        self.ratings[away] = self.get(away) + k * (away_score - exp_away)

    def predict_moneyline_prob(self, home: str, away: str) -> Tuple[float, float]:
        p_home = self.expected(home, away, True)
        return p_home, 1.0 - p_home

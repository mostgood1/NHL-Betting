from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import numpy as np


@dataclass
class PoissonConfig:
    home_attack: float = 1.05
    away_attack: float = 0.95


class PoissonGoals:
    """
    Simple home/away mean goals model.
    lambda_home = base_mu * home_attack
    lambda_away = base_mu * away_attack
    where base_mu can be set from recent league average.
    """

    def __init__(self, base_mu: float = 3.05, cfg: PoissonConfig | None = None):
        self.base_mu = base_mu
        self.cfg = cfg or PoissonConfig()

    def lambdas(self) -> Tuple[float, float]:
        return self.base_mu * self.cfg.home_attack, self.base_mu * self.cfg.away_attack

    def lambdas_from_total_split(self, total_line: float, p_home_ml: float) -> Tuple[float, float]:
        """Derive matchup lambdas from a total and ML split.

        We split total_line across teams proportional to ML win probabilities,
        lightly regularized by base_mu to avoid extreme allocations.
        """
        total_line = float(total_line) if total_line is not None else 2 * self.base_mu
        p_home = float(p_home_ml)
        p_home = min(max(p_home, 0.05), 0.95)  # clamp
        p_away = 1.0 - p_home
        # Soft allocation: 70% by ML split, 30% by baseline split
        lam_h0, lam_a0 = self.lambdas()
        lam_h = 0.7 * (total_line * p_home) + 0.3 * lam_h0
        lam_a = 0.7 * (total_line * p_away) + 0.3 * lam_a0
        return lam_h, lam_a

    def score_matrix(self, max_goals: int = 10, lam_h: float | None = None, lam_a: float | None = None) -> np.ndarray:
        lam_h = lam_h if lam_h is not None else self.base_mu * self.cfg.home_attack
        lam_a = lam_a if lam_a is not None else self.base_mu * self.cfg.away_attack
        i = np.arange(0, max_goals + 1)
        j = np.arange(0, max_goals + 1)
        # Use Python's math.factorial (vectorized) to avoid numpy.math dependency issues
        p_h = np.exp(-lam_h) * np.power(lam_h, i) / np.vectorize(math.factorial)(i)
        p_a = np.exp(-lam_a) * np.power(lam_a, j) / np.vectorize(math.factorial)(j)
        return np.outer(p_h, p_a)

    def probs(self, total_line: float = 6.0, pline: float = -1.5, max_goals: int = 10, lam_h: float | None = None, lam_a: float | None = None) -> Dict[str, float]:
        mat = self.score_matrix(max_goals, lam_h=lam_h, lam_a=lam_a)
        # Moneyline
        p_home = np.tril(mat, -1).sum()  # home goals > away goals
        p_away = np.triu(mat, 1).sum()
        p_draw = np.trace(mat)

        # Totals
        totals = np.arange(0, 2 * max_goals + 1)
        conv = np.convolve(mat.sum(axis=1), mat.sum(axis=0))  # distribution of total goals
        # Over/Under at half-lines using cumulative
        over = conv[totals > total_line].sum()
        under = conv[totals < total_line].sum()

        # Puck line (home -1.5)
        ph_pl = 0.0
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                if h - a > 1.5:
                    ph_pl += mat[h, a]
        pa_pl = 1.0 - ph_pl

        return {
            "home_ml": float(p_home + 0.5 * p_draw),
            "away_ml": float(p_away + 0.5 * p_draw),
            "over": float(over),
            "under": float(under),
            "home_puckline_-1.5": float(ph_pl),
            "away_puckline_+1.5": float(pa_pl),
        }

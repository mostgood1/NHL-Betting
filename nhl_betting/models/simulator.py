from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SimConfig:
    n_sims: int = 20000
    max_goals_period: int = 10  # cap for safety when using matrix methods
    random_state: Optional[int] = 42


def _rng(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    # Use PCG64 for speed and reproducibility
    return np.random.default_rng(seed)


def derive_team_lambdas(total_mean: float, diff_mean: float) -> Tuple[float, float]:
    """Derive team goal means (per game) from total and differential.

    lambda_home = (total_mean + diff_mean) / 2
    lambda_away = (total_mean - diff_mean) / 2
    Clamps to [0.05, 8.0] to avoid extreme/unphysical values.
    """
    lh = float((total_mean + diff_mean) * 0.5)
    la = float((total_mean - diff_mean) * 0.5)
    lh = max(0.05, min(8.0, lh))
    la = max(0.05, min(8.0, la))
    return lh, la


def simulate_from_period_lambdas(
    home_periods: List[float],
    away_periods: List[float],
    total_line: Optional[float] = None,
    puck_line: float = -1.5,
    cfg: SimConfig = SimConfig(),
) -> Dict[str, float]:
    """Monte Carlo simulate outcomes from per-period expected goals for each team.

    Args:
        home_periods: [P1, P2, P3] expected home goals
        away_periods: [P1, P2, P3] expected away goals
        total_line: Optional totals line (e.g., 6.0) for over/under
        puck_line: Spread for home side (typically -1.5)
        cfg: Simulation config

    Returns:
        Probabilities dict including home_ml, away_ml, over, under,
        and puckline cover for home -1.5 / away +1.5.
    """
    g = _rng(cfg.random_state)
    n = int(cfg.n_sims)

    # Sample goals per period as independent Poisson
    hp = np.array(home_periods, dtype=np.float64)
    ap = np.array(away_periods, dtype=np.float64)
    hp = np.maximum(hp, 0.0)
    ap = np.maximum(ap, 0.0)

    # Draw Poisson samples for each period, shape (n, 3)
    h_samples = np.column_stack([g.poisson(lam=max(0.0, float(hp[i])), size=n) for i in range(3)])
    a_samples = np.column_stack([g.poisson(lam=max(0.0, float(ap[i])), size=n) for i in range(3)])

    home_goals = h_samples.sum(axis=1)
    away_goals = a_samples.sum(axis=1)
    diff = home_goals - away_goals
    total = home_goals + away_goals

    # Moneyline: assign half of draws to each side (OT/SO resolution proxy)
    wins_h = (diff > 0).sum()
    wins_a = (diff < 0).sum()
    draws = (diff == 0).sum()
    p_home_ml = (wins_h + 0.5 * draws) / n
    p_away_ml = (wins_a + 0.5 * draws) / n

    # Totals
    if total_line is not None:
        tl = float(total_line)
        over = (total > tl).sum() / n
        under = (total < tl).sum() / n
    else:
        over = np.nan
        under = np.nan

    # Puck line: home -1.5 cover probability
    # For a half-line, strict inequality is correct
    ph_pl = (diff > abs(puck_line)).sum() / n
    pa_pl = 1.0 - ph_pl

    return {
        "home_ml": float(p_home_ml),
        "away_ml": float(p_away_ml),
        "over": float(over),
        "under": float(under),
        "home_puckline_-1.5": float(ph_pl),
        "away_puckline_+1.5": float(pa_pl),
    }


def simulate_from_totals_diff(
    total_mean: float,
    diff_mean: float,
    total_line: Optional[float] = None,
    puck_line: float = -1.5,
    cfg: SimConfig = SimConfig(),
) -> Dict[str, float]:
    """Monte Carlo simulate outcomes from overall total and differential means.

    Uses independent team-level Poisson with lambdas derived from total/diff.
    """
    lh, la = derive_team_lambdas(total_mean, diff_mean)
    g = _rng(cfg.random_state)
    n = int(cfg.n_sims)

    home_goals = g.poisson(lam=lh, size=n)
    away_goals = g.poisson(lam=la, size=n)
    diff = home_goals - away_goals
    total = home_goals + away_goals

    wins_h = (diff > 0).sum()
    wins_a = (diff < 0).sum()
    draws = (diff == 0).sum()
    p_home_ml = (wins_h + 0.5 * draws) / n
    p_away_ml = (wins_a + 0.5 * draws) / n

    if total_line is not None:
        tl = float(total_line)
        over = (total > tl).sum() / n
        under = (total < tl).sum() / n
    else:
        over = np.nan
        under = np.nan

    ph_pl = (diff > abs(puck_line)).sum() / n
    pa_pl = 1.0 - ph_pl

    return {
        "home_ml": float(p_home_ml),
        "away_ml": float(p_away_ml),
        "over": float(over),
        "under": float(under),
        "home_puckline_-1.5": float(ph_pl),
        "away_puckline_+1.5": float(pa_pl),
    }

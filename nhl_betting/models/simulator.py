from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SimConfig:
    n_sims: int = 20000
    max_goals_period: int = 10  # cap for safety when using matrix methods
    random_state: Optional[int] = 42
    # Overdispersion control (Gamma-Poisson mixture). When set, simulates
    # Negative Binomial-like goals by sampling team-specific lambda from Gamma(k, scale=mean/k).
    # Larger k -> lower dispersion; k=None or k<=0 -> standard Poisson.
    overdispersion_k: Optional[float] = None
    # Shared pace correlation: when set (>0), apply a game-level Gamma(shared_k, scale=1/shared_k)
    # multiplier (mean=1) to both team lambdas to induce positive correlation in scoring.
    shared_k: Optional[float] = None
    # Empty-net behavior: probability that a team leading by exactly 1 goal scores an additional
    # late empty-net goal, converting a 1-goal win into 2+. Applied post-simulation.
    empty_net_p: Optional[float] = None
    # Optional scaling to allow a smaller empty-net probability when leading by 2.
    # If set (>0), apply empty_net_p * empty_net_two_goal_scale when diff==2.
    empty_net_two_goal_scale: Optional[float] = None


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

    # Draw goals per period; optionally apply shared pace (correlation) and overdispersion
    # Shared multiplier per simulation (mean=1) induces positive correlation in team scoring.
    if cfg.shared_k is not None and float(cfg.shared_k) > 0.0:
        shared = g.gamma(shape=float(cfg.shared_k), scale=1.0 / float(cfg.shared_k), size=n)
    else:
        shared = np.ones(n, dtype=np.float64)

    if cfg.overdispersion_k is not None and float(cfg.overdispersion_k) > 0.0:
        k = float(cfg.overdispersion_k)
        h_samples = np.empty((n, 3), dtype=np.int64)
        a_samples = np.empty((n, 3), dtype=np.int64)
        for i in range(3):
            lam_h = max(0.0, float(hp[i])) * shared
            lam_a = max(0.0, float(ap[i])) * shared
            # Vectorized Gamma-Poisson mixture; zeros where lambda <= 0
            gamma_h = g.gamma(shape=k, scale=np.clip(lam_h, 0.0, None) / k, size=n)
            h_samples[:, i] = g.poisson(lam=np.clip(gamma_h, 0.0, None))
            gamma_a = g.gamma(shape=k, scale=np.clip(lam_a, 0.0, None) / k, size=n)
            a_samples[:, i] = g.poisson(lam=np.clip(gamma_a, 0.0, None))
    else:
        # Standard Poisson per period with optional shared multiplier
        h_samples = np.column_stack([
            g.poisson(lam=np.clip(max(0.0, float(hp[i])) * shared, 0.0, None), size=n) for i in range(3)
        ])
        a_samples = np.column_stack([
            g.poisson(lam=np.clip(max(0.0, float(ap[i])) * shared, 0.0, None), size=n) for i in range(3)
        ])

    home_goals = h_samples.sum(axis=1)
    away_goals = a_samples.sum(axis=1)
    # Optional empty-net adjustment: if leading by exactly 1, add 1 goal with probability empty_net_p
    try:
        if cfg.empty_net_p is not None and float(cfg.empty_net_p) > 0.0:
            p_en = float(cfg.empty_net_p)
            # Home leads by 1
            mask_h = (home_goals - away_goals) == 1
            if np.any(mask_h):
                add_h = g.binomial(n=1, p=p_en, size=int(mask_h.sum()))
                idx_h = np.where(mask_h)[0]
                home_goals[idx_h] += add_h
            # Away leads by 1
            mask_a = (away_goals - home_goals) == 1
            if np.any(mask_a):
                add_a = g.binomial(n=1, p=p_en, size=int(mask_a.sum()))
                idx_a = np.where(mask_a)[0]
                away_goals[idx_a] += add_a
            # Optional: chance of a second empty-net goal when leading by 2
            if cfg.empty_net_two_goal_scale is not None and float(cfg.empty_net_two_goal_scale) > 0.0:
                p_en2 = float(p_en * float(cfg.empty_net_two_goal_scale))
                mask_h2 = (home_goals - away_goals) == 2
                if np.any(mask_h2) and p_en2 > 0.0:
                    add_h2 = g.binomial(n=1, p=p_en2, size=int(mask_h2.sum()))
                    idx_h2 = np.where(mask_h2)[0]
                    home_goals[idx_h2] += add_h2
                mask_a2 = (away_goals - home_goals) == 2
                if np.any(mask_a2) and p_en2 > 0.0:
                    add_a2 = g.binomial(n=1, p=p_en2, size=int(mask_a2.sum()))
                    idx_a2 = np.where(mask_a2)[0]
                    away_goals[idx_a2] += add_a2
    except Exception:
        pass
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

    # Optional shared multiplier (correlated pace)
    if cfg.shared_k is not None and float(cfg.shared_k) > 0.0:
        shared = g.gamma(shape=float(cfg.shared_k), scale=1.0 / float(cfg.shared_k), size=n)
    else:
        shared = np.ones(n, dtype=np.float64)

    # Optionally apply overdispersion via Gamma-Poisson mixture
    if cfg.overdispersion_k is not None and float(cfg.overdispersion_k) > 0.0:
        k = float(cfg.overdispersion_k)
        if lh > 0:
            gamma_h = g.gamma(shape=k, scale=(lh * shared) / k, size=n)
            home_goals = g.poisson(lam=np.clip(gamma_h, 0.0, None))
        else:
            home_goals = np.zeros(n, dtype=np.int64)
        if la > 0:
            gamma_a = g.gamma(shape=k, scale=(la * shared) / k, size=n)
            away_goals = g.poisson(lam=np.clip(gamma_a, 0.0, None))
        else:
            away_goals = np.zeros(n, dtype=np.int64)
    else:
        home_goals = g.poisson(lam=np.clip(lh * shared, 0.0, None), size=n)
        away_goals = g.poisson(lam=np.clip(la * shared, 0.0, None), size=n)

    # Optional empty-net adjustment
    try:
        if cfg.empty_net_p is not None and float(cfg.empty_net_p) > 0.0:
            p_en = float(cfg.empty_net_p)
            mask_h = (home_goals - away_goals) == 1
            if np.any(mask_h):
                add_h = g.binomial(n=1, p=p_en, size=int(mask_h.sum()))
                idx_h = np.where(mask_h)[0]
                home_goals[idx_h] += add_h
            mask_a = (away_goals - home_goals) == 1
            if np.any(mask_a):
                add_a = g.binomial(n=1, p=p_en, size=int(mask_a.sum()))
                idx_a = np.where(mask_a)[0]
                away_goals[idx_a] += add_a
            # Optional second empty-net chance when leading by 2
            if cfg.empty_net_two_goal_scale is not None and float(cfg.empty_net_two_goal_scale) > 0.0:
                p_en2 = float(p_en * float(cfg.empty_net_two_goal_scale))
                mask_h2 = (home_goals - away_goals) == 2
                if np.any(mask_h2) and p_en2 > 0.0:
                    add_h2 = g.binomial(n=1, p=p_en2, size=int(mask_h2.sum()))
                    idx_h2 = np.where(mask_h2)[0]
                    home_goals[idx_h2] += add_h2
                mask_a2 = (away_goals - home_goals) == 2
                if np.any(mask_a2) and p_en2 > 0.0:
                    add_a2 = g.binomial(n=1, p=p_en2, size=int(mask_a2.sum()))
                    idx_a2 = np.where(mask_a2)[0]
                    away_goals[idx_a2] += add_a2
    except Exception:
        pass
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math
import numpy as np


def _dc_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon–Coles low-score correction factor.

    See: Dixon & Coles (1997). Applies a dependence correction for low scores.
    """
    # Clamp rho to a sensible range to avoid negative probabilities
    r = max(-0.2, min(0.2, float(rho)))
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * r
    if x == 0 and y == 1:
        return 1.0 + lam_h * r
    if x == 1 and y == 0:
        return 1.0 + lam_a * r
    if x == 1 and y == 1:
        return 1.0 - r
    return 1.0


@dataclass
class DCConfig:
    rho: float = -0.05  # small negative correlation tends to fit hockey/soccer better
    max_goals: int = 10


def dc_score_matrix(lam_h: float, lam_a: float, rho: float = -0.05, max_goals: int = 10) -> np.ndarray:
    """Compute Dixon–Coles-adjusted joint score matrix for home/away goals.

    Returns an (max_goals+1) x (max_goals+1) matrix of probabilities that sums to ~1.
    """
    lam_h = float(lam_h); lam_a = float(lam_a)
    if lam_h < 0 or lam_a < 0 or not (math.isfinite(lam_h) and math.isfinite(lam_a)):
        raise ValueError("Invalid lambdas for DC model")
    i = np.arange(0, max_goals + 1)
    j = np.arange(0, max_goals + 1)
    base_h = np.exp(-lam_h) * np.power(lam_h, i) / np.vectorize(math.factorial)(i)
    base_a = np.exp(-lam_a) * np.power(lam_a, j) / np.vectorize(math.factorial)(j)
    mat = np.outer(base_h, base_a)
    # Apply DC tau adjustments only for low scores
    tau = np.ones_like(mat)
    for x in (0, 1):
        for y in (0, 1):
            if x <= max_goals and y <= max_goals:
                tau[x, y] = _dc_tau(x, y, lam_h, lam_a, rho)
    mat = mat * tau
    # Renormalize to ensure it sums to 1
    s = float(mat.sum())
    if s <= 0 or not math.isfinite(s):
        raise ValueError("Degenerate DC matrix")
    mat /= s
    return mat


def dc_market_probs(lam_h: float, lam_a: float, total_line: float = 6.0, puckline: float = -1.5, rho: float = -0.05, max_goals: int = 10) -> Dict[str, float]:
    """Compute ML, Totals, and Puckline probabilities from DC-adjusted score matrix.
    """
    mat = dc_score_matrix(lam_h, lam_a, rho=rho, max_goals=max_goals)
    # Moneyline (include half of draw as OT/SO win for both sides)
    p_home = np.tril(mat, -1).sum()
    p_away = np.triu(mat, 1).sum()
    p_draw = float(np.trace(mat))
    # Totals via convolution of marginals
    totals = np.arange(0, 2 * max_goals + 1)
    conv = np.convolve(mat.sum(axis=1), mat.sum(axis=0))
    over = conv[totals > total_line].sum()
    under = conv[totals < total_line].sum()
    # Puckline (home -1.5)
    ph_pl = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h - a > 1.5:
                ph_pl += float(mat[h, a])
    pa_pl = 1.0 - ph_pl
    return {
        "home_ml": float(p_home + 0.5 * p_draw),
        "away_ml": float(p_away + 0.5 * p_draw),
        "over": float(over),
        "under": float(under),
        "home_puckline_-1.5": float(ph_pl),
        "away_puckline_+1.5": float(pa_pl),
    }

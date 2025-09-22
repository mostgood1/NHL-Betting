from __future__ import annotations

from typing import Tuple


def american_to_decimal(american: float) -> float:
    if american > 0:
        return 1.0 + american / 100.0
    else:
        return 1.0 + 100.0 / abs(american)


def decimal_to_implied_prob(decimal_odds: float) -> float:
    return 1.0 / decimal_odds


def remove_vig_two_way(p1: float, p2: float) -> Tuple[float, float]:
    """
    Given implied probabilities (with vig) for a two-way market, normalize to sum to 1.
    """
    s = p1 + p2
    if s <= 0:
        return p1, p2
    return p1 / s, p2 / s


def ev_unit(prob: float, decimal_odds: float) -> float:
    """Expected value per 1 unit stake for a single-outcome bet."""
    return prob * (decimal_odds - 1.0) - (1.0 - prob)


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """
    Kelly fraction for a single outcome given true win probability and decimal odds.
    b = decimal_odds - 1
    f* = (b*p - (1-p)) / b
    Clamp at 0 if negative.
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    f = (b * prob - (1.0 - prob)) / b
    return max(0.0, f)


def kelly_stake(prob: float, decimal_odds: float, bankroll: float, fraction: float = 1.0) -> float:
    """
    Stake size using Kelly criterion: bankroll * (fraction * kelly_fraction)
    """
    f = kelly_fraction(prob, decimal_odds)
    return max(0.0, bankroll * fraction * f)


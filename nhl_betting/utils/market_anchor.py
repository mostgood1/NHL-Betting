from __future__ import annotations

from typing import Optional, Tuple
import math

from .odds import american_to_decimal, decimal_to_implied_prob, remove_vig_two_way


def _clean_price(v) -> Optional[float]:
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            f = float(v)
            return f if math.isfinite(f) else None
        s = str(v).strip().replace(",", "")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def implied_pair_from_american(home_odds, away_odds) -> Optional[Tuple[float, float]]:
    """Return no-vig implied probabilities (home, away) from American home/away lines, if both present."""
    h = _clean_price(home_odds); a = _clean_price(away_odds)
    if h is None or a is None:
        return None
    dec_h = american_to_decimal(h); dec_a = american_to_decimal(a)
    if dec_h is None or dec_a is None:
        return None
    imp_h = decimal_to_implied_prob(dec_h); imp_a = decimal_to_implied_prob(dec_a)
    nv_h, nv_a = remove_vig_two_way(imp_h, imp_a)
    return nv_h, nv_a


def implied_pair_from_two_sided(odds_a, odds_b) -> Optional[Tuple[float, float]]:
    """Generic two-sided market (e.g., Over/Under). Returns no-vig (sideA, sideB)."""
    return implied_pair_from_american(odds_a, odds_b)


def blend_probability(p_model: float, p_market: Optional[float], w_market: float = 0.25) -> float:
    """Blend a model probability toward the no-vig market probability.

    p = (1-w)*model + w*market, with safe clamps.
    """
    try:
        pm = float(p_model)
        if not (0 <= pm <= 1) or not math.isfinite(pm):
            return pm
    except Exception:
        return p_model
    try:
        wm = float(w_market)
    except Exception:
        wm = 0.25
    wm = max(0.0, min(1.0, wm))
    if p_market is None or not (0.0 <= p_market <= 1.0) or not math.isfinite(p_market):
        return pm
    return (1.0 - wm) * pm + wm * float(p_market)

"""Period-specific betting recommendations and analysis.

This module provides betting recommendations for:
- Period 1/2/3 totals (over/under)
- First 10 minutes goal probability
- Period-by-period winner predictions
- Comparative period strength analysis
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from scipy.stats import poisson


@dataclass
class PeriodBet:
    """Represents a period-specific betting opportunity."""
    date: str
    home: str
    away: str
    period: str  # "P1", "P2", "P3", "FIRST_10MIN"
    market: str  # "TOTAL", "WINNER", "FIRST_GOAL"
    line: float
    projection: float
    probability: float
    book_odds: Optional[float] = None
    ev: Optional[float] = None
    confidence: str = "MEDIUM"  # LOW, MEDIUM, HIGH


def period_total_probability(
    home_lambda: float,
    away_lambda: float,
    line: float,
    side: str = "over"
) -> float:
    """Calculate probability of period total going over/under line.
    
    Args:
        home_lambda: Expected goals for home team in period
        away_lambda: Expected goals for away team in period
        line: Total goals line (e.g., 1.5)
        side: "over" or "under"
    
    Returns:
        Probability (0-1)
    """
    total_lambda = home_lambda + away_lambda
    threshold = int(np.floor(line + 1e-9))
    
    if side == "over":
        return float(poisson.sf(threshold, mu=total_lambda))
    else:
        return float(poisson.cdf(threshold, mu=total_lambda))


def first_10min_goal_probability(
    p1_home_lambda: float,
    p1_away_lambda: float,
    scale_factor: float = 0.5  # ~50% of P1 goals happen in first 10 min
) -> float:
    """Estimate probability of at least one goal in first 10 minutes.
    
    Assumes first 10 min represents ~50% of period 1's expected goals.
    """
    # Scale down period 1 lambdas to first 10 min
    lambda_10min = (p1_home_lambda + p1_away_lambda) * scale_factor
    
    # P(at least 1 goal) = 1 - P(0 goals)
    prob_zero = poisson.pmf(0, mu=lambda_10min)
    return float(1 - prob_zero)


def period_winner_probability(
    home_lambda: float,
    away_lambda: float
) -> tuple[float, float, float]:
    """Calculate probabilities for period winner.
    
    Returns:
        (p_home_win, p_away_win, p_tie)
    """
    max_goals = 10
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob_matrix[h, a] = (
                poisson.pmf(h, mu=home_lambda) * 
                poisson.pmf(a, mu=away_lambda)
            )
    
    # Home wins: h > a
    p_home = np.sum(np.tril(prob_matrix, k=-1))
    
    # Away wins: a > h
    p_away = np.sum(np.triu(prob_matrix, k=1))
    
    # Tie: h == a
    p_tie = np.sum(np.diag(prob_matrix))
    
    return float(p_home), float(p_away), float(p_tie)


def analyze_period_bets(
    predictions_df: pd.DataFrame,
    period_lines: Optional[Dict[str, float]] = None,
    min_ev: float = 0.05,
    min_prob: float = 0.55
) -> List[PeriodBet]:
    """Analyze period-specific betting opportunities from predictions.
    
    Args:
        predictions_df: DataFrame with period projections
        period_lines: Dict of market lines (default uses common lines)
        min_ev: Minimum expected value to recommend
        min_prob: Minimum probability to recommend
    
    Returns:
        List of PeriodBet recommendations
    """
    if period_lines is None:
        period_lines = {
            "P1_TOTAL": 1.5,  # Common period 1 total line
            "P2_TOTAL": 1.5,
            "P3_TOTAL": 1.5,
            "FIRST_10MIN": 0.5,  # At least 1 goal in first 10
        }
    
    bets = []
    
    for _, game in predictions_df.iterrows():
        date = game.get("date_et", game.get("date", ""))
        home = game.get("home", "")
        away = game.get("away", "")
        
        # Period 1 totals
        p1_home = game.get("period1_home_proj", 0)
        p1_away = game.get("period1_away_proj", 0)
        
        if p1_home and p1_away:
            p1_total = p1_home + p1_away
            p1_over = period_total_probability(p1_home, p1_away, period_lines["P1_TOTAL"], "over")
            p1_under = 1 - p1_over
            
            # Period 1 over
            if p1_over >= min_prob:
                bets.append(PeriodBet(
                    date=date,
                    home=home,
                    away=away,
                    period="P1",
                    market="OVER",
                    line=period_lines["P1_TOTAL"],
                    projection=p1_total,
                    probability=p1_over,
                    confidence="HIGH" if p1_over >= 0.65 else "MEDIUM"
                ))
            
            # Period 1 under
            if p1_under >= min_prob:
                bets.append(PeriodBet(
                    date=date,
                    home=home,
                    away=away,
                    period="P1",
                    market="UNDER",
                    line=period_lines["P1_TOTAL"],
                    projection=p1_total,
                    probability=p1_under,
                    confidence="HIGH" if p1_under >= 0.65 else "MEDIUM"
                ))
            
            # First 10 minutes
            first_10_prob = first_10min_goal_probability(p1_home, p1_away)
            if first_10_prob >= min_prob:
                bets.append(PeriodBet(
                    date=date,
                    home=home,
                    away=away,
                    period="FIRST_10MIN",
                    market="YES_GOAL",
                    line=0.5,
                    projection=first_10_prob,
                    probability=first_10_prob,
                    confidence="HIGH" if first_10_prob >= 0.70 else "MEDIUM"
                ))
            
            # Period 1 winner
            p1_home_win, p1_away_win, p1_tie = period_winner_probability(p1_home, p1_away)
            if p1_home_win >= min_prob:
                bets.append(PeriodBet(
                    date=date,
                    home=home,
                    away=away,
                    period="P1",
                    market="HOME_WINNER",
                    line=0,
                    projection=p1_home,
                    probability=p1_home_win,
                    confidence="HIGH" if p1_home_win >= 0.60 else "MEDIUM"
                ))
            if p1_away_win >= min_prob:
                bets.append(PeriodBet(
                    date=date,
                    home=home,
                    away=away,
                    period="P1",
                    market="AWAY_WINNER",
                    line=0,
                    projection=p1_away,
                    probability=p1_away_win,
                    confidence="HIGH" if p1_away_win >= 0.60 else "MEDIUM"
                ))
        
        # Period 2 totals
        p2_home = game.get("period2_home_proj", 0)
        p2_away = game.get("period2_away_proj", 0)
        
        if p2_home and p2_away:
            p2_total = p2_home + p2_away
            p2_over = period_total_probability(p2_home, p2_away, period_lines["P2_TOTAL"], "over")
            p2_under = 1 - p2_over
            
            if p2_over >= min_prob:
                bets.append(PeriodBet(
                    date=date,
                    home=home,
                    away=away,
                    period="P2",
                    market="OVER",
                    line=period_lines["P2_TOTAL"],
                    projection=p2_total,
                    probability=p2_over,
                    confidence="HIGH" if p2_over >= 0.65 else "MEDIUM"
                ))
            
            if p2_under >= min_prob:
                bets.append(PeriodBet(
                    date=date,
                    home=home,
                    away=away,
                    period="P2",
                    market="UNDER",
                    line=period_lines["P2_TOTAL"],
                    projection=p2_total,
                    probability=p2_under,
                    confidence="HIGH" if p2_under >= 0.65 else "MEDIUM"
                ))
        
        # Period 3 totals
        p3_home = game.get("period3_home_proj", 0)
        p3_away = game.get("period3_away_proj", 0)
        
        if p3_home and p3_away:
            p3_total = p3_home + p3_away
            p3_over = period_total_probability(p3_home, p3_away, period_lines["P3_TOTAL"], "over")
            p3_under = 1 - p3_over
            
            if p3_over >= min_prob:
                bets.append(PeriodBet(
                    date=date,
                    home=home,
                    away=away,
                    period="P3",
                    market="OVER",
                    line=period_lines["P3_TOTAL"],
                    projection=p3_total,
                    probability=p3_over,
                    confidence="HIGH" if p3_over >= 0.65 else "MEDIUM"
                ))
            
            if p3_under >= min_prob:
                bets.append(PeriodBet(
                    date=date,
                    home=home,
                    away=away,
                    period="P3",
                    market="UNDER",
                    line=period_lines["P3_TOTAL"],
                    projection=p3_total,
                    probability=p3_under,
                    confidence="HIGH" if p3_under >= 0.65 else "MEDIUM"
                ))
    
    # Sort by probability descending
    bets.sort(key=lambda x: x.probability, reverse=True)
    
    return bets


def format_period_bets_table(bets: List[PeriodBet]) -> pd.DataFrame:
    """Format period bets as DataFrame for display."""
    if not bets:
        return pd.DataFrame()
    
    records = []
    for bet in bets:
        records.append({
            "date": bet.date,
            "matchup": f"{bet.away} @ {bet.home}",
            "period": bet.period,
            "market": bet.market,
            "line": bet.line,
            "projection": round(bet.projection, 2),
            "probability": round(bet.probability * 100, 1),
            "confidence": bet.confidence,
        })
    
    return pd.DataFrame(records)

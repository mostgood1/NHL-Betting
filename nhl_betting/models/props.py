from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson


@dataclass
class PropsConfig:
    window: int = 10


class SkaterShotsModel:
    def __init__(self, cfg: PropsConfig | None = None):
        self.cfg = cfg or PropsConfig()

    def player_lambda(self, df: pd.DataFrame, player: str, team: Optional[str] = None) -> float:
        pdf = df[(df["player"] == player) & (df["role"] == "skater")].dropna(subset=["shots"]).copy()
        pdf = pdf.sort_values("date").tail(self.cfg.window)
        if pdf.empty:
            return 2.0
        return float(pdf["shots"].mean())

    def prob_over(self, lam: float, line: float, max_x: int = 15) -> float:
        # Use stable survival function to avoid factorial overflow
        threshold = int(np.floor(line + 1e-9))
        return float(poisson.sf(threshold, mu=lam))


class GoalieSavesModel:
    def __init__(self, cfg: PropsConfig | None = None):
        self.cfg = cfg or PropsConfig()

    def player_lambda(self, df: pd.DataFrame, player: str) -> float:
        pdf = df[(df["player"] == player) & (df["role"] == "goalie")].dropna(subset=["saves"]).copy()
        pdf = pdf.sort_values("date").tail(self.cfg.window)
        if pdf.empty:
            return 25.0
        return float(pdf["saves"].mean())

    def prob_over(self, lam: float, line: float, max_x: int = 60) -> float:
        threshold = int(np.floor(line + 1e-9))
        return float(poisson.sf(threshold, mu=lam))


class SkaterGoalsModel:
    def __init__(self, cfg: PropsConfig | None = None):
        self.cfg = cfg or PropsConfig()

    def player_lambda(self, df: pd.DataFrame, player: str) -> float:
        pdf = df[(df["player"] == player) & (df["role"] == "skater")].dropna(subset=["goals"]).copy()
        pdf = pdf.sort_values("date").tail(self.cfg.window)
        if pdf.empty:
            return 0.3
        return float(pdf["goals"].mean())

    def prob_over(self, lam: float, line: float, max_x: int = 5) -> float:
        threshold = int(np.floor(line + 1e-9))
        return float(poisson.sf(threshold, mu=lam))


class SkaterAssistsModel:
    def __init__(self, cfg: PropsConfig | None = None):
        self.cfg = cfg or PropsConfig()

    def player_lambda(self, df: pd.DataFrame, player: str) -> float:
        pdf = df[(df["player"] == player) & (df["role"] == "skater")].dropna(subset=["assists"]).copy()
        pdf = pdf.sort_values("date").tail(self.cfg.window)
        if pdf.empty:
            return 0.4
        return float(pdf["assists"].mean())

    def prob_over(self, lam: float, line: float, max_x: int = 5) -> float:
        threshold = int(np.floor(line + 1e-9))
        return float(poisson.sf(threshold, mu=lam))


class SkaterPointsModel:
    def __init__(self, cfg: PropsConfig | None = None):
        self.cfg = cfg or PropsConfig()

    def player_lambda(self, df: pd.DataFrame, player: str) -> float:
        # Points = goals + assists
        pdf = df[(df["player"] == player) & (df["role"] == "skater")].dropna(subset=["goals", "assists"]).copy()
        pdf = pdf.sort_values("date").tail(self.cfg.window)
        if pdf.empty:
            return 0.7
        pts = (pdf["goals"].astype(float) + pdf["assists"].astype(float))
        return float(pts.mean())

    def prob_over(self, lam: float, line: float, max_x: int = 8) -> float:
        threshold = int(np.floor(line + 1e-9))
        return float(poisson.sf(threshold, mu=lam))


class SkaterBlocksModel:
    def __init__(self, cfg: PropsConfig | None = None):
        self.cfg = cfg or PropsConfig()

    def player_lambda(self, df: pd.DataFrame, player: str) -> float:
        pdf = df[(df["player"] == player) & (df["role"] == "skater")].dropna(subset=["blocked"]).copy()
        pdf = pdf.sort_values("date").tail(self.cfg.window)
        if pdf.empty:
            return 1.5
        return float(pdf["blocked"].mean())

    def prob_over(self, lam: float, line: float, max_x: int = 15) -> float:
        threshold = int(np.floor(line + 1e-9))
        return float(poisson.sf(threshold, mu=lam))

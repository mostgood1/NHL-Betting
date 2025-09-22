from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


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
        # Use Poisson CDF complement at half-lines
        threshold = int(np.floor(line + 1e-9))
        xs = np.arange(0, max_x + 1)
        pmf = np.exp(-lam) * np.power(lam, xs) / np.vectorize(np.math.factorial)(xs)
        return float(pmf[xs > threshold].sum())


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
        xs = np.arange(0, max_x + 1)
        pmf = np.exp(-lam) * np.power(lam, xs) / np.vectorize(np.math.factorial)(xs)
        return float(pmf[xs > threshold].sum())


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
        xs = np.arange(0, max_x + 1)
        pmf = np.exp(-lam) * np.power(lam, xs) / np.vectorize(np.math.factorial)(xs)
        return float(pmf[xs > threshold].sum())

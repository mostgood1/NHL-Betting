from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Set

import numpy as np
import pandas as pd
from scipy.stats import poisson


@dataclass
class PropsConfig:
    window: int = 10
def _normalize_name(s: str) -> str:
    import unicodedata, re
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _name_variants(full_name: str) -> Set[str]:
    """Generate reasonable variants for matching player names in historical data.

    Examples:
    - "Tyler Bertuzzi" -> {"Tyler Bertuzzi", "T Bertuzzi", "T. Bertuzzi"}
    - Handles extra spaces and diacritics.
    """
    full = _normalize_name(full_name)
    parts = full.split(" ")
    out: Set[str] = {full}
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        if first:
            ini = first[0]
            out.add(f"{ini} {last}")
            out.add(f"{ini}. {last}")
    return { _normalize_name(x) for x in out }


def _select_player_rows(df: pd.DataFrame, player: str, role: str, metric_cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "player", "role", *metric_cols])
    try:
        candidates = _name_variants(player)
        pdf = df[df["role"] == role].copy()
        # Normalize player names in the view for matching
        pdf["_p_norm"] = pdf["player"].astype(str).map(_normalize_name)
        pdf = pdf[pdf["_p_norm"].isin(candidates)]
        # Clean to expected columns
        keep = [c for c in ["date", "player", "role", *metric_cols] if c in pdf.columns]
        pdf = pdf[keep] if keep else pdf
        return pdf
    except Exception:
        return pd.DataFrame(columns=["date", "player", "role", *metric_cols])



class SkaterShotsModel:
    def __init__(self, cfg: PropsConfig | None = None):
        self.cfg = cfg or PropsConfig()

    def player_lambda(self, df: pd.DataFrame, player: str, team: Optional[str] = None) -> float:
        pdf = _select_player_rows(df, player, role="skater", metric_cols=["shots"]).dropna(subset=["shots"]).copy()
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
        pdf = _select_player_rows(df, player, role="goalie", metric_cols=["saves"]).dropna(subset=["saves"]).copy()
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
        pdf = _select_player_rows(df, player, role="skater", metric_cols=["goals"]).dropna(subset=["goals"]).copy()
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
        pdf = _select_player_rows(df, player, role="skater", metric_cols=["assists"]).dropna(subset=["assists"]).copy()
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
        pdf = _select_player_rows(df, player, role="skater", metric_cols=["goals","assists"]).dropna(subset=["goals", "assists"]).copy()
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
        pdf = _select_player_rows(df, player, role="skater", metric_cols=["blocked"]).dropna(subset=["blocked"]).copy()
        pdf = pdf.sort_values("date").tail(self.cfg.window)
        if pdf.empty:
            return 1.5
        return float(pdf["blocked"].mean())

    def prob_over(self, lam: float, line: float, max_x: int = 15) -> float:
        threshold = int(np.floor(line + 1e-9))
        return float(poisson.sf(threshold, mu=lam))

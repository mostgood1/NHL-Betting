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


def _unwrap_dictish_name(val: str) -> str:
    """If the name is serialized like "{'default': 'N. Schmaltz'}", extract the default/name.

    Returns the original string if parsing fails.
    """
    try:
        s = str(val or "").strip()
        if s.startswith("{") and s.endswith("}"):
            import ast
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                v = d.get("default") or d.get("name") or d.get("fullName")
                if isinstance(v, str) and v.strip():
                    return v.strip()
    except Exception:
        pass
    return str(val or "")


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
    """Return per-player rows for a given role with guaranteed metric columns present.

    This helper is defensive: if expected metric columns are missing from the source
    data, it will create them as NaN so downstream dropna/sort logic is safe.
    """
    expected = ["date", "player", "role", *metric_cols]
    if df is None or df.empty:
        # Ensure the returned frame has the expected schema
        return pd.DataFrame(columns=expected)
    try:
        candidates = _name_variants(player)
        pdf = df[df.get("role", "").astype(str).str.lower() == str(role).lower()].copy()
        # Normalize player names in the view for matching
        if "player" in pdf.columns:
            # Unwrap dict-like serialized names, then normalize
            pdf["_p_norm"] = pdf["player"].astype(str).map(_unwrap_dictish_name).map(_normalize_name)
            pdf = pdf[pdf["_p_norm"].isin(candidates)]
        else:
            # No player column; return empty schema
            return pd.DataFrame(columns=expected)
        # If 'date' is missing but 'date_key' exists (common in calibration), copy it
        if "date" not in pdf.columns and "date_key" in pdf.columns:
            pdf["date"] = pdf["date_key"]
        # Materialize any missing metric columns as NaN
        for col in metric_cols:
            if col not in pdf.columns:
                pdf[col] = np.nan
        # Build final view with the expected schema order (missing added above)
        keep = [c for c in expected if c in pdf.columns]
        if len(keep) < len(expected):
            # If any core columns like date/player/role were still missing, add them
            for core in ["date", "player", "role"]:
                if core not in pdf.columns:
                    pdf[core] = "" if core != "date" else None
            keep = expected
        return pdf[keep]
    except Exception:
        return pd.DataFrame(columns=expected)



class SkaterShotsModel:
    def __init__(self, cfg: PropsConfig | None = None):
        self.cfg = cfg or PropsConfig()

    def player_lambda(self, df: pd.DataFrame, player: str, team: Optional[str] = None) -> float:
        pdf = _select_player_rows(df, player, role="skater", metric_cols=["shots"]).copy()
        # Coerce metric and drop missing
        pdf["shots"] = pd.to_numeric(pdf.get("shots"), errors="coerce")
        pdf = pdf.dropna(subset=["shots"]).copy()
        # History (df) is already chronological per player; avoid re-sorting for speed
        pdf = pdf.tail(self.cfg.window)
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
        pdf = _select_player_rows(df, player, role="goalie", metric_cols=["saves"]).copy()
        pdf["saves"] = pd.to_numeric(pdf.get("saves"), errors="coerce")
        pdf = pdf.dropna(subset=["saves"]).copy()
        pdf = pdf.tail(self.cfg.window)
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
        pdf = _select_player_rows(df, player, role="skater", metric_cols=["goals"]).copy()
        pdf["goals"] = pd.to_numeric(pdf.get("goals"), errors="coerce")
        pdf = pdf.dropna(subset=["goals"]).copy()
        pdf = pdf.tail(self.cfg.window)
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
        pdf = _select_player_rows(df, player, role="skater", metric_cols=["assists"]).copy()
        pdf["assists"] = pd.to_numeric(pdf.get("assists"), errors="coerce")
        pdf = pdf.dropna(subset=["assists"]).copy()
        pdf = pdf.tail(self.cfg.window)
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
        pdf = _select_player_rows(df, player, role="skater", metric_cols=["goals","assists"]).copy()
        pdf["goals"] = pd.to_numeric(pdf.get("goals"), errors="coerce")
        pdf["assists"] = pd.to_numeric(pdf.get("assists"), errors="coerce")
        pdf = pdf.dropna(subset=["goals", "assists"]).copy()
        pdf = pdf.tail(self.cfg.window)
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
        pdf = _select_player_rows(df, player, role="skater", metric_cols=["blocked"]).copy()
        pdf["blocked"] = pd.to_numeric(pdf.get("blocked"), errors="coerce")
        pdf = pdf.dropna(subset=["blocked"]).copy()
        pdf = pdf.tail(self.cfg.window)
        if pdf.empty:
            return 1.5
        return float(pdf["blocked"].mean())

    def prob_over(self, lam: float, line: float, max_x: int = 15) -> float:
        threshold = int(np.floor(line + 1e-9))
        return float(poisson.sf(threshold, mu=lam))

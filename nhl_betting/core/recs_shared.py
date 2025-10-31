from __future__ import annotations

import math
from pathlib import Path
import os
from typing import Optional, List, Dict

import pandas as pd

from ..utils.io import PROC_DIR
from ..data.nhl_api_web import NHLWebClient


# --------- helpers ---------

def _num(v):
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            fv = float(v)
            return fv if math.isfinite(fv) else None
        s = str(v).strip().replace(",", "")
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def _american_to_decimal(american: float | int | None) -> float | None:
    if american is None:
        return None
    try:
        a = float(american)
        return 1.0 + (a / 100.0) if a > 0 else 1.0 + (100.0 / abs(a))
    except Exception:
        return None


def _price_with_fallback(row: pd.Series, market_key: str, odds_key: str) -> float | None:
    price_val = _num(row.get(odds_key)) if odds_key in row else None
    if price_val is None:
        close_map = {
            "home_ml_odds": "close_home_ml_odds",
            "away_ml_odds": "close_away_ml_odds",
            "over_odds": "close_over_odds",
            "under_odds": "close_under_odds",
            "home_pl_-1.5_odds": "close_home_pl_-1.5_odds",
            "away_pl_+1.5_odds": "close_away_pl_+1.5_odds",
        }
        ck = close_map.get(odds_key)
        if ck and ck in row:
            price_val = _num(row.get(ck))
    # Market-specific default American odds when price is missing
    if price_val is None:
        if market_key == "first10":
            price_val = -150.0  # updated default for First-10 market
        elif market_key in ("totals", "puckline", "periods"):
            price_val = -110.0
    return price_val


def _ensure_ev(row: pd.Series, prob_key: str, odds_key: str, ev_key: str, market_key: str = "") -> pd.Series:
    try:
        ev_present = (ev_key in row) and pd.notna(row.get(ev_key))
        if ev_present:
            return row
        p = None
        if prob_key in row and pd.notna(row.get(prob_key)):
            pv = float(row.get(prob_key))
            if 0.0 <= pv <= 1.0 and math.isfinite(pv):
                p = pv
        price = _price_with_fallback(row, market_key or "", odds_key)
        if (p is not None) and (price is not None):
            dec = _american_to_decimal(price)
            if dec is not None and math.isfinite(dec):
                row[ev_key] = round(float(p) * (dec - 1.0) - (1.0 - float(p)), 4)
    except Exception:
        return row
    return row


# --------- public API ---------

def recompute_edges_and_recommendations(date_str: str, min_ev: float = 0.0) -> List[Dict]:
    """Compute EVs, write edges_{date}.csv and recommendations_{date}.csv, return recs list."""
    pred_path = PROC_DIR / f"predictions_{date_str}.csv"
    if not pred_path.exists():
        return []
    df = pd.read_csv(pred_path)
    if df is None or df.empty:
        return []
    # First-10 probs if only lambda present
    if "p_f10_yes" not in df.columns:
        df["p_f10_yes"] = pd.NA
    if "p_f10_no" not in df.columns:
        df["p_f10_no"] = pd.NA
    # Team-level First-10 scoring/allowing (derived) placeholders
    if "p_f10_home_scores" not in df.columns:
        df["p_f10_home_scores"] = pd.NA
    if "p_f10_away_scores" not in df.columns:
        df["p_f10_away_scores"] = pd.NA
    if "p_f10_home_allows" not in df.columns:
        df["p_f10_home_allows"] = pd.NA
    if "p_f10_away_allows" not in df.columns:
        df["p_f10_away_allows"] = pd.NA
    try:
        # Optional early-window factor: convert period-1 goal rate to first-10 rate (default 10/20)
        # Precedence: env var > first10_eval.json (p1_scale) > model_calibration.json > default 0.55
        def _clamp(v: float, lo: float = 0.35, hi: float = 0.7) -> float:
            try:
                return max(lo, min(hi, float(v)))
            except Exception:
                return v
        _F10_EARLY_FACTOR = None
        # 1) Environment override
        try:
            _env_val = os.getenv("F10_EARLY_FACTOR")
            if _env_val is not None:
                _F10_EARLY_FACTOR = float(_env_val)
        except Exception:
            _F10_EARLY_FACTOR = None
        # 2) Data-driven evaluation file
        if _F10_EARLY_FACTOR is None or (not math.isfinite(_F10_EARLY_FACTOR)) or _F10_EARLY_FACTOR <= 0:
            try:
                import json as _json
                _eval_path = PROC_DIR / "first10_eval.json"
                if _eval_path.exists():
                    _obj = _json.loads(_eval_path.read_text(encoding="utf-8"))
                    _p1_scale = _obj.get("p1_scale")
                    if _p1_scale is not None:
                        _F10_EARLY_FACTOR = float(_p1_scale)
            except Exception:
                _F10_EARLY_FACTOR = None
        # 3) Model calibration file fallback
        if _F10_EARLY_FACTOR is None or (not math.isfinite(_F10_EARLY_FACTOR)) or _F10_EARLY_FACTOR <= 0:
            try:
                import json as _json
                _cal_path = PROC_DIR / "model_calibration.json"
                if _cal_path.exists():
                    _obj = _json.loads(_cal_path.read_text(encoding="utf-8"))
                    _f_cal = _obj.get("f10_early_factor")
                    if _f_cal is not None:
                        _F10_EARLY_FACTOR = float(_f_cal)
            except Exception:
                _F10_EARLY_FACTOR = None
        # 4) Sensible default
        if _F10_EARLY_FACTOR is None or (not math.isfinite(_F10_EARLY_FACTOR)) or _F10_EARLY_FACTOR <= 0:
            _F10_EARLY_FACTOR = 0.55
        # Clamp to avoid extreme/unrealistic probabilities from overfitting
        _F10_EARLY_FACTOR = _clamp(_F10_EARLY_FACTOR)
    # Optional team-rate blend configuration
        _F10_BLEND_ALPHA = None  # weight for P1-based estimate vs team-based
        _F10_BLEND_ENABLE = str(os.getenv("FIRST10_BLEND", "0")).lower() in ("1","true","yes")
        # Try calibration file for blend alpha
        if _F10_BLEND_ENABLE:
            try:
                import json as _json
                _cal_path = PROC_DIR / "model_calibration.json"
                if _cal_path.exists():
                    _obj = _json.loads(_cal_path.read_text(encoding="utf-8"))
                    _ba = _obj.get("f10_blend_alpha")
                    if _ba is not None:
                        _F10_BLEND_ALPHA = float(_ba)
            except Exception:
                _F10_BLEND_ALPHA = None
            # Env override
            try:
                _ba_env = os.getenv("F10_BLEND_ALPHA")
                if _ba_env is not None:
                    _F10_BLEND_ALPHA = float(_ba_env)
            except Exception:
                pass
            if _F10_BLEND_ALPHA is None or (not math.isfinite(_F10_BLEND_ALPHA)):
                _F10_BLEND_ALPHA = 0.7
            _F10_BLEND_ALPHA = max(0.0, min(1.0, _F10_BLEND_ALPHA))
            # Load team rates
            try:
                import json as _json
                _tr_path = PROC_DIR / "first10_team_rates.json"
                _TEAM_RATES = _json.loads(_tr_path.read_text(encoding="utf-8")) if _tr_path.exists() else {}
            except Exception:
                _TEAM_RATES = {}
            # Load team scoring/allowing splits (optional)
            try:
                import json as _json
                _spl_path = PROC_DIR / "first10_team_splits.json"
                _TEAM_SPLITS = _json.loads(_spl_path.read_text(encoding="utf-8")) if _spl_path.exists() else {}
            except Exception:
                _TEAM_SPLITS = {}
            # Compute league baseline from team rates if available
            try:
                _TEAM_LEAGUE_P = float(
                    sum(float((v or {}).get("yes_rate", 0.0)) for v in (_TEAM_RATES or {}).values())
                ) / max(1.0, float(len(_TEAM_RATES))) if _TEAM_RATES else 0.62
            except Exception:
                _TEAM_LEAGUE_P = 0.62
            # Estimate slate-average P1 sum to pace-adjust team prior
            try:
                s1 = pd.to_numeric(df.get("period1_home_proj"), errors="coerce")
                s2 = pd.to_numeric(df.get("period1_away_proj"), errors="coerce")
                sums = (s1.fillna(0.0) + s2.fillna(0.0))
                sums = sums[sums > 0]
                _P1_MEAN_SUM = float(sums.mean()) if len(sums) > 0 else 2.0
            except Exception:
                _P1_MEAN_SUM = 2.0
            # Pace exponent (how strongly pace widens/narrows team prior)
            try:
                _PACE_EXP = float(os.getenv("F10_PACE_EXP", "0.6"))
            except Exception:
                _PACE_EXP = 0.6
        else:
            _TEAM_RATES = {}
            _TEAM_SPLITS = {}
            _TEAM_LEAGUE_P = 0.62
            _P1_MEAN_SUM = 2.0
            _PACE_EXP = 0.6
        # Logit helpers for smoother blending (prevents linear squashing around 0.5)
        def _sigmoid(x: float) -> float:
            try:
                # Guard against overflow
                if x > 20:
                    return 1.0
                if x < -20:
                    return 0.0
                return 1.0 / (1.0 + math.exp(-x))
            except Exception:
                return 0.5
        def _logit(p: float, eps: float = 1e-6) -> float:
            try:
                pp = min(1.0 - eps, max(eps, float(p)))
                return math.log(pp / (1.0 - pp))
            except Exception:
                return 0.0
        for i, r in df.iterrows():
            p_yes = None
            # Team-driven estimate from period1 projections (preferred)
            h1 = float(r.get("period1_home_proj")) if pd.notna(r.get("period1_home_proj")) else None
            a1 = float(r.get("period1_away_proj")) if pd.notna(r.get("period1_away_proj")) else None
            lam_h10 = lam_a10 = None
            if h1 is not None and a1 is not None and (h1 >= 0 and a1 >= 0):
                try:
                    lam_h10 = _F10_EARLY_FACTOR * float(h1)
                    lam_a10 = _F10_EARLY_FACTOR * float(a1)
                    lam10 = lam_h10 + lam_a10
                    if math.isfinite(lam10) and lam10 >= 0:
                        p_p1 = 1.0 - math.exp(-lam10)
                        p_yes = p_p1
                except Exception:
                    lam_h10 = lam_a10 = None
            # Fallback to existing fields if team-driven unavailable
            if p_yes is None:
                if pd.notna(r.get("first_10min_prob")):
                    p_yes = float(r.get("first_10min_prob"))
                elif pd.notna(r.get("first_10min_proj")):
                    lam10 = float(r.get("first_10min_proj"))
                    if math.isfinite(lam10) and lam10 >= 0:
                        p_yes = 1.0 - math.exp(-lam10)
            # Optional blend with team rates
            if _F10_BLEND_ENABLE:
                try:
                    team_h = str(r.get("home") or ""); team_a = str(r.get("away") or "")
                    # Prefer scoring/allowing splits if available
                    hs = (_TEAM_SPLITS.get(team_h) or {}).get("scores_rate") if _TEAM_SPLITS else None
                    ha = (_TEAM_SPLITS.get(team_h) or {}).get("allows_rate") if _TEAM_SPLITS else None
                    as_ = (_TEAM_SPLITS.get(team_a) or {}).get("scores_rate") if _TEAM_SPLITS else None
                    aa = (_TEAM_SPLITS.get(team_a) or {}).get("allows_rate") if _TEAM_SPLITS else None
                    def _clip01(x):
                        try:
                            return max(0.0, min(1.0, float(x)))
                        except Exception:
                            return None
                    hs = _clip01(hs); ha = _clip01(ha); as_ = _clip01(as_); aa = _clip01(aa)
                    if all(v is not None for v in (hs, ha, as_, aa)):
                        # Combine team score vs opponent allow: 1 - (1 - s)*(1 - a)
                        p_home_score = 1.0 - (1.0 - hs) * (1.0 - aa)
                        p_away_score = 1.0 - (1.0 - as_) * (1.0 - ha)
                        # Pace adjustment toward league baseline
                        pace_ratio = None
                        try:
                            if (h1 is not None) and (a1 is not None) and (_P1_MEAN_SUM and _P1_MEAN_SUM > 0):
                                pace_ratio = (float(h1) + float(a1)) / float(_P1_MEAN_SUM)
                        except Exception:
                            pace_ratio = None
                        if pace_ratio is not None and math.isfinite(pace_ratio):
                            pace_ratio = max(0.7, min(1.3, pace_ratio))
                            p_home_score = _TEAM_LEAGUE_P + (p_home_score - _TEAM_LEAGUE_P) * (pace_ratio ** _PACE_EXP)
                            p_away_score = _TEAM_LEAGUE_P + (p_away_score - _TEAM_LEAGUE_P) * (pace_ratio ** _PACE_EXP)
                            p_home_score = max(0.0, min(1.0, p_home_score))
                            p_away_score = max(0.0, min(1.0, p_away_score))
                        # Combine to overall yes
                        p_team = 1.0 - (1.0 - p_home_score) * (1.0 - p_away_score)
                        # Blend with P1 probability if available (logit blend)
                        if p_yes is not None:
                            p_lin = float(p_yes)
                            p_yes = _sigmoid(_F10_BLEND_ALPHA * _logit(p_lin) + (1.0 - _F10_BLEND_ALPHA) * _logit(p_team))
                        else:
                            p_yes = p_team
                        # Expose component scores to CSV
                        try:
                            df.at[i, "p_f10_home_scores"] = p_home_score
                            df.at[i, "p_f10_away_scores"] = p_away_score
                            df.at[i, "p_f10_home_allows"] = 1.0 - (1.0 - ha)  # store raw allow proxy as rate
                            df.at[i, "p_f10_away_allows"] = 1.0 - (1.0 - aa)
                        except Exception:
                            pass
                    else:
                        # Fallback to yes_rate average if splits missing
                        th = float((_TEAM_RATES.get(team_h) or {}).get("yes_rate", 0.0)) if _TEAM_RATES else None
                        ta = float((_TEAM_RATES.get(team_a) or {}).get("yes_rate", 0.0)) if _TEAM_RATES else None
                        if th is not None and ta is not None and (0.0 <= th <= 1.0) and (0.0 <= ta <= 1.0):
                            p_team = max(0.0, min(1.0, 0.5 * (th + ta)))
                            pace_ratio = None
                            try:
                                if (h1 is not None) and (a1 is not None) and (_P1_MEAN_SUM and _P1_MEAN_SUM > 0):
                                    pace_ratio = (float(h1) + float(a1)) / float(_P1_MEAN_SUM)
                            except Exception:
                                pace_ratio = None
                            if pace_ratio is not None and math.isfinite(pace_ratio):
                                pace_ratio = max(0.7, min(1.3, pace_ratio))
                                p_team = _TEAM_LEAGUE_P + (p_team - _TEAM_LEAGUE_P) * (pace_ratio ** _PACE_EXP)
                                p_team = max(0.0, min(1.0, p_team))
                            if p_yes is not None:
                                p_lin = float(p_yes)
                                p_yes = _sigmoid(_F10_BLEND_ALPHA * _logit(p_lin) + (1.0 - _F10_BLEND_ALPHA) * _logit(p_team))
                            else:
                                p_yes = p_team
                except Exception:
                    pass
            if p_yes is not None:
                df.at[i, "p_f10_yes"] = max(0.0, min(1.0, float(p_yes)))
                df.at[i, "p_f10_no"] = 1.0 - float(df.at[i, "p_f10_yes"]) if pd.notna(df.at[i, "p_f10_yes"]) else pd.NA
            # Compute team-level first-10 score/allow from team-driven lambdas if available
            if lam_h10 is not None and lam_a10 is not None:
                try:
                    df.at[i, "p_f10_home_scores"] = max(0.0, min(1.0, 1.0 - math.exp(-lam_h10)))
                    df.at[i, "p_f10_away_scores"] = max(0.0, min(1.0, 1.0 - math.exp(-lam_a10)))
                    df.at[i, "p_f10_home_allows"] = df.at[i, "p_f10_away_scores"]
                    df.at[i, "p_f10_away_allows"] = df.at[i, "p_f10_home_scores"]
                except Exception:
                    pass
            # As a final consistency step, if component score rates are present, set p_f10_yes to their combined probability
            try:
                ph = float(df.at[i, "p_f10_home_scores"]) if pd.notna(df.at[i, "p_f10_home_scores"]) else None
                pa = float(df.at[i, "p_f10_away_scores"]) if pd.notna(df.at[i, "p_f10_away_scores"]) else None
                if ph is not None and pa is not None:
                    pt = 1.0 - (1.0 - ph) * (1.0 - pa)
                    df.at[i, "p_f10_yes"] = max(0.0, min(1.0, pt))
                    df.at[i, "p_f10_no"] = 1.0 - df.at[i, "p_f10_yes"]
            except Exception:
                pass
    except Exception:
        pass
    # Ensure EVs
    for i, r in df.iterrows():
        def compute_and_assign(prob_key, odds_key, ev_key, market_key: str):
            rr = _ensure_ev(r.copy(), prob_key, odds_key, ev_key, market_key)
            val = rr.get(ev_key)
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                df.at[i, ev_key] = val
        compute_and_assign("p_home_ml", "home_ml_odds", "ev_home_ml", "moneyline")
        compute_and_assign("p_away_ml", "away_ml_odds", "ev_away_ml", "moneyline")
        compute_and_assign("p_over", "over_odds", "ev_over", "totals")
        compute_and_assign("p_under", "under_odds", "ev_under", "totals")
        compute_and_assign("p_home_pl_-1.5", "home_pl_-1.5_odds", "ev_home_pl_-1.5", "puckline")
        compute_and_assign("p_away_pl_+1.5", "away_pl_+1.5_odds", "ev_away_pl_+1.5", "puckline")
        compute_and_assign("p_f10_yes", "f10_yes_odds", "ev_f10_yes", "first10")
        compute_and_assign("p_f10_no", "f10_no_odds", "ev_f10_no", "first10")
        for pn in (1, 2, 3):
            compute_and_assign(f"p{pn}_over_prob", f"p{pn}_over_odds", f"ev_p{pn}_over", "periods")
            compute_and_assign(f"p{pn}_under_prob", f"p{pn}_under_odds", f"ev_p{pn}_under", "periods")
    # Persist EVs
    df.to_csv(pred_path, index=False)
    # Edges
    ev_cols = [c for c in df.columns if c.startswith("ev_")]
    if ev_cols:
        edges = df.melt(id_vars=["date", "home", "away"], value_vars=ev_cols, var_name="market", value_name="ev").dropna()
        edges = edges.sort_values("ev", ascending=False)
        edges_path = PROC_DIR / f"edges_{date_str}.csv"
        edges.to_csv(edges_path, index=False)
    # Recommendations (never against model; tie-break EV)
    recs: list[dict] = []
    def _ev(row: pd.Series, market: str, prob_key: str, odds_key: str, ev_key: str) -> float | None:
        if ev_key in row and pd.notna(row.get(ev_key)):
            try:
                val = float(row.get(ev_key))
                return val if math.isfinite(val) else None
            except Exception:
                pass
        p = None
        if prob_key in row and pd.notna(row.get(prob_key)):
            try:
                p = float(row.get(prob_key))
            except Exception:
                p = None
        price = _price_with_fallback(row, market, odds_key)
        if (p is None) or (price is None):
            return None
        dec = _american_to_decimal(price)
        if dec is None:
            return None
        return p * (dec - 1.0) - (1.0 - p)
    def _add(row, market, bet, prob_key, ev_key, odds_key, book_key=None):
        evv = _ev(row, market, prob_key, odds_key, ev_key)
        if evv is None or evv < float(min_ev):
            return
        price = _price_with_fallback(row, market, odds_key)
        try:
            prob = float(row.get(prob_key)) if pd.notna(row.get(prob_key)) else None
        except Exception:
            prob = None
        recs.append({
            "date": row.get("date"),
            "home": row.get("home"),
            "away": row.get("away"),
            "market": market,
            "bet": bet,
            "prob": prob,
            "price": price,
            "book": (row.get(book_key) if book_key else None),
            "ev": round(float(evv), 4) if evv is not None else None,
        })
    for _, r in df.iterrows():
        # Moneyline
        ph = _num(r.get("p_home_ml")); pa = _num(r.get("p_away_ml"))
        if (ph is not None) and (pa is not None):
            if ph > pa:
                _add(r, "moneyline", "home_ml", "p_home_ml", "ev_home_ml", "home_ml_odds", "home_ml_book")
            elif pa > ph:
                _add(r, "moneyline", "away_ml", "p_away_ml", "ev_away_ml", "away_ml_odds", "away_ml_book")
            else:
                evh = _ev(r, "moneyline", "p_home_ml", "home_ml_odds", "ev_home_ml")
                eva = _ev(r, "moneyline", "p_away_ml", "away_ml_odds", "ev_away_ml")
                if (evh is not None) and (eva is not None):
                    if evh >= eva:
                        _add(r, "moneyline", "home_ml", "p_home_ml", "ev_home_ml", "home_ml_odds", "home_ml_book")
                    else:
                        _add(r, "moneyline", "away_ml", "p_away_ml", "ev_away_ml", "away_ml_odds", "away_ml_book")
        # Totals
        po = _num(r.get("p_over")); pu = _num(r.get("p_under"))
        if (po is not None) and (pu is not None):
            if po > pu:
                _add(r, "totals", "over", "p_over", "ev_over", "over_odds", "over_book")
            elif pu > po:
                _add(r, "totals", "under", "p_under", "ev_under", "under_odds", "under_book")
            else:
                evo = _ev(r, "totals", "p_over", "over_odds", "ev_over")
                evu = _ev(r, "totals", "p_under", "under_odds", "ev_under")
                if (evo is not None) and (evu is not None):
                    if evo >= evu:
                        _add(r, "totals", "over", "p_over", "ev_over", "over_odds", "over_book")
                    else:
                        _add(r, "totals", "under", "p_under", "ev_under", "under_odds", "under_book")
        # Puckline
        php = _num(r.get("p_home_pl_-1.5")); pap = _num(r.get("p_away_pl_+1.5"))
        if (php is not None) and (pap is not None):
            if php > pap:
                _add(r, "puckline", "home_pl_-1.5", "p_home_pl_-1.5", "ev_home_pl_-1.5", "home_pl_-1.5_odds", "home_pl_-1.5_book")
            elif pap > php:
                _add(r, "puckline", "away_pl_+1.5", "p_away_pl_+1.5", "ev_away_pl_+1.5", "away_pl_+1.5_odds", "away_pl_+1.5_book")
            else:
                evhpl = _ev(r, "puckline", "p_home_pl_-1.5", "home_pl_-1.5_odds", "ev_home_pl_-1.5")
                evapl = _ev(r, "puckline", "p_away_pl_+1.5", "away_pl_+1.5_odds", "ev_away_pl_+1.5")
                if (evhpl is not None) and (evapl is not None):
                    if evhpl >= evapl:
                        _add(r, "puckline", "home_pl_-1.5", "p_home_pl_-1.5", "ev_home_pl_-1.5", "home_pl_-1.5_odds", "home_pl_-1.5_book")
                    else:
                        _add(r, "puckline", "away_pl_+1.5", "p_away_pl_+1.5", "ev_away_pl_+1.5", "away_pl_+1.5_odds", "away_pl_+1.5_book")
        # First-10
        py = _num(r.get("p_f10_yes")); pn = _num(r.get("p_f10_no"))
        if (py is not None) and (pn is not None):
            if py > pn:
                _add(r, "first10", "f10_yes", "p_f10_yes", "ev_f10_yes", "f10_yes_odds")
            elif pn > py:
                _add(r, "first10", "f10_no", "p_f10_no", "ev_f10_no", "f10_no_odds")
            else:
                evy = _ev(r, "first10", "p_f10_yes", "f10_yes_odds", "ev_f10_yes")
                evn = _ev(r, "first10", "p_f10_no", "f10_no_odds", "ev_f10_no")
                if (evy is not None) and (evn is not None):
                    if evy >= evn:
                        _add(r, "first10", "f10_yes", "p_f10_yes", "ev_f10_yes", "f10_yes_odds")
                    else:
                        _add(r, "first10", "f10_no", "p_f10_no", "ev_f10_no", "f10_no_odds")
        # Periods
        for n in (1, 2, 3):
            pov = _num(r.get(f"p{n}_over_prob")); pun = _num(r.get(f"p{n}_under_prob"))
            if (pov is not None) and (pun is not None):
                if pov > pun:
                    _add(r, "periods", f"p{n}_over", f"p{n}_over_prob", f"ev_p{n}_over", f"p{n}_over_odds")
                elif pun > pov:
                    _add(r, "periods", f"p{n}_under", f"p{n}_under_prob", f"ev_p{n}_under", f"p{n}_under_odds")
                else:
                    evo = _ev(r, "periods", f"p{n}_over_prob", f"p{n}_over_odds", f"ev_p{n}_over")
                    evu = _ev(r, "periods", f"p{n}_under_prob", f"p{n}_under_odds", f"ev_p{n}_under")
                    if (evo is not None) and (evu is not None):
                        if evo >= evu:
                            _add(r, "periods", f"p{n}_over", f"p{n}_over_prob", f"ev_p{n}_over", f"p{n}_over_odds")
                        else:
                            _add(r, "periods", f"p{n}_under", f"p{n}_under_prob", f"ev_p{n}_under", f"p{n}_under_odds")
    # Persist recommendations (robust to empty list)
    outp = PROC_DIR / f"recommendations_{date_str}.csv"
    try:
        if recs:
            df_recs = pd.DataFrame(recs).sort_values("ev", ascending=False)
        else:
            # Write empty file with stable schema so downstream readers don't break
            df_recs = pd.DataFrame(columns=[
                "date","home","away","market","bet","prob","price","book","ev"
            ])
        df_recs.to_csv(outp, index=False)
    except Exception:
        # Best-effort: still attempt to write minimal CSV if something odd in sort/types
        try:
            pd.DataFrame(recs).to_csv(outp, index=False)
        except Exception:
            pass
    return recs


def backfill_settlement_for_date(date_str: str, *, force: bool = False) -> dict:
    pred_csv_path = PROC_DIR / f"predictions_{date_str}.csv"
    if not pred_csv_path.exists():
        return {"skipped": True, "reason": "no_predictions"}
    try:
        df = pd.read_csv(pred_csv_path)
    except Exception:
        return {"skipped": True, "reason": "read_error"}
    if df.empty:
        return {"skipped": True, "reason": "empty_predictions"}
    client = NHLWebClient()
    try:
        sb = client.scoreboard_day(date_str)
    except Exception:
        sb = []
    def _abbr(x: str) -> str:
        try:
            from ..web.teams import get_team_assets
            return (get_team_assets(str(x)).get("abbr") or "").upper()
        except Exception:
            return ""
    sb_idx = {}
    for g in (sb or []):
        try:
            hk = _abbr(g.get("home")); ak = _abbr(g.get("away"))
            if hk and ak:
                sb_idx[(hk, ak)] = g
        except Exception:
            continue
    pbp_cache: dict[int, dict] = {}
    backfilled = 0
    # Ensure new columns exist
    if "result_first10_home" not in df.columns:
        df["result_first10_home"] = pd.NA
    if "result_first10_away" not in df.columns:
        df["result_first10_away"] = pd.NA
    for i, r in df.iterrows():
        try:
            hk = _abbr(r.get("home")); ak = _abbr(r.get("away"))
            g = sb_idx.get((hk, ak))
            if not g:
                continue
            fh = int(g.get("home_goals")) if g.get("home_goals") is not None else None
            fa = int(g.get("away_goals")) if g.get("away_goals") is not None else None
            if fh is None or fa is None:
                continue
            actual_total = fh + fa
            df.at[i, "final_home_goals"] = fh
            df.at[i, "final_away_goals"] = fa
            df.at[i, "actual_home_goals"] = fh
            df.at[i, "actual_away_goals"] = fa
            df.at[i, "actual_total"] = actual_total
            if pd.isna(r.get("winner_actual")) or not r.get("winner_actual"):
                df.at[i, "winner_actual"] = r.get("home") if fh > fa else (r.get("away") if fa > fh else "Draw")
            tl = _num(r.get("close_total_line_used")) or _num(r.get("total_line_used")) or _num(r.get("pl_line_used"))
            if (pd.isna(r.get("result_total")) or not r.get("result_total")) and (tl is not None):
                if actual_total > tl:
                    df.at[i, "result_total"] = "Over"
                elif actual_total < tl:
                    df.at[i, "result_total"] = "Under"
                else:
                    df.at[i, "result_total"] = "Push"
            if pd.isna(r.get("result_ats")) or not r.get("result_ats"):
                diff = fh - fa
                df.at[i, "result_ats"] = "home_-1.5" if diff > 1.5 else "away_+1.5"
            game_pk = g.get("gamePk")
            if game_pk is not None:
                if game_pk not in pbp_cache:
                    try:
                        pbp_cache[game_pk] = client.play_by_play(int(game_pk))
                    except Exception:
                        pbp_cache[game_pk] = {}
                pbp = pbp_cache.get(game_pk) or {}
                plays = pbp.get("plays") if isinstance(pbp, dict) else None
                goals_by_period = {1: 0, 2: 0, 3: 0}
                first10_yes = False
                home_scored10 = False
                away_scored10 = False
                if isinstance(plays, list):
                    for p_ in plays:
                        try:
                            tkey = (str(p_.get("typeDescKey") or p_.get("type")) or "").lower()
                            # Only treat actual scoring events as goals; do NOT match substrings like 'shot-on-goal'
                            if tkey != "goal":
                                continue
                            per = None
                            try:
                                pdsc = p_.get("periodDescriptor") or {}
                                per = int(pdsc.get("number") or pdsc.get("period") or p_.get("period") or 0)
                            except Exception:
                                per = int(p_.get("period") or 0)
                            if per in goals_by_period:
                                goals_by_period[per] += 1
                            tr = p_.get("timeRemaining") or p_.get("timeInPeriod") or p_.get("time")
                            mm = ss = None
                            if isinstance(tr, str) and ":" in tr:
                                parts = tr.split(":")
                                try:
                                    mm = int(parts[0]); ss = int(parts[1])
                                except Exception:
                                    mm, ss = None, None
                            if per == 1 and (mm is not None and ss is not None):
                                secs = mm * 60 + ss
                                # Use field semantics when available:
                                # - timeRemaining: first 10 means remaining >= 600
                                # - timeInPeriod: first 10 means elapsed <= 600
                                # - time (unknown): assume elapsed clock
                                try:
                                    if p_.get("timeRemaining") is not None:
                                        if secs >= 600:
                                            first10_yes = True
                                            # attribute team if available
                                            try:
                                                # Pre-compute expected abbreviations
                                                home_abbr = _abbr(r.get("home")); away_abbr = _abbr(r.get("away"))
                                                tri = ""
                                                tm = p_.get("team") or {}
                                                tri = (tm.get("triCode") or tm.get("abbrev") or tm.get("abbreviation") or "").upper()
                                                if not tri:
                                                    tri = str(p_.get("teamAbbrev") or p_.get("teamTriCode") or "").upper()
                                                if not tri:
                                                    det = p_.get("details") or {}
                                                    tri = str(det.get("eventOwnerAbbrev") or det.get("teamTriCode") or det.get("clubCode") or "").upper()
                                                side = None
                                                if not tri:
                                                    det = p_.get("details") or {}
                                                    side = (det.get("eventOwner") or det.get("owner") or p_.get("club") or "").lower()
                                                if tri and home_abbr and tri == home_abbr:
                                                    home_scored10 = True
                                                elif tri and away_abbr and tri == away_abbr:
                                                    away_scored10 = True
                                                elif side in ("home", "h"):
                                                    home_scored10 = True
                                                elif side in ("away", "a"):
                                                    away_scored10 = True
                                            except Exception:
                                                pass
                                    elif p_.get("timeInPeriod") is not None:
                                        if secs <= 600:
                                            first10_yes = True
                                            try:
                                                # Pre-compute expected abbreviations
                                                home_abbr = _abbr(r.get("home")); away_abbr = _abbr(r.get("away"))
                                                tri = ""
                                                tm = p_.get("team") or {}
                                                tri = (tm.get("triCode") or tm.get("abbrev") or tm.get("abbreviation") or "").upper()
                                                if not tri:
                                                    tri = str(p_.get("teamAbbrev") or p_.get("teamTriCode") or "").upper()
                                                if not tri:
                                                    det = p_.get("details") or {}
                                                    tri = str(det.get("eventOwnerAbbrev") or det.get("teamTriCode") or det.get("clubCode") or "").upper()
                                                side = None
                                                if not tri:
                                                    det = p_.get("details") or {}
                                                    side = (det.get("eventOwner") or det.get("owner") or p_.get("club") or "").lower()
                                                if tri and home_abbr and tri == home_abbr:
                                                    home_scored10 = True
                                                elif tri and away_abbr and tri == away_abbr:
                                                    away_scored10 = True
                                                elif side in ("home", "h"):
                                                    home_scored10 = True
                                                elif side in ("away", "a"):
                                                    away_scored10 = True
                                            except Exception:
                                                pass
                                    else:
                                        # Assume elapsed clock if ambiguous
                                        if secs <= 600:
                                            first10_yes = True
                                            try:
                                                # Pre-compute expected abbreviations
                                                home_abbr = _abbr(r.get("home")); away_abbr = _abbr(r.get("away"))
                                                tri = ""
                                                tm = p_.get("team") or {}
                                                tri = (tm.get("triCode") or tm.get("abbrev") or tm.get("abbreviation") or "").upper()
                                                if not tri:
                                                    tri = str(p_.get("teamAbbrev") or p_.get("teamTriCode") or "").upper()
                                                if not tri:
                                                    det = p_.get("details") or {}
                                                    tri = str(det.get("eventOwnerAbbrev") or det.get("teamTriCode") or det.get("clubCode") or "").upper()
                                                side = None
                                                if not tri:
                                                    det = p_.get("details") or {}
                                                    side = (det.get("eventOwner") or det.get("owner") or p_.get("club") or "").lower()
                                                if tri and home_abbr and tri == home_abbr:
                                                    home_scored10 = True
                                                elif tri and away_abbr and tri == away_abbr:
                                                    away_scored10 = True
                                                elif side in ("home", "h"):
                                                    home_scored10 = True
                                                elif side in ("away", "a"):
                                                    away_scored10 = True
                                            except Exception:
                                                pass
                                except Exception:
                                    pass
                        except Exception:
                            continue
                # Fallback attribution via NHL Stats API when first10 is true but team unknown
                try:
                    if first10_yes and (not home_scored10 and not away_scored10) and (game_pk is not None):
                        import requests as _rq
                        resp = _rq.get(f"https://statsapi.web.nhl.com/api/v1/game/{int(game_pk)}/feed/live", timeout=20)
                        if resp.ok:
                            js = resp.json() or {}
                            plays_node = (js.get('liveData') or {}).get('plays') or {}
                            all_plays = plays_node.get('allPlays') or []
                            scoring_idxs = plays_node.get('scoringPlays') or []
                            home_abbr = _abbr(r.get("home")); away_abbr = _abbr(r.get("away"))
                            for idx in scoring_idxs:
                                try:
                                    ev = all_plays[idx] if isinstance(idx, int) and idx < len(all_plays) else None
                                    if not ev:
                                        continue
                                    per = int((ev.get('about') or {}).get('period') or 0)
                                    clk = str((ev.get('about') or {}).get('periodTime') or '')
                                    mm = ss = None
                                    if isinstance(clk, str) and ':' in clk:
                                        parts = clk.split(':')
                                        mm = int(parts[0]); ss = int(parts[1])
                                    if per == 1 and (mm is not None and ss is not None) and (mm*60+ss) <= 600:
                                        tri = ( (ev.get('team') or {}).get('triCode') or (ev.get('team') or {}).get('abbreviation') or '' ).upper()
                                        if tri and home_abbr and tri == home_abbr:
                                            home_scored10 = True
                                        elif tri and away_abbr and tri == away_abbr:
                                            away_scored10 = True
                                        break
                                except Exception:
                                    continue
                except Exception:
                    pass
                try:
                    if force or pd.isna(r.get("result_first10")) or not r.get("result_first10"):
                        df.at[i, "result_first10"] = "Yes" if first10_yes else "No"
                    # Always update home/away first10 scorer flags when force or missing
                    if force or pd.isna(r.get("result_first10_home")):
                        df.at[i, "result_first10_home"] = "Yes" if home_scored10 else "No"
                    if force or pd.isna(r.get("result_first10_away")):
                        df.at[i, "result_first10_away"] = "Yes" if away_scored10 else "No"
                except Exception:
                    pass
                for pn in (1, 2, 3):
                    try:
                        ln = _num(r.get(f"close_p{pn}_total_line")) or _num(r.get(f"p{pn}_total_line"))
                        if ln is None:
                            continue
                        actual_p = goals_by_period.get(pn)
                        if actual_p is None:
                            continue
                        res_key = f"result_p{pn}_total"
                        if pd.isna(r.get(res_key)) or not r.get(res_key):
                            if abs(ln - round(ln)) < 1e-9 and int(round(ln)) == int(actual_p):
                                df.at[i, res_key] = "Push"
                            elif float(actual_p) > float(ln):
                                df.at[i, res_key] = "Over"
                            else:
                                df.at[i, res_key] = "Under"
                    except Exception:
                        continue
            backfilled += 1
        except Exception:
            continue
    df.to_csv(pred_csv_path, index=False)
    return {"ok": True, "date": date_str, "rows_backfilled": int(backfilled)}


def reconcile_extended(date_str: str, flat_stake: float = 100.0) -> dict:
    path = PROC_DIR / f"predictions_{date_str}.csv"
    if not path.exists():
        return {"status": "no-predictions", "date": date_str}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"status": "read-failed", "date": date_str, "error": str(e)}
    if df.empty:
        return {"status": "empty", "date": date_str}
    picks = []
    def add(r, market, bet, ev_key, price_key, result_field=None):
        ev = r.get(ev_key)
        if ev is None or (isinstance(ev, float) and pd.isna(ev)):
            return
        price = r.get({
            "home_ml_odds": "close_home_ml_odds",
            "away_ml_odds": "close_away_ml_odds",
            "over_odds": "close_over_odds",
            "under_odds": "close_under_odds",
            "home_pl_-1.5_odds": "close_home_pl_-1.5_odds",
            "away_pl_+1.5_odds": "close_away_pl_+1.5_odds",
        }.get(price_key, price_key)) or r.get(price_key)
        picks.append({
            "date": r.get("date"),
            "home": r.get("home"),
            "away": r.get("away"),
            "market": market,
            "bet": bet,
            "ev": float(ev),
            "price": price,
            "result": (r.get(result_field) if result_field else None),
            "winner_actual": r.get("winner_actual"),
            "actual_total": r.get("actual_total"),
            "total_line_used": r.get("close_total_line_used") or r.get("total_line_used"),
            "final_home_goals": r.get("final_home_goals"),
            "final_away_goals": r.get("final_away_goals"),
            "result_first10": r.get("result_first10"),
            "result_p1_total": r.get("result_p1_total"),
            "result_p2_total": r.get("result_p2_total"),
            "result_p3_total": r.get("result_p3_total"),
        })
    for _, r in df.iterrows():
        add(r, "moneyline", "home_ml", "ev_home_ml", "home_ml_odds")
        add(r, "moneyline", "away_ml", "ev_away_ml", "away_ml_odds")
        add(r, "totals", "over", "ev_over", "over_odds", "result_total")
        add(r, "totals", "under", "ev_under", "under_odds", "result_total")
        add(r, "puckline", "home_pl_-1.5", "ev_home_pl_-1.5", "home_pl_-1.5_odds", "result_ats")
        add(r, "puckline", "away_pl_+1.5", "ev_away_pl_+1.5", "away_pl_+1.5_odds", "result_ats")
        add(r, "first10", "f10_yes", "ev_f10_yes", "f10_yes_odds", "result_first10")
        add(r, "first10", "f10_no", "ev_f10_no", "f10_no_odds", "result_first10")
        add(r, "periods", "p1_over", "ev_p1_over", "p1_over_odds", "result_p1_total")
        add(r, "periods", "p1_under", "ev_p1_under", "p1_under_odds", "result_p1_total")
        add(r, "periods", "p2_over", "ev_p2_over", "p2_over_odds", "result_p2_total")
        add(r, "periods", "p2_under", "ev_p2_under", "p2_under_odds", "result_p2_total")
        add(r, "periods", "p3_over", "ev_p3_over", "p3_over_odds", "result_p3_total")
        add(r, "periods", "p3_under", "ev_p3_under", "p3_under_odds", "result_p3_total")
    def to_dec(american):
        return _american_to_decimal(_num(american))
    pnl = 0.0; staked = 0.0; wins = losses = pushes = decided = 0
    rows = []
    for p in picks:
        stake = flat_stake
        dec = to_dec(p.get("price")) if p.get("price") is not None else None
        rl = None
        try:
            mkt = (p.get("market") or "").lower(); bet = (p.get("bet") or "").lower()
            if mkt == "moneyline":
                wa = p.get("winner_actual")
                if isinstance(wa, str):
                    if bet == "home_ml": rl = "win" if wa == p.get("home") else "loss"
                    elif bet == "away_ml": rl = "win" if wa == p.get("away") else "loss"
            elif mkt == "totals":
                at = p.get("actual_total"); tl = p.get("total_line_used")
                if at is not None and tl is not None:
                    atf = float(at); tlf = float(tl)
                    if abs(tlf - round(tlf)) < 1e-9 and int(round(tlf)) == int(atf): rl = "push"
                    elif bet == "over": rl = "win" if atf > tlf else "loss"
                    elif bet == "under": rl = "win" if atf < tlf else "loss"
                if rl is None and isinstance(p.get("result"), str):
                    rlow = p.get("result").lower()
                    if rlow == "push": rl = "push"
                    elif (bet == "over" and rlow == "over") or (bet == "under" and rlow == "under"): rl = "win"
                    else: rl = "loss"
            elif mkt == "puckline":
                fh = _num(p.get("final_home_goals")); fa = _num(p.get("final_away_goals"))
                if fh is not None and fa is not None:
                    diff = fh - fa
                    if bet == "home_pl_-1.5": rl = "win" if diff > 1.5 else "loss"
                    elif bet == "away_pl_+1.5": rl = "win" if diff < -1.5 else "loss"
            elif mkt == "first10":
                rf = (p.get("result_first10") or "").lower()
                if rf in ("yes", "no"):
                    if bet == "f10_yes": rl = "win" if rf == "yes" else "loss"
                    elif bet == "f10_no": rl = "win" if rf == "no" else "loss"
            elif mkt == "periods":
                import re
                m = re.match(r"p([123])_(over|under)", bet)
                if m:
                    pn = m.group(1); side = m.group(2)
                    rf = p.get(f"result_p{pn}_total")
                    if isinstance(rf, str):
                        rlow = rf.lower()
                        if rlow == "push": rl = "push"
                        elif rlow == side: rl = "win"
                        else: rl = "loss"
        except Exception:
            rl = None
        if isinstance(rl, str):
            if rl == "push":
                pushes += 1
                rows.append({**p, "stake": stake, "payout": 0.0})
            elif rl == "win":
                wins += 1
                if dec is not None:
                    pnl += stake * (dec - 1.0)
                staked += stake; decided += 1
                rows.append({**p, "stake": stake, "payout": (stake * (dec - 1.0)) if dec is not None else None})
            elif rl == "loss":
                losses += 1
                pnl -= stake
                staked += stake; decided += 1
                rows.append({**p, "stake": stake, "payout": -stake})
        else:
            rows.append({**p, "stake": stake, "payout": None})
    summary = {"date": date_str, "picks": len(picks), "decided": decided, "wins": wins, "losses": losses, "pushes": pushes, "staked": staked, "pnl": pnl, "roi": (pnl / staked) if staked > 0 else None}
    out = {"summary": summary, "rows": rows}
    # Persist reconciliation JSON
    out_path = PROC_DIR / f"reconciliation_{date_str}.json"
    try:
        import json
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    except Exception:
        pass
    return {"status": "ok", **summary}

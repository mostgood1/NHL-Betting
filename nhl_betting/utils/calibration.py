from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Optional, Dict, Any

import numpy as np


@dataclass
class BinaryCalibration:
    # p' = sigmoid((logit(p) + b) / t)
    t: float = 1.0  # temperature (t>0); t>1 flattens, t<1 sharpens
    b: float = 0.0  # bias (log-odds intercept shift)

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1 - 1e-6)
        logit = np.log(p / (1 - p))
        adj = (logit + self.b) / max(self.t, 1e-6)
        out = 1.0 / (1.0 + np.exp(-adj))
        return np.clip(out, 1e-6, 1 - 1e-6)


def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def fit_temp_shift(prob: Iterable[float], y: Iterable[int],
                   t_grid: Optional[Iterable[float]] = None,
                   b_grid: Optional[Iterable[float]] = None,
                   metric: str = "logloss") -> BinaryCalibration:
    """
    Grid-search a simple calibration: p' = sigmoid((logit(p) + b) / t)
    - prob: iterable of probabilities in (0,1)
    - y: iterable of binary targets (0/1)
    - t_grid: search values for temperature
    - b_grid: search values for bias (log-odds shift)
    - metric: 'logloss' or 'brier'
    Returns BinaryCalibration with best (t,b) minimizing chosen metric.
    """
    p = np.asarray(list(prob), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    m = (~np.isnan(p)) & (~np.isnan(y_arr))
    p = p[m]
    y_arr = y_arr[m]
    if p.size == 0:
        return BinaryCalibration()
    if t_grid is None:
        t_grid = np.arange(0.7, 1.51, 0.05)
    if b_grid is None:
        b_grid = np.array([-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15])
    best = BinaryCalibration()
    best_score = float("inf")
    for t in t_grid:
        for b in b_grid:
            cal = BinaryCalibration(t=float(t), b=float(b))
            p_adj = cal.apply(p)
            score = _logloss(p_adj, y_arr) if metric == "logloss" else _brier(p_adj, y_arr)
            if score < best_score:
                best_score = score
                best = cal
    return best


def save_calibration(path: Path, moneyline: BinaryCalibration, totals: BinaryCalibration, meta: Optional[Dict[str, Any]] = None) -> None:
    obj: Dict[str, Any] = {
        "moneyline": {"t": moneyline.t, "b": moneyline.b},
        "totals": {"t": totals.t, "b": totals.b},
    }
    if meta:
        obj["meta"] = meta
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_calibration(path: Path) -> Tuple[BinaryCalibration, BinaryCalibration]:
    if not path.exists():
        return BinaryCalibration(), BinaryCalibration()
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    ml = obj.get("moneyline", {})
    tt = obj.get("totals", {})
    return BinaryCalibration(float(ml.get("t", 1.0)), float(ml.get("b", 0.0))), BinaryCalibration(float(tt.get("t", 1.0)), float(tt.get("b", 0.0)))


def summarize_binary(y_true: np.ndarray, p_pred: np.ndarray) -> Dict[str, float]:
    m = (~np.isnan(y_true)) & (~np.isnan(p_pred))
    y = y_true[m].astype(float)
    p = np.clip(p_pred[m].astype(float), 1e-6, 1 - 1e-6)
    if y.size == 0:
        return {"n": 0, "logloss": float("nan"), "brier": float("nan"), "acc": float("nan")}
    acc = float(np.mean(((p >= 0.5).astype(int) == y)))
    return {
        "n": int(y.size),
        "logloss": _logloss(p, y),
        "brier": _brier(p, y),
        "acc": acc,
    }


# ==== Props calibration helpers ====
def load_props_stats_calibration_map(path: Path) -> Dict[tuple[str, float], BinaryCalibration]:
    """Load props stats calibration JSON and return a mapping (market,line)->BinaryCalibration.

    If multiple windows exist for a given (market,line), choose the one with lowest post-calibration Brier if available,
    otherwise lowest pre-calibration Brier.
    """
    try:
        import json as _json
        if not path.exists() or path.stat().st_size <= 0:
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = _json.load(f)
        groups = data.get("groups", []) or []
        best: Dict[tuple[str, float], tuple[float, BinaryCalibration]] = {}
        for g in groups:
            try:
                mkt = str(g.get("market") or "").upper()
                line = float(g.get("line"))
                brier = float(g.get("brier")) if g.get("brier") is not None else float("inf")
                cp = g.get("calibration_params") or {}
                t = float(cp.get("t", 1.0))
                b = float(cp.get("b", 0.0))
                brier_post = cp.get("brier_post")
                score = float(brier_post) if brier_post is not None else brier
                cal = BinaryCalibration(t=t, b=b)
                key = (mkt, line)
                prev = best.get(key)
                if (prev is None) or (score < prev[0]):
                    best[key] = (score, cal)
            except Exception:
                continue
        # Flatten to calibration map
        out: Dict[tuple[str, float], BinaryCalibration] = {}
        for k, v in best.items():
            out[k] = v[1]
        return out
    except Exception:
        return {}

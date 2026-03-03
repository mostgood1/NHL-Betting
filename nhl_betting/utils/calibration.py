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


@dataclass
class IsotonicCalibration:
    """Monotone calibration via isotonic regression (piecewise-linear).

    Stored as breakpoints (x, y) where x is raw probability and y is calibrated.
    """

    x: np.ndarray
    y: np.ndarray

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        try:
            if self.x is None or self.y is None:
                return p
            if len(self.x) < 2 or len(self.y) < 2:
                return p
            x = np.asarray(self.x, dtype=float)
            y = np.asarray(self.y, dtype=float)
            # Ensure sorted by x
            o = np.argsort(x)
            x = x[o]
            y = y[o]
            out = np.interp(p, x, y, left=float(y[0]), right=float(y[-1]))
            return np.clip(out, 1e-6, 1 - 1e-6)
        except Exception:
            return p


def fit_isotonic(prob: Iterable[float], y: Iterable[int]) -> IsotonicCalibration:
    """Fit isotonic regression using PAVA on grouped unique probabilities."""
    p = np.asarray(list(prob), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    m = (~np.isnan(p)) & (~np.isnan(y_arr))
    p = p[m]
    y_arr = y_arr[m]
    if p.size < 2:
        return IsotonicCalibration(x=np.asarray([1e-6, 1 - 1e-6], dtype=float), y=np.asarray([0.5, 0.5], dtype=float))

    # Sort by p
    o = np.argsort(p)
    p = p[o]
    y_arr = y_arr[o]

    # Group by identical p to reduce noise and improve stability.
    uniq_p: list[float] = []
    w: list[float] = []
    y_mean: list[float] = []
    i = 0
    n = int(p.size)
    while i < n:
        j = i + 1
        while j < n and float(p[j]) == float(p[i]):
            j += 1
        cnt = float(j - i)
        uniq_p.append(float(p[i]))
        w.append(cnt)
        y_mean.append(float(np.mean(y_arr[i:j])))
        i = j

    x = np.asarray(uniq_p, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    yb = np.asarray(y_mean, dtype=float)

    # PAVA for non-decreasing yb with weights.
    # Maintain blocks with (sum_w, sum_wy) so mean = sum_wy/sum_w.
    sum_w = list(w_arr.tolist())
    sum_wy = list((w_arr * yb).tolist())
    x_end = list(x.tolist())
    k = 0
    while k < len(sum_w) - 1:
        mean_k = sum_wy[k] / max(1e-12, sum_w[k])
        mean_n = sum_wy[k + 1] / max(1e-12, sum_w[k + 1])
        if mean_k <= mean_n + 1e-12:
            k += 1
            continue
        # Merge blocks k and k+1
        sum_w[k] = float(sum_w[k] + sum_w[k + 1])
        sum_wy[k] = float(sum_wy[k] + sum_wy[k + 1])
        x_end[k] = float(x_end[k + 1])
        del sum_w[k + 1]
        del sum_wy[k + 1]
        del x_end[k + 1]
        if k > 0:
            k -= 1

    x_fit = np.asarray(x_end, dtype=float)
    y_fit = np.asarray([sum_wy[i] / max(1e-12, sum_w[i]) for i in range(len(sum_w))], dtype=float)

    # Ensure endpoints for stable interpolation.
    x_fit = np.clip(x_fit, 1e-6, 1 - 1e-6)
    y_fit = np.clip(y_fit, 1e-6, 1 - 1e-6)
    if x_fit.size < 2:
        x_fit = np.asarray([1e-6, 1 - 1e-6], dtype=float)
        y_fit = np.asarray([float(y_fit[0]) if y_fit.size else 0.5, float(y_fit[0]) if y_fit.size else 0.5], dtype=float)
    if float(x_fit[0]) > 1e-6:
        x_fit = np.insert(x_fit, 0, 1e-6)
        y_fit = np.insert(y_fit, 0, float(y_fit[0]))
    if float(x_fit[-1]) < 1 - 1e-6:
        x_fit = np.append(x_fit, 1 - 1e-6)
        y_fit = np.append(y_fit, float(y_fit[-1]))

    return IsotonicCalibration(x=x_fit, y=y_fit)


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

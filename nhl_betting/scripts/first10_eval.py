"""Compute FIRST_10 evaluation metrics and write JSON summary.

Usage (PowerShell):
  .\.venv\Scripts\Activate.ps1; python -m nhl_betting.scripts.first10_eval --start 2023-10-01 --end 2025-10-17 --out data/processed/first10_eval.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from nhl_betting.utils.io import RAW_DIR, PROC_DIR

RAW_PATH = RAW_DIR / "games_with_features.csv"


def prob_yes_from_lambda(lam: float) -> float:
    return 1.0 - np.exp(-max(0.0, float(lam)))


def metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p.astype(float), 1e-9, 1 - 1e-9)
    y = y_true.astype(int)
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    # Calibration deciles
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    cal = []
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            continue
        p_bin = p[idx]
        y_bin = y[idx]
        cal.append({"decile": i, "mean_pred": float(np.mean(p_bin)), "mean_obs": float(np.mean(y_bin)), "n": int(len(idx))})
    return {"brier": brier, "logloss": logloss, "calibration": cal}


essential_cols = [
    "date","home","away","home_goals","away_goals",
    "period1_home_goals","period1_away_goals","goals_first_10min","period_source"
]


def derive_lambda(row: pd.Series, p1_scale: float, total_scale: float, p1_share: float = 0.35) -> float:
    lam = None
    try:
        h = row.get("period1_home_goals"); a = row.get("period1_away_goals")
        if pd.notna(h) and pd.notna(a):
            lam = (float(h) + float(a)) * float(p1_scale)
    except Exception:
        lam = None
    if lam is None:
        try:
            total = float(row.get("home_goals", 0)) + float(row.get("away_goals", 0))
            lam = total * (p1_share * 0.5) * float(total_scale) / (0.35 * 0.5)
        except Exception:
            lam = 0.0
    return max(0.0, float(lam))


def run(start: Optional[str], end: Optional[str], out_json: str, p1_scale: float = 0.55, total_scale: float = 0.15) -> Path:
    if not RAW_PATH.exists():
        raise FileNotFoundError(RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    # Normalize date
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    except Exception:
        df["date"] = df["date"].astype(str)
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    df = df.sort_values("date")
    # Keep rows with likely real period/first10
    def is_int_like(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        return (s - s.round()).abs() < 1e-9
    has_p1_int = is_int_like(df.get("period1_home_goals", 0)) & is_int_like(df.get("period1_away_goals", 0))
    has_first10_int = is_int_like(df.get("goals_first_10min", 0))
    real_mask = has_p1_int | has_first10_int
    dfr = df[real_mask].copy()
    if dfr.empty:
        raise RuntimeError("No real-ish rows for evaluation")
    # Ground truth
    if has_first10_int.loc[dfr.index].any():
        y = (dfr.get("goals_first_10min", 0).fillna(0).astype(float) > 0.0).astype(int).values
    else:
        p1_total = (dfr.get("period1_home_goals", 0).fillna(0).astype(float) + dfr.get("period1_away_goals", 0).fillna(0).astype(float))
        y = (p1_total > 0.0).astype(int).values
    # Predictions
    lam = dfr.apply(lambda r: derive_lambda(r, p1_scale, total_scale), axis=1).astype(float).values
    p = np.array([prob_yes_from_lambda(x) for x in lam], dtype=float)
    m = metrics(y, p)
    # Build JSON
    out = {
        "start": start,
        "end": end,
        "samples": int(len(dfr)),
        "p1_scale": float(p1_scale),
        "total_scale": float(total_scale),
        "brier": m["brier"],
        "logloss": m["logloss"],
        "calibration": m["calibration"],
    }
    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[first10-eval] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--out", type=str, default=str(PROC_DIR / "first10_eval.json"))
    ap.add_argument("--p1-scale", type=float, default=0.55)
    ap.add_argument("--total-scale", type=float, default=0.15)
    args = ap.parse_args()
    run(args.start, args.end, args.out, args.p1_scale, args.total_scale)

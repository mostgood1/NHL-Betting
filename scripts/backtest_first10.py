"""Backtest first-10 goal probability calibration.

This script evaluates our first-10 YES-goal probability using historical data.
It supports deriving lambda from Period 1 goals or from total goals when P1 is
unavailable, and performs a small grid search over scaling factors.

Outputs:
- Console summary of Brier score, LogLoss, and calibration by decile
- CSV with per-game metrics (optional)

Usage (PowerShell):
  .\.venv\Scripts\Activate.ps1; python scripts/backtest_first10.py --start 2023-10-01 --end 2025-10-17 --out data/processed/first10_backtest.csv

Notes:
- Requires data/raw/games_with_features.csv with columns including:
  date, home, away, period1_home_goals, period1_away_goals, goals_first_10min, home_goals, away_goals
- If period goals missing, the prepare pipeline estimates them; first 10 min may be estimated too.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

RAW_PATH = Path("data/raw/games_with_features.csv")


def sigmoid_clip(x: float, lo: float = 1e-6, hi: float = 1 - 1e-6) -> float:
    return max(lo, min(hi, float(x)))


def prob_yes_from_lambda(lam: float) -> float:
    # P(at least one goal) for Poisson with mean lam
    return 1.0 - np.exp(-max(0.0, float(lam)))


def metrics(y_true: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p.astype(float), 1e-9, 1 - 1e-9)
    y = y_true.astype(int)
    brier = np.mean((p - y) ** 2)
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    # Calibration by decile
    order = np.argsort(p)
    bins = np.array_split(order, 10)
    cal = []
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            cal.append((i, np.nan, np.nan, 0))
            continue
        p_bin = p[idx]
        y_bin = y[idx]
        cal.append((i, float(np.mean(p_bin)), float(np.mean(y_bin)), int(len(idx))))
    return {
        "brier": float(brier),
        "logloss": float(logloss),
        "calibration": cal,
    }


def derive_lambda_from_p1(row: pd.Series, scale: float) -> Optional[float]:
    h = row.get("period1_home_goals")
    a = row.get("period1_away_goals")
    if pd.notna(h) and pd.notna(a):
        try:
            return max(0.0, (float(h) + float(a)) * float(scale))
        except Exception:
            return None
    return None


def derive_lambda_from_total(row: pd.Series, total_scale: float, p1_share: float = 0.35) -> Optional[float]:
    # Approx: first-10 lambda as share of total via P1 share and first-10/P1 fraction (default 0.5)
    try:
        total = float(row.get("home_goals", 0)) + float(row.get("away_goals", 0))
        lam = max(0.0, total * (p1_share * 0.5) * float(total_scale) / (0.35 * 0.5))
        return lam
    except Exception:
        return None


def run_backtest(df: pd.DataFrame, start: Optional[str], end: Optional[str], out_csv: Optional[str], source: Optional[str] = None) -> None:
    df = df.copy()
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]
    df = df.sort_values("date")

    # Optional filter by period_source (e.g., 'pbp', 'api')
    if source and "period_source" in df.columns:
        df = df[df["period_source"].astype(str).str.lower() == source.lower()]

    # Restrict to rows with integer-like period 1 goals (heuristic for real data)
    def _is_int_like(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        return (s - s.round()).abs() < 1e-9

    has_p1_int = _is_int_like(df.get("period1_home_goals", 0)) & _is_int_like(df.get("period1_away_goals", 0))
    has_first10_int = _is_int_like(df.get("goals_first_10min", 0))
    real_mask = has_p1_int | has_first10_int
    df_real = df[real_mask].copy()
    dropped = len(df) - len(df_real)
    if dropped > 0:
        print(f"[info] Filtered out {dropped} rows likely using synthesized period splits; kept {len(df_real)} real-ish rows")
    if df_real.empty:
        print("[error] No real-like rows remain; cannot backtest meaningfully")
        return

    # Ground truth: prefer integer-like first_10 goals; else use (P1 > 0) proxy on integer-like P1
    if "goals_first_10min" in df_real.columns and has_first10_int.loc[df_real.index].any():
        y_series = (df_real.get("goals_first_10min", 0).fillna(0).astype(float) > 0.0).astype(int)
    else:
        p1_total = (
            df_real.get("period1_home_goals", 0).fillna(0).astype(float)
            + df_real.get("period1_away_goals", 0).fillna(0).astype(float)
        )
        y_series = (p1_total > 0.0).astype(int)
    y = y_series.values

    # Grid search scales
    p1_scales = [0.45, 0.5, 0.55]
    total_scales = [0.15, 0.175, 0.20]

    best = None
    results = []

    for s in p1_scales:
        # Prefer P1-derived
        lam_p1 = df_real.apply(lambda r: derive_lambda_from_p1(r, s), axis=1)
        # Fallbacks using total-derived
        for ts in total_scales:
            lam_total = df_real.apply(lambda r: derive_lambda_from_total(r, ts), axis=1)
            lam = lam_p1.fillna(lam_total)
            p = lam.fillna(0.0).astype(float).map(prob_yes_from_lambda).values
            m = metrics(y, p)
            results.append({
                "p1_scale": s,
                "total_scale": ts,
                "brier": m["brier"],
                "logloss": m["logloss"],
            })
            if best is None or m["brier"] < best["brier"]:
                best = {"p1_scale": s, "total_scale": ts, **m}

    print("\n=== First-10 Backtest Summary ===")
    print(f"Samples: {len(df_real)} | Start: {df_real['date'].min()} | End: {df_real['date'].max()}")
    print(f"Best (by Brier): P1_SCALE={best['p1_scale']} TOTAL_SCALE={best['total_scale']} \n  Brier={best['brier']:.4f}  LogLoss={best['logloss']:.4f}")
    print("\nCalibration (deciles):")
    for i, p_hat, p_obs, n in best["calibration"]:
        if np.isnan(p_hat):
            continue
        print(f"  decile {i}: mean_pred={p_hat:.3f} | mean_obs={p_obs:.3f} | n={n}")

    if out_csv:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved grid results to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD")
    ap.add_argument("--out", type=str, default=None, help="Optional CSV output for grid results")
    ap.add_argument("--source", type=str, default=None, help="Filter by period_source (pbp, api)")
    args = ap.parse_args()

    if not RAW_PATH.exists():
        print(f"[error] {RAW_PATH} not found")
        return
    df = pd.read_csv(RAW_PATH)
    # Coerce date to string format for comparisons; file typically has ISO-like strings
    if not np.issubdtype(df["date"].dtype, np.datetime64):
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        except Exception:
            df["date"] = df["date"].astype(str)

    run_backtest(df, args.start, args.end, args.out, args.source)


if __name__ == "__main__":
    main()

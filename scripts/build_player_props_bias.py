"""Build per-player, per-market reconciliation bias from recent vs long-term form.

This script computes a multiplicative bias for each player and market by comparing
recent performance (EWMA over last N games) to a longer-term baseline. The bias
can be applied to model lambdas to better reflect current role and form.

Outputs: data/processed/player_props_bias_{date}.csv with columns:
  date, player, market, bias, games_recent, recent_mean, long_mean

Usage:
  python scripts/build_player_props_bias.py --date 2025-11-14 --recent-games 8 --long-games 40
"""
from __future__ import annotations

import argparse
from pathlib import Path
import math
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


MARKET_TO_COL = {
    "SOG": "shots",
    "GOALS": "goals",
    "ASSISTS": "assists",
    "POINTS": "points",  # computed as goals+assists if not present
    "BLOCKS": "blocked",
    "SAVES": "saves",
}


def _safe_mean(vals: pd.Series) -> float:
    if vals is None or len(vals) == 0:
        return float("nan")
    try:
        return float(pd.to_numeric(vals, errors="coerce").dropna().astype(float).mean())
    except Exception:
        return float("nan")


def _compute_bias(df: pd.DataFrame, recent_games: int, long_games: int, recency_alpha: float = 0.3) -> pd.DataFrame:
    out_rows = []
    # Normalize names and ensure needed columns
    df = df.copy()
    if "player" not in df.columns:
        return pd.DataFrame(columns=["player","market","bias","games_recent","recent_mean","long_mean"])  # empty

    # Make points if possible
    if "points" not in df.columns and set(["goals","assists"]).issubset(set(df.columns)):
        try:
            df["points"] = pd.to_numeric(df["goals"], errors="coerce").fillna(0) + pd.to_numeric(df["assists"], errors="coerce").fillna(0)
        except Exception:
            pass

    # Defensive coercion
    for col in ["shots","goals","assists","points","blocked","saves"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Order by date if available
    if "date" in df.columns:
        try:
            df = df.sort_values("date")
        except Exception:
            pass

    # Iterate per player
    for player, pdf in df.groupby("player", sort=False):
        # Keep only numeric rows
        pdf = pdf.copy()
        for mkt, col in MARKET_TO_COL.items():
            if col not in pdf.columns:
                continue
            vals = pd.to_numeric(pdf[col], errors="coerce").dropna().astype(float)
            if vals.empty:
                continue
            # Long-term baseline: last long_games
            v_long = vals.tail(long_games)
            long_mean = _safe_mean(v_long)
            if not math.isfinite(long_mean) or long_mean <= 0:
                # Without a stable long-term mean, skip creating a bias
                continue
            # Recent EWMA over last recent_games
            v_recent = vals.tail(recent_games)
            if v_recent.empty:
                continue
            al = max(0.0, min(0.99, float(recency_alpha)))
            # EWMA weights newer games more
            n = len(v_recent)
            idx = np.arange(n)
            w = (1.0 - al) ** (n - 1 - idx)
            w /= w.sum()
            recent_mean = float(np.dot(v_recent.values, w))
            # Shrink ratio toward 1 based on sample size
            raw_ratio = (recent_mean / long_mean) if long_mean > 0 else 1.0
            s = min(1.0, len(v_recent) / max(1.0, recent_games))
            ratio = 1.0 + s * (raw_ratio - 1.0)
            # Clip to reasonable range
            bias = float(max(0.6, min(1.4, ratio)))
            out_rows.append({
                "player": str(player),
                "market": mkt,
                "bias": bias,
                "games_recent": int(len(v_recent)),
                "recent_mean": float(recent_mean),
                "long_mean": float(long_mean),
            })
    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="As-of date YYYY-MM-DD (written to output)")
    ap.add_argument("--recent-games", type=int, default=8, help="Recent games window for EWMA")
    ap.add_argument("--long-games", type=int, default=40, help="Long-term baseline window")
    args = ap.parse_args()

    stats_path = RAW / "player_game_stats.csv"
    if not stats_path.exists():
        print("player_game_stats.csv not found at", stats_path)
        return
    try:
        df = pd.read_csv(stats_path)
    except Exception as e:
        print("failed to read stats:", e)
        return
    bias_df = _compute_bias(df, recent_games=args.recent_games, long_games=args.long_games)
    if bias_df is None or bias_df.empty:
        print("no bias rows computed; writing empty file for visibility")
        bias_df = pd.DataFrame(columns=["player","market","bias","games_recent","recent_mean","long_mean"])
    out = bias_df.copy()
    out.insert(0, "date", args.date)
    out_path = PROC / f"player_props_bias_{args.date}.csv"
    out.to_csv(out_path, index=False)
    print("wrote", out_path, "rows=", len(out))


if __name__ == "__main__":
    main()

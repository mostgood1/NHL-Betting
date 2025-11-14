"""Analyze props projections and recommendations uniformity for a given date.

Usage:
    python scripts/analyze_props_uniformity.py --date YYYY-MM-DD [--warn-std 0.05] [--min-nn-share 0.05]

Outputs a textual summary:
  - Source share per market (nn / trad / fallback)
  - Lambda dispersion stats (mean/std/min/max)
  - Top/Bottom 5 lambda players per market
  - Recommendation EV spread (p95 - p5) per market
  - Warnings if std < warn-std or nn share < min-nn-share
Exit code: 0 success, 2 if any warning triggered (optional CI gating)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

PROC = Path("data/processed")

def _load_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()

def summarize(date: str, warn_std: float, min_nn_share: float) -> int:
    proj_path = PROC / f"props_projections_all_{date}.csv"
    rec_path = PROC / f"props_recommendations_{date}.csv"
    proj = _load_csv(proj_path)
    recs = _load_csv(rec_path)
    if proj.empty:
        print(f"[uniformity] projections empty for {date}: {proj_path}")
        return 0
    if "proj_lambda" not in proj.columns:
        print("[uniformity] proj_lambda missing")
        return 0

    print(f"[uniformity] Loaded {len(proj)} projection rows; markets={sorted(proj['market'].unique())}")
    warnings = []
    # Source shares
    if "source" in proj.columns:
        src = proj.groupby(["market","source"]).size().reset_index(name="count")
        print("\nSource shares:")
        for _, r in src.iterrows():
            print(f"  {r['market']:<7} {r['source']:<12} {r['count']}")
    else:
        print("[uniformity] source column missing; cannot compute shares")
    # Dispersion
    disp = proj.groupby("market")["proj_lambda"].agg(["mean","std","min","max","count"]).reset_index()
    print("\nLambda dispersion:")
    for _, r in disp.iterrows():
        print(f"  {r['market']:<7} mean={r['mean']:.2f} std={r['std']:.2f} min={r['min']:.2f} max={r['max']:.2f} n={int(r['count'])}")
        if r['std'] < warn_std:
            warnings.append(f"low std {r['market']}={r['std']:.3f} < {warn_std}")
    # Top/Bottom players per market
    print("\nTop / Bottom 5 lambdas per market:")
    for mkt, g in proj.groupby("market"):
        g_sorted = g.sort_values("proj_lambda")
        bot = g_sorted.head(5)[["player","proj_lambda"]]
        top = g_sorted.tail(5)[["player","proj_lambda"]]
        print(f"  {mkt} bottom5: " + ", ".join(f"{p}={v:.2f}" for p,v in bot.values))
        print(f"  {mkt} top5:    " + ", ".join(f"{p}={v:.2f}" for p,v in top.values))
    # Recommendation EV spread
    if not recs.empty:
        if {"ev_over","market"}.issubset(recs.columns):
            print("\nRecommendation EV spread (over side):")
            for mkt, g in recs.groupby("market"):
                ev = pd.to_numeric(g["ev_over"], errors="coerce").dropna()
                if ev.empty:
                    continue
                p95 = float(ev.quantile(0.95))
                p05 = float(ev.quantile(0.05))
                spread = p95 - p05
                print(f"  {mkt:<7} p05={p05:.4f} p95={p95:.4f} spread={spread:.4f}")
        else:
            print("[uniformity] recommendations missing ev_over/market columns")
    else:
        print("[uniformity] recommendations empty; skipping EV spread")
    # NN share warnings
    if "source" in proj.columns:
        nn_share = proj.assign(is_nn=proj['source'].astype(str).str.startswith('nn')).groupby('market')['is_nn'].mean().reset_index()
        for _, r in nn_share.iterrows():
            if r['is_nn'] < min_nn_share:
                warnings.append(f"low nn share {r['market']}={r['is_nn']:.3f} < {min_nn_share}")
        print("\nNN source share:")
        for _, r in nn_share.iterrows():
            print(f"  {r['market']:<7} nn_share={r['is_nn']:.3f}")
    # Final warnings
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
        return 2
    print("\n[uniformity] OK: no warnings")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    ap.add_argument("--warn-std", type=float, default=0.05, help="Warn if std < threshold")
    ap.add_argument("--min-nn-share", type=float, default=0.05, help="Warn if nn share < threshold")
    args = ap.parse_args()
    code = summarize(args.date, args.warn_std, args.min_nn_share)
    raise SystemExit(code)

if __name__ == "__main__":
    main()

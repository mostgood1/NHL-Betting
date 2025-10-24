from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
import pandas as pd

# Ensure project root import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.utils.io import PROC_DIR


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _ymd(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _ev_band(ev: float) -> str:
    try:
        e = float(ev)
    except Exception:
        return "nan"
    # bands of 0.02 (2%)
    start = int((e // 0.02) * 2)
    lo = round(start / 100.0, 2)
    hi = round(lo + 0.02, 2)
    return f"[{lo:.2f},{hi:.2f})"


def main(start: str, end: str):
    s = _parse_date(start); e = _parse_date(end)
    if e < s:
        s, e = e, s
    log_path = PROC_DIR / "reconciliations_log.csv"
    if not log_path.exists():
        print({"status": "no-log", "path": str(log_path)})
        return 0
    df = pd.read_csv(log_path)
    # Filter by date
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    except Exception:
        df["date"] = df["date"].astype(str).str[:10]
    mask = (df["date"] >= _ymd(s)) & (df["date"] <= _ymd(e))
    df = df[mask].copy()
    if df.empty:
        print({"status": "no-rows", "start": start, "end": end})
        return 0
    # Keep only decided rows (payout not null)
    decided = df[df["payout"].notna()].copy()
    decided["stake"] = pd.to_numeric(decided.get("stake", 0.0), errors="coerce").fillna(0.0)
    decided["payout"] = pd.to_numeric(decided.get("payout", 0.0), errors="coerce").fillna(0.0)
    decided["ev"] = pd.to_numeric(decided.get("ev", float("nan")), errors="coerce")

    # Overall summary
    staked = float(decided["stake"].sum())
    pnl = float(decided["payout"].sum())
    roi = (pnl / staked) if staked > 0 else None
    wins = int((decided["payout"] > 0).sum())
    losses = int((decided["payout"] < 0).sum())
    pushes = int((decided["payout"] == 0).sum())

    # By market
    by_mkt = []
    for mkt, g in decided.groupby("market"):
        s2 = float(g["stake"].sum()); p2 = float(g["payout"].sum())
        roi2 = (p2 / s2) if s2 > 0 else None
        by_mkt.append({"market": mkt, "bets": int(len(g)), "staked": s2, "pnl": p2, "roi": roi2})

    # EV bands
    bands = []
    if "ev" in decided.columns:
        decided["ev_band"] = decided["ev"].apply(_ev_band)
        for b, g in decided.groupby("ev_band"):
            s3 = float(g["stake"].sum()); p3 = float(g["payout"].sum())
            roi3 = (p3 / s3) if s3 > 0 else None
            bands.append({"ev_band": b, "bets": int(len(g)), "staked": s3, "pnl": p3, "roi": roi3})
        # Sort bands by lower bound
        def _band_key(x: dict) -> float:
            try:
                return float(x["ev_band"].split(",")[0].strip("["))
            except Exception:
                return -1.0
        bands.sort(key=_band_key)

    out = {
        "start": start,
        "end": end,
        "overall": {"bets": int(len(decided)), "wins": wins, "losses": losses, "pushes": pushes, "staked": staked, "pnl": pnl, "roi": roi},
        "by_market": by_mkt,
        "by_ev_band": bands,
    }
    print(out)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: summarize_reconciliations.py START END (YYYY-MM-DD)")
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))

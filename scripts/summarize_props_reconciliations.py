from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

PROC = Path("data/processed")


def _ymd(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def main(start: str | None = None, end: str | None = None) -> int:
    path = PROC / "props_reconciliations_log.csv"
    if not path.exists():
        print({"status": "no-log", "path": str(path)})
        return 0
    df = pd.read_csv(path)
    # Optional date filter
    if start and end:
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            s = datetime.strptime(start, "%Y-%m-%d"); e = datetime.strptime(end, "%Y-%m-%d")
            if e < s:
                s, e = e, s
            df = df[(df["date"] >= _ymd(s)) & (df["date"] <= _ymd(e))].copy()
        except Exception:
            pass
    df = df[df["payout"].notna()].copy()
    if df.empty:
        print({"status": "no-rows"})
        return 0
    df["stake"] = pd.to_numeric(df.get("stake", 0.0), errors="coerce").fillna(0.0)
    df["payout"] = pd.to_numeric(df.get("payout", 0.0), errors="coerce").fillna(0.0)
    staked = float(df["stake"].sum())
    pnl = float(df["payout"].sum())
    roi = (pnl / staked) if staked > 0 else None
    overall = {"bets": int(len(df)), "staked": staked, "pnl": pnl, "roi": roi}
    by_mkt = []
    for mkt, g in df.groupby("market"):
        s2 = float(g["stake"].sum()); p2 = float(g["payout"].sum())
        roi2 = (p2 / s2) if s2 > 0 else None
        by_mkt.append({"market": mkt, "bets": int(len(g)), "staked": s2, "pnl": p2, "roi": roi2})
    print({"overall": overall, "by_market": by_mkt})
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >= 2:
        sys.exit(main(args[0], args[1]))
    elif len(args) == 1:
        sys.exit(main(args[0], args[0]))
    else:
        sys.exit(main())

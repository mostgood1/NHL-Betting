from datetime import datetime, timedelta
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
import os
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.scripts.daily_update import (
    _ensure_predictions_csv,
    capture_closing_for_date,
    archive_finals_for_date,
    reconcile_date,
    reconcile_props_date,
)

# Default EV threshold to 2% unless overridden by RECON_EV_THRESHOLD env
EV_THRESHOLD = float(os.getenv("RECON_EV_THRESHOLD", "0.02") or 0.02)

def _ymd(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def main(start: str, end: str) -> int:
    try:
        s = datetime.strptime(start, "%Y-%m-%d"); e = datetime.strptime(end, "%Y-%m-%d")
    except Exception:
        print("Usage: reconcile_range.py START END (YYYY-MM-DD)")
        return 1
    if e < s:
        s, e = e, s
    d = s
    agg = {"picks": 0, "decided": 0, "wins": 0, "losses": 0, "pushes": 0, "staked": 0.0, "pnl": 0.0}
    while d <= e:
        ds = _ymd(d)
        try:
            # Ensure predictions CSV exists for the day (enables reconciliation on missed days)
            try:
                _ensure_predictions_csv(ds, verbose=False)
            except Exception:
                pass
            # Capture pre-game closing odds snapshot and archive finals for robust outcomes
            try:
                capture_closing_for_date(ds, prefer_book=None, best_of_all=True, verbose=False)
            except Exception:
                pass
            try:
                archive_finals_for_date(ds, verbose=False)
            except Exception:
                pass
            # Reconcile games (EV filter via env)
            res = reconcile_date(ds, verbose=False, ev_threshold=EV_THRESHOLD)
            if isinstance(res, dict) and res.get("status") == "ok":
                agg["picks"] += int(res.get("picks", 0))
                agg["decided"] += int(res.get("decided", 0))
                agg["wins"] += int(res.get("wins", 0))
                agg["losses"] += int(res.get("losses", 0))
                agg["pushes"] += int(res.get("pushes", 0))
                agg["staked"] += float(res.get("staked", 0.0))
                agg["pnl"] += float(res.get("pnl", 0.0))
            # Reconcile props (best-effort; ignore failures)
            try:
                reconcile_props_date(ds, verbose=False)
            except Exception:
                pass
            print({"date": ds, "status": res.get("status") if isinstance(res, dict) else "ok"})
        except Exception as ex:
            print({"date": ds, "error": str(ex)})
        d += timedelta(days=1)
    roi = (agg["pnl"] / agg["staked"]) if agg["staked"] else None
    print({"summary": {**agg, "roi": roi}})
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: reconcile_range.py START END (YYYY-MM-DD)")
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))

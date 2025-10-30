from __future__ import annotations

import sys
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.scripts.daily_update import (
    _ensure_predictions_csv,
    capture_closing_for_date,
)
from nhl_betting.utils.io import PROC_DIR
from nhl_betting.core.recs_shared import (
    recompute_edges_and_recommendations,
    backfill_settlement_for_date,
    reconcile_extended,
)
import pandas as pd


def _ymd(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


## Local numeric helper kept for small cases
def _num(v):
    try:
        return float(v)
    except Exception:
        return None


async def process_day(date_str: str, verbose: bool = False) -> dict:
    summary: dict[str, object] = {"date": date_str}
    # 1) Ensure predictions CSV exists (no-odds run is fine; we'll capture closings next)
    try:
        _ensure_predictions_csv(date_str, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"[backfill] ensure predictions failed {date_str}: {e}")
    # 2) Capture pre-game closings to populate close_* odds used by EV calc fallbacks
    try:
        capture_closing_for_date(date_str, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"[backfill] closings failed {date_str}: {e}")
    # 3) Recompute EVs/edges and persist recommendations (shared)
    try:
        recompute_edges_and_recommendations(date_str, min_ev=0.0)
        summary["edges_recs"] = True
    except Exception as e:
        if verbose:
            print(f"[backfill] recompute edges/recs failed {date_str}: {e}")
        summary["edges_recs"] = False
    # 4) Backfill settlement including First-10 and period totals via PBP (shared)
    try:
        r = backfill_settlement_for_date(date_str)
        summary["settlement_rows"] = int(r.get("rows_backfilled", 0)) if isinstance(r, dict) else None
    except Exception as e:
        if verbose:
            print(f"[backfill] settlement failed {date_str}: {e}")
        summary["settlement_rows"] = None
    # 5) Reconciliation including First-10/periods (shared)
    try:
        rec = reconcile_extended(date_str, flat_stake=100.0)
        summary["reconciliation_written"] = (rec.get("status") == "ok")
    except Exception as e:
        if verbose:
            print(f"[backfill] reconciliation failed {date_str}: {e}")
        summary["reconciliation_written"] = False
    return summary


def main(start: str, end: str, verbose: bool = True) -> int:
    try:
        s = _parse_date(start)
        e = _parse_date(end)
    except Exception:
        print("Usage: backfill_recommendations_range.py START END (YYYY-MM-DD)")
        return 1
    if e < s:
        s, e = e, s
    d = s
    agg = {"picks": 0, "decided": 0, "wins": 0, "losses": 0, "pushes": 0, "staked": 0.0, "pnl": 0.0}
    while d <= e:
        ds = _ymd(d)
        try:
            res = asyncio.run(process_day(ds, verbose=verbose))
            if verbose:
                print({k: v for k, v in res.items() if k != "rows"})
            # If reconciliation JSON exists, aggregate a simple running summary
            try:
                path = PROC_DIR / f"reconciliation_{ds}.json"
                if path.exists():
                    obj = json.loads(path.read_text(encoding="utf-8"))
                    summ = obj.get("summary") or {}
                    agg["picks"] += int(summ.get("picks", 0) or 0)
                    agg["decided"] += int(summ.get("decided", 0) or 0)
                    agg["wins"] += int(summ.get("wins", 0) or 0)
                    agg["losses"] += int(summ.get("losses", 0) or 0)
                    agg["pushes"] += int(summ.get("pushes", 0) or 0)
                    agg["staked"] += float(summ.get("staked", 0.0) or 0.0)
                    agg["pnl"] += float(summ.get("pnl", 0.0) or 0.0)
            except Exception:
                pass
        except Exception as ex:
            print({"date": ds, "error": str(ex)})
        d += timedelta(days=1)
    roi = (agg["pnl"] / agg["staked"]) if agg["staked"] else None
    print({"summary": {**agg, "roi": roi}})
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: backfill_recommendations_range.py START END (YYYY-MM-DD)")
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2], verbose=True))

from datetime import datetime, timedelta
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
import os
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.cli import recompute_ev as _recompute_ev

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: recompute_ev_range.py START END (YYYY-MM-DD)")
        sys.exit(1)
    start, end = sys.argv[1], sys.argv[2]
    s = datetime.strptime(start, "%Y-%m-%d"); e = datetime.strptime(end, "%Y-%m-%d")
    d = s
    while d <= e:
        ds = d.strftime("%Y-%m-%d")
        try:
            _recompute_ev(ds, prefer_closing=True)
            print({"status": "ok", "date": ds})
        except SystemExit:
            print({"status": "exit", "date": ds})
        except Exception as ex:
            print({"date": ds, "error": str(ex)})
        d += timedelta(days=1)

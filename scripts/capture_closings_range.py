from datetime import datetime, timedelta
import sys, os
from pathlib import Path

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.scripts.daily_update import capture_closing_for_date

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: capture_closings_range.py START END (YYYY-MM-DD)")
        sys.exit(1)
    start, end = sys.argv[1], sys.argv[2]
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    d = s
    while d <= e:
        ds = d.strftime("%Y-%m-%d")
        try:
            res = capture_closing_for_date(ds, prefer_book=None, best_of_all=True, verbose=False)
            print(res)
        except Exception as ex:
            print({"date": ds, "error": str(ex)})
        d += timedelta(days=1)

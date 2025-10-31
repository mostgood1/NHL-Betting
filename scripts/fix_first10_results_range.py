import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure repo root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.core.recs_shared import backfill_settlement_for_date

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")

def _ymd(d: datetime) -> str:
    return d.strftime('%Y-%m-%d')


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print('Usage: fix_first10_results_range.py START END (YYYY-MM-DD)')
        return 1
    s = _parse_date(argv[1]); e = _parse_date(argv[2])
    if e < s:
        s, e = e, s
    d = s
    total = 0
    while d <= e:
        ds = _ymd(d)
        try:
            r = backfill_settlement_for_date(ds, force=True)
            print({'date': ds, 'rows_backfilled': r.get('rows_backfilled') if isinstance(r, dict) else None})
            total += int(r.get('rows_backfilled', 0) or 0) if isinstance(r, dict) else 0
        except Exception as ex:
            print({'date': ds, 'error': str(ex)})
        d += timedelta(days=1)
    print({'total_rows_backfilled': total})
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

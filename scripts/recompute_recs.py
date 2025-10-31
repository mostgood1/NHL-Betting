import sys
import os
from datetime import datetime, timedelta

# Ensure repo root in path
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.core.recs_shared import recompute_edges_and_recommendations

def _ymd(dt: datetime) -> str:
    return dt.strftime('%Y-%m-%d')


def main(argv):
    if len(argv) < 2:
        today = datetime.now()
        dates = [_ymd(today), _ymd(today + timedelta(days=1))]
    else:
        dates = argv[1:]
    os.environ.setdefault('FIRST10_BLEND', '1')
    print('[recompute] dates:', dates)
    for d in dates:
        try:
            recompute_edges_and_recommendations(d, min_ev=0.0)
            print('[ok]', d)
        except SystemExit:
            print('[warn] recompute exited for', d)
        except Exception as e:
            print('[err] recompute failed for', d, e)
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

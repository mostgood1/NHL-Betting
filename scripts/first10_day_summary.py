import sys
import os
import pandas as pd

# Ensure repo root on path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nhl_betting.utils.io import PROC_DIR

def main(argv):
    if len(argv) < 2:
        print('Usage: first10_day_summary.py YYYY-MM-DD')
        return 1
    d = argv[1]
    p = PROC_DIR / f'predictions_{d}.csv'
    if not p.exists():
        print(f'[err] predictions not found {p}')
        return 2
    df = pd.read_csv(p)
    if 'result_first10' not in df.columns:
        print('[err] no result_first10 column')
        return 3
    m = df['result_first10'].astype(str).str.lower().isin(['yes','no'])
    sub = df[m].copy()
    print('[date]', d, 'rows', len(sub))
    print(sub[['home','away','result_first10']].to_string(index=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

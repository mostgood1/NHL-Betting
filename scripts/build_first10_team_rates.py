import sys
import os
import json
import glob
import argparse
import pandas as pd
from datetime import datetime, date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from nhl_betting.utils.io import PROC_DIR


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Build Bayesian-smoothed First-10 team rates with optional recency weighting.")
    parser.add_argument('start', nargs='?', default='2023-10-01', help='Start date YYYY-MM-DD (inclusive)')
    parser.add_argument('end', nargs='?', default=datetime.today().strftime('%Y-%m-%d'), help='End date YYYY-MM-DD (inclusive)')
    # Recency weighting options
    env_current_start = os.getenv('FIRST10_CURRENT_START')
    default_current_start = env_current_start or f"{datetime.today().year}-10-07"
    parser.add_argument('--current-start', dest='current_start', default=default_current_start, help='Current season start date YYYY-MM-DD (for weighting buckets)')
    parser.add_argument('--w-current', dest='w_current', type=float, default=float(os.getenv('FIRST10_WEIGHT_CURRENT', '3.0')), help='Weight for current season games')
    parser.add_argument('--w-last', dest='w_last', type=float, default=float(os.getenv('FIRST10_WEIGHT_LAST', '1.5')), help='Weight for last season games')
    parser.add_argument('--w-prev', dest='w_prev', type=float, default=float(os.getenv('FIRST10_WEIGHT_PREV', '1.0')), help='Weight for two seasons ago (and earlier within range)')
    parser.add_argument('--prior-k', dest='prior_k', type=float, default=float(os.getenv('FIRST10_PRIOR_K', '50')), help='Bayesian prior strength (effective games) toward league average')
    return parser.parse_args(argv[1:])


def main(argv):
    args = _parse_args(argv)
    start = args.start
    end = args.end
    paths = sorted(glob.glob(str(PROC_DIR / 'predictions_*.csv')))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, usecols=['date','home','away','result_first10'])
            m = df['result_first10'].astype(str).str.lower().isin(['yes','no'])
            df = df[m].copy()
            if start:
                df = df[df['date'] >= start]
            if end:
                df = df[df['date'] <= end]
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        print('[warn] no data found')
        return 1
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=['date','home','away'], keep='first')
    data['y'] = data['result_first10'].str.lower().eq('yes').astype(int)
    # Recency weighting by season buckets
    def _to_date(x):
        try:
            return datetime.strptime(str(x), '%Y-%m-%d').date()
        except Exception:
            return None
    current_start_d = _to_date(args.current_start)
    if current_start_d is None:
        # Fallback to Oct 7 of current year
        current_start_d = date(datetime.today().year, 10, 7)
    last_start_d = date(current_start_d.year - 1, current_start_d.month, current_start_d.day)
    prev_start_d = date(current_start_d.year - 2, current_start_d.month, current_start_d.day)
    data['_date'] = data['date'].apply(_to_date)
    def _weight(d: date | None) -> float:
        if d is None:
            return args.w_prev
        if d >= current_start_d:
            return args.w_current
        if d >= last_start_d:
            return args.w_last
        # earlier within range
        return args.w_prev
    data['w'] = data['_date'].apply(_weight).astype(float)
    # Weighted league mean for prior
    sw = float(data['w'].sum())
    sy = float((data['w'] * data['y']).sum())
    league_p = (sy / sw) if sw > 0 else data['y'].mean()
    # Bayesian smoothing parameters
    k = float(args.prior_k)  # prior strength in effective games
    a0 = league_p * k
    b0 = (1.0 - league_p) * k
    out = {}
    teams = pd.unique(pd.concat([data['home'], data['away']], ignore_index=True).dropna().astype(str))
    for t in teams:
        sub = data[(data['home'].astype(str).eq(t)) | (data['away'].astype(str).eq(t))]
        n = len(sub)
        # Weighted counts
        n_eff = float(sub['w'].sum()) if n > 0 else 0.0
        s_eff = float((sub['w'] * sub['y']).sum()) if n > 0 else 0.0
        # Beta posterior mean with weighted effective counts
        denom = (a0 + b0 + n_eff)
        p_hat = ((a0 + s_eff) / denom) if denom > 0 else league_p
        out[str(t)] = {
            'games': int(n),
            'games_eff': float(round(n_eff, 3)),
            'yes_rate': float(p_hat),
            'raw_rate': float(sub['y'].mean()) if n>0 else None,
        }
    # Save JSON
    path = PROC_DIR / 'first10_team_rates.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"[ok] wrote {path} with {len(out)} teams; league_p={league_p:.3f}; weights=(current={args.w_current}, last={args.w_last}, prev={args.w_prev}); current_start={current_start_d}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

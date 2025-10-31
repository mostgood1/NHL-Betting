import sys, os, json, glob
import argparse
import pandas as pd
from datetime import datetime, date

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from nhl_betting.utils.io import PROC_DIR


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Build First-10 scoring/allowing splits per team with smoothing and recency weighting.")
    p.add_argument('start', nargs='?', default='2023-10-01', help='Start date YYYY-MM-DD (inclusive)')
    p.add_argument('end', nargs='?', default=datetime.today().strftime('%Y-%m-%d'), help='End date YYYY-MM-DD (inclusive)')
    env_current_start = os.getenv('FIRST10_CURRENT_START')
    default_current_start = env_current_start or f"{datetime.today().year}-10-07"
    p.add_argument('--current-start', default=default_current_start, help='Current season start date YYYY-MM-DD')
    p.add_argument('--w-current', type=float, default=float(os.getenv('FIRST10_WEIGHT_CURRENT', '3.0')))
    p.add_argument('--w-last', type=float, default=float(os.getenv('FIRST10_WEIGHT_LAST', '1.5')))
    p.add_argument('--w-prev', type=float, default=float(os.getenv('FIRST10_WEIGHT_PREV', '1.0')))
    p.add_argument('--prior-k', type=float, default=float(os.getenv('FIRST10_PRIOR_K', '50')))
    return p.parse_args(argv[1:])


def _to_date(x):
    try:
        return datetime.strptime(str(x), '%Y-%m-%d').date()
    except Exception:
        return None


def main(argv):
    args = _parse_args(argv)
    paths = sorted(glob.glob(str(PROC_DIR / 'predictions_*.csv')))
    frames = []
    usecols = ['date','home','away','result_first10_home','result_first10_away']
    for p in paths:
        try:
            df = pd.read_csv(p, usecols=usecols)
            if args.start:
                df = df[df['date'] >= args.start]
            if args.end:
                df = df[df['date'] <= args.end]
            m_home = df['result_first10_home'].astype(str).str.lower().isin(['yes','no'])
            m_away = df['result_first10_away'].astype(str).str.lower().isin(['yes','no'])
            df = df[m_home & m_away].copy()
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        print('[warn] no rows for splits')
        return 0
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=['date','home','away'], keep='first')
    data['_date'] = data['date'].apply(_to_date)
    current_start = _to_date(args.current_start) or date(datetime.today().year, 10, 7)
    last_start = date(current_start.year - 1, current_start.month, current_start.day)
    prev_start = date(current_start.year - 2, current_start.month, current_start.day)
    def _w(d):
        if d is None: return args.w_prev
        if d >= current_start: return args.w_current
        if d >= last_start: return args.w_last
        return args.w_prev
    data['w'] = data['_date'].apply(_w).astype(float)
    # Targets
    data['y_home_scores'] = data['result_first10_home'].astype(str).str.lower().eq('yes').astype(int)
    data['y_away_scores'] = data['result_first10_away'].astype(str).str.lower().eq('yes').astype(int)
    # League means (weighted)
    sw = float(data['w'].sum())
    league_scores = float((data['w'] * (data['y_home_scores'] | data['y_away_scores']).astype(int)).sum()) / sw if sw>0 else 0.62
    league_home = float((data['w'] * data['y_home_scores']).sum()) / sw if sw>0 else 0.31
    league_away = float((data['w'] * data['y_away_scores']).sum()) / sw if sw>0 else 0.31
    k = float(args.prior_k)
    out = {}
    teams = pd.unique(pd.concat([data['home'], data['away']], ignore_index=True).dropna().astype(str))
    for t in teams:
        d_home = data[data['home'].astype(str).eq(t)]
        d_away = data[data['away'].astype(str).eq(t)]
        n = len(d_home) + len(d_away)
        # Weighted games and events
        n_eff = float(d_home['w'].sum() + d_away['w'].sum())
        s_eff_scores = float((d_home['w'] * d_home['y_home_scores']).sum() + (d_away['w'] * d_away['y_away_scores']).sum())
        s_eff_allows = float((d_home['w'] * d_home['y_away_scores']).sum() + (d_away['w'] * d_away['y_home_scores']).sum())
        # Posterior means toward league_home/league_away
        denom_h = k + (d_home['w'].sum() + d_away['w'].sum())
        scores_rate = ((league_home * (k/2)) + (league_away * (k/2)) + s_eff_scores) / max(1.0, denom_h)
        allows_rate = ((league_home * (k/2)) + (league_away * (k/2)) + s_eff_allows) / max(1.0, denom_h)
        # Clip
        scores_rate = float(max(0.0, min(1.0, scores_rate)))
        allows_rate = float(max(0.0, min(1.0, allows_rate)))
        out[str(t)] = {
            'games': int(n),
            'games_eff': float(round(n_eff, 3)),
            'scores_rate': scores_rate,
            'allows_rate': allows_rate,
        }
    path = PROC_DIR / 'first10_team_splits.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"[ok] wrote {path} with {len(out)} teams; league_home={league_home:.3f} league_away={league_away:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

import os
import sys
import glob
import pandas as pd
from collections import defaultdict
from datetime import datetime

PROC_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
PROC_DIR = os.path.abspath(PROC_DIR)

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, '%Y-%m-%d')


def load_all_results(start: str | None = None, end: str | None = None):
    paths = sorted(glob.glob(os.path.join(PROC_DIR, 'predictions_*.csv')))
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, usecols=['date','home','away','result_first10'])
            # Keep only rows with explicit yes/no
            m = df['result_first10'].astype(str).str.lower().isin(['yes','no'])
            df = df[m].copy()
            # Filter by date window if provided
            if start or end:
                try:
                    if start:
                        df = df[df['date'] >= start]
                    if end:
                        df = df[df['date'] <= end]
                except Exception:
                    pass
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=['date','home','away','result_first10'])
    out = pd.concat(frames, ignore_index=True)
    # Deduplicate by unique game key within day
    out = out.drop_duplicates(subset=['date','home','away'], keep='first')
    out['res_yes'] = out['result_first10'].astype(str).str.lower().eq('yes').astype(int)
    return out


def summarize(df: pd.DataFrame):
    if df.empty:
        print('[warn] no result_first10 data found')
        return 2
    n = len(df)
    overall = df['res_yes'].mean()
    print(f'[overall] N={n}  yes_rate={overall:.3f} ({overall*100:.1f}%)')
    # Per-team (in games involving the team)
    teams = sorted(pd.unique(pd.concat([df['home'], df['away']], ignore_index=True).dropna().astype(str)))
    rows = []
    for t in teams:
        m = (df['home'].astype(str).eq(t)) | (df['away'].astype(str).eq(t))
        sub = df[m]
        if len(sub) == 0:
            continue
        rows.append({
            'team': t,
            'games': len(sub),
            'yes_rate': sub['res_yes'].mean(),
            'yes_pct': 100*sub['res_yes'].mean(),
        })
    team_df = pd.DataFrame(rows).sort_values('yes_rate', ascending=False)
    print('\n[by_team] (top 10 by yes_rate)')
    print(team_df.head(10).to_string(index=False, formatters={'yes_rate':lambda v: f'{v:.3f}','yes_pct':lambda v: f'{v:.1f}'}))
    print('\n[by_team] (bottom 10 by yes_rate)')
    print(team_df.tail(10).to_string(index=False, formatters={'yes_rate':lambda v: f'{v:.3f}','yes_pct':lambda v: f'{v:.1f}'}))
    # Spread summary
    if not team_df.empty:
        print('\n[spread] teams=', len(team_df))
        print('  min = {:.3f} ({:.1f}%)'.format(team_df['yes_rate'].min(), team_df['yes_rate'].min()*100))
        print('  p25 = {:.3f} ({:.1f}%)'.format(team_df['yes_rate'].quantile(0.25), team_df['yes_rate'].quantile(0.25)*100))
        print('  median = {:.3f} ({:.1f}%)'.format(team_df['yes_rate'].median(), team_df['yes_rate'].median()*100))
        print('  p75 = {:.3f} ({:.1f}%)'.format(team_df['yes_rate'].quantile(0.75), team_df['yes_rate'].quantile(0.75)*100))
        print('  max = {:.3f} ({:.1f}%)'.format(team_df['yes_rate'].max(), team_df['yes_rate'].max()*100))
        print('  std = {:.3f}'.format(team_df['yes_rate'].std()))
        # Optional: home vs away split per team
        rows_home = []
        rows_away = []
        for t in teams:
            home_sub = df[df['home'].astype(str).eq(t)]
            away_sub = df[df['away'].astype(str).eq(t)]
            rows_home.append({'team':t,'games_home':len(home_sub),'home_yes_rate': home_sub['res_yes'].mean() if len(home_sub)>0 else float('nan')})
            rows_away.append({'team':t,'games_away':len(away_sub),'away_yes_rate': away_sub['res_yes'].mean() if len(away_sub)>0 else float('nan')})
        home_df = pd.DataFrame(rows_home)
        away_df = pd.DataFrame(rows_away)
        join = pd.merge(home_df, away_df, on='team', how='outer')
        print('\n[team home/away split] (first 10)')
        print(join.sort_values('home_yes_rate', ascending=False).head(10).to_string(index=False, formatters={'home_yes_rate':lambda v: f'{v:.3f}' if pd.notna(v) else '—','away_yes_rate':lambda v: f'{v:.3f}' if pd.notna(v) else '—'}))
    return 0


def main(argv: list[str]) -> int:
    start = argv[1] if len(argv) >= 2 else None
    end = argv[2] if len(argv) >= 3 else None
    if start and (not end):
        print(f"[info] filtering start={start}")
    if start and end:
        print(f"[info] filtering window {start}..{end}")
    df = load_all_results(start, end)
    return summarize(df)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

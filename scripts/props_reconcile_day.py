import sys
from pathlib import Path
import pandas as pd
import numpy as np

from nhl_betting.data.collect import collect_player_game_stats, _parse_boxscore_players
from nhl_betting.data.nhl_api import NHLClient as StatsClient
from nhl_betting.data.nhl_api_web import NHLWebClient as WebClient
from nhl_betting.data.rosters import fetch_current_roster
from nhl_betting.web.teams import get_team_assets

PROC = Path('data/processed')

MARKET_MAP = {
    'SOG': 'shots',
    'GOALS': 'goals',
    'ASSISTS': 'assists',
    'POINTS': 'points',
    'BLOCKS': 'blocked',
    'SAVES': 'saves',
}

CANDIDATE_REC_FILES = [
    'props_recommendations_combined_{d}.csv',
    'props_recommendations_sim_{d}.csv',
    'props_recommendations_{d}.csv',
    'props_recommendations_nolines_{d}.csv',
]

def _to_abbr(team: str) -> str | None:
    try:
        s = str(team or '').strip()
        if not s:
            return None
        if len(s) <= 3:
            return s.upper()
        a = get_team_assets(s).get('abbr')
        return str(a).upper() if a else None
    except Exception:
        return None

def load_actuals(date: str) -> pd.DataFrame:
    # Try direct Stats API schedule+boxscores for the day, then Web API; fallback to historical collector.
    def _direct_stats(day: str) -> pd.DataFrame:
        try:
            sc = StatsClient()
            games = sc.schedule(day, day)
            rows = []
            for g in games:
                if getattr(g, 'home_goals', None) is None or getattr(g, 'away_goals', None) is None:
                    continue
                box = sc.boxscore(getattr(g, 'gamePk'))
                rows.extend(_parse_boxscore_players(box, getattr(g, 'gamePk'), getattr(g, 'gameDate'), getattr(g, 'home'), getattr(g, 'away')))
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()
    def _direct_web(day: str) -> pd.DataFrame:
        try:
            wc = WebClient()
            games = wc.schedule_day(day)
            rows = []
            for g in games:
                if getattr(g, 'home_goals', None) is None or getattr(g, 'away_goals', None) is None:
                    continue
                box = wc.boxscore(getattr(g, 'gamePk'))
                rows.extend(_parse_boxscore_players(box, getattr(g, 'gamePk'), getattr(g, 'gameDate'), getattr(g, 'home'), getattr(g, 'away')))
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()
    df = _direct_stats(date)
    if df is None or df.empty:
        df = _direct_web(date)
    if df is None or df.empty:
        df = collect_player_game_stats(date, date, source='stats')
        if df is None or df.empty:
            df = collect_player_game_stats(date, date, source='web')
        if df is None:
            df = pd.DataFrame()
        df = df[df['date'].astype(str).str.startswith(date)] if not df.empty else df
    # normalize team to abbr
    df['team'] = df.get('team', pd.Series([None]*len(df))).astype(str).map(lambda x: _to_abbr(x) or (str(x).upper() if isinstance(x, str) else None))
    # compute points
    df['points'] = pd.to_numeric(df.get('goals'), errors='coerce') + pd.to_numeric(df.get('assists'), errors='coerce')
    return df

def find_recs(date: str) -> pd.DataFrame:
    for tmpl in CANDIDATE_REC_FILES:
        p = PROC / tmpl.format(d=date)
        if p.exists():
            try:
                df = pd.read_csv(p)
                if df is None or df.empty or (len(df.columns) == 0):
                    continue
                df['__source'] = p.name
                return df
            except Exception:
                continue
    return pd.DataFrame()


def evaluate_day(date: str) -> pd.DataFrame:
    recs = find_recs(date)
    if recs is None or recs.empty:
        raise FileNotFoundError(f'No recommendations file found for {date} under {PROC}')
    act = load_actuals(date)
    if act is None or act.empty:
        raise RuntimeError(f'No actual boxscores found for {date}')
    # normalize team
    recs['team'] = recs.get('team', '').astype(str).map(lambda x: _to_abbr(x) or (str(x).upper() if isinstance(x, str) else None))
    # Enrich recs with player_id via current roster (best effort)
    def _norm_name(s: str) -> str:
        try:
            x = str(s or '').strip().lower()
            return ' '.join(x.split())
        except Exception:
            return ''
    pid_map: dict[tuple[str,str], int] = {}
    for ab in sorted(set(recs['team'].dropna().astype(str))):
        try:
            players = fetch_current_roster(ab)
            for p in players or []:
                pid_map[(ab, _norm_name(getattr(p, 'full_name', '')))] = int(getattr(p, 'player_id', 0) or 0)
        except Exception:
            continue
    recs['player_id'] = recs.apply(lambda r: pid_map.get((str(r.get('team') or ''), _norm_name(r.get('player') or ''))), axis=1)
    # Prefer join on player_id when present and non-null
    join_on = ['team','player_id'] if {'team','player_id'}.issubset(recs.columns) and {'team','player_id'}.issubset(act.columns) else ['team','player']
    left = recs.copy(); right = act.copy()
    # If joining on names, gently normalize
    if join_on[1] == 'player':
        def _norm(n):
            try:
                s = str(n or '').strip().lower()
                return ' '.join(s.split())
            except Exception:
                return None
        left['player'] = left.get('player', '').map(_norm)
        right['player'] = right.get('player', '').map(_norm)
    merged = pd.merge(left, right, on=join_on, how='inner', suffixes=('','_act'))
    if merged.empty and join_on == ['team','player']:
        # attempt normalized join
        def _norm(n):
            try:
                s = str(n or '').strip().lower()
                return ' '.join(s.split())
            except Exception:
                return None
        l = left.copy(); r = right.copy()
        l['player_norm'] = l.get('player','').map(_norm)
        r['player_norm'] = r.get('player','').map(_norm)
        merged = pd.merge(l, r, on=['team','player_norm'], how='inner', suffixes=('','_act'))
    if merged.empty:
        raise RuntimeError('Failed to join recs to actuals')
    # Map market to actual column
    merged['market'] = merged.get('market', '').astype(str).str.upper()
    merged['actual'] = np.nan
    for mkt, col in MARKET_MAP.items():
        sel = merged['market'] == mkt
        if col in merged.columns:
            merged.loc[sel, 'actual'] = pd.to_numeric(merged[col], errors='coerce')
        else:
            # from actual columns
            merged.loc[sel, 'actual'] = pd.to_numeric(merged.get(col), errors='coerce')
    # Evaluate outcome
    line = pd.to_numeric(merged.get('line'), errors='coerce')
    side = merged.get('side') if 'side' in merged.columns else merged.get('chosen_side')
    side = side.astype(str).str.title() if side is not None else pd.Series(['Over']*len(merged))
    res = []
    for i, r in merged.iterrows():
        a = r.get('actual'); l = r.get('line'); s = r.get('side') or r.get('chosen_side') or 'Over'
        if pd.isna(a) or pd.isna(l):
            out = 'unknown'
        else:
            if s == 'Over':
                out = 'win' if float(a) > float(l) else ('push' if float(a) == float(l) else 'loss')
            else:
                out = 'win' if float(a) < float(l) else ('push' if float(a) == float(l) else 'loss')
        res.append(out)
    merged['result'] = res
    keep = ['date','team','player','market','line','side','actual','result','ev','price','book','__source']
    for k in keep:
        if k not in merged.columns:
            merged[k] = None
    out = merged[keep]
    out_path = PROC / f'props_reconciliations_{date}.csv'
    out.to_csv(out_path, index=False)
    print(f'[props-reconcile] wrote {out_path} rows={len(out)}')
    # Summary
    summ = out.groupby('result').size().to_dict()
    print(f'[props-reconcile] summary: {summ}')
    return out

if __name__ == '__main__':
    date = sys.argv[1] if len(sys.argv) > 1 else None
    if not date:
        print('Usage: props_reconcile_day.py YYYY-MM-DD')
        sys.exit(2)
    try:
        evaluate_day(date)
    except Exception as e:
        print(f'[props-reconcile] ERROR: {e}')
        sys.exit(1)

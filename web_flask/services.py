from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from nhl_betting.web.teams import get_team_assets

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / 'data' / 'processed'
ODDS_TEAM_DIR = BASE / 'data' / 'odds' / 'team'

# Simple in-memory caches keyed by date for performance
_PERIOD_CACHE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_PLAYER_CACHE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}


def to_abbr(name: str | None) -> str | None:
    s = str(name or '').strip()
    if not s:
        return None
    if len(s) <= 3:
        return s.upper()
    a = get_team_assets(s).get('abbr')
    return str(a).upper() if a else None


def logo_url_for_abbr(abbr: str | None) -> str | None:
    s = str(abbr or '').upper()
    if not s:
        return None
    # Default to NHL assets dark svg for consistency
    return f"https://assets.nhle.com/logos/nhl/svg/{s}_dark.svg"


def list_games(date: str) -> List[Dict[str, Any]]:
    p = PROC / f'predictions_{date}.csv'
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    games: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        home = str(r.get('home') or '')
        away = str(r.get('away') or '')
        games.append({
            'home': home,
            'away': away,
            'home_abbr': to_abbr(home),
            'away_abbr': to_abbr(away),
            'date_et': r.get('date_et'),
        })
    return games


def load_edges(date: str) -> pd.DataFrame:
    # Prefer sim-based edges when available
    for p in [PROC / f'edges_sim_{date}.csv', PROC / f'edges_{date}.csv']:
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                continue
    return pd.DataFrame()


def load_team_odds(date: str) -> pd.DataFrame:
    dirp = ODDS_TEAM_DIR / f'date={date}'
    p = dirp / 'oddsapi.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    # Attempt to aggregate into a simple per-match snapshot with Home/Away ML and Totals
    # Raw file format: one row per bookmaker/outcome (market in {h2h, spreads, totals})
    required = {'home', 'away', 'market', 'outcome_name', 'outcome_price', 'outcome_point'}
    if not required.issubset(set(df.columns)):
        return df
    # Filter to h2h and totals
    h2h = df[df['market'] == 'h2h'].copy()
    totals = df[df['market'] == 'totals'].copy()
    # Normalize types
    for c in ['outcome_price', 'outcome_point']:
        if c in h2h.columns:
            h2h[c] = pd.to_numeric(h2h[c], errors='coerce')
        if c in totals.columns:
            totals[c] = pd.to_numeric(totals[c], errors='coerce')
    # Build home/away ML by selecting a single bookmaker (prefer DraftKings), else first
    def pick_ml(group: pd.DataFrame, team_name: str) -> float | None:
        g = group[group['outcome_name'] == team_name]
        if g.empty:
            return None
        dk = g[g['bookmaker_key'].str.lower() == 'draftkings'] if 'bookmaker_key' in g.columns else pd.DataFrame()
        row = dk.iloc[0] if not dk.empty else g.iloc[0]
        val = row.get('outcome_price')
        try:
            return float(val)
        except Exception:
            return None
    # Totals Over/Under similarly (prefer DraftKings)
    def pick_totals(group: pd.DataFrame, label: str) -> float | None:
        g = group[group['outcome_name'].str.lower() == label]
        if g.empty:
            return None
        dk = g[g['bookmaker_key'].str.lower() == 'draftkings'] if 'bookmaker_key' in g.columns else pd.DataFrame()
        row = dk.iloc[0] if not dk.empty else g.iloc[0]
        val = row.get('outcome_price')
        try:
            return float(val)
        except Exception:
            return None
    # Aggregate per (home, away) pair
    rows: List[Dict[str, Any]] = []
    for (home, away), g in h2h.groupby(['home', 'away']):
        g_tot = totals[(totals['home'] == home) & (totals['away'] == away)]
        rows.append({
            'home': home,
            'away': away,
            'home_ml': pick_ml(g, home),
            'away_ml': pick_ml(g, away),
            'home_totals_over': pick_totals(g_tot, 'over'),
            'home_totals_under': pick_totals(g_tot, 'under'),
        })
    return pd.DataFrame(rows)


def load_props_recs(date: str) -> pd.DataFrame:
    # Prefer sim-based recommendations, then combined, then model-only
    for name in [PROC / f'props_recommendations_sim_{date}.csv', PROC / f'props_recommendations_combined_{date}.csv', PROC / f'props_recommendations_{date}.csv']:
        if name.exists():
            try:
                df = pd.read_csv(name)
                df['__source'] = name.name
                return df
            except Exception:
                continue
    return pd.DataFrame()


def load_sim_boxscores_samples(date: str) -> pd.DataFrame:
    # Prefer the aggregated per-player sim file (smaller) first for responsiveness
    p3 = PROC / f'props_boxscores_sim_{date}.csv'
    if p3.exists():
        try:
            return pd.read_csv(p3)
        except Exception:
            return pd.DataFrame()
    # Then position-level boxscores
    p2 = PROC / f'sim_boxscores_pos_{date}.csv'
    if p2.exists():
        try:
            return pd.read_csv(p2)
        except Exception:
            return pd.DataFrame()
    # Finally, fall back to the very large per-sample file
    p = PROC / f'props_boxscores_sim_samples_{date}.csv'
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

def _normalize_team_abbr_col(df: pd.DataFrame) -> pd.DataFrame:
    def to_team_abbr(val: str) -> str:
        s = str(val or '').strip()
        if not s:
            return ''
        if len(s) <= 3:
            return s.upper()
        a = get_team_assets(s).get('abbr')
        return str(a).upper() if a else s.upper()
    if 'team' in df.columns:
        df['team_abbr'] = df['team'].apply(to_team_abbr)
    else:
        df['team_abbr'] = ''
    return df

def get_period_cache(date: str) -> Dict[str, List[Dict[str, Any]]]:
    if date in _PERIOD_CACHE:
        return _PERIOD_CACHE[date]
    samples = load_sim_boxscores_samples(date)
    if samples is None or samples.empty:
        _PERIOD_CACHE[date] = {}
        return _PERIOD_CACHE[date]
    df = samples.copy()
    df = _normalize_team_abbr_col(df)
    if 'period' not in df.columns:
        df['period'] = 0
    # blocks/blocked fix
    if 'blocked' not in df.columns and 'blocks' in df.columns:
        df['blocked'] = df['blocks']
    cols = ['goals','shots','blocked','saves']
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    agg = df.groupby(['team_abbr','period'])[cols].sum().reset_index()
    period_map: Dict[str, List[Dict[str, Any]]] = {}
    for team, g in agg.groupby('team_abbr'):
        period_map[team] = g.sort_values('period').to_dict(orient='records')
    _PERIOD_CACHE[date] = period_map
    return _PERIOD_CACHE[date]

def get_player_cache(date: str, top_n: int = 12) -> Dict[str, List[Dict[str, Any]]]:
    key = f"{date}:{top_n}"
    if key in _PLAYER_CACHE:
        return _PLAYER_CACHE[key]
    samples = load_sim_boxscores_samples(date)
    if samples is None or samples.empty:
        _PLAYER_CACHE[key] = {}
        return _PLAYER_CACHE[key]
    df = samples.copy()
    df = _normalize_team_abbr_col(df)
    # blocks/blocked and TOI
    if 'blocked' not in df.columns and 'blocks' in df.columns:
        df['blocked'] = df['blocks']
    if 'toi_min' not in df.columns and 'toi_sec' in df.columns:
        df['toi_min'] = pd.to_numeric(df['toi_sec'], errors='coerce').fillna(0.0) / 60.0
    cols = ['goals','assists','points','shots','blocked','saves','toi_min']
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    if 'player' not in df.columns:
        df['player'] = df.get('full_name', '')
    agg = df.groupby(['team_abbr','player'])[cols].mean().reset_index()
    agg['rank_key'] = agg['points']*3 + agg['shots']
    team_players: Dict[str, List[Dict[str, Any]]] = {}
    for team, g in agg.groupby('team_abbr'):
        g2 = g.sort_values('rank_key', ascending=False).drop(columns=['rank_key'])
        team_players[team] = g2.head(top_n).to_dict(orient='records')
    _PLAYER_CACHE[key] = team_players
    return _PLAYER_CACHE[key]


def aggregate_period_boxscores(samples: pd.DataFrame, home_abbr: str, away_abbr: str) -> Dict[str, Any]:
    if samples is None or samples.empty:
        return {'home': [], 'away': []}
    # Expect columns: team (name or abbr), period (1..3), goals, shots, blocked, saves
    df = samples.copy()
    if 'team' not in df.columns:
        return {'home': [], 'away': []}
    # Normalize to abbreviations
    def to_team_abbr(val: str) -> str:
        s = str(val or '').strip()
        if not s:
            return ''
        if len(s) <= 3:
            return s.upper()
        a = get_team_assets(s).get('abbr')
        return str(a).upper() if a else s.upper()
    df['team_abbr'] = df['team'].apply(to_team_abbr)
    # normalize period
    if 'period' not in df.columns:
        df['period'] = 0
    # numeric
    # Support both 'blocks' and 'blocked' from different generators
    if 'blocked' not in df.columns and 'blocks' in df.columns:
        df['blocked'] = df['blocks']
    cols = ['goals','shots','blocked','saves']
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    # Team period expected metrics should SUM over players, not mean
    agg = df.groupby(['team_abbr','period'])[cols].sum().reset_index()
    home = agg[agg['team_abbr'] == home_abbr].sort_values('period').to_dict(orient='records')
    away = agg[agg['team_abbr'] == away_abbr].sort_values('period').to_dict(orient='records')
    return {'home': home, 'away': away}


def line_score_from_periods(per_period: Dict[str, Any]) -> Dict[str, Any]:
    def per_team(rows: List[Dict[str, Any]]) -> List[float]:
        if not rows:
            return [0.0, 0.0, 0.0, 0.0]
        by_p = {int(r.get('period', 0) or 0): float(r.get('goals', 0) or 0.0) for r in rows}
        p1 = round(by_p.get(1, 0.0), 2)
        p2 = round(by_p.get(2, 0.0), 2)
        p3 = round(by_p.get(3, 0.0), 2)
        total = round(p1 + p2 + p3, 2)
        return [p1, p2, p3, total]
    return {
        'home': per_team(per_period.get('home', [])),
        'away': per_team(per_period.get('away', [])),
    }


def first10_goal_prob(per_period: Dict[str, Any]) -> float:
    # Approximate expected goals in first 10 minutes from period 1 averages
    def p1_goals(rows: List[Dict[str, Any]]) -> float:
        for r in rows:
            if int(r.get('period', 0) or 0) == 1:
                return float(r.get('goals', 0) or 0.0)
        return 0.0
    home_p1 = p1_goals(per_period.get('home', []))
    away_p1 = p1_goals(per_period.get('away', []))
    # Scale 20-minute period to 10 minutes
    lam = max((home_p1 + away_p1) * 0.5, 0.0)
    # Poisson: P(at least one goal) = 1 - e^{-lambda}
    try:
        import math
        prob = 1.0 - math.exp(-lam)
    except Exception:
        prob = 0.0
    return round(prob * 100.0, 0)


def aggregate_player_boxscores(samples: pd.DataFrame, team_abbr: str, top_n: int = 12) -> List[Dict[str, Any]]:
    if samples is None or samples.empty:
        return []
    df = samples.copy()
    # Map blocks and TOI
    if 'blocked' not in df.columns and 'blocks' in df.columns:
        df['blocked'] = df['blocks']
    # Convert TOI seconds to minutes if available
    if 'toi_min' not in df.columns and 'toi_sec' in df.columns:
        try:
            df['toi_min'] = pd.to_numeric(df['toi_sec'], errors='coerce').fillna(0.0) / 60.0
        except Exception:
            df['toi_min'] = 0.0
    cols = ['goals','assists','points','shots','blocked','saves','toi_min']
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    # Normalize team to abbreviations for filtering
    def to_team_abbr(val: str) -> str:
        s = str(val or '').strip()
        if not s:
            return ''
        if len(s) <= 3:
            return s.upper()
        a = get_team_assets(s).get('abbr')
        return str(a).upper() if a else s.upper()
    df['team_abbr'] = df.get('team', '').apply(to_team_abbr)
    df = df[df['team_abbr'] == team_abbr]
    if 'player' not in df.columns:
        df['player'] = df.get('full_name', '')
    agg = df.groupby(['team','player'])[cols].mean().reset_index()
    # Mask empty TOI to None for cleaner display
    try:
        agg['toi_min'] = agg['toi_min'].apply(lambda x: None if (pd.isna(x) or x <= 0) else round(float(x), 1))
    except Exception:
        pass
    # rank by points then shots
    agg['rank_key'] = agg['points']*3 + agg['shots']
    agg = agg.sort_values('rank_key', ascending=False)
    return agg.drop(columns=['rank_key']).head(top_n).to_dict(orient='records')


def betting_recommendations(edges: pd.DataFrame, home: str, away: str, ev_thr: float = 0.10) -> List[Dict[str, Any]]:
    if edges is None or edges.empty:
        return []
    df = edges.copy()
    df = df[(df['home'] == home) & (df['away'] == away)] if {'home','away'}.issubset(df.columns) else df
    recs: List[Dict[str, Any]] = []
    def add_if(col: str, label: str):
        if col in df.columns:
            v = float(df[col].iloc[0])
            if v >= ev_thr:
                recs.append({'market': label, 'ev': round(v, 4)})
    add_if('ev_home_ml', 'Home ML')
    add_if('ev_away_ml', 'Away ML')
    add_if('ev_over', 'Totals Over')
    add_if('ev_under', 'Totals Under')
    # Puckline if present
    for side in ['home','away']:
        for pl in ['+1.5','-1.5']:
            col = f'ev_{side}_pl_{pl}'
            if col in df.columns:
                v = float(df[col].iloc[0])
                if v >= ev_thr:
                    recs.append({'market': f'{side.title()} PL {pl}', 'ev': round(v, 4)})
    return recs


def props_table(recs_df: pd.DataFrame, teams: Tuple[str,str]) -> List[Dict[str, Any]]:
    if recs_df is None or recs_df.empty:
        return []
    home, away = teams
    df = recs_df.copy()
    df['team'] = df.get('team','').astype(str).str.upper()
    df = df[df['team'].isin([home, away])]
    # If EV is missing (e.g., nolines source), fallback to probability columns
    if 'ev' not in df.columns or df['ev'].isna().all():
        prob_col = None
        for c in ['chosen_prob','p_over','p_under']:
            if c in df.columns and not df[c].isna().all():
                prob_col = c
                break
        if prob_col:
            df['ev'] = df[prob_col]
    # Prefer to show source when book missing
    if 'book' not in df.columns or df['book'].isna().all():
        df['book'] = df.get('__source', df.get('source', None))
    keep = ['team','player','market','line','side','price','book','ev']
    for k in keep:
        if k not in df.columns:
            df[k] = None
    # Deduplicate: one row per (team, player, market, line, side) with best EV
    try:
        df['ev'] = pd.to_numeric(df['ev'], errors='coerce')
    except Exception:
        pass
    dedup_cols = ['team','player','market','line','side']
    try:
        idx = df.groupby(dedup_cols)['ev'].idxmax()
        df = df.loc[idx]
    except Exception:
        df = df.drop_duplicates(subset=dedup_cols, keep='first')
    df = df.sort_values(by=['ev'], ascending=False)
    return df[keep].head(30).to_dict(orient='records')

def load_player_props_projections(date: str) -> pd.DataFrame:
    p = PROC / f'props_projections_all_{date}.csv'
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def player_table_from_projections(proj_df: pd.DataFrame, team_abbr: str, toi_cache: List[Dict[str, Any]], top_n: int = 12) -> List[Dict[str, Any]]:
    if proj_df is None or proj_df.empty:
        return []
    df = proj_df.copy()
    # Filter to team abbreviations directly present in projections
    df['team'] = df.get('team', '').astype(str).str.upper()
    df = df[df['team'] == str(team_abbr).upper()]
    wanted = ['SOG', 'BLOCKS', 'POINTS', 'GOALS', 'ASSISTS']
    df = df[df['market'].isin(wanted)] if 'market' in df.columns else df
    # Pivot to player rows: columns shots, blocked, points
    if not {'player', 'market', 'proj_lambda'}.issubset(df.columns):
        return []
    piv = df.pivot_table(index='player', columns='market', values='proj_lambda', aggfunc='first').reset_index()
    # Normalize missing columns
    for c in wanted:
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv.rename(columns={'SOG': 'shots', 'BLOCKS': 'blocked', 'POINTS': 'points', 'GOALS': 'goals', 'ASSISTS': 'assists'})
    # Attach TOI from cache if available (by player name)
    toi_map = {}
    for r in toi_cache or []:
        name = str(r.get('player') or '')
        if name:
            toi_map[name] = r.get('toi_min')
    piv['toi_min'] = piv['player'].map(toi_map).fillna(pd.NA)
    # Ranking key
    piv['rank_key'] = piv['points']*3 + piv['shots']
    piv = piv.sort_values('rank_key', ascending=False)
    # Ensure numeric and fill missing G/A
    for c in ['goals','assists']:
        if c not in piv.columns:
            piv[c] = 0.0
    cols = ['player', 'goals', 'assists', 'points', 'shots', 'blocked', 'toi_min']
    return piv[cols].head(top_n).to_dict(orient='records')


def sportswriter_writeup(home: str, away: str, per_period: Dict[str, Any], home_players: List[Dict[str, Any]], away_players: List[Dict[str, Any]], edges_recs: List[Dict[str, Any]], props_recs: List[Dict[str, Any]]) -> str:
    def goals_sum(rows: List[Dict[str, Any]]) -> float:
        return round(sum(float(r.get('goals', 0) or 0.0) for r in rows), 2)
    home_goals = goals_sum(per_period.get('home', []))
    away_goals = goals_sum(per_period.get('away', []))
    tone = 'edge' if home_goals != away_goals else 'toss-up'
    lead = f"{home} vs {away}: projected {home_goals}-{away_goals} {tone}."
    # top players
    def top_line(players: List[Dict[str, Any]]) -> str:
        if not players:
            return ''
        top = players[0]
        return f"{top.get('player')} drives offense ({top.get('points')} pts, {top.get('shots')} SOG avg)."
    home_top = top_line(home_players)
    away_top = top_line(away_players)
    # edges summary
    edges_txt = ''
    if edges_recs:
        best = sorted(edges_recs, key=lambda x: x['ev'], reverse=True)[0]
        edges_txt = f"Model leans {best['market']} (EV {best['ev']})."
    # props highlight
    props_txt = ''
    if props_recs:
        p = props_recs[0]
        props_txt = f"Props: {p['player']} {p['market']} {p['side']} {p['line']} @ {p['book']} (EV {p['ev']})."
    return ' '.join([lead, home_top, away_top, edges_txt, props_txt]).strip()

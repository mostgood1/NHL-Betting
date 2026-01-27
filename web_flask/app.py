from __future__ import annotations
import os
import datetime as dt
from pathlib import Path
import sys

from flask import Flask, render_template, redirect, url_for, request, abort
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
# Ensure repo root is on sys.path for module imports when launching directly
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

PROC = BASE / 'data' / 'processed'
from web_flask.services import (
    list_games,
    load_edges,
    load_team_odds,
    load_props_recs,
    load_sim_boxscores_samples,
    aggregate_period_boxscores,
    aggregate_player_boxscores,
    betting_recommendations,
    props_table,
    sportswriter_writeup,
    to_abbr,
    get_period_cache,
    get_player_cache,
    load_player_props_projections,
    player_table_from_projections,
)


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route('/')
    def index():
        today = dt.date.today().strftime('%Y-%m-%d')
        return redirect(url_for('game_cards_for_date', date=today))

    @app.context_processor
    def inject_today():
        return {'today': dt.date.today().strftime('%Y-%m-%d')}

    @app.route('/games/<date>')
    def games_for_date(date: str):
        pred_path = PROC / f'predictions_{date}.csv'
        edges_path = PROC / f'edges_{date}.csv'
        games_df = pd.DataFrame()
        edges = pd.DataFrame()
        try:
            if pred_path.exists():
                games_df = pd.read_csv(pred_path)
            if edges_path.exists():
                edges = pd.read_csv(edges_path)
        except Exception:
            pass
        # Keep simple columns
        cols = [
            'date_et', 'home', 'away', 'home_ml', 'away_ml',
            'totals_line', 'ev_home_ml', 'ev_away_ml', 'ev_over', 'ev_under'
        ]
        games = []
        if not games_df.empty:
            g2 = games_df[[c for c in cols if c in games_df.columns]].copy()
            for _, r in g2.iterrows():
                home = str(r.get('home') or '')
                away = str(r.get('away') or '')
                games.append({
                    **{k: r.get(k) for k in g2.columns},
                    'home_abbr': to_abbr(home),
                    'away_abbr': to_abbr(away),
                })
        return render_template('games.html', date=date, games=games, edges=edges.to_dict(orient='records'))

    @app.route('/games/')
    def games_today_redirect():
        return redirect(url_for('games_for_date', date=dt.date.today().strftime('%Y-%m-%d')))

    @app.route('/props/<date>')
    def props_for_date(date: str):
        combined = PROC / f'props_recommendations_combined_{date}.csv'
        sim = PROC / f'props_recommendations_sim_{date}.csv'
        recs = pd.DataFrame()
        src = None
        try:
            if combined.exists():
                recs = pd.read_csv(combined)
                src = combined.name
            elif sim.exists():
                recs = pd.read_csv(sim)
                src = sim.name
        except Exception:
            pass
        # Display subset for clarity
        keep = ['team','player','market','line','side','price','ev','book']
        if not recs.empty:
            recs = recs[[c for c in keep if c in recs.columns]]
        return render_template('props.html', date=date, source=src, rows=recs.to_dict(orient='records'))

    @app.route('/props/')
    def props_today_redirect():
        return redirect(url_for('props_for_date', date=dt.date.today().strftime('%Y-%m-%d')))

    @app.route('/boxscores/<date>')
    def boxscores_for_date(date: str):
        # Prefer actuals when available; fallback to sim props boxscores
        actual_p = PROC / f'player_props_vs_actuals_{date}.csv'
        sim_p = PROC / f'props_boxscores_sim_{date}.csv'
        df = pd.DataFrame()
        source = None
        try:
            if actual_p.exists():
                df = pd.read_csv(actual_p)
                source = actual_p.name
            elif sim_p.exists():
                df = pd.read_csv(sim_p)
                source = sim_p.name
        except Exception:
            pass
        # Basic columns for display; group by team then player
        cols = ['team','player','goals','assists','points','shots','blocked','saves','toi_min']
        if not df.empty:
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            df = df[cols]
        rows = df.to_dict(orient='records')
        return render_template('boxscores.html', date=date, source=source, rows=rows)

    @app.route('/boxscores/')
    def boxscores_today_redirect():
        return redirect(url_for('boxscores_for_date', date=dt.date.today().strftime('%Y-%m-%d')))

    @app.route('/game-card/<date>/<home>/<away>')
    def game_card(date: str, home: str, away: str):
        # odds and edges
        edges = load_edges(date)
        team_odds = load_team_odds(date)
        # betting recs from edges
        # Recover team names from abbr using predictions
        preds = pd.read_csv(PROC / f'predictions_{date}.csv') if (PROC / f'predictions_{date}.csv').exists() else pd.DataFrame()
        # find proper display names
        home_name = home; away_name = away
        if not preds.empty and {'home','away'}.issubset(preds.columns):
            for _, r in preds.iterrows():
                if to_abbr(r.get('home')) == home:
                    home_name = str(r.get('home'))
                if to_abbr(r.get('away')) == away:
                    away_name = str(r.get('away'))
        edge_recs = betting_recommendations(edges, home_name, away_name)
        # props recs filtered to teams
        props_recs_df = load_props_recs(date)
        props_rows = props_table(props_recs_df, (home, away))
        # projected boxscores from sim samples (use caches for speed)
        period_cache = get_period_cache(date)
        player_cache = get_player_cache(date)
        proj_df = load_player_props_projections(date)
        per_period = {
            'home': period_cache.get(home, []),
            'away': period_cache.get(away, []),
        }
        # Prefer player projections for points/shots/blocks, merge TOI from cache for variability
        home_players = player_table_from_projections(proj_df, home, player_cache.get(home, [])) if proj_df is not None and not proj_df.empty else player_cache.get(home, [])
        away_players = player_table_from_projections(proj_df, away, player_cache.get(away, [])) if proj_df is not None and not proj_df.empty else player_cache.get(away, [])
        # simple odds subset for this matchup (best effort)
        odds_rows = []
        if team_odds is not None and not team_odds.empty:
            # attempt to match display names
            m = team_odds.copy()
            m_cols = ['home','away','home_ml','away_ml','home_totals_over','home_totals_under']
            for c in m_cols:
                if c not in m.columns:
                    m[c] = None
            for _, r in m.iterrows():
                if str(r.get('home')).strip() == home_name and str(r.get('away')).strip() == away_name:
                    odds_rows.append({k: r.get(k) for k in m_cols})
                    break
        # Fallback to predictions-derived odds if aggregation didn't yield a row
        if (not odds_rows) and not preds.empty:
            row = preds[(preds['home'] == home_name) & (preds['away'] == away_name)] if {'home','away'}.issubset(preds.columns) else pd.DataFrame()
            if not row.empty:
                r0 = row.iloc[0]
                odds_rows.append({
                    'home': home_name,
                    'away': away_name,
                    'home_ml': r0.get('home_ml_odds'),
                    'away_ml': r0.get('away_ml_odds'),
                    'home_totals_over': r0.get('over_odds'),
                    'home_totals_under': r0.get('under_odds'),
                })
        # write-up
        writeup = sportswriter_writeup(home, away, per_period, home_players, away_players, edge_recs, props_rows)
        return render_template('game_card.html', date=date, home=home, away=away, home_name=home_name, away_name=away_name, odds=odds_rows, edge_recs=edge_recs, per_period=per_period, home_players=home_players, away_players=away_players, props_rows=props_rows, writeup=writeup)

    @app.route('/game-cards/<date>')
    def game_cards_for_date(date: str):
        edges = load_edges(date)
        team_odds = load_team_odds(date)
        props_recs_df = load_props_recs(date)
        # Precompute period and player caches once per request
        period_cache = get_period_cache(date)
        player_cache = get_player_cache(date)
        proj_df = load_player_props_projections(date)
        games = list_games(date)
        preds = pd.read_csv(PROC / f'predictions_{date}.csv') if (PROC / f'predictions_{date}.csv').exists() else pd.DataFrame()
        cards = []
        for g in games:
            home_name = g.get('home') or ''
            away_name = g.get('away') or ''
            home_abbr = g.get('home_abbr') or to_abbr(home_name) or ''
            away_abbr = g.get('away_abbr') or to_abbr(away_name) or ''
            edge_recs = betting_recommendations(edges, home_name, away_name)
            props_rows = props_table(props_recs_df, (home_abbr, away_abbr))
            per_period = {
                'home': period_cache.get(home_abbr, []),
                'away': period_cache.get(away_abbr, []),
            }
            # Fallback: derive per-period projections from predictions if sim samples didn't match
            if (not per_period.get('home')) and (not per_period.get('away')) and not preds.empty:
                row = preds[(preds['home'] == home_name) & (preds['away'] == away_name)] if {'home','away'}.issubset(preds.columns) else pd.DataFrame()
                if not row.empty:
                    r0 = row.iloc[0]
                    def mk(team_key: str) -> list:
                        p1 = float(r0.get(f'period1_{team_key}_proj') or 0.0)
                        p2 = float(r0.get(f'period2_{team_key}_proj') or 0.0)
                        p3 = float(r0.get(f'period3_{team_key}_proj') or 0.0)
                        tot = float(r0.get(f'proj_{team_key}_goals') or 0.0)
                        return [
                            {'team': home_abbr if team_key=='home' else away_abbr, 'period': 1, 'goals': p1, 'shots': None, 'blocked': None, 'saves': None},
                            {'team': home_abbr if team_key=='home' else away_abbr, 'period': 2, 'goals': p2, 'shots': None, 'blocked': None, 'saves': None},
                            {'team': home_abbr if team_key=='home' else away_abbr, 'period': 3, 'goals': p3, 'shots': None, 'blocked': None, 'saves': None},
                            {'team': home_abbr if team_key=='home' else away_abbr, 'period': 0, 'goals': tot, 'shots': None, 'blocked': None, 'saves': None},
                        ]
                    per_period = {'home': mk('home'), 'away': mk('away')}
            home_players = player_table_from_projections(proj_df, home_abbr, player_cache.get(home_abbr, [])) if proj_df is not None and not proj_df.empty else player_cache.get(home_abbr, [])
            away_players = player_table_from_projections(proj_df, away_abbr, player_cache.get(away_abbr, [])) if proj_df is not None and not proj_df.empty else player_cache.get(away_abbr, [])
            from web_flask.services import logo_url_for_abbr, line_score_from_periods, first10_goal_prob
            logos = {
                'home': logo_url_for_abbr(home_abbr),
                'away': logo_url_for_abbr(away_abbr),
            }
            line_score = line_score_from_periods(per_period)
            # Prefer predictions for first10 prob when available
            first10 = first10_goal_prob(per_period)
            if not edge_recs and not props_rows and not home_players and not away_players and not per_period.get('home') and not per_period.get('away') and not preds.empty:
                # Try to populate at least first10 from predictions row
                row = preds[(preds['home'] == home_name) & (preds['away'] == away_name)] if {'home','away'}.issubset(preds.columns) else pd.DataFrame()
                if not row.empty:
                    r0 = row.iloc[0]
                    p_f10_yes = r0.get('p_f10_yes')
                    if pd.notna(p_f10_yes):
                        try:
                            first10 = round(float(p_f10_yes) * 100.0, 0)
                        except Exception:
                            pass
            # If line score exists but sums are effectively zero, use predictions prob
            if first10 == 0 and not preds.empty:
                row = preds[(preds['home'] == home_name) & (preds['away'] == away_name)] if {'home','away'}.issubset(preds.columns) else pd.DataFrame()
                if not row.empty:
                    r0 = row.iloc[0]
                    p_f10_yes = r0.get('p_f10_yes')
                    if pd.notna(p_f10_yes):
                        try:
                            first10 = round(float(p_f10_yes) * 100.0, 0)
                        except Exception:
                            pass
            odds_rows = []
            if team_odds is not None and not team_odds.empty:
                m = team_odds.copy()
                m_cols = ['home','away','home_ml','away_ml','home_totals_over','home_totals_under']
                for c in m_cols:
                    if c not in m.columns:
                        m[c] = None
                for _, r in m.iterrows():
                    if str(r.get('home')).strip() == home_name and str(r.get('away')).strip() == away_name:
                        odds_rows.append({k: r.get(k) for k in m_cols})
                        break
            # Fallback to predictions-derived odds
            if (not odds_rows) and not preds.empty:
                row = preds[(preds['home'] == home_name) & (preds['away'] == away_name)] if {'home','away'}.issubset(preds.columns) else pd.DataFrame()
                if not row.empty:
                    r0 = row.iloc[0]
                    odds_rows.append({
                        'home': home_name,
                        'away': away_name,
                        'home_ml': r0.get('home_ml_odds'),
                        'away_ml': r0.get('away_ml_odds'),
                        'home_totals_over': r0.get('over_odds'),
                        'home_totals_under': r0.get('under_odds'),
                    })
            writeup = sportswriter_writeup(home_abbr, away_abbr, per_period, home_players, away_players, edge_recs, props_rows)
            cards.append({
                'home': home_abbr,
                'away': away_abbr,
                'home_name': home_name,
                'away_name': away_name,
                'logos': logos,
                'odds': odds_rows,
                'edge_recs': edge_recs,
                'per_period': per_period,
                'line_score': line_score,
                'first10': first10,
                'home_players': home_players,
                'away_players': away_players,
                'props_rows': props_rows,
                'writeup': writeup,
            })
        return render_template('game_cards.html', date=date, cards=cards)

    @app.route('/game-cards/')
    def game_cards_today_redirect():
        return redirect(url_for('game_cards_for_date', date=dt.date.today().strftime('%Y-%m-%d')))

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from nhl_betting.cli import predict_core, featurize, train
from nhl_betting.data.nhl_api_web import NHLWebClient
from nhl_betting.utils.io import RAW_DIR, PROC_DIR, save_df
from nhl_betting.data.odds_api import OddsAPIClient, normalize_snapshot_to_rows
from nhl_betting.cli import props_fetch_bovada as _props_fetch_bovada
from nhl_betting.cli import props_predict as _props_predict
from nhl_betting.data.collect import collect_player_game_stats
from nhl_betting.utils.odds import american_to_decimal
from nhl_betting.models.elo import Elo
from nhl_betting.models.trends import TrendAdjustments, team_keys


def _today_et() -> datetime:
    try:
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now(timezone.utc)


def _ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _vprint(verbose: bool, *args, **kwargs):
    if verbose:
        print(*args, **kwargs)


def update_models_history_window(years_back: int = 2, verbose: bool = False) -> None:
    """Fetch recent seasons' schedule/results, overwrite games.csv, featurize and train.

    By default, fetch from Sep 1 of (current_year - years_back) through Aug 1 of (current_year + 1).
    """
    now = datetime.now(timezone.utc)
    start = f"{now.year - years_back}-09-01"
    end = f"{now.year + 1}-08-01"
    _vprint(verbose, f"[models] Fetching schedule/results window…")
    client = NHLWebClient()
    games = client.schedule_range(start, end)
    rows = []
    for g in games:
        rows.append({
            "gamePk": g.gamePk,
            "date": g.gameDate,
            "season": g.season,
            "type": g.gameType,
            "home": g.home,
            "away": g.away,
            "home_goals": g.home_goals,
            "away_goals": g.away_goals,
        })
    df = pd.DataFrame(rows)
    out = RAW_DIR / "games.csv"
    save_df(df, out)
    _vprint(verbose, f"[models] Saved {len(df)} games to {out}")
    # Rebuild features and retrain Elo/base_mu
    _vprint(verbose, "[models] Building features…")
    featurize()
    _vprint(verbose, "[models] Training Elo/base_mu…")
    train()
    _vprint(verbose, "[models] Models updated.")


def quick_retune_from_yesterday(verbose: bool = False, trends_decay: float = 0.98, reset_trends: bool = False) -> dict:
    """Apply a quick Elo and base_mu retune using only yesterday's completed NHL games (ET).

    - Loads existing Elo ratings and config (base_mu). If missing, no-op.
    - Fetches yesterday's ET slate via NHLWebClient and filters completed NHL vs NHL games.
    - Updates Elo with those results and blends base_mu slightly toward yesterday's average total.
    - Persists updated ratings and config.
    """
    from nhl_betting.utils.io import MODEL_DIR
    ratings_path = MODEL_DIR / "elo_ratings.json"
    cfg_path = MODEL_DIR / "config.json"
    if not ratings_path.exists() or not cfg_path.exists():
        _vprint(verbose, "[retune] Ratings/config missing; skip quick retune (bootstrap models first).")
        return {"status": "missing-models"}
    # Load
    with open(ratings_path, "r", encoding="utf-8") as f:
        ratings = json.load(f)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_mu = float(cfg.get("base_mu", 3.05))
    elo = Elo()
    elo.ratings = ratings
    # Fetch yesterday ET games
    y_et = _today_et().date() - timedelta(days=1)
    y_str = y_et.strftime("%Y-%m-%d")
    client = NHLWebClient()
    games = client.schedule_range(y_str, y_str)
    # Filter to completed NHL vs NHL
    try:
        from nhl_betting.web.teams import get_team_assets as _assets
        def is_nhl(t: str) -> bool:
            try:
                return bool(_assets(str(t)).get("abbr"))
            except Exception:
                return True
    except Exception:
        def is_nhl(t: str) -> bool:
            return True
    # Load existing subgroup adjustments
    trends = TrendAdjustments.load()
    # Optionally reset trends, else apply a mild decay toward 0 to avoid drift
    if reset_trends:
        trends.ml_home.clear()
        trends.goals.clear()
        _vprint(verbose, "[retune] Trends reset to empty.")
    else:
        try:
            decay = float(trends_decay)
        except Exception:
            decay = 0.98
        if decay < 0.0 or decay > 1.0:
            decay = 0.98
        if trends.ml_home:
            for k in list(trends.ml_home.keys()):
                trends.ml_home[k] = float(round(trends.ml_home[k] * decay, 6))
        if trends.goals:
            for k in list(trends.goals.keys()):
                trends.goals[k] = float(round(trends.goals[k] * decay, 6))
        if verbose:
            _vprint(verbose, f"[retune] Applied trends decay factor {decay} to {len(trends.ml_home)} ml and {len(trends.goals)} goal keys.")
    upd = 0
    tot_goals = 0
    tot_games = 0
    # Smoothing/learning rates
    alpha_ml = 0.2  # EMA for moneyline delta (applied to home side only)
    alpha_goals = 0.2  # EMA for goals lambda delta per team/div/conf
    cap_ml = 0.05  # cap absolute ML adjustment per key
    cap_goals = 0.30  # cap absolute goals lambda delta per key (goals per game)
    for g in games:
        try:
            if g.home_goals is None or g.away_goals is None:
                continue
            if not (is_nhl(g.home) and is_nhl(g.away)):
                continue
            hg = int(g.home_goals)
            ag = int(g.away_goals)
            # 1) Compute expectation BEFORE elo update (based on prior ratings)
            exp_home = elo.expected(g.home, g.away, True)
            # 2) Update Elo ratings with the game result
            elo.update_game(g.home, g.away, hg, ag)
            # 3) Compute residuals for subgroup trends
            # Moneyline residual: actual outcome - expected
            actual_home = 1.0 if hg > ag else 0.0
            ml_resid = actual_home - exp_home  # positive means home outperformed expectation
            # Blend into team/div/conf home ML adjustment
            h_team, h_div, h_conf = team_keys(g.home)
            for k in (h_team, h_div, h_conf):
                if not k:
                    continue
                prev = float(trends.ml_home.get(k, 0.0))
                new = (1 - alpha_ml) * prev + alpha_ml * ml_resid
                # cap
                new = max(-cap_ml, min(cap_ml, new))
                trends.ml_home[k] = float(round(new, 5))
            # Goals residuals: actual team goals - baseline expectation (use base_mu from cfg)
            # We approximate per-team residual as (goals - base_mu), which our Poisson splits adjust later.
            try:
                base_mu = float(cfg.get("base_mu", 3.05))
            except Exception:
                base_mu = 3.05
            gh_resid = hg - base_mu
            ga_resid = ag - base_mu
            for k in (h_team, h_div, h_conf):
                if not k:
                    continue
                prev = float(trends.goals.get(k, 0.0))
                new = (1 - alpha_goals) * prev + alpha_goals * gh_resid
                new = max(-cap_goals, min(cap_goals, new))
                trends.goals[k] = float(round(new, 4))
            a_team, a_div, a_conf = team_keys(g.away)
            for k in (a_team, a_div, a_conf):
                if not k:
                    continue
                prev = float(trends.goals.get(k, 0.0))
                new = (1 - alpha_goals) * prev + alpha_goals * ga_resid
                new = max(-cap_goals, min(cap_goals, new))
                trends.goals[k] = float(round(new, 4))
            tot_goals += (hg + ag)
            tot_games += 1
            upd += 1
        except Exception:
            continue
    # Persist Elo if any updates
    if upd > 0:
        with open(ratings_path, "w", encoding="utf-8") as f:
            json.dump(elo.ratings, f, indent=2)
        # Persist trends as well
        try:
            trends.save()
        except Exception:
            pass
    # Lightly blend base_mu toward yesterday's average
    if tot_games > 0:
        y_avg_per_team = (tot_goals / (2 * tot_games))
        new_mu = 0.99 * base_mu + 0.01 * y_avg_per_team
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump({"base_mu": float(new_mu)}, f, indent=2)
    _vprint(verbose, f"[retune] Applied {upd} game(s). base_mu -> {float(new_mu) if tot_games>0 else base_mu:.3f}")
    # Report sample of adjustments for visibility
    try:
        ml_n = len(trends.ml_home)
        gl_n = len(trends.goals)
    except Exception:
        ml_n = gl_n = 0
    return {"status": "ok", "games": upd, "base_mu": float(new_mu) if tot_games>0 else base_mu, "ml_keys": ml_n, "goals_keys": gl_n}


def make_predictions(days_ahead: int = 2, verbose: bool = False) -> None:
    # Only generate for ET today and ET tomorrow (days_ahead default=2)
    base = _today_et().astimezone(timezone.utc)  # drive by calendar day; game dates are ISO UTC in predictions
    horizon = min(2, max(1, days_ahead))
    _vprint(verbose, f"[predict] Generating predictions for {horizon} day(s)…")
    for i in range(0, horizon):
        d = _ymd(base + timedelta(days=i))
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Try Bovada first
        try:
            predict_core(date=d, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
            _vprint(verbose, f"[predict] {d}: Bovada OK")
        except Exception as e:
            _vprint(verbose, f"[predict] {d}: Bovada failed: {e}")
        # Fallback to Odds API (DK preferred)
        try:
            predict_core(date=d, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
            _vprint(verbose, f"[predict] {d}: OddsAPI OK")
        except Exception as e:
            _vprint(verbose, f"[predict] {d}: OddsAPI failed: {e}")
        # Ensure file exists even without odds
        try:
            predict_core(date=d, source="web", odds_source="csv")
            _vprint(verbose, f"[predict] {d}: Ensured predictions CSV exists")
        except Exception:
            pass


def _team_abbr(name: str) -> str:
    try:
        from nhl_betting.web.teams import get_team_assets as _assets
        return (_assets(str(name)).get("abbr") or "").upper()
    except Exception:
        return ""


def capture_closing_for_date(date: str, prefer_book: str | None = None, best_of_all: bool = True, verbose: bool = False) -> dict:
    """Capture pre-game closing odds for each matchup on a date and persist into predictions_{date}.csv.

    Strategy: for each game row in predictions_{date}.csv, query The Odds API historical snapshot
    at the game's commence time (UTC). Use best-of-all across books by default to maximize coverage
    and store prices into close_* columns if not already set (first-write wins).
    """
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        _vprint(verbose, f"[close] No predictions file for {date}; skipping closings.")
        return {"status": "no-file", "date": date}
    df = pd.read_csv(path)
    if df.empty:
        _vprint(verbose, f"[close] predictions_{date}.csv is empty; skipping.")
        return {"status": "empty", "date": date}
    try:
        client = OddsAPIClient()
    except Exception as e:
        _vprint(verbose, f"[close] OddsAPI not configured: {e}")
        return {"status": "no-oddsapi", "error": str(e)}
    updated = 0
    # Ensure close_* columns exist
    close_cols = [
        "close_home_ml_odds","close_away_ml_odds","close_over_odds","close_under_odds",
        "close_home_pl_-1.5_odds","close_away_pl_+1.5_odds","close_total_line_used",
        "close_home_ml_book","close_away_ml_book","close_over_book","close_under_book",
        "close_home_pl_-1.5_book","close_away_pl_+1.5_book","close_snapshot",
    ]
    for c in close_cols:
        if c not in df.columns:
            df[c] = pd.NA
    # Iterate games
    for idx, r in df.iterrows():
        try:
            # If moneyline closings already set, skip
            if pd.notna(r.get("close_home_ml_odds")) or pd.notna(r.get("close_away_ml_odds")):
                continue
            iso = str(r.get("date"))
            # Parse ISO to ensure Z suffix
            try:
                dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
            except Exception:
                # If it's a plain date without time, skip specific snapshot and use midnight UTC
                dt = datetime.strptime(iso[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            snapshot = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            # Odds snapshots: try NHL, then preseason
            snap, _ = client.historical_odds_snapshot(
                sport="icehockey_nhl",
                snapshot_iso=snapshot,
                regions="us",
                markets="h2h,totals,spreads",
                odds_format="american",
            )
            df_odds = normalize_snapshot_to_rows(snap, bookmaker=prefer_book, best_of_all=best_of_all)
            if df_odds is None or df_odds.empty:
                snap2, _ = client.historical_odds_snapshot(
                    sport="icehockey_nhl_preseason",
                    snapshot_iso=snapshot,
                    regions="us",
                    markets="h2h,totals,spreads",
                    odds_format="american",
                )
                df_odds = normalize_snapshot_to_rows(snap2, bookmaker=prefer_book, best_of_all=best_of_all)
            if df_odds is None or df_odds.empty:
                _vprint(verbose, f"[close] No odds snapshot matched for {date} row {idx}; skipping.")
                continue
            # Map odds rows to team abbr for robust matching
            df_odds["home_abbr"] = df_odds["home"].apply(_team_abbr)
            df_odds["away_abbr"] = df_odds["away"].apply(_team_abbr)
            h_abbr = _team_abbr(r.get("home"))
            a_abbr = _team_abbr(r.get("away"))
            m = df_odds[(df_odds["home_abbr"] == h_abbr) & (df_odds["away_abbr"] == a_abbr)]
            if m.empty:
                # Try reversed just in case
                m = df_odds[(df_odds["home_abbr"] == a_abbr) & (df_odds["away_abbr"] == h_abbr)]
            if m.empty:
                continue
            o = m.iloc[0]
            # Write close_* first time only
            def set_first(dst, val):
                try:
                    cur = df.at[idx, dst]
                    if pd.isna(cur) or cur is None:
                        df.at[idx, dst] = val
                except Exception:
                    pass
            set_first("close_home_ml_odds", o.get("home_ml"))
            set_first("close_away_ml_odds", o.get("away_ml"))
            set_first("close_over_odds", o.get("over"))
            set_first("close_under_odds", o.get("under"))
            set_first("close_home_pl_-1.5_odds", o.get("home_pl_-1.5"))
            set_first("close_away_pl_+1.5_odds", o.get("away_pl_+1.5"))
            set_first("close_total_line_used", o.get("total_line"))
            set_first("close_home_ml_book", o.get("home_ml_book"))
            set_first("close_away_ml_book", o.get("away_ml_book"))
            set_first("close_over_book", o.get("over_book"))
            set_first("close_under_book", o.get("under_book"))
            set_first("close_home_pl_-1.5_book", o.get("home_pl_-1.5_book"))
            set_first("close_away_pl_+1.5_book", o.get("away_pl_+1.5_book"))
            set_first("close_snapshot", snapshot)
            updated += 1
        except Exception:
            continue
    # Persist if any updates
    if updated > 0:
        df.to_csv(path, index=False)
    _vprint(verbose, f"[close] Updated {updated} matchup(s) for {date}.")
    return {"status": "ok", "date": date, "updated": updated}


def reconcile_date(date: str, bankroll: float = 1000.0, flat_stake: float = 100.0, verbose: bool = False) -> dict:
    """Write reconciliation summary/rows for a given date to data/processed/reconciliation_{date}.json.

    Mirrors the web API logic for totals/puckline; moneyline requires explicit winner/price mapping and
    is omitted unless present.
    """
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return {"status": "no-predictions", "date": date}
    df = pd.read_csv(path)
    if df.empty:
        return {"status": "empty", "date": date}
    picks = []
    def add_pick(r: pd.Series, market: str, bet: str, ev_key: str, price_key: str, result_field: str | None = None):
        ev = r.get(ev_key)
        if ev is None or (isinstance(ev, float) and pd.isna(ev)):
            return
        try:
            evf = float(ev)
        except Exception:
            return
        if evf <= 0:
            return
        close_map = {
            "home_ml_odds": "close_home_ml_odds",
            "away_ml_odds": "close_away_ml_odds",
            "over_odds": "close_over_odds",
            "under_odds": "close_under_odds",
            "home_pl_-1.5_odds": "close_home_pl_-1.5_odds",
            "away_pl_+1.5_odds": "close_away_pl_+1.5_odds",
        }
        close_key = close_map.get(price_key)
        price = r.get(close_key)
        if price is None or (isinstance(price, float) and pd.isna(price)):
            price = r.get(price_key)
        res = r.get(result_field) if result_field else None
        picks.append({
            "date": r.get("date"),
            "home": r.get("home"),
            "away": r.get("away"),
            "market": market,
            "bet": bet,
            "ev": evf,
            "price": price,
            "result": res,
        })
    for _, r in df.iterrows():
        add_pick(r, "moneyline", "home_ml", "ev_home_ml", "home_ml_odds", None)
        add_pick(r, "moneyline", "away_ml", "ev_away_ml", "away_ml_odds", None)
        add_pick(r, "totals", "over", "ev_over", "over_odds", "result_total")
        add_pick(r, "totals", "under", "ev_under", "under_odds", "result_total")
        add_pick(r, "puckline", "home_pl_-1.5", "ev_home_pl_-1.5", "home_pl_-1.5_odds", "result_ats")
        add_pick(r, "puckline", "away_pl_+1.5", "ev_away_pl_+1.5", "away_pl_+1.5_odds", "result_ats")
    def american_to_decimal_local(american):
        if american is None or (isinstance(american, float) and pd.isna(american)):
            return None
        try:
            a = float(american)
        except Exception:
            return None
        return 1.0 + (a / 100.0) if a > 0 else 1.0 + (100.0 / abs(a))
    pnl = 0.0
    staked = 0.0
    wins = losses = pushes = 0
    decided = 0
    rows = []
    for p in picks:
        stake = flat_stake
        dec = american_to_decimal_local(p.get("price")) if p.get("price") is not None else None
        res = p.get("result")
        if isinstance(res, str):
            rl = res.lower()
            if rl == "push":
                pushes += 1
                rows.append({**p, "stake": stake, "payout": 0.0})
                continue
            if rl == "win":
                wins += 1
                if dec:
                    pnl += stake * (dec - 1.0)
            elif rl == "loss":
                losses += 1
                pnl -= stake
            decided += 1
            staked += stake
            rows.append({**p, "stake": stake, "payout": (stake * (dec - 1.0)) if (dec and rl == 'win') else (-stake if rl == 'loss' else 0.0)})
        else:
            rows.append({**p, "stake": stake, "payout": None})
    summary = {
        "date": date,
        "picks": len(picks),
        "decided": decided,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "staked": staked,
        "pnl": pnl,
        "roi": (pnl / staked) if staked > 0 else None,
    }
    out = {
        "summary": summary,
        "rows": rows,
    }
    with open(PROC_DIR / f"reconciliation_{date}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    _vprint(verbose, f"[recon] Games {date}: picks={summary['picks']} decided={summary['decided']} pnl={summary['pnl']:.2f} roi={summary['roi'] if summary['roi'] is not None else 'n/a'}")
    # Also append to a long-lived CSV log for model tuning
    try:
        log_path = PROC_DIR / "reconciliations_log.csv"
        log_df = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        rows_df = pd.DataFrame(rows)
        # Add computed stake/payout if missing in rows
        if "stake" not in rows_df.columns:
            rows_df["stake"] = flat_stake
        # Dedup on keys to avoid duplicates across runs
        keys = ["date","home","away","market","bet"]
        if not log_df.empty:
            combined = pd.concat([log_df, rows_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=keys, keep="first")
            combined.to_csv(log_path, index=False)
        else:
            rows_df.to_csv(log_path, index=False)
    except Exception:
        pass
    return {"status": "ok", **summary}


def run(days_ahead: int = 2, years_back: int = 2, reconcile_yesterday: bool = True, verbose: bool = False, bootstrap_models: bool = False, trends_decay: float = 0.98, reset_trends: bool = False, skip_props: bool = False) -> None:
    _vprint(verbose, "[run] Starting daily update…")
    t_start = time.perf_counter()
    # 1) Optionally (re)build models from history
    if bootstrap_models:
        t0 = time.perf_counter()
        update_models_history_window(years_back=years_back, verbose=verbose)
        _vprint(verbose, f"[run] Models updated in {time.perf_counter() - t0:.1f}s")
    # 1b) Quick retune from yesterday's completed games
    t_rt = time.perf_counter()
    quick_retune_from_yesterday(verbose=verbose, trends_decay=trends_decay, reset_trends=reset_trends)
    _vprint(verbose, f"[run] Quick retune completed in {time.perf_counter() - t_rt:.1f}s")
    # 2) Generate predictions for upcoming days
    t1 = time.perf_counter()
    make_predictions(days_ahead=min(2, days_ahead), verbose=verbose)
    _vprint(verbose, f"[run] Predictions generated in {time.perf_counter() - t1:.1f}s")
    # 3) Capture closings for yesterday's ET slate and reconcile
    recon_games = recon_props = None
    if reconcile_yesterday:
        y_et = _today_et().date() - timedelta(days=1)
        y_str = y_et.strftime("%Y-%m-%d")
        _vprint(verbose, f"[run] Reconciling previous ET day: {y_str}")
        t2 = time.perf_counter()
        try:
            capture_closing_for_date(y_str, verbose=verbose)
        except Exception:
            pass
        recon_games = reconcile_date(y_str, verbose=verbose)
        # Also reconcile props for yesterday
        if not skip_props:
            try:
                recon_props = reconcile_props_date(y_str, verbose=verbose)
            except Exception:
                pass
        _vprint(verbose, f"[run] Reconciliation completed in {time.perf_counter() - t2:.1f}s")
    # End-of-run summary
    total_dur = time.perf_counter() - t_start
    try:
        from nhl_betting.models.trends import TrendAdjustments
        tr = TrendAdjustments.load()
        tr_ml = len(tr.ml_home)
        tr_goals = len(tr.goals)
    except Exception:
        tr_ml = tr_goals = 0
    pred_dates = []
    try:
        base = _today_et().astimezone(timezone.utc)
        for i in range(0, min(2, max(1, days_ahead))):
            pred_dates.append(_ymd(base + timedelta(days=i)))
    except Exception:
        pass
    if verbose:
        _vprint(verbose, f"[summary] Predicted: {', '.join(pred_dates)}; Trends: ml_keys={tr_ml}, goals_keys={tr_goals}; Duration: {total_dur:.1f}s")
    else:
        print(f"[summary] Predicted: {', '.join(pred_dates)}; Trends: ml_keys={tr_ml}, goals_keys={tr_goals}; Duration: {total_dur:.1f}s")
    _vprint(verbose, "[run] Daily update complete.")


def reconcile_props_date(date: str, flat_stake: float = 100.0, verbose: bool = False) -> dict:
    """Reconcile previous day's props (SOG/GOALS/SAVES) OVER bets with positive EV.

    Steps:
      - Fetch Bovada props odds for the date and run props predictions
      - Ensure player boxscore stats for that date are collected
      - Compare outcomes vs lines, compute PnL summary
      - Persist reconciliation_{date}.json (props) and append to props_reconciliations_log.csv
    """
    # 1) Fetch props odds and run predictions (writes PROC_DIR/props_predictions.csv)
    odds_csv = str(RAW_DIR / f"bovada_props_{date}.csv")
    try:
        _props_fetch_bovada.callback(date=date, out_csv=odds_csv)  # typer command function is callable
        _vprint(verbose, f"[props] Fetched Bovada props odds for {date}")
    except Exception as e:
        # If fetch fails, continue only if odds file already exists
        _vprint(verbose, f"[props] Fetch props failed: {e}")
    preds_tmp = PROC_DIR / "props_predictions.csv"
    if not preds_tmp.exists():
        try:
            _props_predict.callback(odds_csv=odds_csv)
            _vprint(verbose, f"[props] Ran props predictions for {date}")
        except Exception:
            # If prediction still missing, bail gracefully
            return {"status": "no-props-predictions", "date": date}
    else:
        # Rebuild predictions using fresh odds file when possible
        try:
            _props_predict.callback(odds_csv=odds_csv)
            _vprint(verbose, f"[props] Refreshed props predictions for {date}")
        except Exception:
            pass
    if not preds_tmp.exists():
        return {"status": "no-props-predictions", "date": date}
    # Move/copy to date-stamped file for persistence
    preds_out = PROC_DIR / f"props_predictions_{date}.csv"
    try:
        pd.read_csv(preds_tmp).to_csv(preds_out, index=False)
    except Exception:
        return {"status": "props-read-failed", "date": date}
    preds = pd.read_csv(preds_out)
    if preds.empty:
        return {"status": "empty-props", "date": date}
    # Only consider positive EV OVER bets
    preds = preds[preds["ev_over"].astype(float) > 0]
    if preds.empty:
        return {"status": "no-positive-ev", "date": date}
    # 2) Ensure player stats exist for the date
    try:
        _vprint(verbose, f"[props] Collecting player game stats for {date}…")
        collect_player_game_stats(date, date, source="stats")
    except Exception:
        pass
    stats_path = RAW_DIR / "player_game_stats.csv"
    if not stats_path.exists():
        return {"status": "no-player-stats", "date": date}
    stats = pd.read_csv(stats_path)
    # Filter to date only and relevant fields
    def fmt_date(x):
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        except Exception:
            return None
    stats = stats[stats["date"].apply(fmt_date) == date]
    # Build picks and compute outcomes
    rows = []
    pnl = 0.0
    staked = 0.0
    wins = losses = pushes = 0
    decided = 0
    for _, r in preds.iterrows():
        market = str(r.get("market")).upper()
        player = str(r.get("player"))
        line = float(r.get("line")) if pd.notna(r.get("line")) else None
        odds = float(r.get("odds")) if pd.notna(r.get("odds")) else None
        if line is None or odds is None:
            continue
        # Find player stat row(s)
        ps = stats[stats["player"] == player]
        actual = None
        if not ps.empty:
            if market == "SOG":
                actual = ps.iloc[0].get("shots")
            elif market == "GOALS":
                actual = ps.iloc[0].get("goals")
            elif market == "SAVES":
                actual = ps.iloc[0].get("saves")
        result = None
        if actual is not None and pd.notna(actual):
            try:
                av = float(actual)
                if av > float(line):
                    result = "win"
                elif av < float(line):
                    result = "loss"
                else:
                    result = "push"
            except Exception:
                result = None
        # Compute payout if decided
        stake = flat_stake
        dec = american_to_decimal(odds) if odds is not None else None
        payout = None
        if result == "win" and dec is not None:
            pnl += stake * (dec - 1.0)
            staked += stake
            decided += 1
            wins += 1
            payout = stake * (dec - 1.0)
        elif result == "loss":
            pnl -= stake
            staked += stake
            decided += 1
            losses += 1
            payout = -stake
        elif result == "push":
            pushes += 1
            payout = 0.0
        rows.append({
            "date": date,
            "market": market,
            "player": player,
            "line": line,
            "odds": odds,
            "ev_over": float(r.get("ev_over")) if pd.notna(r.get("ev_over")) else None,
            "actual": actual,
            "result": result,
            "stake": stake,
            "payout": payout,
        })
    summary = {
        "date": date,
        "picks": len(rows),
        "decided": decided,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "staked": staked,
        "pnl": pnl,
        "roi": (pnl / staked) if staked > 0 else None,
    }
    out = {"summary": summary, "rows": rows}
    with open(PROC_DIR / f"reconciliation_props_{date}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    _vprint(verbose, f"[recon] Props {date}: picks={summary['picks']} decided={summary['decided']} pnl={summary['pnl']:.2f} roi={summary['roi'] if summary['roi'] is not None else 'n/a'}")
    # Append to props log CSV
    try:
        log_path = PROC_DIR / "props_reconciliations_log.csv"
        log_df = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
        rows_df = pd.DataFrame(rows)
        keys = ["date","market","player","line"]
        if not log_df.empty:
            combined = pd.concat([log_df, rows_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=keys, keep="first")
            combined.to_csv(log_path, index=False)
        else:
            rows_df.to_csv(log_path, index=False)
    except Exception:
        pass
    return {"status": "ok", **summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Daily update: refresh models, predictions, and reconciliation.")
    ap.add_argument("--days-ahead", type=int, default=2, help="How many days of predictions to generate starting today (ET)")
    ap.add_argument("--years-back", type=int, default=2, help="How many years back to include when rebuilding models (by season start)")
    ap.add_argument("--no-reconcile", action="store_true", help="Skip reconciliation step")
    ap.add_argument("--verbose", action="store_true", help="Print step-by-step progress messages")
    ap.add_argument("--bootstrap-models", action="store_true", help="Rebuild models from historical window before daily steps")
    ap.add_argument("--skip-props", action="store_true", help="Skip props reconciliation for previous day")
    ap.add_argument("--trends-decay", type=float, default=0.98, help="Daily decay factor applied to trend adjustments (0-1)")
    ap.add_argument("--reset-trends", action="store_true", help="Reset trend adjustments before retune")
    args = ap.parse_args()
    run(
        days_ahead=args.days_ahead,
        years_back=args.years_back,
        reconcile_yesterday=(not args.no_reconcile),
        verbose=args.verbose,
        bootstrap_models=args.bootstrap_models,
        trends_decay=args.trends_decay,
        reset_trends=args.reset_trends,
        skip_props=args.skip_props,
    )

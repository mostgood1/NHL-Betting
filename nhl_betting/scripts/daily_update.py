from __future__ import annotations

import argparse
import subprocess
import json
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Ensure the repository root is on sys.path when this file is executed directly
# (e.g., python path/to/nhl_betting/scripts/daily_update.py)
try:
    import sys
    from pathlib import Path
    _THIS = Path(__file__).resolve()
    # Repo root is three levels up: nhl_betting/scripts/daily_update.py -> nhl_betting -> repo root
    _ROOT = _THIS.parent.parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
except Exception:
    pass

# CRITICAL: Import cli.py BEFORE pandas to avoid DLL conflicts with torch/onnx
# cli.py imports torch/onnx first, then pandas. If we import pandas here first,
# torch/onnx will fail to load later, disabling NN models.
from nhl_betting.cli import predict_core, featurize, train
from nhl_betting.cli import props_fetch_bovada as _props_fetch_bovada
from nhl_betting.cli import props_predict as _props_predict
from nhl_betting.cli import props_build_dataset as _props_build_dataset

# NOW safe to import pandas (after cli.py has loaded torch/onnx)
import pandas as pd

from nhl_betting.data.nhl_api_web import NHLWebClient
from nhl_betting.utils.io import RAW_DIR, PROC_DIR, save_df
from nhl_betting.data.odds_api import OddsAPIClient, normalize_snapshot_to_rows
from nhl_betting.data import player_props as props_data
from nhl_betting.data.rosters import build_all_team_roster_snapshots
from nhl_betting.data.collect import collect_player_game_stats
from nhl_betting.utils.odds import american_to_decimal
from nhl_betting.models.elo import Elo, EloConfig
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


def ensure_props_recs_history(dates: list[str], verbose: bool = False) -> None:
    """Append per-day recommendations into a rolling history CSV used by the web app.

    Writes/updates data/processed/props_recommendations_history.csv.
    """
    try:
        hist_path = PROC_DIR / "props_recommendations_history.csv"
        try:
            hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
        except Exception:
            hist = pd.DataFrame()
        for d in dates or []:
            try:
                p = PROC_DIR / f"props_recommendations_{d}.csv"
                if not p.exists() or p.stat().st_size == 0:
                    continue
                df = pd.read_csv(p)
                if df is None or df.empty:
                    continue
                # Ensure date column present
                if 'date' not in df.columns:
                    df['date'] = d
                # Normalize minimal columns
                keep = [c for c in ['date','player','team','market','line','over_price','under_price','book','p_over','ev','proj_lambda','ev_over'] if c in df.columns]
                if keep:
                    df = df[keep]
                # Merge into history with simple de-dup on (date, player, market, line)
                if hist is not None and not hist.empty:
                    combined = pd.concat([hist, df], ignore_index=True)
                else:
                    combined = df
                if not combined.empty:
                    subset = [c for c in ['date','player','market','line'] if c in combined.columns]
                    if subset:
                        combined = combined.drop_duplicates(subset=subset, keep='last')
                hist = combined
                _vprint(verbose, f"[history] appended recs for {d} ({len(df)} rows)")
            except Exception as e:
                _vprint(verbose, f"[history] skip {d}: {e}")
        if hist is not None and not hist.empty:
            save_df(hist, hist_path)
            _vprint(verbose, f"[history] wrote {hist_path.name} with {len(hist)} rows")
    except Exception as e:
        _vprint(verbose, f"[history] failed: {e}")


def build_player_props_vs_actuals(date: str, verbose: bool = False) -> None:
    """Produce player_props_vs_actuals_{date}.csv by joining canonical lines with realized stats.

    Mirrors the logic in web endpoint /api/player-props-reconciliation but runs locally.
    """
    try:
        # Load canonical lines parquet (bovada + oddsapi)
        base = PROC_DIR.parent / "props" / f"player_props_lines/date={date}"
        parts = []
        for name in ("bovada.parquet", "oddsapi.parquet"):
            p = base / name
            if p.exists():
                try:
                    parts.append(pd.read_parquet(p))
                except Exception:
                    pass
        lines = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if lines.empty:
            save_df(pd.DataFrame(), PROC_DIR / f"player_props_vs_actuals_{date}.csv")
            _vprint(verbose, f"[vs_actuals] no lines for {date}; wrote empty cache")
            return
        # Ensure stats exist and load
        try:
            collect_player_game_stats(date, date, source="stats")
        except Exception:
            pass
        stats_path = RAW_DIR / "player_game_stats.csv"
        stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
        if stats.empty:
            save_df(pd.DataFrame(), PROC_DIR / f"player_props_vs_actuals_{date}.csv")
            _vprint(verbose, f"[vs_actuals] no stats for {date}; wrote empty cache")
            return
        stats['date_key'] = pd.to_datetime(stats['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        stats_day = stats[stats['date_key'] == date].copy()
        left = lines.rename(columns={"date":"date_key","player_name":"player"}).copy()
        keep_stats = [c for c in ['player','shots','goals','assists','saves','blocked'] if c in stats_day.columns]
        right = stats_day[['date_key'] + keep_stats]
        merged = left.merge(right, on=['date_key','player'], how='left', suffixes=('', '_act'))
        def _act_row(row):
            m = str(row.get('market') or '').upper()
            if m == 'SOG': return row.get('shots')
            if m == 'GOALS': return row.get('goals')
            if m == 'ASSISTS': return row.get('assists')
            if m == 'POINTS':
                try: return float((row.get('goals') or 0)) + float((row.get('assists') or 0))
                except Exception: return None
            if m == 'SAVES': return row.get('saves')
            if m == 'BLOCKS': return row.get('blocked')
            return None
        merged['actual'] = merged.apply(_act_row, axis=1)
        out_cols = [c for c in ['date_key','player','team','market','line','over_price','under_price','book','actual'] if c in merged.columns]
        out = merged[out_cols].rename(columns={'date_key':'date'}) if out_cols else merged
        save_df(out, PROC_DIR / f"player_props_vs_actuals_{date}.csv")
        _vprint(verbose, f"[vs_actuals] wrote player_props_vs_actuals_{date}.csv ({len(out)} rows)")
    except Exception as e:
        _vprint(verbose, f"[vs_actuals] failed for {date}: {e}")


def _ensure_predictions_csv(date: str, verbose: bool = False) -> None:
    """Ensure a predictions_{date}.csv exists; if missing, generate a minimal one without odds.

    This allows reconciliation to proceed even if the prior day wasn't run locally.
    """
    path = PROC_DIR / f"predictions_{date}.csv"
    if path.exists() and path.stat().st_size > 0:
        return
    try:
        predict_core(date=date, source="web", odds_source="csv")
        # Check if file was actually created and non-empty (no-game days won't create a file)
        if path.exists() and path.stat().st_size > 0:
            _vprint(verbose, f"[ensure] predictions_{date}.csv created (no odds)")
        else:
            _vprint(verbose, f"[ensure] no eligible games for {date}; predictions CSV not created")
    except Exception as e:
        _vprint(verbose, f"[ensure] failed to create predictions_{date}.csv: {e}")


def _maybe_calibrate_props_stats(verbose: bool = False) -> None:
    """Calibrate stats-only props models once per day (writes processed JSON).

    Runs only if player_game_stats.csv exists and last calibration is older than ~20 hours.
    """
    try:
        # Allow quick opt-out via environment
        import os as _os
        if str(_os.environ.get("SKIP_PROPS_CALIBRATION", "")).strip() in ("1", "true", "yes"): 
            _vprint(verbose, "[props] calibration skipped by SKIP_PROPS_CALIBRATION env")
            return
        from nhl_betting.utils.io import RAW_DIR as _RAW
        stats_path = _RAW / "player_game_stats.csv"
        if not stats_path.exists() or stats_path.stat().st_size < 10:
            return
        # Determine if we need to refresh calibration
        out_path = PROC_DIR / "props_stats_calibration.json"
        import datetime as _dt
        if out_path.exists():
            mtime = _dt.datetime.fromtimestamp(out_path.stat().st_mtime)
            if (_dt.datetime.now() - mtime).total_seconds() < 20 * 3600:
                return
        # Choose season window (Sep 1 of last season to yesterday ET)
        today_et = _today_et().date()
        season_start_year = today_et.year - 1
        start = f"{season_start_year}-09-01"
        end = (today_et - _dt.timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            from nhl_betting.cli import props_stats_calibration as _props_cal
            if hasattr(_props_cal, 'callback') and callable(getattr(_props_cal, 'callback')):
                _props_cal.callback(start=start, end=end, windows="5,10,20", bins=10, output_json=str(out_path))
            else:
                _props_cal(start=start, end=end, windows="5,10,20", bins=10, output_json=str(out_path))
            _vprint(verbose, f"[props] stats calibration refreshed {start}..{end}")
        except Exception as e:
            _vprint(verbose, f"[props] calibration skipped: {e}")
    except Exception:
        return


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
    # Respect tuned Elo parameters if present
    try:
        k = float(cfg.get("elo_k", 20.0))
    except Exception:
        k = 20.0
    try:
        ha = float(cfg.get("elo_home_adv", 50.0))
    except Exception:
        ha = 50.0
    elo = Elo(EloConfig(k=k, home_adv=ha))
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
        # Preserve existing keys (elo_k, elo_home_adv, etc.) when updating base_mu
        try:
            cfg["base_mu"] = float(new_mu)
        except Exception:
            cfg = {"base_mu": float(new_mu)}
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
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
        # Ensure file exists only if missing; don't overwrite an existing file that may contain odds
        try:
            from nhl_betting.utils.io import PROC_DIR as _PROC
            path = _PROC / f"predictions_{d}.csv"
            if not (path.exists() and path.stat().st_size > 0):
                predict_core(date=d, source="web", odds_source="csv")
                if path.exists() and path.stat().st_size > 0:
                    _vprint(verbose, f"[predict] {d}: Ensured predictions CSV exists")
                else:
                    _vprint(verbose, f"[predict] {d}: No eligible games; predictions not created")
        except Exception:
            pass


def collect_props_canonical(days_ahead: int = 1, verbose: bool = False) -> dict:
    """Collect canonical player props Parquet partitions for yesterday and upcoming days.

    For each target date (yesterday ET and the next N-1 days including today when days_ahead>=1):
      - Try Bovada props collection into data/props/player_props_lines/date=YYYY-MM-DD/bovada.parquet
      - If zero combined rows, fallback to The Odds API (if ODDS_API_KEY configured)
    Returns a summary dict with per-date counts.
    """
    base_et = _today_et()
    targets = []
    # Yesterday ET
    targets.append((base_et.date() - timedelta(days=1)).strftime("%Y-%m-%d"))
    # Today + optional future window
    for i in range(0, max(1, days_ahead)):
        targets.append((base_et + timedelta(days=i)).strftime("%Y-%m-%d"))
    # Dedup preserve order
    seen = set(); ordered = []
    for d in targets:
        if d not in seen:
            seen.add(d); ordered.append(d)
    out = {"dates": ordered, "counts": {}, "paths": {}}
    cfg_b = props_data.PropsCollectionConfig(output_root="data/props", book="bovada", source="bovada")
    cfg_o = props_data.PropsCollectionConfig(output_root="data/props", book="oddsapi", source="oddsapi")
    # Build roster snapshot once to aid player_id normalization (best-effort)
    roster_df = None
    try:
        roster_df = build_all_team_roster_snapshots()
    except Exception as e:
        _vprint(verbose, f"[props] roster snapshot failed: {e}")
    # Also build a master roster lookup with image URLs and team abbreviations for web enrichment
    try:
        from nhl_betting.cli import roster_master as _roster_master
        if hasattr(_roster_master, 'callback') and callable(getattr(_roster_master, 'callback')):
            _roster_master.callback(date=_ymd(base_et))
        else:
            _roster_master(date=_ymd(base_et))
    except Exception as e:
        _vprint(verbose, f"[props] roster master build skipped: {e}")
    import os as _os
    include_bovada = str(_os.environ.get("PROPS_INCLUDE_BOVADA", "")).strip().lower() in ("1","true","yes")
    for d in ordered:
        # Prefer OddsAPI; optionally add Bovada fallback or include both via env
        cnt_b, path_b = 0, None
        cnt_o, path_o = 0, None
        try:
            res_o = props_data.collect_and_write(d, roster_df=roster_df, cfg=cfg_o)
            cnt_o = int(res_o.get("combined_count") or 0)
            path_o = res_o.get("output_path")
        except Exception as e_o:
            cnt_o, path_o = 0, None
            _vprint(verbose, f"[props] oddsapi failed for {d}: {e_o}")
        # Fallback to Bovada only if OddsAPI empty OR explicitly included
        if include_bovada or cnt_o == 0:
            try:
                res_b = props_data.collect_and_write(d, roster_df=roster_df, cfg=cfg_b)
                cnt_b = int(res_b.get("combined_count") or 0)
                path_b = res_b.get("output_path")
            except Exception as e_b:
                cnt_b, path_b = 0, None
                _vprint(verbose, f"[props] bovada failed for {d}: {e_b}")
        out["counts"][d] = {"bovada": cnt_b, "oddsapi": cnt_o, "combined": cnt_b + cnt_o}
        out["paths"][d] = {"bovada": path_b, "oddsapi": path_o}
        if verbose:
            _vprint(verbose, f"[props] {d}: bovada={cnt_b} oddsapi={cnt_o} total={cnt_b+cnt_o}")
    return out


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


def archive_finals_for_date(date: str, verbose: bool = False) -> dict:
    """Archive final scores and derive outcome/correctness fields into predictions_{date}.csv.

    Ensures downstream reconciliation and web UI do not rely on dynamic backfilling.

    Steps:
      1. Load predictions_{date}.csv (skip if missing/empty)
      2. Fetch scoreboard for the date (NHLWebClient)
      3. Map each prediction row to scoreboard game (team abbreviations)
      4. Fill final_home_goals/final_away_goals (and actual_* equivalents)
      5. Derive winner_actual, winner_model, winner_correct
      6. Derive result_total, result_ats, totals_pick_correct, ats_pick_correct, total_diff
      7. Force game_state to FINAL when final scores present
      8. Persist updated CSV (only if any change)
    """
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return {"status": "no-file", "date": date}
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"status": "read-failed", "date": date, "error": str(e)}
    if df.empty:
        return {"status": "empty", "date": date}
    try:
        client = NHLWebClient()
        sb = client.scoreboard_day(date)
    except Exception as e:
        if verbose:
            print(f"[archive] scoreboard fetch failed {date}: {e}")
        sb = []
    # Build scoreboard index keyed by (home_abbr, away_abbr)
    from nhl_betting.web.teams import get_team_assets as _assets
    def _abbr(x: str) -> str:
        try:
            return (_assets(str(x)).get("abbr") or "").upper()
        except Exception:
            return ""
    sb_idx = {}
    for g in sb:
        try:
            hk = _abbr(g.get("home"))
            ak = _abbr(g.get("away"))
            if hk and ak:
                sb_idx[(hk, ak)] = g
        except Exception:
            continue
    # Ensure columns exist
    need_cols = [
        "final_home_goals","final_away_goals","actual_home_goals","actual_away_goals","actual_total",
        "winner_actual","winner_model","winner_correct","result_total","result_ats","totals_pick_correct","ats_pick_correct","total_diff","game_state"
    ]
    for c in need_cols:
        if c not in df.columns:
            df[c] = pd.NA
    import math
    changed = 0
    for idx, r in df.iterrows():
        try:
            hk = _abbr(r.get("home")); ak = _abbr(r.get("away"))
            g = sb_idx.get((hk, ak))
            if g:
                hg = g.get("home_goals")
                ag = g.get("away_goals")
                if hg is not None and ag is not None:
                    try:
                        hg_i = int(hg); ag_i = int(ag)
                    except Exception:
                        hg_i = ag_i = None
                    if hg_i is not None and ag_i is not None:
                        # Final & actual goals
                        for col, val in (
                            ("final_home_goals", hg_i),("final_away_goals", ag_i),
                            ("actual_home_goals", hg_i),("actual_away_goals", ag_i),
                        ):
                            cur = df.at[idx, col]
                            if (pd.isna(cur) or cur in (None, "")) and val is not None:
                                df.at[idx, col] = val; changed += 1
                        # actual_total
                        at_cur = df.at[idx, "actual_total"]
                        if (pd.isna(at_cur) or at_cur in (None, "")):
                            df.at[idx, "actual_total"] = hg_i + ag_i; changed += 1
                        # winner_actual
                        wa_cur = df.at[idx, "winner_actual"]
                        if (pd.isna(wa_cur) or wa_cur in (None, "")):
                            if hg_i > ag_i:
                                df.at[idx, "winner_actual"] = r.get("home"); changed += 1
                            elif ag_i > hg_i:
                                df.at[idx, "winner_actual"] = r.get("away"); changed += 1
                            else:
                                df.at[idx, "winner_actual"] = "Draw"; changed += 1
                        # winner_model (probabilities)
                        if pd.isna(df.at[idx, "winner_model"]) or df.at[idx, "winner_model"] in (None, ""):
                            try:
                                ph = float(r.get("p_home_ml")) if pd.notna(r.get("p_home_ml")) else None
                                pa = float(r.get("p_away_ml")) if pd.notna(r.get("p_away_ml")) else None
                                if ph is not None and pa is not None:
                                    df.at[idx, "winner_model"] = r.get("home") if ph >= pa else r.get("away"); changed += 1
                            except Exception:
                                pass
                        # winner_correct
                        if (pd.isna(df.at[idx, "winner_correct"]) or df.at[idx, "winner_correct"] in (None, "")) and pd.notna(df.at[idx, "winner_actual"]) and pd.notna(df.at[idx, "winner_model"]):
                            df.at[idx, "winner_correct"] = (df.at[idx, "winner_actual"] == df.at[idx, "winner_model"]); changed += 1
                        # total_line determination
                        total_line = None
                        for key in ("close_total_line_used","total_line_used","pl_line_used"):
                            v = r.get(key)
                            if v is None or (isinstance(v, float) and math.isnan(v)):
                                continue
                            try:
                                total_line = float(v); break
                            except Exception:
                                continue
                        if total_line is not None and (pd.isna(df.at[idx, "result_total"]) or df.at[idx, "result_total"] in (None, "")):
                            act_tot = hg_i + ag_i
                            if act_tot > total_line:
                                df.at[idx, "result_total"] = "Over"; changed += 1
                            elif act_tot < total_line:
                                df.at[idx, "result_total"] = "Under"; changed += 1
                            else:
                                df.at[idx, "result_total"] = "Push"; changed += 1
                        # result_ats (puck line ±1.5)
                        if pd.isna(df.at[idx, "result_ats"]) or df.at[idx, "result_ats"] in (None, ""):
                            diff = hg_i - ag_i
                            df.at[idx, "result_ats"] = "home_-1.5" if diff > 1.5 else "away_+1.5"; changed += 1
                        # totals_pick_correct
                        if (pd.isna(df.at[idx, "totals_pick_correct"]) or df.at[idx, "totals_pick_correct"] in (None, "")) and pd.notna(df.at[idx, "result_total"]) and str(df.at[idx, "result_total"]).lower() not in ("push",):
                            t_pick = r.get("totals_pick")
                            if t_pick and isinstance(t_pick, str):
                                df.at[idx, "totals_pick_correct"] = (t_pick == df.at[idx, "result_total"]); changed += 1
                        # ats_pick_correct
                        if (pd.isna(df.at[idx, "ats_pick_correct"]) or df.at[idx, "ats_pick_correct"] in (None, "")) and pd.notna(df.at[idx, "result_ats"]):
                            a_pick = r.get("ats_pick")
                            if a_pick and isinstance(a_pick, str):
                                df.at[idx, "ats_pick_correct"] = (a_pick == df.at[idx, "result_ats"]); changed += 1
                        # total_diff (model_total - actual_total)
                        if pd.notna(r.get("model_total")) and (pd.isna(df.at[idx, "total_diff"]) or df.at[idx, "total_diff"] in (None, "")):
                            try:
                                df.at[idx, "total_diff"] = round(float(r.get("model_total")) - float(hg_i + ag_i), 2); changed += 1
                            except Exception:
                                pass
                        # Force FINAL state
                        cur_state = df.at[idx, "game_state"] if "game_state" in df.columns else None
                        if cur_state is None or str(cur_state).strip() == "" or "FINAL" not in str(cur_state).upper():
                            df.at[idx, "game_state"] = "FINAL"; changed += 1
        except Exception:
            continue
    if changed > 0:
        try:
            df.to_csv(path, index=False)
            if verbose:
                print(f"[archive] {date}: archived finals / derived outcomes (changes={changed})")
        except Exception as e:
            return {"status": "write-failed", "date": date, "changed": changed, "error": str(e)}
    return {"status": "ok", "date": date, "changed": int(changed)}


def reconcile_date(date: str, bankroll: float = 1000.0, flat_stake: float = 100.0, verbose: bool = False, ev_threshold: float = 0.0) -> dict:
    """Write reconciliation summary/rows for a given date to data/processed/reconciliation_{date}.json.

    Mirrors the web API logic for totals/puckline; moneyline requires explicit winner/price mapping and
    is omitted unless present.
    """
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return {"status": "no-predictions", "date": date}
    # Short-circuit on empty files to avoid EmptyDataError
    try:
        if path.exists() and path.stat().st_size < 10:
            return {"status": "empty", "date": date}
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {"status": "empty", "date": date}
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
        # Apply EV threshold filter (default 0 keeps behavior unchanged)
        if evf <= float(ev_threshold):
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


def run(days_ahead: int = 2, years_back: int = 2, reconcile_yesterday: bool = True, verbose: bool = False, bootstrap_models: bool = False, trends_decay: float = 0.98, reset_trends: bool = False, skip_props: bool = False, git_push: bool = True, git_remote: str = "origin", git_branch: str | None = None, recon_ev_threshold: float = 0.0) -> None:
    _vprint(verbose, "[run] Starting daily update…")
    t_start = time.perf_counter()
    # 1) Optionally (re)build models from history
    if bootstrap_models:
        t0 = time.perf_counter()
        update_models_history_window(years_back=years_back, verbose=verbose)
        _vprint(verbose, f"[run] Models updated in {time.perf_counter() - t0:.1f}s")
    # 1a) Ensure prior-day predictions CSV exists (enables reconciliation if prior run missed)
    try:
        y_et = _today_et().date() - timedelta(days=1)
        _ensure_predictions_csv(y_et.strftime('%Y-%m-%d'), verbose=verbose)
    except Exception:
        pass
    # 1b) Quick retune from yesterday's completed games
    t_rt = time.perf_counter()
    quick_retune_from_yesterday(verbose=verbose, trends_decay=trends_decay, reset_trends=reset_trends)
    _vprint(verbose, f"[run] Quick retune completed in {time.perf_counter() - t_rt:.1f}s")
    # 2) Generate predictions for upcoming days
    t1 = time.perf_counter()
    make_predictions(days_ahead=min(2, days_ahead), verbose=verbose)
    _vprint(verbose, f"[run] Predictions generated in {time.perf_counter() - t1:.1f}s")
    # 2b) Collect canonical player props (yesterday + today/tomorrow) and refresh modeling dataset (rolling window)
    if not skip_props:
        t1b = time.perf_counter()
        try:
            coll = collect_props_canonical(days_ahead=min(2, days_ahead), verbose=verbose)
        except Exception as e:
            coll = {"error": str(e)}
        # Rebuild/refresh modeling dataset on a rolling window (Sep 1 of last season to today)
        try:
            today_et = _today_et().date()
            season_start_year = today_et.year - 1
            start = f"{season_start_year}-09-01"
            end = today_et.strftime("%Y-%m-%d")
            # Call the Typer command function robustly
            def _call_typer_or_func(cmd, **kwargs):
                if hasattr(cmd, 'callback') and callable(getattr(cmd, 'callback')):
                    return cmd.callback(**kwargs)
                elif callable(cmd):
                    return cmd(**kwargs)
                else:
                    raise RuntimeError('Unsupported command object for props build dataset')
            from nhl_betting.utils.io import RAW_DIR
            out_csv_path = (RAW_DIR.parent / "props" / "props_modeling_dataset.csv")
            out_csv_path.parent.mkdir(parents=True, exist_ok=True)
            # Skip rebuild if file is fresh (modified within last 3 hours) to keep runs predictable
            try:
                if out_csv_path.exists():
                    import datetime as _dt
                    mtime = _dt.datetime.fromtimestamp(out_csv_path.stat().st_mtime)
                    if (_dt.datetime.now() - mtime).total_seconds() < 3 * 3600:
                        _vprint(verbose, f"[run] Skipping props modeling dataset rebuild (fresh as of {mtime:%H:%M}).")
                    else:
                        _vprint(verbose, f"[run] Building props modeling dataset {start}..{end}…")
                        _call_typer_or_func(_props_build_dataset, start=start, end=end, output_csv=str(out_csv_path.resolve()))
                else:
                    _vprint(verbose, f"[run] Building props modeling dataset {start}..{end}…")
                    _call_typer_or_func(_props_build_dataset, start=start, end=end, output_csv=str(out_csv_path.resolve()))
            except Exception as ie:
                _vprint(verbose, f"[run] props dataset build skipped quickly due to error: {ie}")
        except Exception as e:
            _vprint(verbose, f"[run] props dataset build skipped quickly due to error: {e}")
        # Precompute model-only projections for slate players using NN models (enabled by default)
        try:
            import os as _os
            # Allow opt-out via PROPS_SKIP_PROJECTIONS=1, otherwise run by default with NN
            if str(_os.environ.get("PROPS_SKIP_PROJECTIONS", "")).strip().lower() not in ("1","true","yes"):
                from nhl_betting.cli import props_project_all as _props_project_all
                def _call_typer_or_func_proj(cmd, **kwargs):
                    if hasattr(cmd, 'callback') and callable(getattr(cmd, 'callback')):
                        return cmd.callback(**kwargs)
                    elif callable(cmd):
                        return cmd(**kwargs)
                    else:
                        raise RuntimeError('Unsupported command object for props project all')
                base = _today_et().date()
                targets = [base.strftime('%Y-%m-%d')]
                if days_ahead and int(days_ahead) > 1:
                    from datetime import timedelta as _td
                    targets.append((base + _td(days=1)).strftime('%Y-%m-%d'))
                for d in targets:
                    try:
                        _vprint(verbose, f"[run] Precomputing NN props projections for {d}…")
                        # Always use NN models (use_nn=True is default)
                        _call_typer_or_func_proj(_props_project_all, date=d, ensure_history_days=365, include_goalies=True, use_nn=True)
                    except Exception as e2:
                        _vprint(verbose, f"[run] props_project_all failed for {d}: {e2}")
            else:
                _vprint(verbose, "[run] Skipping props_projections_all precompute (PROPS_SKIP_PROJECTIONS=1)")
        except Exception as e:
            _vprint(verbose, f"[run] precompute props projections_all skipped: {e}")
        # Calibrate stats-only props models (periodic)
        try:
            _maybe_calibrate_props_stats(verbose=verbose)
        except Exception:
            pass
        # Precompute props recommendations for ET today (+1 day) to speed up web UI
        try:
            from nhl_betting.cli import props_recommendations as _props_recs
            # Helper that tolerates both Typer command and plain function
            def _call_typer_or_func_recs(cmd, **kwargs):
                if hasattr(cmd, 'callback') and callable(getattr(cmd, 'callback')):
                    return cmd.callback(**kwargs)
                elif callable(cmd):
                    return cmd(**kwargs)
                else:
                    raise RuntimeError('Unsupported command object for props recommendations')
            # Helper to normalize ev/proj columns for downstream web routes
            def _normalize_recs_file(path):
                try:
                    if not path.exists() or path.stat().st_size == 0:
                        return False
                    df = pd.read_csv(path)
                    if df is None or df.empty:
                        return False
                    cols = set(df.columns)
                    changed = False
                    if ('ev' not in cols) and ('ev_over' in cols):
                        try:
                            df['ev'] = pd.to_numeric(df['ev_over'], errors='coerce')
                        except Exception:
                            df['ev'] = df['ev_over']
                        changed = True
                    if ('proj' not in cols) and ('proj_lambda' in cols):
                        try:
                            df['proj'] = pd.to_numeric(df['proj_lambda'], errors='coerce')
                        except Exception:
                            df['proj'] = df['proj_lambda']
                        changed = True
                    if changed:
                        save_df(df, path)
                        return True
                except Exception:
                    return False
                return False
            base = _today_et().date()
            targets = [base.strftime('%Y-%m-%d')]
            if days_ahead and int(days_ahead) > 1:
                from datetime import timedelta as _td
                targets.append((base + _td(days=1)).strftime('%Y-%m-%d'))
            # Track artifacts to explicitly stage for git (belt-and-suspenders)
            produced_artifacts: list[str] = []
            for d in targets:
                try:
                    _vprint(verbose, f"[run] Building props recommendations for {d}…")
                    _call_typer_or_func_recs(_props_recs, date=d, min_ev=0.0, top=200, market="")
                    # Normalize and record artifact
                    outp = PROC_DIR / f"props_recommendations_{d}.csv"
                    try:
                        if _normalize_recs_file(outp):
                            _vprint(verbose, f"[run] normalized columns in {outp.name}")
                    except Exception:
                        pass
                    produced_artifacts.append(str(outp))
                except Exception as e2:
                    _vprint(verbose, f"[run] props recommendations failed for {d}: {e2}")
                    # Write an empty placeholder to allow reconciliation to proceed
                    try:
                        outp = PROC_DIR / f"props_recommendations_{d}.csv"
                        if not outp.exists():
                            pd.DataFrame(columns=["date","player_id","player_name","team","market","line","over_price","under_price","book","ev","p_over"]).to_csv(outp, index=False)
                            _vprint(verbose, f"[run] wrote empty props_recommendations_{d}.csv placeholder")
                        produced_artifacts.append(str(outp))
                    except Exception:
                        pass
            # Append into history CSV for web charts/tables
            try:
                ensure_props_recs_history(targets, verbose=verbose)
            except Exception:
                pass
            # Persist list of artifacts on the instance for later git stage
            try:
                globals()["_DAILY_UPDATE_PRODUCED_ARTIFACTS"] = produced_artifacts
            except Exception:
                pass
        except Exception as e:
            _vprint(verbose, f"[run] precompute props recommendations skipped: {e}")
        _vprint(verbose, f"[run] Props collection/dataset in {time.perf_counter() - t1b:.1f}s")
    # 3) Capture closings for yesterday's ET slate and reconcile
    recon_games = recon_props = None
    if reconcile_yesterday:
        y_et = _today_et().date() - timedelta(days=1)
        y_str = y_et.strftime("%Y-%m-%d")
        _vprint(verbose, f"[run] Reconciling previous ET day: {y_str}")
        t2 = time.perf_counter()
        try:
            # First archive finals/outcomes to ensure predictions CSV has settled fields
            archive_finals_for_date(y_str, verbose=verbose)
        except Exception:
            pass
        try:
            capture_closing_for_date(y_str, verbose=verbose)
        except Exception:
            pass
        recon_games = reconcile_date(y_str, verbose=verbose, ev_threshold=recon_ev_threshold)
        # Also reconcile props for yesterday
        if not skip_props:
            try:
                recon_props = reconcile_props_date(y_str, verbose=verbose)
            except Exception:
                pass
            # Build per-day props vs actuals cache for web and commit
            try:
                build_player_props_vs_actuals(y_str, verbose=verbose)
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
    # 4) Optionally commit and push changes to git (code + tracked data updates)
    if git_push:
        try:
            # Determine repo root and current branch
            root = _ROOT if '_ROOT' in globals() else Path(__file__).resolve().parents[2]
            def _run(cmd: list[str]) -> subprocess.CompletedProcess:
                return subprocess.run(cmd, cwd=str(root), capture_output=True, text=True, check=False)
            # Ensure we are inside a git repository
            chk = _run(["git", "rev-parse", "--is-inside-work-tree"])
            if chk.returncode != 0 or (chk.stdout or '').strip() != 'true':
                _vprint(verbose, "[git] Not a git repository; skipping push.")
            else:
                # Stage all tracked/untracked (respects .gitignore)
                _run(["git", "add", "-A"]).stdout
                # Explicitly stage key processed artifacts we rely on in web (safety against ignore edge cases)
                try:
                    produced = globals().get("_DAILY_UPDATE_PRODUCED_ARTIFACTS", []) or []
                    for p in produced:
                        _run(["git", "add", p])
                except Exception:
                    pass
                # CRITICAL: Force-add essential CSV files that power the web app (predictions, edges, roster)
                # These files are core data files that MUST be in git for Render to serve them
                try:
                    for d in pred_dates:
                        # Predictions CSVs - game predictions with period breakdowns
                        pred_file = root / "data" / "processed" / f"predictions_{d}.csv"
                        if pred_file.exists():
                            _run(["git", "add", "-f", str(pred_file)])
                        # Edges CSVs - betting edges with EV calculations
                        edges_file = root / "data" / "processed" / f"edges_{d}.csv"
                        if edges_file.exists():
                            _run(["git", "add", "-f", str(edges_file)])
                        # Roster CSVs - player-team mappings
                        roster_file = root / "data" / "processed" / f"roster_{d}.csv"
                        if roster_file.exists():
                            _run(["git", "add", "-f", str(roster_file)])
                        # Props projections CSVs - NN player stat projections
                        props_proj_file = root / "data" / "processed" / f"props_projections_all_{d}.csv"
                        if props_proj_file.exists():
                            _run(["git", "add", "-f", str(props_proj_file)])
                    _vprint(verbose, f"[git] Force-added essential CSV files for: {', '.join(pred_dates)}")
                except Exception as e:
                    _vprint(verbose, f"[git] Warning: failed to force-add CSV files: {e}")
                # Skip commit if no changes
                status = _run(["git", "status", "--porcelain"]).stdout
                if status.strip():
                    # Get current branch if none provided
                    branch = git_branch
                    if not branch:
                        br = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
                        branch = br or "master"
                    # Compose message
                    now_et = _today_et().strftime('%Y-%m-%d %H:%M %Z')
                    msg = f"[auto] daily update: {', '.join(pred_dates)} @ {now_et}"
                    cm = _run(["git", "commit", "-m", msg])
                    if cm.returncode != 0:
                        _vprint(verbose, f"[git] Commit returned code {cm.returncode}: {cm.stderr.strip()}")
                    # Push
                    ps = _run(["git", "push", git_remote, branch])
                    if ps.returncode == 0:
                        print(f"[git] Pushed to {git_remote}/{branch}: {msg}")
                    else:
                        _vprint(verbose, f"[git] Push failed code {ps.returncode}: {ps.stderr.strip()}")
                else:
                    _vprint(verbose, "[git] No changes to commit.")
        except Exception as e:
            _vprint(verbose, f"[git] Skipped due to error: {e}")
    _vprint(verbose, "[run] Daily update complete.")


def reconcile_props_date(date: str, flat_stake: float = 100.0, verbose: bool = False) -> dict:
    """Reconcile previous day's props recommendations (ALL markets) with actual outcomes.

    Uses canonical recommendations_{date}.csv (generated from Parquet) across markets:
    SOG, GOALS, ASSISTS, POINTS, SAVES, BLOCKS. If recommendations are missing, attempts
    to build them on the fly for that date.

    Steps:
      - Ensure props recommendations file exists for the date
      - Ensure player boxscore stats for that date are collected
      - For each pick, compute actual vs line and settle based on chosen side
      - Compute PnL summary and persist reconciliation_{date}.json; append to props log CSV
    """
    # 1) Ensure recommendations exist for the date (build if missing)
    recs_path = PROC_DIR / f"props_recommendations_{date}.csv"
    if not recs_path.exists():
        try:
            from nhl_betting.cli import props_recommendations as _props_recs
            def _call_typer_or_func_recs(cmd, **kwargs):
                if hasattr(cmd, 'callback') and callable(getattr(cmd, 'callback')):
                    return cmd.callback(**kwargs)
                elif callable(cmd):
                    return cmd(**kwargs)
                else:
                    raise RuntimeError('Unsupported command object for props recommendations')
            _vprint(verbose, f"[props] Building recommendations for {date}…")
            _call_typer_or_func_recs(_props_recs, date=date, min_ev=0.0, top=200, market="")
        except Exception as e:
            return {"status": "no-recommendations", "date": date, "error": str(e)}
    try:
        recs = pd.read_csv(recs_path)
    except Exception:
        return {"status": "recs-read-failed", "date": date}
    if recs.empty:
        return {"status": "empty-recommendations", "date": date}
    # Only consider positive EV picks (either Over or Under)
    try:
        recs = recs[recs["ev"].astype(float) > 0]
    except Exception:
        pass
    if recs.empty:
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
    try:
        stats = pd.read_csv(stats_path)
    except Exception as e:
        return {"status": "stats-read-failed", "date": date, "error": str(e)}
    # Normalize to ET calendar day for robust matching
    def _to_et(s):
        try:
            dt = pd.to_datetime(s, utc=True)
            return dt.tz_convert("America/New_York").strftime("%Y-%m-%d")
        except Exception:
            try:
                # Fallback parse
                return pd.to_datetime(str(s)[:19]).tz_localize('UTC').tz_convert('America/New_York').strftime('%Y-%m-%d')
            except Exception:
                return None
    stats["date_et"] = stats["date"].apply(_to_et)
    stats = stats[stats["date_et"] == date]
    # Extract/normalize player names in stats for matching
    import ast, re, unicodedata
    def _extract_player_text(v):
        if v is None:
            return ""
        try:
            if isinstance(v, str) and v.strip().startswith("{") and v.strip().endswith("}"):
                d = ast.literal_eval(v)
                if isinstance(d, dict):
                    for k in ("default","en","name","fullName","full_name"):
                        if d.get(k):
                            return str(d.get(k))
            if isinstance(v, dict):
                for k in ("default","en","name","fullName","full_name"):
                    if v.get(k):
                        return str(v.get(k))
            return str(v)
        except Exception:
            return str(v)
    def _norm_name(s: str) -> str:
        s = (s or "").strip()
        s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode()
        s = re.sub(r"\s+"," ", s)
        return s.lower()
    stats["player_text_raw"] = stats["player"].apply(_extract_player_text)
    stats["player_norm"] = stats["player_text_raw"].apply(_norm_name)
    stats["player_nodot"] = stats["player_norm"].str.replace(".", "", regex=False)
    # Build picks and compute outcomes
    rows = []
    pnl = 0.0
    staked = 0.0
    wins = losses = pushes = 0
    decided = 0
    for _, r in recs.iterrows():
        market = str(r.get("market") or "").upper()
        player = str(r.get("player") or "")
        line = float(r.get("line")) if pd.notna(r.get("line")) else None
        side = str(r.get("side") or "")
        # Choose corresponding odds
        odds = None
        if side.lower() == "over":
            odds = float(r.get("over_price")) if pd.notna(r.get("over_price")) else None
        elif side.lower() == "under":
            odds = float(r.get("under_price")) if pd.notna(r.get("under_price")) else None
        if line is None or odds is None or not player:
            continue
        # Flexible player match
        def _variants(full: str):
            full = (full or "").strip()
            parts = [p for p in full.split(" ") if p]
            vs = set()
            if full:
                n = _norm_name(full)
                vs.add(n)
                vs.add(n.replace(".", ""))
            if len(parts) >= 2:
                first, last = parts[0], parts[-1]
                init_last = f"{first[0]}. {last}"
                n2 = _norm_name(init_last)
                vs.add(n2)
                vs.add(n2.replace(".", ""))
            return vs
        vs = _variants(player)
        ps = stats[(stats["player_norm"].isin(vs)) | (stats["player_nodot"].isin(vs))]
        actual = None
        if not ps.empty:
            row = ps.iloc[0]
            if market == "SOG":
                actual = row.get("shots")
            elif market == "GOALS":
                actual = row.get("goals")
            elif market == "ASSISTS":
                actual = row.get("assists")
            elif market == "POINTS":
                try:
                    actual = float((row.get("goals") or 0)) + float((row.get("assists") or 0))
                except Exception:
                    actual = None
            elif market == "SAVES":
                actual = row.get("saves")
            elif market == "BLOCKS":
                actual = row.get("blocked")
        result = None
        if actual is not None and pd.notna(actual):
            try:
                av = float(actual)
                # Determine outcome relative to chosen side
                if av == float(line):
                    result = "push"
                else:
                    over_res = (av > float(line))
                    result = "win" if ((side.lower()=="over" and over_res) or (side.lower()=="under" and not over_res)) else "loss"
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
            "side": side,
            "odds": odds,
            "ev": float(r.get("ev")) if pd.notna(r.get("ev")) else None,
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
    ap.add_argument("--recon-ev-threshold", type=float, default=0.0, help="Minimum EV required to include a pick in reconciliation (0.0 keeps previous behavior)")
    # Git auto-push controls (enabled by default)
    ap.add_argument("--no-git-push", action="store_true", help="Disable final git commit/push step")
    ap.add_argument("--git-remote", type=str, default="origin", help="Remote name to push to (default: origin)")
    ap.add_argument("--git-branch", type=str, default=None, help="Branch to push (default: current branch)")
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
        git_push=(not args.no_git_push),
        git_remote=args.git_remote,
        git_branch=args.git_branch,
        recon_ev_threshold=args.recon_ev_threshold)
    

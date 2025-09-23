from __future__ import annotations

import argparse
import json
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


def _today_et() -> datetime:
    try:
        return datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        return datetime.now(timezone.utc)


def _ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def update_models_history_window(years_back: int = 2) -> None:
    """Fetch recent seasons' schedule/results, overwrite games.csv, featurize and train.

    By default, fetch from Sep 1 of (current_year - years_back) through Aug 1 of (current_year + 1).
    """
    now = datetime.now(timezone.utc)
    start = f"{now.year - years_back}-09-01"
    end = f"{now.year + 1}-08-01"
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
    # Rebuild features and retrain Elo/base_mu
    featurize()
    train()


def make_predictions(days_ahead: int = 2) -> None:
    # Only generate for ET today and ET tomorrow (days_ahead default=2)
    base = _today_et().astimezone(timezone.utc)  # drive by calendar day; game dates are ISO UTC in predictions
    for i in range(0, min(2, max(1, days_ahead))):
        d = _ymd(base + timedelta(days=i))
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Try Bovada first
        try:
            predict_core(date=d, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
        except Exception:
            pass
        # Fallback to Odds API (DK preferred)
        try:
            predict_core(date=d, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
        except Exception:
            pass
        # Ensure file exists even without odds
        try:
            predict_core(date=d, source="web", odds_source="csv")
        except Exception:
            pass


def _team_abbr(name: str) -> str:
    try:
        from nhl_betting.web.teams import get_team_assets as _assets
        return (_assets(str(name)).get("abbr") or "").upper()
    except Exception:
        return ""


def capture_closing_for_date(date: str, prefer_book: str | None = None, best_of_all: bool = True) -> dict:
    """Capture pre-game closing odds for each matchup on a date and persist into predictions_{date}.csv.

    Strategy: for each game row in predictions_{date}.csv, query The Odds API historical snapshot
    at the game's commence time (UTC). Use best-of-all across books by default to maximize coverage
    and store prices into close_* columns if not already set (first-write wins).
    """
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return {"status": "no-file", "date": date}
    df = pd.read_csv(path)
    if df.empty:
        return {"status": "empty", "date": date}
    try:
        client = OddsAPIClient()
    except Exception as e:
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
    return {"status": "ok", "date": date, "updated": updated}


def reconcile_date(date: str, bankroll: float = 1000.0, flat_stake: float = 100.0) -> dict:
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


def run(days_ahead: int = 2, years_back: int = 2, reconcile_yesterday: bool = True) -> None:
    # 1) Update models (features + Elo) from recent seasons including latest finals
    update_models_history_window(years_back=years_back)
    # 2) Generate predictions for upcoming days
    make_predictions(days_ahead=min(2, days_ahead))
    # 3) Capture closings for yesterday's ET slate and reconcile
    if reconcile_yesterday:
        y_et = _today_et().date() - timedelta(days=1)
        y_str = y_et.strftime("%Y-%m-%d")
        try:
            capture_closing_for_date(y_str)
        except Exception:
            pass
        reconcile_date(y_str)
        # Also reconcile props for yesterday
        try:
            reconcile_props_date(y_str)
        except Exception:
            pass


def reconcile_props_date(date: str, flat_stake: float = 100.0) -> dict:
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
    except Exception:
        # If fetch fails, continue only if odds file already exists
        pass
    preds_tmp = PROC_DIR / "props_predictions.csv"
    if not preds_tmp.exists():
        try:
            _props_predict.callback(odds_csv=odds_csv)
        except Exception:
            # If prediction still missing, bail gracefully
            return {"status": "no-props-predictions", "date": date}
    else:
        # Rebuild predictions using fresh odds file when possible
        try:
            _props_predict.callback(odds_csv=odds_csv)
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
    args = ap.parse_args()
    run(days_ahead=args.days_ahead, years_back=args.years_back, reconcile_yesterday=(not args.no_reconcile))

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from nhl_betting.cli import predict_core, featurize, train
from nhl_betting.data.nhl_api_web import NHLWebClient
from nhl_betting.utils.io import RAW_DIR, PROC_DIR, save_df


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
    base = _today_et().astimezone(timezone.utc)  # drive by calendar day; game dates are ISO UTC in predictions
    for i in range(0, max(1, days_ahead)):
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
    return {"status": "ok", **summary}


def run(days_ahead: int = 2, years_back: int = 2, reconcile_yesterday: bool = True) -> None:
    # 1) Update models (features + Elo) from recent seasons including latest finals
    update_models_history_window(years_back=years_back)
    # 2) Generate predictions for upcoming days
    make_predictions(days_ahead=days_ahead)
    # 3) Reconcile yesterday's ET slate
    if reconcile_yesterday:
        y_et = _today_et().date() - timedelta(days=1)
        reconcile_date(y_et.strftime("%Y-%m-%d"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Daily update: refresh models, predictions, and reconciliation.")
    ap.add_argument("--days-ahead", type=int, default=2, help="How many days of predictions to generate starting today (ET)")
    ap.add_argument("--years-back", type=int, default=2, help="How many years back to include when rebuilding models (by season start)")
    ap.add_argument("--no-reconcile", action="store_true", help="Skip reconciliation step")
    args = ap.parse_args()
    run(days_ahead=args.days_ahead, years_back=args.years_back, reconcile_yesterday=(not args.no_reconcile))

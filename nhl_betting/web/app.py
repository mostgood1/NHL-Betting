from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..utils.io import RAW_DIR, PROC_DIR
from ..utils.io import MODEL_DIR as _MODEL_DIR
from ..data.nhl_api_web import NHLWebClient
from .teams import get_team_assets
from ..cli import predict_core, fetch as cli_fetch, train as cli_train
import asyncio

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="NHL Betting")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html"]))


def _today_ymd() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _has_any_odds_df(df: pd.DataFrame) -> bool:
    try:
        if df is None or df.empty:
            return False
        cols = [
            "home_ml_odds",
            "away_ml_odds",
            "over_odds",
            "under_odds",
            "home_pl_-1.5_odds",
            "away_pl_+1.5_odds",
        ]
        present_cols = [c for c in cols if c in df.columns]
        if not present_cols:
            return False
        return any(df[c].notna().any() for c in present_cols)
    except Exception:
        return False


@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.on_event("startup")
async def _bootstrap_models_if_missing():
    # Ensure Elo ratings and config exist; if missing, fetch ~two seasons and train.
    try:
        from ..utils.io import MODEL_DIR
        ratings_path = MODEL_DIR / "elo_ratings.json"
        cfg_path = MODEL_DIR / "config.json"
        if ratings_path.exists() and cfg_path.exists():
            return
        # Build a two-season window ending last season end (Aug 1 current year)
        now = datetime.now(timezone.utc)
        end = f"{now.year}-08-01"
        start_year = now.year - 2
        start = f"{start_year}-09-01"

        async def _do_bootstrap():
            try:
                await asyncio.to_thread(cli_fetch, start, end, "web")
                await asyncio.to_thread(cli_train)
            except Exception:
                # Ignore failures; the app can still serve with on-demand prediction
                pass

        # Schedule in background so startup isn't blocked on Render
        asyncio.create_task(_do_bootstrap())
    except Exception:
        # Don't block startup if bootstrap scheduling fails
        pass


@app.get("/")
async def cards(date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD")):
    date = date or _today_ymd()
    note_msg = None
    # Ensure models exist (Elo/config); if missing, do a quick bootstrap inline
    try:
        ratings_path = _MODEL_DIR / "elo_ratings.json"
        cfg_path = _MODEL_DIR / "config.json"
        if not ratings_path.exists() or not cfg_path.exists():
            # Build a one-season window to be faster
            now = datetime.now(timezone.utc)
            start = f"{now.year-1}-09-01"
            end = f"{now.year}-08-01"
            try:
                await asyncio.to_thread(cli_fetch, start, end, "web")
                await asyncio.to_thread(cli_train)
            except Exception:
                pass
    except Exception:
        pass
    # Ensure we have predictions for the date; run inline if missing
    pred_path = PROC_DIR / f"predictions_{date}.csv"
    if not pred_path.exists():
        # Attempt Bovada first
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            predict_core(date=date, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
        except Exception:
            pass
        # Fallback to Odds API if no odds captured
        if pred_path.exists():
            try:
                tmp = pd.read_csv(pred_path)
            except Exception:
                tmp = pd.DataFrame()
            if not _has_any_odds_df(tmp):
                try:
                    predict_core(date=date, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
                except Exception:
                    pass
    df = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    # If predictions exist but odds are missing, try Bovada then Odds API to populate
    if pred_path.exists() and not _has_any_odds_df(df):
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            predict_core(date=date, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
            df = pd.read_csv(pred_path)
        except Exception:
            pass
        if not _has_any_odds_df(df):
            try:
                predict_core(date=date, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
                df = pd.read_csv(pred_path)
            except Exception:
                pass
    # If no games for requested date, try to find the next available slate within 10 days
    if df.empty:
        try:
            client = NHLWebClient()
            from datetime import timedelta
            base = pd.to_datetime(date)
            for i in range(1, 11):
                d2 = (base + timedelta(days=i)).strftime("%Y-%m-%d")
                games = client.schedule_range(d2, d2)
                # Filter to known NHL teams using assets
                elig = []
                for g in games:
                    try:
                        h_ok = bool(get_team_assets(str(getattr(g, "home", "")).strip()).get("abbr"))
                        a_ok = bool(get_team_assets(str(getattr(g, "away", "")).strip()).get("abbr"))
                        if h_ok and a_ok:
                            elig.append(g)
                    except Exception:
                        pass
                if elig:
                    # Generate predictions for this next slate
                    snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    try:
                        predict_core(date=d2, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
                    except Exception:
                        pass
                    alt_path = PROC_DIR / f"predictions_{d2}.csv"
                    if alt_path.exists():
                        try:
                            df2 = pd.read_csv(alt_path)
                        except Exception:
                            df2 = pd.DataFrame()
                        if not df2.empty:
                            df = df2
                            note_msg = f"No games on {date}. Showing next slate on {d2}."
                            date = d2
                            break
        except Exception:
            pass
    rows = df.to_dict(orient="records") if not df.empty else []
    # Attach team assets; convert UTC to local time string
    def to_local(iso_utc: str) -> str:
        try:
            ts = pd.to_datetime(iso_utc, utc=True)
            # Convert to system local timezone via tzlocal None, then append TZ offset
            local_ts = ts.tz_convert(None)
            # Show local date/time; we cannot reliably get TZ abbrev without pytz/zoneinfo in this context
            return local_ts.strftime("%Y-%m-%d %I:%M %p")
        except Exception:
            return iso_utc
    for r in rows:
        h = get_team_assets(str(r.get("home", "")))
        a = get_team_assets(str(r.get("away", "")))
        r["home_abbr"] = h.get("abbr")
        r["home_logo"] = h.get("logo_dark") or h.get("logo_light")
        r["away_abbr"] = a.get("abbr")
        r["away_logo"] = a.get("logo_dark") or a.get("logo_light")
        if r.get("date"):
            r["local_time"] = to_local(r["date"])

    template = env.get_template("cards.html")
    html = template.render(date=date, rows=rows, note=note_msg)
    return HTMLResponse(content=html)


@app.get("/api/predictions")
async def api_predictions(date: Optional[str] = Query(None)):
    date = date or _today_ymd()
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return JSONResponse({"error": "No predictions for date", "date": date}, status_code=404)
    df = pd.read_csv(path)
    return JSONResponse(df.to_dict(orient="records"))

@app.get("/api/props")
async def api_props(
    market: Optional[str] = Query(None, description="Filter by market: SOG, SAVES, GOALS"),
    min_ev: float = Query(0.0, description="Minimum EV threshold for ev_over"),
    top: int = Query(50, description="Top N to return after filtering/sorting by EV desc"),
):
    path = PROC_DIR / "props_predictions.csv"
    if not path.exists():
        return JSONResponse({"error": "No props predictions found. Generate props_predictions.csv with the CLI."}, status_code=404)
    df = pd.read_csv(path)
    if market:
        df = df[df["market"].str.upper() == market.upper()]
    if "ev_over" in df.columns:
        df = df[df["ev_over"].astype(float) >= float(min_ev)]
        df = df.sort_values("ev_over", ascending=False)
    if top and top > 0:
        df = df.head(top)
    return JSONResponse(df.to_dict(orient="records"))

@app.get("/api/last-updated")
async def api_last_updated(date: Optional[str] = Query(None)):
    date = date or _today_ymd()
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return JSONResponse({"date": date, "last_modified": None})
    try:
        import os, datetime as _dt
        ts = _dt.datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        return JSONResponse({"date": date, "last_modified": ts.isoformat()})
    except Exception:
        return JSONResponse({"date": date, "last_modified": None})

@app.get("/props")
async def props_page(
    market: Optional[str] = Query(None, description="Filter by market: SOG, SAVES, GOALS"),
    min_ev: float = Query(0.0, description="Minimum EV threshold for ev_over"),
    top: int = Query(50, description="Top N to display"),
):
    # Reuse API logic
    resp = await api_props(market=market, min_ev=min_ev, top=top)
    rows = []
    if isinstance(resp, JSONResponse):
        try:
            import json as _json
            rows = _json.loads(resp.body)
        except Exception:
            rows = []
    template = env.get_template("props.html")
    html = template.render(rows=rows, market=market or "All", min_ev=min_ev, top=top)
    return HTMLResponse(content=html)


@app.get("/api/edges")
async def api_edges(date: Optional[str] = Query(None)):
    date = date or _today_ymd()
    path = PROC_DIR / f"edges_{date}.csv"
    if not path.exists():
        return JSONResponse([], status_code=200)
    df = pd.read_csv(path)
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/api/refresh-odds")
async def api_refresh_odds(
    date: Optional[str] = Query(None),
    snapshot: Optional[str] = Query(None),
    bankroll: float = Query(0.0, description="Bankroll for Kelly sizing; 0 disables"),
    kelly_fraction_part: float = Query(0.5, description="Kelly fraction, e.g., 0.5 for half-Kelly"),
):
    date = date or _today_ymd()
    if not snapshot:
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Try Bovada first; fall back to Odds API if no odds
    try:
        predict_core(
            date=date,
            source="web",
            odds_source="bovada",
            snapshot=snapshot,
            odds_best=True,
            bankroll=bankroll,
            kelly_fraction_part=kelly_fraction_part,
        )
    except Exception:
        pass
    # Check whether odds present; if not, attempt Odds API fallback
    try:
        df = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
    except Exception:
        df = pd.DataFrame()
    if not _has_any_odds_df(df):
        try:
            predict_core(
                date=date,
                source="web",
                odds_source="oddsapi",
                snapshot=snapshot,
                odds_best=False,
                odds_bookmaker="draftkings",
                bankroll=bankroll,
                kelly_fraction_part=kelly_fraction_part,
            )
        except Exception as e:
            # Return ok even if odds fallback fails; frontend will still render predictions
            return JSONResponse({"status": "partial", "message": "Bovada failed and Odds API fallback failed; predictions updated without odds.", "date": date}, status_code=200)
    return {"status": "ok", "date": date, "snapshot": snapshot, "bankroll": bankroll, "kelly_fraction_part": kelly_fraction_part}


@app.get("/api/recommendations")
async def api_recommendations(
    date: Optional[str] = Query(None),
    min_ev: float = Query(0.0, description="Minimum EV threshold to include"),
    top: int = Query(20, description="Top N recommendations to return"),
    markets: str = Query("all", description="Comma-separated filters: moneyline,totals,puckline"),
    bankroll: float = Query(0.0, description="If > 0, compute Kelly stake using provided bankroll"),
    kelly_fraction_part: float = Query(0.5, description="Kelly fraction; used only if bankroll>0"),
):
    date = date or _today_ymd()
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return JSONResponse({"error": "No predictions for date", "date": date}, status_code=404)
    df = pd.read_csv(path)

    # Helper to add a rec if EV present and above threshold
    recs = []
    def add_rec(row: pd.Series, market_key: str, label: str, prob_key: str, ev_key: str, edge_key: str, odds_key: str, book_key: Optional[str] = None):
        if ev_key in row and pd.notna(row[ev_key]) and float(row[ev_key]) >= min_ev:
            rec = {
                "date": row.get("date"),
                "home": row.get("home"),
                "away": row.get("away"),
                "market": market_key,
                "bet": label,
                "model_prob": float(row.get(prob_key)) if prob_key in row and pd.notna(row.get(prob_key)) else None,
                "ev": float(row.get(ev_key)),
                "edge": float(row.get(edge_key)) if edge_key in row and pd.notna(row.get(edge_key)) else None,
                "price": float(row.get(odds_key)) if odds_key in row and pd.notna(row.get(odds_key)) else None,
                "book": row.get(book_key) if book_key and (book_key in row) and pd.notna(row.get(book_key)) else None,
                "total_line_used": float(row.get("total_line_used")) if "total_line_used" in row and pd.notna(row.get("total_line_used")) else None,
            }
            # Result mapping (if actuals exist)
            res = None
            try:
                if market_key == "moneyline" and isinstance(rec.get("bet"), str):
                    winner_actual = row.get("winner_actual")
                    if winner_actual:
                        if rec["bet"] == "home_ml":
                            res = "Win" if winner_actual == row.get("home") else "Loss"
                        elif rec["bet"] == "away_ml":
                            res = "Win" if winner_actual == row.get("away") else "Loss"
                elif market_key == "totals" and isinstance(rec.get("bet"), str):
                    rt = row.get("result_total")
                    if rt:
                        want = "Over" if rec["bet"].lower() == "over" else "Under"
                        if rt == "Push":
                            res = "Push"
                        else:
                            res = "Win" if rt == want else "Loss"
                elif market_key == "puckline" and isinstance(rec.get("bet"), str):
                    ra = row.get("result_ats")
                    if ra:
                        want = rec["bet"]  # matches 'home_pl_-1.5' or 'away_pl_+1.5'
                        res = "Win" if ra == want else "Loss"
            except Exception:
                res = None
            if res:
                rec["result"] = res
            # Compute stake if bankroll provided and we have prob+odds
            if bankroll > 0 and rec["model_prob"] is not None and rec["price"] is not None:
                # Kelly uses decimal odds; our price stored is American. Convert from American to decimal.
                from ..utils.odds import american_to_decimal, kelly_stake
                try:
                    dec = american_to_decimal(rec["price"])
                    rec["stake"] = round(kelly_stake(rec["model_prob"], dec, bankroll, kelly_fraction_part), 2)
                except Exception:
                    rec["stake"] = None
            recs.append(rec)

    # Market filters
    f_markets = set([m.strip().lower() for m in markets.split(",")]) if markets and markets != "all" else {"moneyline", "totals", "puckline"}

    for _, r in df.iterrows():
        # Moneyline
        if "moneyline" in f_markets:
            add_rec(r, "moneyline", "home_ml", "p_home_ml", "ev_home_ml", "edge_home_ml", "home_ml_odds", "home_ml_book")
            add_rec(r, "moneyline", "away_ml", "p_away_ml", "ev_away_ml", "edge_away_ml", "away_ml_odds", "away_ml_book")
        # Totals
        if "totals" in f_markets:
            add_rec(r, "totals", "over", "p_over", "ev_over", "edge_over", "over_odds", "over_book")
            add_rec(r, "totals", "under", "p_under", "ev_under", "edge_under", "under_book")
        # Puck line
        if "puckline" in f_markets:
            add_rec(r, "puckline", "home_pl_-1.5", "p_home_pl_-1.5", "ev_home_pl_-1.5", "edge_home_pl_-1.5", "home_pl_-1.5_odds", "home_pl_-1.5_book")
            add_rec(r, "puckline", "away_pl_+1.5", "p_away_pl_+1.5", "ev_away_pl_+1.5", "edge_away_pl_+1.5", "away_pl_+1.5_odds", "away_pl_+1.5_book")

    # Sort by EV and take top N
    recs_sorted = sorted(recs, key=lambda x: x["ev"], reverse=True)[: top if top and top > 0 else len(recs)]
    return JSONResponse(recs_sorted)


@app.get("/recommendations")
async def recommendations(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD"),
    min_ev: float = Query(0.0, description="Minimum EV threshold to include"),
    top: int = Query(20, description="Top N recommendations to show"),
    markets: str = Query("all", description="Comma-separated filters: moneyline,totals,puckline"),
    bankroll: float = Query(0.0, description="If > 0, show Kelly stake using provided bankroll"),
    kelly_fraction_part: float = Query(0.5, description="Kelly fraction; used only if bankroll>0"),
    high_ev: float = Query(0.05, description="EV threshold for High confidence grouping (e.g., 0.05 for 5%)"),
):
    date = date or _today_ymd()
    # Ensure predictions exist
    pred_path = PROC_DIR / f"predictions_{date}.csv"
    if not pred_path.exists():
        snapshot = datetime.now(timezone.utc).replace(hour=18, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            predict_core(date=date, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=True, bankroll=bankroll, kelly_fraction_part=kelly_fraction_part)
        except Exception:
            pass
    # Build recommendations via API to share logic
    recs = await api_recommendations(date=date, min_ev=min_ev, top=top, markets=markets, bankroll=bankroll, kelly_fraction_part=kelly_fraction_part)
    data = recs.body  # JSONResponse
    try:
        import json as _json
        rows = _json.loads(data)
    except Exception:
        rows = []
    # Compute confidence groupings (NFL-style): High (ev>=high_ev), Low (0<=ev<high_ev), Other (ev<0)
    EV_HIGH = float(high_ev)
    def group_row(r):
        try:
            ev = float(r.get("ev"))
        except Exception:
            ev = -999
        if ev >= EV_HIGH:
            return "high"
        elif ev >= 0:
            return "low"
        else:
            return "other"
    rows_high = [r for r in rows if group_row(r) == "high"]
    rows_low = [r for r in rows if group_row(r) == "low"]
    rows_other = [r for r in rows if group_row(r) == "other"]
    # Sort within groups by EV desc
    rows_high.sort(key=lambda x: x.get("ev", 0), reverse=True)
    rows_low.sort(key=lambda x: x.get("ev", 0), reverse=True)
    rows_other.sort(key=lambda x: x.get("ev", 0), reverse=True)
    # Summary metrics (overall and per-group)
    def american_to_decimal_local(american):
        try:
            a = float(american)
        except Exception:
            return None
        if a > 0:
            return 1.0 + (a / 100.0)
        else:
            return 1.0 + (100.0 / abs(a))
    def compute_summary(subrows):
        wins = losses = pushes = 0
        staked = 0.0
        pnl = 0.0
        decided = 0
        for r in subrows:
            res = (r.get("result") or "").lower()
            # Determine stake
            stake = None
            try:
                if bankroll and float(bankroll) > 0 and r.get("stake") is not None:
                    stake = float(r.get("stake"))
            except Exception:
                stake = None
            if stake is None:
                stake = 100.0  # flat stake assumption
            # Determine price; fallback -110 for spreads/totals when missing
            price = r.get("price")
            if price is None and r.get("market") in ("totals", "puckline"):
                price = -110
            if price is None and r.get("market") == "moneyline":
                price = -110
            dec = american_to_decimal_local(price) if price is not None else None
            if res in ("win", "loss", "push"):
                if res == "win":
                    wins += 1
                    if dec:
                        pnl += stake * (dec - 1.0)
                elif res == "loss":
                    losses += 1
                    pnl -= stake
                else:
                    pushes += 1
                staked += stake
        decided = wins + losses
        acc = (wins / decided) if decided > 0 else None
        roi = (pnl / staked) if staked > 0 else None
        return {
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "picks": len(subrows),
            "accuracy": acc,
            "stake": staked,
            "pnl": pnl,
            "roi": roi,
        }
    summary_overall = compute_summary(rows)
    summary_high = compute_summary(rows_high)
    summary_low = compute_summary(rows_low)
    summary_other = compute_summary(rows_other)
    # Market counts for top bar (based on displayed rows)
    counts = {"moneyline": 0, "totals": 0, "puckline": 0}
    for r in rows:
        m = (r.get("market") or "").lower()
        if m in counts:
            counts[m] += 1
    template = env.get_template("recommendations.html")
    html = template.render(
        date=date,
        rows=rows,
        rows_high=rows_high,
        rows_low=rows_low,
        rows_other=rows_other,
        summary_overall=summary_overall,
        summary_high=summary_high,
        summary_low=summary_low,
        summary_other=summary_other,
        counts=counts,
        total_picks=len(rows),
        min_ev=min_ev,
        top=top,
        markets=markets,
        bankroll=bankroll,
        kelly_fraction_part=kelly_fraction_part,
        high_ev=high_ev,
    )
    return HTMLResponse(content=html)


@app.get("/api/odds-coverage")
async def api_odds_coverage(date: Optional[str] = Query(None)):
    date = date or _today_ymd()
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return JSONResponse({"error": "No predictions for date", "date": date}, status_code=404)
    df = pd.read_csv(path)
    rows = []
    ml_count = 0
    totals_count = 0
    pl_count = 0
    for _, r in df.iterrows():
        has_ml = pd.notna(r.get("home_ml_odds")) and pd.notna(r.get("away_ml_odds"))
        has_totals = pd.notna(r.get("over_odds")) and pd.notna(r.get("under_odds"))
        has_pl = pd.notna(r.get("home_pl_-1.5_odds")) and pd.notna(r.get("away_pl_+1.5_odds"))
        ml_count += 1 if has_ml else 0
        totals_count += 1 if has_totals else 0
        pl_count += 1 if has_pl else 0
        ml_books = list({
            r.get("home_ml_book") if pd.notna(r.get("home_ml_book")) else None,
            r.get("away_ml_book") if pd.notna(r.get("away_ml_book")) else None,
        } - {None})
        totals_books = list({
            r.get("over_book") if pd.notna(r.get("over_book")) else None,
            r.get("under_book") if pd.notna(r.get("under_book")) else None,
        } - {None})
        pl_books = list({
            r.get("home_pl_-1.5_book") if pd.notna(r.get("home_pl_-1.5_book")) else None,
            r.get("away_pl_+1.5_book") if pd.notna(r.get("away_pl_+1.5_book")) else None,
        } - {None})
        rows.append({
            "date": r.get("date"),
            "home": r.get("home"),
            "away": r.get("away"),
            "has_moneyline": has_ml,
            "has_totals": has_totals,
            "has_puckline": has_pl,
            "ml_books": ml_books,
            "totals_books": totals_books,
            "puckline_books": pl_books,
        })
    summary = {
        "date": date,
        "games": int(len(df)),
        "moneyline_covered": int(ml_count),
        "totals_covered": int(totals_count),
        "puckline_covered": int(pl_count),
    }
    return JSONResponse({"summary": summary, "rows": rows})


@app.get("/odds-coverage")
async def odds_coverage(date: Optional[str] = Query(None)):
    date = date or _today_ymd()
    resp = await api_odds_coverage(date=date)
    payload = {}
    if isinstance(resp, JSONResponse):
        try:
            import json as _json
            payload = _json.loads(resp.body)
        except Exception:
            payload = {"summary": {"date": date}, "rows": []}
    template = env.get_template("odds_coverage.html")
    html = template.render(summary=payload.get("summary", {}), rows=payload.get("rows", []))
    return HTMLResponse(content=html)

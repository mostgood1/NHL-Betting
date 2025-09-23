from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
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
from ..data.nhl_api import NHLClient as NHLStatsClient
from .teams import get_team_assets
from ..cli import predict_core, fetch as cli_fetch, train as cli_train
import asyncio
from ..data.bovada import BovadaClient

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="NHL Betting")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html"]))


def _today_ymd() -> str:
    """Return today's date in US/Eastern to align the slate with 'tonight'."""
    try:
        et = ZoneInfo("America/New_York")
        return datetime.now(et).strftime("%Y-%m-%d")
    except Exception:
        # Fallback to UTC if zoneinfo not available
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _iso_to_et_date(iso_utc: str) -> str:
    """Convert an ISO UTC timestamp (e.g., 2025-09-22T23:00:00Z) to an ET YYYY-MM-DD date string."""
    if not iso_utc:
        return ""
    try:
        s = str(iso_utc).replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(s)
        et = ZoneInfo("America/New_York")
        dt_et = dt_utc.astimezone(et)
        return dt_et.strftime("%Y-%m-%d")
    except Exception:
        try:
            # Best-effort fallback: treat as UTC naive
            dt_utc = datetime.fromisoformat(str(iso_utc)[:19])
            et = ZoneInfo("America/New_York")
            dt_et = dt_utc.replace(tzinfo=timezone.utc).astimezone(et)
            return dt_et.strftime("%Y-%m-%d")
        except Exception:
            return ""


def _is_live_day(date: str) -> bool:
    """Return True if any game for the date is currently LIVE/in progress.

    Uses the NHL Web API scoreboard; treats states containing LIVE/IN/PROGRESS as live.
    """
    try:
        client = NHLWebClient()
        rows = client.scoreboard_day(date)
        for r in rows:
            st = str(r.get("gameState") or "").upper()
            if any(k in st for k in ["LIVE", "IN", "PROGRESS", "CRIT"]):
                return True
    except Exception:
        pass
    return False


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


def _merge_preserve_odds(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    """Fill any missing odds/book fields in df_new from df_old by matching games.

    Match on date (YYYY-MM-DD) and normalized home/away names. Only fills when df_new is NaN/null
    and df_old has a value. Returns a new DataFrame (does not mutate inputs).
    """
    if df_new is None or df_new.empty:
        return df_new
    if df_old is None or df_old.empty:
        return df_new
    def norm_team(s: str) -> str:
        import re, unicodedata
        if s is None:
            return ""
        s = str(s)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s
    def date_key(x) -> str:
        try:
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        except Exception:
            return None
    # Build lookup from old
    old_idx = {}
    for _, r in df_old.iterrows():
        k = (date_key(r.get("date")), norm_team(r.get("home")), norm_team(r.get("away")))
        old_idx[k] = r
    # Columns to preserve
    cand_cols = [
        "home_ml_odds","away_ml_odds","over_odds","under_odds","home_pl_-1.5_odds","away_pl_+1.5_odds",
        "home_ml_book","away_ml_book","over_book","under_book","home_pl_-1.5_book","away_pl_+1.5_book",
        "total_line_used","total_line",
    ]
    cols = [c for c in cand_cols if c in df_new.columns or c in df_old.columns]
    rows = []
    for _, r in df_new.iterrows():
        k = (date_key(r.get("date")), norm_team(r.get("home")), norm_team(r.get("away")))
        if k in old_idx:
            ro = old_idx[k]
            for c in cols:
                # If new missing and old present, fill
                new_has = (c in r and pd.notna(r.get(c)))
                old_has = (c in ro and pd.notna(ro.get(c)))
                if (not new_has) and old_has:
                    r[c] = ro.get(c)
        rows.append(r)
    return pd.DataFrame(rows, columns=df_new.columns)


def _capture_closing_for_game(date: str, home_abbr: str, away_abbr: str, snapshot: Optional[str] = None) -> dict:
    """Persist first-seen 'closing' odds into predictions_{date}.csv for reconciliation.

    We match the row by team abbreviations; then copy current odds fields into close_* columns
    if they are missing. Returns a small status dict.
    """
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return {"status": "no-file", "date": date}
    df = pd.read_csv(path)
    if df.empty:
        return {"status": "empty", "date": date}
    from .teams import get_team_assets as _assets
    def to_abbr(x):
        try:
            return (_assets(str(x)).get("abbr") or "").upper()
        except Exception:
            return ""
    # Build mask
    m = (df.apply(lambda r: to_abbr(r.get("home")) == (home_abbr or "").upper() and to_abbr(r.get("away")) == (away_abbr or "").upper(), axis=1))
    if not m.any():
        return {"status": "not-found", "home_abbr": home_abbr, "away_abbr": away_abbr}
    idx = df.index[m][0]
    # Ensure close_* columns exist
    def ensure(col):
        if col not in df.columns:
            df[col] = pd.NA
    closing_cols = [
        "close_home_ml_odds","close_away_ml_odds","close_over_odds","close_under_odds",
        "close_home_pl_-1.5_odds","close_away_pl_+1.5_odds","close_total_line_used",
        "close_home_ml_book","close_away_ml_book","close_over_book","close_under_book",
        "close_home_pl_-1.5_book","close_away_pl_+1.5_book","close_snapshot",
    ]
    for c in closing_cols:
        ensure(c)
    # Helper to set first
    def set_first(dst_col, src_col):
        try:
            cur = df.at[idx, dst_col]
            if pd.isna(cur) or cur is None:
                if src_col in df.columns and pd.notna(df.at[idx, src_col]):
                    df.at[idx, dst_col] = df.at[idx, src_col]
        except Exception:
            pass
    set_first("close_home_ml_odds", "home_ml_odds")
    set_first("close_away_ml_odds", "away_ml_odds")
    set_first("close_over_odds", "over_odds")
    set_first("close_under_odds", "under_odds")
    set_first("close_home_pl_-1.5_odds", "home_pl_-1.5_odds")
    set_first("close_away_pl_+1.5_odds", "away_pl_+1.5_odds")
    set_first("close_total_line_used", "total_line_used")
    set_first("close_home_ml_book", "home_ml_book")
    set_first("close_away_ml_book", "away_ml_book")
    set_first("close_over_book", "over_book")
    set_first("close_under_book", "under_book")
    set_first("close_home_pl_-1.5_book", "home_pl_-1.5_book")
    set_first("close_away_pl_+1.5_book", "away_pl_+1.5_book")
    # snapshot
    try:
        if pd.isna(df.at[idx, "close_snapshot"]) or df.at[idx, "close_snapshot"] is None:
            df.at[idx, "close_snapshot"] = snapshot or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass
    # Persist
    df.to_csv(path, index=False)
    return {"status": "ok", "date": date, "home_abbr": home_abbr, "away_abbr": away_abbr}


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


async def _ensure_models(quick: bool = False) -> None:
    """
    Ensure Elo ratings and config exist. If missing, fetch schedule and train.
    Tries multiple sources to avoid preseason/offseason gaps.
    quick=True limits to ~1 season for speed; otherwise ~2 seasons.
    """
    try:
        ratings_path = _MODEL_DIR / "elo_ratings.json"
        cfg_path = _MODEL_DIR / "config.json"
        if ratings_path.exists() and cfg_path.exists():
            return
        now = datetime.now(timezone.utc)
        if quick:
            start = f"{now.year-1}-09-01"
            end = f"{now.year}-08-01"
        else:
            start = f"{now.year-2}-09-01"
            end = f"{now.year}-08-01"
        # Try WEB source first
        try:
            await asyncio.to_thread(cli_fetch, start, end, "web")
        except Exception:
            pass
        # If RAW games seems empty or ratings still missing after training, try STATS as fallback
        try:
            await asyncio.to_thread(cli_train)
        except Exception:
            pass
        if not ratings_path.exists() or not cfg_path.exists():
            try:
                await asyncio.to_thread(cli_fetch, start, end, "stats")
                await asyncio.to_thread(cli_train)
            except Exception:
                pass
    except Exception:
        # Silent failure; callers may try again
        pass

@app.get("/")
async def cards(date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD")):
    date = date or _today_ymd()
    note_msg = None
    live_now = _is_live_day(date)
    # Consider a slate 'settled' if it is strictly before today's ET date (independent of live scoreboard noise)
    try:
        et_today = _today_ymd()
        settled = (str(date) < str(et_today))
    except Exception:
        settled = False
    if settled:
        note_msg = note_msg or "Finalized slate (prior day). Background updates are disabled; showing saved closing numbers."
    # Capture any existing predictions to preserve odds if updates fail/are partial
    try:
        df_old_global = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
    except Exception:
        df_old_global = pd.DataFrame()
    # Ensure models exist (Elo/config); if missing, do a quick bootstrap inline (only needed for non-settled views)
    if not settled:
        try:
            await _ensure_models(quick=True)
        except Exception:
            pass
    # Ensure we have predictions for the date; run inline if missing
    pred_path = PROC_DIR / f"predictions_{date}.csv"
    if not pred_path.exists():
        # Attempt Bovada first
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if (not live_now) and (not settled):
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
                    if not settled:
                        try:
                            predict_core(date=date, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
                        except Exception:
                            pass
        # If file still doesn't exist, at least generate predictions without odds (allowed during live to show something)
        if not pred_path.exists():
            try:
                predict_core(date=date, source="web", odds_source="csv")
            except Exception:
                pass
    df = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    # Also ensure neighbor-day predictions exist so late ET games (crossing UTC midnight) can be surfaced
    if not settled:
        try:
            nd = (datetime.fromisoformat(date) + timedelta(days=1)).strftime("%Y-%m-%d")
            next_path = PROC_DIR / f"predictions_{nd}.csv"
            if not next_path.exists():
                try:
                    # Cheapest generation to create rows; odds can be injected later
                    predict_core(date=nd, source="web", odds_source="csv")
                except Exception:
                    pass
        except Exception:
            pass
    # If predictions exist but odds are missing, try Bovada then Odds API to populate
    if pred_path.exists() and not _has_any_odds_df(df) and (not live_now) and (not settled):
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Preserve any existing odds if present in old df
        try:
            df_old = pd.read_csv(pred_path)
        except Exception:
            df_old = pd.DataFrame()
        try:
            predict_core(date=date, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
            df = pd.read_csv(pred_path)
            if not df_old.empty:
                df = _merge_preserve_odds(df_old, df)
                df.to_csv(pred_path, index=False)
        except Exception:
            pass
        if not _has_any_odds_df(df):
            try:
                predict_core(date=date, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
                df = pd.read_csv(pred_path)
                if not df_old.empty:
                    df = _merge_preserve_odds(df_old, df)
                    df.to_csv(pred_path, index=False)
            except Exception:
                pass
    # If no games for requested date, first try alternate schedule source, then try to find the next available slate within 10 days
    if df.empty:
        # Try using the NHL stats API as an alternate source for schedule
        try:
            # If stats API has games, generate predictions using that source
            stats_client = NHLStatsClient()
            stats_games = stats_client.schedule(date, date)
            if stats_games:
                snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                try:
                    predict_core(date=date, source="stats", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
                    df_alt = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
                    if not df_alt.empty:
                        df = df_alt
                except Exception:
                    pass
        except Exception:
            pass
    if df.empty:
        try:
            client = NHLWebClient()
            base = pd.to_datetime(date)
            for i in range(1, 11):
                d2 = (base + timedelta(days=i)).strftime("%Y-%m-%d")
                # Load schedule for that day
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
                    # If odds pipeline failed to produce a file, just generate without odds
                    if not alt_path.exists():
                        try:
                            predict_core(date=d2, source="web", odds_source="csv")
                        except Exception:
                            pass
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
    # Final odds preservation pass: if we had older data, fill missing odds/book fields
    if not settled:
        try:
            if not df.empty and not df_old_global.empty:
                df = _merge_preserve_odds(df_old_global, df)
                df.to_csv(PROC_DIR / f"predictions_{date}.csv", index=False)
        except Exception:
            pass
    # Cross-midnight inclusion: merge neighbor-day predictions and filter to ET date bucket
    try:
        frames = []
        if not df.empty:
            frames.append(df)
        # Previous and next day files, if they exist
        base_dt = datetime.fromisoformat(date)
        pd_str = (base_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        nd_str = (base_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        for d_nei in (pd_str, nd_str):
            p = PROC_DIR / f"predictions_{d_nei}.csv"
            if p.exists():
                try:
                    dfn = pd.read_csv(p)
                    if not dfn.empty:
                        frames.append(dfn)
                except Exception:
                    pass
        if frames:
            dfall = pd.concat(frames, ignore_index=True)
            # Compute ET date for each row from ISO 'date' if available, else try 'gameDate'
            def _row_et_date(x):
                v = x.get("date") if isinstance(x, dict) else None
                if not v and hasattr(x, "get"):
                    v = x.get("gameDate")
                if not v and isinstance(x, pd.Series):
                    v = x.get("gameDate")
                return _iso_to_et_date(v)
            try:
                et_dates = dfall.apply(lambda r: _iso_to_et_date(r.get("date") if pd.notna(r.get("date")) else r.get("gameDate")), axis=1)
            except Exception:
                et_dates = dfall.apply(lambda r: _row_et_date(r), axis=1)
            dfall = dfall[et_dates == date]
            # Drop potential duplicates (same home/away)
            if {"home","away"}.issubset(dfall.columns):
                dfall = dfall.drop_duplicates(subset=["home","away"], keep="first")
            df = dfall
    except Exception:
        pass
    rows = df.to_dict(orient="records") if not df.empty else []
    # For settled slates, mark rows as FINAL to avoid relying on live scoreboard
    if settled:
        try:
            for r in rows:
                r["game_state"] = r.get("game_state") or "FINAL"
        except Exception:
            pass
    # Enrich rows for settled slates: final scores from scoreboard
    if settled and rows:
        try:
            client = NHLWebClient()
            sb = client.scoreboard_day(date)
        except Exception:
            sb = []
        # Build lookup by abbr pair
        def _abbr(x: str) -> str:
            try:
                return (get_team_assets(str(x)).get("abbr") or "").upper()
            except Exception:
                return ""
        sb_idx = {}
        try:
            for g in sb:
                hk = _abbr(g.get("home"))
                ak = _abbr(g.get("away"))
                if hk and ak:
                    sb_idx[(hk, ak)] = g
        except Exception:
            pass
        for r in rows:
            # Final scores
            try:
                hk = _abbr(r.get("home"))
                ak = _abbr(r.get("away"))
                g = sb_idx.get((hk, ak))
                if g:
                    if g.get("home_goals") is not None:
                        r["final_home_goals"] = int(g.get("home_goals"))
                    if g.get("away_goals") is not None:
                        r["final_away_goals"] = int(g.get("away_goals"))
                    # Ensure FINAL label visible
                    r["game_state"] = r.get("game_state") or g.get("gameState") or "FINAL"
            except Exception:
                pass
    # Build a recommendation (best EV) for all rows; result only for completed games
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    for r in rows:
        # Candidates
        cands = []
        ev_h = _to_float(r.get("ev_home_ml")); ev_a = _to_float(r.get("ev_away_ml"))
        if ev_h is not None:
            cands.append({"market": "moneyline", "bet": "home_ml", "label": "Home ML", "ev": ev_h, "odds": r.get("home_ml_odds"), "book": r.get("home_ml_book")})
        if ev_a is not None:
            cands.append({"market": "moneyline", "bet": "away_ml", "label": "Away ML", "ev": ev_a, "odds": r.get("away_ml_odds"), "book": r.get("away_ml_book")})
        ev_o = _to_float(r.get("ev_over")); ev_u = _to_float(r.get("ev_under"))
        if ev_o is not None:
            cands.append({"market": "totals", "bet": "over", "label": "Over", "ev": ev_o, "odds": r.get("over_odds"), "book": r.get("over_book")})
        if ev_u is not None:
            cands.append({"market": "totals", "bet": "under", "label": "Under", "ev": ev_u, "odds": r.get("under_odds"), "book": r.get("under_book")})
        ev_hpl = _to_float(r.get("ev_home_pl_-1.5")); ev_apl = _to_float(r.get("ev_away_pl_+1.5"))
        if ev_hpl is not None:
            cands.append({"market": "puckline", "bet": "home_pl_-1.5", "label": "Home -1.5", "ev": ev_hpl, "odds": r.get("home_pl_-1.5_odds"), "book": r.get("home_pl_-1.5_book")})
        if ev_apl is not None:
            cands.append({"market": "puckline", "bet": "away_pl_+1.5", "label": "Away +1.5", "ev": ev_apl, "odds": r.get("away_pl_+1.5_odds"), "book": r.get("away_pl_+1.5_book")})
        best = None
        if cands:
            best = sorted(cands, key=lambda x: (x.get("ev") if x.get("ev") is not None else -999), reverse=True)[0]
        # Confidence by EV thresholds
        conf = None
        try:
            evv = best.get("ev") if best else None
            if evv is not None:
                if evv >= 0.05:
                    conf = "High"
                elif evv >= 0.02:
                    conf = "Medium"
                elif evv >= 0:
                    conf = "Low"
        except Exception:
            conf = None
        rec_res = None; rec_ok = None
        if best:
            m = best["market"]; b = best["bet"]
            if settled:
                if m == "moneyline":
                    wact = r.get("winner_actual")
                    if isinstance(wact, str) and wact:
                        want = r.get("home") if b == "home_ml" else r.get("away")
                        rec_ok = (wact == want)
                        rec_res = "Win" if rec_ok else "Loss"
                elif m == "totals":
                    rt = r.get("result_total")
                    if isinstance(rt, str) and rt:
                        if rt == "Push":
                            rec_res = "Push"; rec_ok = None
                        else:
                            want = "Over" if b == "over" else "Under"
                            rec_ok = (rt == want)
                            rec_res = "Win" if rec_ok else "Loss"
                elif m == "puckline":
                    ra = r.get("result_ats")
                    if isinstance(ra, str) and ra:
                        rec_ok = (ra == b)
                        rec_res = "Win" if rec_ok else "Loss"
            r["rec_market"] = best.get("market")
            r["rec_bet"] = best.get("bet")
            r["rec_label"] = best.get("label")
            r["rec_ev"] = best.get("ev")
            r["rec_odds"] = best.get("odds")
            r["rec_book"] = best.get("book")
            r["rec_result"] = rec_res
            r["rec_success"] = rec_ok
            r["rec_confidence"] = conf
        # Add model pick (moneyline highest probability)
        try:
            ph = float(r.get("p_home_ml")) if r.get("p_home_ml") is not None else None
            pa = float(r.get("p_away_ml")) if r.get("p_away_ml") is not None else None
            if ph is not None and pa is not None:
                if ph >= pa:
                    r["model_pick"] = "Home ML"
                    r["model_pick_prob"] = ph
                else:
                    r["model_pick"] = "Away ML"
                    r["model_pick_prob"] = pa
        except Exception:
            pass
    # Load inferred odds as a tertiary display fallback (not persisted): inferred_odds_{date}.csv
    inferred_map = {}
    try:
        inf_path = PROC_DIR / f"inferred_odds_{date}.csv"
        if inf_path.exists():
            dfi = pd.read_csv(inf_path)
            def norm_team(s: str) -> str:
                import re, unicodedata
                if s is None:
                    return ""
                s = str(s)
                s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
                s = s.lower()
                s = re.sub(r"[^a-z0-9]+", "", s)
                return s
            for _, ir in dfi.iterrows():
                key = (norm_team(ir.get("home")), norm_team(ir.get("away")), str(ir.get("market")))
                try:
                    inferred_map[key] = float(ir.get("american_inferred")) if pd.notna(ir.get("american_inferred")) else None
                except Exception:
                    inferred_map[key] = None
    except Exception:
        inferred_map = {}
    # Keep UTC ISO in rows; client formats to user local time
    def to_local(iso_utc: str) -> str:
        return iso_utc
    for r in rows:
        h = get_team_assets(str(r.get("home", "")))
        a = get_team_assets(str(r.get("away", "")))
        r["home_abbr"] = h.get("abbr")
        r["home_logo"] = h.get("logo_dark") or h.get("logo_light")
        r["away_abbr"] = a.get("abbr")
        r["away_logo"] = a.get("logo_dark") or a.get("logo_light")
        # Compute display odds (fallback to closing, then inferred) and presence flag
        try:
            import math
            def _has(v):
                return (v is not None) and (not (isinstance(v, float) and math.isnan(v))) and (str(v).strip() != "")
            def _fb(primary, closev):
                return primary if _has(primary) else (closev if _has(closev) else None)
            def _norm(s: str) -> str:
                import re, unicodedata
                if s is None:
                    return ""
                s = str(s)
                s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
                s = s.lower()
                s = re.sub(r"[^a-z0-9]+", "", s)
                return s
            # Moneyline
            r["disp_home_ml_odds"] = _fb(r.get("home_ml_odds"), r.get("close_home_ml_odds"))
            r["disp_away_ml_odds"] = _fb(r.get("away_ml_odds"), r.get("close_away_ml_odds"))
            r["disp_home_ml_book"] = _fb(r.get("home_ml_book"), r.get("close_home_ml_book"))
            r["disp_away_ml_book"] = _fb(r.get("away_ml_book"), r.get("close_away_ml_book"))
            # Inferred fallback for ML
            hn = _norm(r.get("home")); an = _norm(r.get("away"))
            if not _has(r.get("disp_home_ml_odds")):
                v = inferred_map.get((hn, an, "home_ml"))
                if _has(v):
                    r["disp_home_ml_odds"] = v
                    r["disp_home_ml_book"] = "Inferred"
            if not _has(r.get("disp_away_ml_odds")):
                v = inferred_map.get((hn, an, "away_ml"))
                if _has(v):
                    r["disp_away_ml_odds"] = v
                    r["disp_away_ml_book"] = "Inferred"
            # Totals
            r["disp_over_odds"] = _fb(r.get("over_odds"), r.get("close_over_odds"))
            r["disp_under_odds"] = _fb(r.get("under_odds"), r.get("close_under_odds"))
            r["disp_over_book"] = _fb(r.get("over_book"), r.get("close_over_book"))
            r["disp_under_book"] = _fb(r.get("under_book"), r.get("close_under_book"))
            r["disp_total_line_used"] = _fb(r.get("total_line_used"), r.get("close_total_line_used"))
            # Inferred fallback for totals (line may remain unknown)
            if not _has(r.get("disp_over_odds")):
                v = inferred_map.get((hn, an, "over"))
                if _has(v):
                    r["disp_over_odds"] = v
                    r["disp_over_book"] = "Inferred"
            if not _has(r.get("disp_under_odds")):
                v = inferred_map.get((hn, an, "under"))
                if _has(v):
                    r["disp_under_odds"] = v
                    r["disp_under_book"] = "Inferred"
            # Puck line
            r["disp_home_pl_-1.5_odds"] = _fb(r.get("home_pl_-1.5_odds"), r.get("close_home_pl_-1.5_odds"))
            r["disp_away_pl_+1.5_odds"] = _fb(r.get("away_pl_+1.5_odds"), r.get("close_away_pl_+1.5_odds"))
            r["disp_home_pl_-1.5_book"] = _fb(r.get("home_pl_-1.5_book"), r.get("close_home_pl_-1.5_book"))
            r["disp_away_pl_+1.5_book"] = _fb(r.get("away_pl_+1.5_book"), r.get("close_away_pl_+1.5_book"))
            # Inferred fallback for puck line
            if not _has(r.get("disp_home_pl_-1.5_odds")):
                v = inferred_map.get((hn, an, "home_pl_-1.5"))
                if _has(v):
                    r["disp_home_pl_-1.5_odds"] = v
                    r["disp_home_pl_-1.5_book"] = "Inferred"
            if not _has(r.get("disp_away_pl_+1.5_odds")):
                v = inferred_map.get((hn, an, "away_pl_+1.5"))
                if _has(v):
                    r["disp_away_pl_+1.5_odds"] = v
                    r["disp_away_pl_+1.5_book"] = "Inferred"
            # Presence: consider display odds (may include inferred) as well
            r["has_any_odds"] = any(_has(r.get(k)) for k in [
                "disp_home_ml_odds","disp_away_ml_odds","disp_over_odds","disp_under_odds",
                "disp_home_pl_-1.5_odds","disp_away_pl_+1.5_odds"
            ])
        except Exception:
            r["has_any_odds"] = False
        # Attach gamePk using fresh schedule lookup for reliable scoreboard polling
        try:
            if r.get("date") and r.get("home") and r.get("away"):
                # Use ET calendar day for schedule lookup to handle cross-midnight games
                dkey = _iso_to_et_date(r["date"]) if r.get("date") else date
                _client = NHLWebClient()
                gms = _client.schedule_day(dkey)
                # Find matching by abbr first, then names
                def _abbr(x):
                    try:
                        return (get_team_assets(str(x)).get("abbr") or "").upper()
                    except Exception:
                        return ""
                h_ab = _abbr(r.get("home"))
                a_ab = _abbr(r.get("away"))
                gid = None
                for g in gms:
                    if _abbr(getattr(g, 'home', '')) == h_ab and _abbr(getattr(g, 'away', '')) == a_ab:
                        gid = getattr(g, 'gamePk', None)
                        break
                if gid is None:
                    for g in gms:
                        if str(getattr(g, 'home', '')).strip() == str(r.get('home')).strip() and str(getattr(g, 'away', '')).strip() == str(r.get('away')).strip():
                            gid = getattr(g, 'gamePk', None)
                            break
                if gid is not None:
                    r["gamePk"] = int(gid)
        except Exception:
            pass
        if r.get("date"):
            r["local_time"] = r["date"]

    if live_now:
        # Informational note: during live games we do not regenerate odds/predictions automatically
        note_msg = note_msg or "Live slate detected. Odds are frozen to previously saved values; no regeneration during live games."
    template = env.get_template("cards.html")
    html = template.render(date=date, rows=rows, note=note_msg, live_now=live_now, settled=settled)
    return HTMLResponse(content=html)


    


def _capture_openers_for_day(date: str) -> dict:
    """Persist first-seen 'opening' odds into predictions_{date}.csv.

    For each row, if open_* columns are missing or empty, copy current odds/book/line fields.
    Idempotent: does not overwrite existing open_* values.
    """
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return {"status": "no-file", "date": date}
    df = pd.read_csv(path)
    if df.empty:
        return {"status": "empty", "date": date}
    def ensure(col: str):
        if col not in df.columns:
            df[col] = pd.NA
    opener_cols = [
        "open_home_ml_odds","open_away_ml_odds","open_over_odds","open_under_odds",
        "open_home_pl_-1.5_odds","open_away_pl_+1.5_odds","open_total_line_used",
        "open_home_ml_book","open_away_ml_book","open_over_book","open_under_book",
        "open_home_pl_-1.5_book","open_away_pl_+1.5_book","open_snapshot",
    ]
    for c in opener_cols:
        ensure(c)
    import pandas as _pd
    updated = 0
    for i, r in df.iterrows():
        def set_first(dst_col, src_col):
            try:
                cur = df.at[i, dst_col]
                if _pd.isna(cur) or cur is None or str(cur).strip() == "":
                    if src_col in df.columns and _pd.notna(df.at[i, src_col]):
                        df.at[i, dst_col] = df.at[i, src_col]
                        return True
            except Exception:
                return False
            return False
        changed = False
        changed |= set_first("open_home_ml_odds", "home_ml_odds")
        changed |= set_first("open_away_ml_odds", "away_ml_odds")
        changed |= set_first("open_over_odds", "over_odds")
        changed |= set_first("open_under_odds", "under_odds")
        changed |= set_first("open_home_pl_-1.5_odds", "home_pl_-1.5_odds")
        changed |= set_first("open_away_pl_+1.5_odds", "away_pl_+1.5_odds")
        changed |= set_first("open_total_line_used", "total_line_used")
        changed |= set_first("open_home_ml_book", "home_ml_book")
        changed |= set_first("open_away_ml_book", "away_ml_book")
        changed |= set_first("open_over_book", "over_book")
        changed |= set_first("open_under_book", "under_book")
        changed |= set_first("open_home_pl_-1.5_book", "home_pl_-1.5_book")
        changed |= set_first("open_away_pl_+1.5_book", "away_pl_+1.5_book")
        if changed:
            updated += 1
            try:
                if _pd.isna(df.at[i, "open_snapshot"]) or df.at[i, "open_snapshot"] is None or str(df.at[i, "open_snapshot"]).strip() == "":
                    df.at[i, "open_snapshot"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                pass
    if updated > 0:
        df.to_csv(path, index=False)
    return {"status": "ok", "updated": int(updated), "date": date}


    


@app.post("/api/capture-openers")
async def api_capture_openers(date: Optional[str] = Query(None)):
    date = date or _today_ymd()
    # Don't capture openers if slate is live; we want pregame numbers
    if _is_live_day(date):
        return JSONResponse({"status": "skipped-live", "date": date})
    res = _capture_openers_for_day(date)
    return JSONResponse(res, status_code=200 if res.get("status") == "ok" else 400)


@app.get("/api/scoreboard")
async def api_scoreboard(date: Optional[str] = Query(None)):
    """Lightweight live scoreboard for a date: state, score, period/clock per game.

    Matches by gamePk when possible, else by team abbreviations.
    """
    date = date or _today_ymd()
    client = NHLWebClient()
    rows = client.scoreboard_day(date)
    # Attach abbreviations for robust client matching
    for r in rows:
        try:
            h = get_team_assets(str(r.get("home", "")))
            a = get_team_assets(str(r.get("away", "")))
            r["home_abbr"] = (h.get("abbr") or "").upper()
            r["away_abbr"] = (a.get("abbr") or "").upper()
        except Exception:
            r["home_abbr"] = ""; r["away_abbr"] = ""
    # For LIVE games, try to enrich with linescore to get precise period/clock
    try:
        for r in rows:
            st = str(r.get("gameState") or "").upper()
            if any(k in st for k in ["LIVE", "IN", "PROGRESS", "CRIT"]) and r.get("gamePk"):
                try:
                    ls = client.linescore(int(r.get("gamePk")))
                    if ls:
                        if ls.get("period") is not None:
                            r["period"] = ls.get("period")
                        if ls.get("clock"):
                            r["clock"] = ls.get("clock")
                except Exception:
                    pass
                # Fallback to Stats API for clock if still missing/empty
                try:
                    if not r.get("clock") or r.get("clock") in ("", None):
                        stats_client = NHLStatsClient()
                        glf = stats_client.game_live_feed(int(r.get("gamePk")))
                        live = (glf or {}).get("liveData", {})
                        ls2 = live.get("linescore", {})
                        # Prefer exact time remaining like "03:25". Stats API sometimes returns "END"; treat that as 0:00
                        clock2 = ls2.get("currentPeriodTimeRemaining")
                        if isinstance(clock2, str) and clock2:
                            if clock2.strip().upper() == "END":
                                clock2 = "0:00"
                            r["clock"] = clock2
                        # Try currentPlay as last resort
                        if not r.get("clock") or r.get("clock") in ("", None):
                            cur = live.get("plays", {}).get("currentPlay", {}).get("about", {})
                            clock3 = cur.get("periodTimeRemaining")
                            if isinstance(clock3, str) and clock3:
                                r["clock"] = clock3
                        # Period enrich
                        if r.get("period") is None:
                            per2 = ls2.get("currentPeriod") or ls2.get("period") or cur.get("period")
                            if per2 is not None:
                                r["period"] = per2
                except Exception:
                    pass
    except Exception:
        pass
    return JSONResponse(rows)


@app.post("/api/capture-closing")
async def api_capture_closing(
    date: Optional[str] = Query(None),
    home_abbr: Optional[str] = Query(None),
    away_abbr: Optional[str] = Query(None),
    snapshot: Optional[str] = Query(None),
):
    date = date or _today_ymd()
    if not home_abbr or not away_abbr:
        return JSONResponse({"status": "missing-params"}, status_code=400)
    res = _capture_closing_for_game(date, home_abbr.strip().upper(), away_abbr.strip().upper(), snapshot)
    code = 200 if res.get("status") in ("ok", "not-found", "no-file", "empty") else 400
    return JSONResponse(res, status_code=code)


@app.get("/api/predictions")
async def api_predictions(date: Optional[str] = Query(None)):
    date = date or _today_ymd()
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return JSONResponse({"error": "No predictions for date", "date": date}, status_code=404)
    df = pd.read_csv(path)
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/api/debug/odds-match")
async def api_debug_odds_match(date: Optional[str] = Query(None)):
    """Debug endpoint: for each game on date, show how Bovada odds would match and what prices were found."""
    date = date or _today_ymd()
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return JSONResponse({"error": "No predictions for date", "date": date}, status_code=404)
    df = pd.read_csv(path)
    if df.empty:
        return JSONResponse({"error": "Empty predictions file", "date": date}, status_code=400)
    # Fetch fresh Bovada odds
    try:
        bc = BovadaClient()
        odds = bc.fetch_game_odds(date)
        if odds is None:
            odds = pd.DataFrame()
    except Exception:
        odds = pd.DataFrame()
    def norm_team(s: str) -> str:
        import re, unicodedata
        if s is None:
            return ""
        s = str(s)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s
    # Prepare odds matching keys
    if not odds.empty:
        odds["date"] = pd.to_datetime(odds["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        odds["home_norm"] = odds["home"].apply(norm_team)
        odds["away_norm"] = odds["away"].apply(norm_team)
        try:
            from .teams import get_team_assets as _assets
            def to_abbr(x):
                try:
                    return (_assets(str(x)).get("abbr") or "").upper()
                except Exception:
                    return ""
            odds["home_abbr"] = odds["home"].apply(to_abbr)
            odds["away_abbr"] = odds["away"].apply(to_abbr)
        except Exception:
            odds["home_abbr"] = ""
            odds["away_abbr"] = ""
    out = []
    for _, r in df.iterrows():
        gh = str(r.get("home"))
        ga = str(r.get("away"))
        key_date = pd.to_datetime(r.get("date")).strftime("%Y-%m-%d") if pd.notna(r.get("date")) else date
        gh_n = norm_team(gh)
        ga_n = norm_team(ga)
        try:
            from .teams import get_team_assets as _assets
            gh_ab = (_assets(gh).get("abbr") or "").upper()
            ga_ab = (_assets(ga).get("abbr") or "").upper()
        except Exception:
            gh_ab = ""; ga_ab = ""
        status = "none"
        found = None
        if odds.empty:
            status = "no-odds-df"
        else:
            m = pd.DataFrame()
            # Try abbr+date
            if gh_ab and ga_ab and {"home_abbr","away_abbr"}.issubset(set(odds.columns)):
                m = odds[(odds["date"] == key_date) & (odds["home_abbr"] == gh_ab) & (odds["away_abbr"] == ga_ab)]
                if not m.empty:
                    status = "date_abbr"
            # Try names+date
            if m.empty:
                m = odds[(odds["date"] == key_date) & (odds["home_norm"] == gh_n) & (odds["away_norm"] == ga_n)]
                if not m.empty:
                    status = "date_names"
            # Try abbr-only
            if m.empty and gh_ab and ga_ab and {"home_abbr","away_abbr"}.issubset(set(odds.columns)):
                m = odds[(odds["home_abbr"] == gh_ab) & (odds["away_abbr"] == ga_ab)]
                if not m.empty:
                    status = "abbr_only"
            # Try names-only
            if m.empty:
                m = odds[(odds["home_norm"] == gh_n) & (odds["away_norm"] == ga_n)]
                if not m.empty:
                    status = "names_only"
            # Try reversed
            if m.empty:
                if gh_ab and ga_ab and {"home_abbr","away_abbr"}.issubset(set(odds.columns)):
                    m = odds[(odds["home_abbr"] == ga_ab) & (odds["away_abbr"] == gh_ab)]
                    if not m.empty:
                        status = "reversed_abbr"
                if m.empty:
                    m = odds[(odds["home_norm"] == ga_n) & (odds["away_norm"] == gh_n)]
                    if not m.empty:
                        status = "reversed_names"
            if not m.empty:
                row = m.iloc[0]
                found = {
                    "date": row.get("date"),
                    "home": row.get("home"),
                    "away": row.get("away"),
                    "home_ml": row.get("home_ml"),
                    "away_ml": row.get("away_ml"),
                    "over": row.get("over"),
                    "under": row.get("under"),
                    "total_line": row.get("total_line"),
                    "home_pl_-1.5": row.get("home_pl_-1.5"),
                    "away_pl_+1.5": row.get("away_pl_+1.5"),
                }
        out.append({
            "game_date": key_date,
            "home": gh,
            "away": ga,
            "match": status,
            "found": found,
            "home_abbr": gh_ab,
            "away_abbr": ga_ab,
            "home_norm": gh_n,
            "away_norm": ga_n,
        })
    return JSONResponse(out)

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
    backfill: bool = Query(False, description="If true, during live slates only fill missing odds without overwriting existing prices"),
):
    date = date or _today_ymd()
    # Do not refresh odds during live games to avoid clobbering saved lines, unless backfill mode is requested
    if _is_live_day(date) and not backfill:
        return JSONResponse({"status": "skipped-live", "date": date, "message": "Live games in progress; odds refresh skipped."}, status_code=200)
    if not snapshot:
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Ensure models exist (Elo/config)
    try:
        await _ensure_models(quick=True)
    except Exception:
        pass
    # Try Bovada first; fall back to Odds API if no odds
    try:
        try:
            df_old = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
        except Exception:
            df_old = pd.DataFrame()
        predict_core(
            date=date,
            source="web",
            odds_source="bovada",
            snapshot=snapshot,
            odds_best=True,
            bankroll=bankroll,
            kelly_fraction_part=kelly_fraction_part,
        )
        # Merge preserve after Bovada run (even if Bovada added odds, preserve any older fields still missing)
        try:
            df_new = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
            if not df_old.empty:
                if backfill:
                    # Backfill mode: keep existing prices, fill only missing from new
                    def _backfill_missing(df_target: pd.DataFrame, df_source: pd.DataFrame) -> pd.DataFrame:
                        if df_target is None or df_target.empty:
                            return df_target
                        if df_source is None or df_source.empty:
                            return df_target
                        def norm_team(s: str) -> str:
                            import re, unicodedata
                            if s is None:
                                return ""
                            s = str(s)
                            s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
                            s = s.lower()
                            s = re.sub(r"[^a-z0-9]+", "", s)
                            return s
                        def date_key(x) -> str:
                            try:
                                return pd.to_datetime(x).strftime("%Y-%m-%d")
                            except Exception:
                                return None
                        src_idx = {}
                        for _, r in df_source.iterrows():
                            k = (date_key(r.get("date")), norm_team(r.get("home")), norm_team(r.get("away")))
                            src_idx[k] = r
                        cols = [
                            "home_ml_odds","away_ml_odds","over_odds","under_odds","home_pl_-1.5_odds","away_pl_+1.5_odds",
                            "home_ml_book","away_ml_book","over_book","under_book","home_pl_-1.5_book","away_pl_+1.5_book",
                            "total_line_used","total_line",
                        ]
                        rows = []
                        for _, r in df_target.iterrows():
                            k = (date_key(r.get("date")), norm_team(r.get("home")), norm_team(r.get("away")))
                            if k in src_idx:
                                rs = src_idx[k]
                                for c in cols:
                                    tgt_has = (c in r and pd.notna(r.get(c)))
                                    src_has = (c in rs and pd.notna(rs.get(c)))
                                    if (not tgt_has) and src_has:
                                        r[c] = rs.get(c)
                            rows.append(r)
                        return pd.DataFrame(rows, columns=df_target.columns)
                    df_keep = _backfill_missing(df_old, df_new)
                    df_keep.to_csv(PROC_DIR / f"predictions_{date}.csv", index=False)
                else:
                    # Normal: fill missing odds in new from old (preserve existing values)
                    df_m = _merge_preserve_odds(df_old, df_new)
                    df_m.to_csv(PROC_DIR / f"predictions_{date}.csv", index=False)
        except Exception:
            pass
    except Exception:
        pass
    # Check whether odds present; if not, attempt Odds API fallback
    try:
        df = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
    except Exception:
        df = pd.DataFrame()
    try:
        df = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
    except Exception:
        df = pd.DataFrame()
    if not _has_any_odds_df(df):
        try:
            try:
                df_old2 = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
            except Exception:
                df_old2 = pd.DataFrame()
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
            # Merge preserve after Odds API run
            try:
                df_new2 = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
                if not df_old2.empty:
                    df_m2 = _merge_preserve_odds(df_old2, df_new2)
                    df_m2.to_csv(PROC_DIR / f"predictions_{date}.csv", index=False)
            except Exception:
                pass
        except Exception:
            pass
    # Ensure we have at least a predictions CSV (even without odds)
    pred_path = PROC_DIR / f"predictions_{date}.csv"
    if not pred_path.exists():
        try:
            predict_core(date=date, source="web", odds_source="csv")
        except Exception:
            pass
    # If still no file or empty, try using stats API as schedule source
    try:
        df = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()
    if df.empty:
        try:
            predict_core(date=date, source="stats", odds_source="csv")
        except Exception:
            pass
    # Final status
    try:
        df2 = pd.read_csv(PROC_DIR / f"predictions_{date}.csv")
        if df2.empty:
            return JSONResponse({"status": "partial", "message": "No odds available and no games found for date; created empty predictions.", "date": date}, status_code=200)
        else:
            # Persist openers automatically on refresh if slate is not live (idempotent)
            try:
                if not _is_live_day(date):
                    _capture_openers_for_day(date)
            except Exception:
                pass
            return {"status": "ok", "date": date, "snapshot": snapshot, "bankroll": bankroll, "kelly_fraction_part": kelly_fraction_part, "backfill": backfill}
    except Exception:
        # Improve diagnostics: indicate whether model files exist
        try:
            exist = {
                "elo": (_MODEL_DIR / "elo_ratings.json").exists(),
                "config": (_MODEL_DIR / "config.json").exists(),
            }
        except Exception:
            exist = {"elo": False, "config": False}
        return JSONResponse({"status": "partial", "message": "Failed to create predictions file.", "date": date, "models_present": exist}, status_code=200)


@app.get("/api/debug/status")
async def api_debug_status(date: Optional[str] = Query(None)):
    """Lightweight debug endpoint to inspect presence of model/data files and sizes."""
    date = date or _today_ymd()
    items = {}
    try:
        items["models"] = {
            "elo_path": str((_MODEL_DIR / "elo_ratings.json").resolve()),
            "elo_exists": (_MODEL_DIR / "elo_ratings.json").exists(),
            "config_path": str((_MODEL_DIR / "config.json").resolve()),
            "config_exists": (_MODEL_DIR / "config.json").exists(),
        }
    except Exception:
        items["models"] = {"elo_exists": False, "config_exists": False}
    try:
        raw_games = RAW_DIR / "games.csv"
        items["raw_games"] = {
            "path": str(raw_games.resolve()),
            "exists": raw_games.exists(),
            "size": raw_games.stat().st_size if raw_games.exists() else 0,
        }
    except Exception:
        items["raw_games"] = {"exists": False}
    try:
        pred = PROC_DIR / f"predictions_{date}.csv"
        items["predictions"] = {
            "path": str(pred.resolve()),
            "exists": pred.exists(),
            "size": pred.stat().st_size if pred.exists() else 0,
        }
    except Exception:
        items["predictions"] = {"exists": False}
    return JSONResponse(items)


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
    med_ev: float = Query(0.02, description="EV threshold for Medium confidence grouping (e.g., 0.02 for 2%)"),
    sort_by: str = Query("ev", description="Sort key within groups: ev, edge, prob, price"),
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
    # Compute confidence groupings (NFL-style):
    # High (ev >= high_ev), Medium (med_ev <= ev < high_ev), Low (0 <= ev < med_ev), Other (ev < 0)
    EV_HIGH = float(high_ev)
    EV_MED = float(med_ev)
    if EV_MED > EV_HIGH:  # safety swap
        EV_MED, EV_HIGH = EV_HIGH, EV_MED
    def group_row(r):
        try:
            ev = float(r.get("ev"))
        except Exception:
            ev = -999
        if ev >= EV_HIGH:
            return "high"
        elif ev >= EV_MED:
            return "medium"
        elif ev >= 0:
            return "low"
        else:
            return "other"
    # Annotate confidence on each row
    for r in rows:
        r["confidence"] = group_row(r)
    rows_high = [r for r in rows if r["confidence"] == "high"]
    rows_medium = [r for r in rows if r["confidence"] == "medium"]
    rows_low = [r for r in rows if r["confidence"] == "low"]
    rows_other = [r for r in rows if r["confidence"] == "other"]
    # Sort within groups by EV desc
    def sort_key_func(sb: str):
        sb = (sb or "").lower()
        if sb == "edge":
            return lambda x: x.get("edge") if x.get("edge") is not None else x.get("edge_pts") or -999
        if sb == "prob":
            return lambda x: x.get("model_prob", -999)
        if sb == "price":
            return lambda x: x.get("price", -999)
        # default ev
        return lambda x: x.get("ev", -999)
    _sk = sort_key_func(sort_by)
    rows_high.sort(key=_sk, reverse=True)
    rows_medium.sort(key=_sk, reverse=True)
    rows_low.sort(key=_sk, reverse=True)
    rows_other.sort(key=_sk, reverse=True)
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
    summary_medium = compute_summary(rows_medium)
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
        rows_medium=rows_medium,
        rows_low=rows_low,
        rows_other=rows_other,
        summary_overall=summary_overall,
        summary_high=summary_high,
        summary_medium=summary_medium,
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
        med_ev=med_ev,
        sort_by=sort_by,
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


@app.get("/api/reconciliation")
async def api_reconciliation(
    date: Optional[str] = Query(None),
    bankroll: float = Query(1000.0, description="Bankroll used for stake calc fallback"),
    flat_stake: float = Query(100.0, description="Fallback flat stake when stake not present"),
):
    """Compare model recommendations vs closing lines and compute simple PnL summary.

    Uses predictions_{date}.csv and close_* fields captured earlier. Assumes one bet per market per game if EV>0.
    """
    date = date or _today_ymd()
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return JSONResponse({"error": "No predictions for date", "date": date}, status_code=404)
    df = pd.read_csv(path)
    if df.empty:
        return JSONResponse({"error": "Empty predictions"}, status_code=400)
    # Build picks (moneyline + totals + puckline) with EV>0
    picks = []
    def add_pick(r: pd.Series, market: str, bet: str, ev_key: str, price_key: str, result_field: Optional[str] = None):
        ev = r.get(ev_key)
        if ev is None or (isinstance(ev, float) and pd.isna(ev)):
            return
        try:
            evf = float(ev)
        except Exception:
            return
        if evf <= 0:
            return
        # Closing price fallback to open/current
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
        # Determine result if available
        res = None
        if result_field and r.get(result_field) is not None:
            res = r.get(result_field)
        picks.append({
            "date": r.get("date"),
            "home": r.get("home"),
            "away": r.get("away"),
            "market": market,
            "bet": bet,
            "ev": evf,
            "price": price,
            "result_field": result_field,
            "result": res,
        })
    for _, r in df.iterrows():
        add_pick(r, "moneyline", "home_ml", "ev_home_ml", "home_ml_odds", None)
        add_pick(r, "moneyline", "away_ml", "ev_away_ml", "away_ml_odds", None)
        add_pick(r, "totals", "over", "ev_over", "over_odds", "result_total")
        add_pick(r, "totals", "under", "ev_under", "under_odds", "result_total")
        add_pick(r, "puckline", "home_pl_-1.5", "ev_home_pl_-1.5", "home_pl_-1.5_odds", "result_ats")
        add_pick(r, "puckline", "away_pl_+1.5", "ev_away_pl_+1.5", "away_pl_+1.5_odds", "result_ats")
    # Compute PnL assuming flat_stake when stake not recorded
    def american_to_decimal_local(american):
        if american is None or (isinstance(american, float) and pd.isna(american)):
            return None
        try:
            a = float(american)
        except Exception:
            return None
        if a > 0:
            return 1.0 + (a / 100.0)
        else:
            return 1.0 + (100.0 / abs(a))
    pnl = 0.0
    staked = 0.0
    wins = losses = pushes = 0
    decided = 0
    rows = []
    for p in picks:
        stake = flat_stake
        dec = american_to_decimal_local(p["price"]) if p.get("price") is not None else None
        res = p.get("result")
        # Interpret results for totals/puckline; moneyline requires winner mapping not included here, so skip unless present later
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
            # undecided or moneyline without explicit result mapping
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


@app.get("/reconciliation")
async def reconciliation(date: Optional[str] = Query(None)):
    date = date or _today_ymd()
    resp = await api_reconciliation(date=date)
    payload = {}
    if isinstance(resp, JSONResponse):
        try:
            import json as _json
            payload = _json.loads(resp.body)
        except Exception:
            payload = {"summary": {"date": date}, "rows": []}
    template = env.get_template("reconciliation.html")
    html = template.render(summary=payload.get("summary", {}), rows=payload.get("rows", []))
    return HTMLResponse(content=html)

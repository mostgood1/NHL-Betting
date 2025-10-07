from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, Header
from fastapi.responses import HTMLResponse, JSONResponse, Response, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..utils.io import RAW_DIR, PROC_DIR
from ..utils.io import MODEL_DIR as _MODEL_DIR
from ..data.nhl_api_web import NHLWebClient
from ..data.nhl_api import NHLClient as NHLStatsClient
from .teams import get_team_assets
from ..cli import predict_core, fetch as cli_fetch, train as cli_train
from ..models.poisson import PoissonGoals
from ..utils.io import save_df
import asyncio
from ..data.bovada import BovadaClient
import base64
import requests

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


def _normalize_date_param(d: Optional[str]) -> str:
    """Normalize 'today'/'yesterday' to ET YYYY-MM-DD; pass-through other values."""
    if not d:
        return _today_ymd()
    s = str(d).strip().lower()
    try:
        et = ZoneInfo("America/New_York")
    except Exception:
        et = timezone.utc
    now_et = datetime.now(et)
    if s == "today":
        return now_et.strftime("%Y-%m-%d")
    if s == "yesterday":
        return (now_et - timedelta(days=1)).strftime("%Y-%m-%d")
    return d


def _const_time_eq(a: str, b: str) -> bool:
    try:
        if a is None or b is None:
            return False
        import hmac
        return hmac.compare_digest(str(a), str(b))
    except Exception:
        return False


def _read_only(date: Optional[str] = None) -> bool:
    """Whether to avoid fetching odds or writing predictions.

    Controlled by env vars WEB_READ_ONLY_PREDICTIONS or WEB_DISABLE_ODDS_FETCH.
    Any truthy value (1/true/yes) enables read-only behavior.
    """
    flag1 = os.getenv("WEB_READ_ONLY_PREDICTIONS", "")
    flag2 = os.getenv("WEB_DISABLE_ODDS_FETCH", "")
    val = (flag1 or flag2 or "").strip().lower()
    return val in ("1", "true", "yes")


def _read_csv_fallback(path: Path) -> pd.DataFrame:
    """Read a CSV trying multiple encodings to handle BOM/UTF-16/Windows-1252.

    Returns empty DataFrame if file missing or empty. Avoids raising on transient
    half-writes by catching EmptyDataError and returning empty.
    """
    if not path or not Path(path).exists():
        return pd.DataFrame()
    # If file exists but is zero-bytes, treat as empty data
    try:
        if Path(path).stat().st_size == 0:
            return pd.DataFrame()
    except Exception:
        pass
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16", "utf-16le", "utf-16be")
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except pd.errors.EmptyDataError:
            # Empty/half-written file; treat as empty
            return pd.DataFrame()
        except Exception as e:
            # Non-decode issues: break and surface to python-engine fallback
            last_err = e
            break
    # Last resort: python engine
    try:
        return pd.read_csv(path, engine="python")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception:
        if last_err:
            # If the last error was decode-related, surface it; otherwise, treat as empty
            try:
                import codecs  # noqa: F401
                raise last_err
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()


def _file_mtime_iso(path: Path) -> Optional[str]:
    """Return file modified time as ISO UTC string (Z) if exists, else None."""
    try:
        if path and Path(path).exists():
            import datetime as _dt
            ts = _dt.datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc)
            return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None
    return None


def _fmt_et(iso_utc: Optional[str]) -> Optional[str]:
    """Format an ISO UTC timestamp into ET human string, e.g., 'Sep 30, 2025 06:32 PM ET'."""
    if not iso_utc:
        return None
    try:
        s = str(iso_utc).replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(s)
        et = ZoneInfo("America/New_York")
        dt_et = dt_utc.astimezone(et)
        return dt_et.strftime("%b %d, %Y %I:%M %p ET")
    except Exception:
        return None


def _last_update_info(date: str) -> dict:
    """Collect last update timestamps for predictions (odds) and recommendations for a date."""
    try:
        pred_p = PROC_DIR / f"predictions_{date}.csv"
        rec_p = PROC_DIR / f"recommendations_{date}.csv"
        pred_iso = _file_mtime_iso(pred_p)
        rec_iso = _file_mtime_iso(rec_p)
        return {
            "predictions_iso": pred_iso,
            "predictions_et": _fmt_et(pred_iso),
            "recommendations_iso": rec_iso,
            "recommendations_et": _fmt_et(rec_iso),
        }
    except Exception:
        return {"predictions_iso": None, "predictions_et": None, "recommendations_iso": None, "recommendations_et": None}


async def _recompute_edges_and_recommendations(date: str) -> None:
    """Recompute EVs/edges and persist edges/recommendations CSVs for a date.

    - Reads predictions_{date}.csv
    - Ensures EV columns exist (if odds present). If missing, recompute from p_* and *_odds
    - Writes edges_{date}.csv (long format) and recommendations_{date}.csv (top-N style via API)
    """
    try:
        pred_path = PROC_DIR / f"predictions_{date}.csv"
        if not pred_path.exists():
            return
        df = _read_csv_fallback(pred_path)
        if df is None or df.empty:
            return
        import math as _math
        from ..utils.odds import american_to_decimal, decimal_to_implied_prob, remove_vig_two_way, ev_unit
        # Helper to parse numeric odds
        def _num(v):
            if v is None:
                return None
            try:
                if isinstance(v, (int, float)):
                    fv = float(v)
                    return fv if _math.isfinite(fv) else None
                s = str(v).strip().replace(",", "")
                if s == "":
                    return None
                return float(s)
            except Exception:
                return None
        # Compute EVs if missing and odds present
        def _ensure_ev(row: pd.Series, prob_key: str, odds_key: str, ev_key: str, edge_key: Optional[str] = None):
            try:
                ev_present = (ev_key in row) and (row.get(ev_key) is not None) and not (isinstance(row.get(ev_key), float) and pd.isna(row.get(ev_key)))
                if ev_present and (edge_key is None or (edge_key in row and pd.notna(row.get(edge_key)))):
                    return row
                p = None
                if prob_key in row and pd.notna(row.get(prob_key)):
                    p = float(row.get(prob_key))
                    if not (0.0 <= p <= 1.0) or not _math.isfinite(p):
                        p = None
                price = _num(row.get(odds_key)) if odds_key in row else None
                # fallback to close_* price
                if price is None:
                    close_map = {
                        "home_ml_odds": "close_home_ml_odds",
                        "away_ml_odds": "close_away_ml_odds",
                        "over_odds": "close_over_odds",
                        "under_odds": "close_under_odds",
                        "home_pl_-1.5_odds": "close_home_pl_-1.5_odds",
                        "away_pl_+1.5_odds": "close_away_pl_+1.5_odds",
                    }
                    ck = close_map.get(odds_key)
                    if ck and (ck in row):
                        price = _num(row.get(ck))
                if (p is not None) and (price is not None):
                    dec = american_to_decimal(price)
                    if dec is not None and _math.isfinite(dec):
                        row[ev_key] = round(ev_unit(p, dec), 4)
                        if edge_key:
                            # edge uses no-vig implied prob from two-way if counterpart present
                            # Infer counterpart odds/prob based on market key pattern
                            counterpart_map = {
                                ("p_home_ml", "home_ml_odds"): ("p_away_ml", "away_ml_odds"),
                                ("p_away_ml", "away_ml_odds"): ("p_home_ml", "home_ml_odds"),
                                ("p_over", "over_odds"): ("p_under", "under_odds"),
                                ("p_under", "under_odds"): ("p_over", "over_odds"),
                                ("p_home_pl_-1.5", "home_pl_-1.5_odds"): ("p_away_pl_+1.5", "away_pl_+1.5_odds"),
                                ("p_away_pl_+1.5", "away_pl_+1.5_odds"): ("p_home_pl_-1.5", "home_pl_-1.5_odds"),
                            }
                            other = counterpart_map.get((prob_key, odds_key))
                            if other:
                                p2 = None
                                if other[0] in row and pd.notna(row.get(other[0])):
                                    try:
                                        p2 = float(row.get(other[0]))
                                        if not (0.0 <= p2 <= 1.0) or not _math.isfinite(p2):
                                            p2 = None
                                    except Exception:
                                        p2 = None
                                price2 = _num(row.get(other[1])) if other[1] in row else None
                                if price2 is None:
                                    close_map2 = {
                                        "home_ml_odds": "close_home_ml_odds",
                                        "away_ml_odds": "close_away_ml_odds",
                                        "over_odds": "close_over_odds",
                                        "under_odds": "close_under_odds",
                                        "home_pl_-1.5_odds": "close_home_pl_-1.5_odds",
                                        "away_pl_+1.5_odds": "close_away_pl_+1.5_odds",
                                    }
                                    ck2 = close_map2.get(other[1])
                                    if ck2 and (ck2 in row):
                                        price2 = _num(row.get(ck2))
                                if (p2 is not None) and (price2 is not None):
                                    dec1 = american_to_decimal(price)
                                    dec2 = american_to_decimal(price2)
                                    if dec1 is not None and dec2 is not None:
                                        imp1 = decimal_to_implied_prob(dec1)
                                        imp2 = decimal_to_implied_prob(dec2)
                                        nv1, nv2 = remove_vig_two_way(imp1, imp2)
                                        # choose appropriate no-vig for this side
                                        nv = nv1 if prob_key in ("p_home_ml", "p_over", "p_home_pl_-1.5") else nv2
                                        row[edge_key] = round(p - nv, 4)
            except Exception:
                return row
            return row
        # Apply to rows
        if not df.empty:
            for i, r in df.iterrows():
                r = _ensure_ev(r, "p_home_ml", "home_ml_odds", "ev_home_ml", "edge_home_ml")
                r = _ensure_ev(r, "p_away_ml", "away_ml_odds", "ev_away_ml", "edge_away_ml")
                r = _ensure_ev(r, "p_over", "over_odds", "ev_over", "edge_over")
                r = _ensure_ev(r, "p_under", "under_odds", "ev_under", "edge_under")
                r = _ensure_ev(r, "p_home_pl_-1.5", "home_pl_-1.5_odds", "ev_home_pl_-1.5", "edge_home_pl_-1.5")
                r = _ensure_ev(r, "p_away_pl_+1.5", "away_pl_+1.5_odds", "ev_away_pl_+1.5", "edge_away_pl_+1.5")
                df.iloc[i] = r
        # Persist predictions with updated EV/edge fields
        df.to_csv(pred_path, index=False)
        _gh_upsert_file_if_configured(pred_path, f"web: update predictions with odds/EV for {date}")
        # Write edges long-form
        ev_cols = [c for c in df.columns if c.startswith("ev_")]
        if ev_cols:
            try:
                edges = df.melt(id_vars=["date", "home", "away"], value_vars=ev_cols, var_name="market", value_name="ev").dropna()
                edges = edges.sort_values("ev", ascending=False)
                edges_path = PROC_DIR / f"edges_{date}.csv"
                edges.to_csv(edges_path, index=False)
                _gh_upsert_file_if_configured(edges_path, f"web: update edges for {date}")
            except Exception:
                pass
        # Regenerate recommendations via API to reuse logic and write recommendations_{date}.csv
        try:
            await api_recommendations(date=date, min_ev=0.0, top=1000, markets="all", bankroll=0.0, kelly_fraction_part=0.5)
            # Push recommendations file if created
            rec_path = PROC_DIR / f"recommendations_{date}.csv"
            if rec_path.exists():
                _gh_upsert_file_if_configured(rec_path, f"web: update recommendations for {date}")
        except Exception:
            pass
    except Exception:
        pass


def _backfill_settlement_for_date(date: str) -> dict:
    """Compute final scores and result fields for a settled slate and persist them.

    Writes back into predictions_{date}.csv and upserts to GitHub regardless of read-only UI flags.
    Returns a brief summary dict with counts.
    """
    pred_csv_path = PROC_DIR / f"predictions_{date}.csv"
    if not pred_csv_path.exists():
        return {"skipped": True, "reason": "no_predictions"}
    try:
        df = _read_csv_fallback(pred_csv_path)
    except Exception as e:
        return {"skipped": True, "reason": f"read_error_{type(e).__name__}"}
    if df.empty:
        return {"skipped": True, "reason": "empty_predictions"}
    # Scoreboard lookup
    try:
        client = NHLWebClient()
        sb = client.scoreboard_day(date)
    except Exception:
        sb = []
    def _abbr(x: str) -> str:
        try:
            return (get_team_assets(str(x)).get("abbr") or "").upper()
        except Exception:
            return ""
    sb_idx = {}
    try:
        for g in sb:
            hk = _abbr(g.get("home")); ak = _abbr(g.get("away"))
            if hk and ak:
                sb_idx[(hk, ak)] = g
    except Exception:
        pass
    backfilled = 0
    import math as _math
    def _num(x):
        try:
            if x is None: return None
            if isinstance(x, (int,float)):
                return float(x)
            s = str(x).strip();
            if s == "": return None
            return float(s)
        except Exception:
            return None
    for i, r in df.iterrows():
        try:
            hk = _abbr(r.get("home")); ak = _abbr(r.get("away"))
            g = sb_idx.get((hk, ak))
            fh = fa = None
            if g:
                if g.get("home_goals") is not None:
                    fh = int(g.get("home_goals"))
                if g.get("away_goals") is not None:
                    fa = int(g.get("away_goals"))
            if fh is None or fa is None:
                continue
            actual_total = fh + fa
            df.at[i, "final_home_goals"] = fh
            df.at[i, "final_away_goals"] = fa
            df.at[i, "actual_home_goals"] = fh
            df.at[i, "actual_away_goals"] = fa
            df.at[i, "actual_total"] = actual_total
            # winner_actual
            if pd.isna(r.get("winner_actual")) or not r.get("winner_actual"):
                df.at[i, "winner_actual"] = r.get("home") if fh > fa else (r.get("away") if fa > fh else "Draw")
            # result_total: prefer close_total_line_used then total_line_used
            total_line = None
            for key in ("close_total_line_used", "total_line_used", "pl_line_used"):
                v = r.get(key)
                nv = _num(v)
                if nv is not None:
                    total_line = nv; break
            if (pd.isna(r.get("result_total")) or not r.get("result_total")) and (total_line is not None):
                if actual_total > total_line:
                    df.at[i, "result_total"] = "Over"
                elif actual_total < total_line:
                    df.at[i, "result_total"] = "Under"
                else:
                    df.at[i, "result_total"] = "Push"
            # result_ats at +/-1.5
            if pd.isna(r.get("result_ats")) or not r.get("result_ats"):
                diff = fh - fa
                df.at[i, "result_ats"] = "home_-1.5" if diff > 1.5 else "away_+1.5"
            # winner_model and correctness
            if pd.isna(r.get("winner_model")) or not r.get("winner_model"):
                try:
                    ph = _num(r.get("p_home_ml")); pa = _num(r.get("p_away_ml"))
                    if ph is not None and pa is not None:
                        df.at[i, "winner_model"] = r.get("home") if ph >= pa else r.get("away")
                except Exception:
                    pass
            if (r.get("winner_actual") or df.at[i, "winner_actual"]) and (r.get("winner_model") or df.at[i, "winner_model"]) and pd.isna(r.get("winner_correct")):
                df.at[i, "winner_correct"] = ( (r.get("winner_actual") or df.at[i, "winner_actual"]) == (r.get("winner_model") or df.at[i, "winner_model"]) )
            # total_diff
            mt = _num(r.get("model_total"))
            if mt is not None and pd.isna(r.get("total_diff")):
                df.at[i, "total_diff"] = round(mt - actual_total, 2)
            # pick correctness
            if r.get("totals_pick") and (r.get("result_total") or df.at[i, "result_total"]) and pd.isna(r.get("totals_pick_correct")):
                rt = r.get("result_total") or df.at[i, "result_total"]
                if rt != "Push":
                    df.at[i, "totals_pick_correct"] = (r.get("totals_pick") == rt)
            if r.get("ats_pick") and (r.get("result_ats") or df.at[i, "result_ats"]) and pd.isna(r.get("ats_pick_correct")):
                ra = r.get("result_ats") or df.at[i, "result_ats"]
                df.at[i, "ats_pick_correct"] = (r.get("ats_pick") == ra)
            backfilled += 1
        except Exception:
            continue
    # Persist and push
    try:
        df.to_csv(pred_csv_path, index=False)
        _gh_upsert_file_if_configured(pred_csv_path, f"web: settlement backfill for {date}")
    except Exception:
        pass
    return {"ok": True, "date": date, "rows_backfilled": backfilled}


def _gh_upsert_file_if_configured(path: Path, message: str) -> dict:
    """Push a file to GitHub if GITHUB_TOKEN and GITHUB_REPO are configured. Best-effort, non-fatal."""
    try:
        token = os.getenv("GITHUB_TOKEN", "").strip()
        repo = os.getenv("GITHUB_REPO", "").strip()
        branch = os.getenv("GITHUB_BRANCH", "master").strip()
        if not token or not repo:
            return {"skipped": True, "reason": "missing_token_or_repo"}
        # Build relative path from repo root; assume working dir at repo root
        rel_path = str(path).replace("\\", "/")
        try:
            # Attempt to strip absolute root up to repo folder if present
            # Find last occurrence of '/NHL-Betting/' or repo name
            parts = rel_path.split("/")
            if "data" in parts:
                idx = parts.index("data")
                rel_path = "/".join(parts[idx-0:])  # from 'data/...'
        except Exception:
            pass
        api = f"https://api.github.com/repos/{repo}/contents/{rel_path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
        # Read content
        with open(path, "rb") as f:
            local_bytes = f.read()
            content_b64 = base64.b64encode(local_bytes).decode("ascii")
        # Get existing SHA if file exists
        sha = None
        remote_same = False
        try:
            r = requests.get(api, params={"ref": branch}, headers=headers, timeout=20)
            if r.status_code == 200:
                body = r.json()
                sha = body.get("sha")
                # If API returns content, compare to avoid no-op commits
                try:
                    enc = body.get("encoding")
                    remote_content = body.get("content")
                    if enc == "base64" and isinstance(remote_content, str):
                        import base64 as _b64
                        rb = _b64.b64decode(remote_content.encode("ascii"))
                        if rb == local_bytes:
                            remote_same = True
                except Exception:
                    remote_same = False
        except Exception:
            sha = None
        if remote_same:
            return {"skipped": True, "reason": "no_change", "path": rel_path}
        data = {
            "message": message,
            "content": content_b64,
            "branch": branch,
        }
        author = os.getenv("GITHUB_COMMIT_AUTHOR", "").strip()
        email = os.getenv("GITHUB_COMMIT_EMAIL", "").strip()
        if author and email:
            data["committer"] = {"name": author, "email": email}
        if sha:
            data["sha"] = sha
        pr = requests.put(api, headers=headers, json=data, timeout=30)
        if pr.status_code not in (200, 201):
            return {"skipped": True, "reason": f"push_failed_{pr.status_code}", "body": pr.text[:300]}
        return {"ok": True, "path": rel_path}
    except Exception as e:
        return {"skipped": True, "reason": f"exception_{type(e).__name__}", "msg": str(e)}


def _df_jsonsafe_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe list of dicts.

    - Replace +/-Inf with NaN
    - Convert NaN to None
    - Coerce datetime-like values to ISO date strings (YYYY-MM-DD)
    - Coerce pandas/NumPy scalars to native Python types
    """
    try:
        if df is None or df.empty:
            return []
        _df = df.replace([np.inf, -np.inf], np.nan).copy()
        # Normalize datetime-like columns to YYYY-MM-DD strings
        try:
            dt_cols = list(_df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, tz]"]).columns)
        except Exception:
            dt_cols = []
        for c in dt_cols:
            try:
                _df[c] = pd.to_datetime(_df[c], errors="coerce").dt.strftime("%Y-%m-%d")
            except Exception:
                # Fallback: cast to string
                try:
                    _df[c] = _df[c].astype(str)
                except Exception:
                    pass
        # Handle object dtype cells that may contain Timestamp
        try:
            import datetime as _dt
            def _coerce(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return None
                try:
                    if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
                        # Format dates as YYYY-MM-DD
                        return (v.date().isoformat() if isinstance(v, pd.Timestamp) else v.isoformat())
                except Exception:
                    pass
                # Convert NumPy scalars to Python types
                try:
                    import numpy as _np
                    if isinstance(v, (_np.integer,)):
                        return int(v)
                    if isinstance(v, (_np.floating,)):
                        return float(v)
                except Exception:
                    pass
                return v
            for c in _df.columns:
                if _df[c].dtype == object:
                    try:
                        _df[c] = _df[c].map(_coerce)
                    except Exception:
                        pass
        except Exception:
            pass
        _df = _df.where(pd.notnull(_df), None)
        return _df.to_dict(orient="records")
    except Exception:
        try:
            return df.fillna(value=None).to_dict(orient="records")
        except Exception:
            return []


@app.get("/health")
def health():
    """Simple health probe that avoids heavy work.

    Returns service status and whether today's predictions file exists.
    """
    try:
        et_today = _today_ymd()
    except Exception:
        et_today = None
    pred_path = None; pred_exists = False
    try:
        if et_today:
            pred_path = PROC_DIR / f"predictions_{et_today}.csv"
            pred_exists = pred_path.exists()
    except Exception:
        pred_exists = False
    return JSONResponse({
        "status": "ok",
        "date_et": et_today,
        "predictions_today": bool(pred_exists),
    })


@app.get("/health/render")
def health_render():
    """Render cards template with an empty slate to validate template compiles."""
    try:
        et_today = _today_ymd()
    except Exception:
        et_today = ""
    try:
        template = env.get_template("cards.html")
        html = template.render(date=et_today, original_date=None, rows=[], note=None, live_now=False, settled=False)
        return HTMLResponse(content=html)
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)


@app.get("/api/status")
def api_status(date: Optional[str] = Query(None)):
    """Return basic diagnostics for a given date (defaults to ET today)."""
    try:
        d = date or _today_ymd()
    except Exception:
        d = date
    info = {"date": d, "predictions_exists": False, "rows": 0, "has_any_odds": False}
    try:
        p = PROC_DIR / f"predictions_{d}.csv"
        if p.exists():
            info["predictions_exists"] = True
            try:
                df = _read_csv_fallback(p)
                info["rows"] = 0 if df is None or df.empty else int(len(df))
                info["has_any_odds"] = _has_any_odds_df(df)
                # Include a tiny sample of columns to confirm shape
                info["columns"] = list(df.columns)[:12]
            except Exception as e:
                info["read_error"] = str(e)
        else:
            info["predictions_exists"] = False
    except Exception as e:
        info["error"] = str(e)
    return JSONResponse(info)


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
            # Avoid overly broad substring matches (e.g., "IN" matches "FINAL").
            # Consider only clear live indicators.
            live_tokens = [
                "LIVE",
                "IN PROGRESS",
                "IN-PROGRESS",
                "IN_PROGRESS",
                "CRIT",  # critical live state
            ]
            if any(tok in st for tok in live_tokens):
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
    # Preserve union of columns so newly filled odds columns aren't dropped
    try:
        out_cols = list(dict.fromkeys(list(df_new.columns) + [c for c in cand_cols if (c in df_new.columns) or (c in df_old.columns)]))
        df_out = pd.DataFrame(rows)
        # Only select columns that exist in df_out
        out_cols = [c for c in out_cols if c in df_out.columns]
        return df_out[out_cols]
    except Exception:
        return pd.DataFrame(rows)


def _capture_closing_for_game(date: str, home_abbr: str, away_abbr: str, snapshot: Optional[str] = None) -> dict:
    """Persist first-seen 'closing' odds into predictions_{date}.csv for reconciliation.

    We match the row by team abbreviations; then copy current odds fields into close_* columns
    if they are missing. Returns a small status dict.
    """
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return {"status": "no-file", "date": date}
    df = _read_csv_fallback(path)
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
    try:
        _gh_upsert_file_if_configured(path, f"web: capture closing odds for {date} {home_abbr}-{away_abbr}")
    except Exception:
        pass
    return {"status": "ok", "date": date, "home_abbr": home_abbr, "away_abbr": away_abbr}


def _capture_closing_for_day(date: str) -> dict:
    """Persist first-seen closing odds for all FINAL games on the given date.

    Iterates the scoreboard for the ET date, finds games in a FINAL state, and captures
    closing odds/books/lines into predictions_{date}.csv using team abbreviations.
    Idempotent: only fills close_* columns if they are currently empty.
    """
    try:
        client = NHLWebClient()
        games = client.scoreboard_day(date)
    except Exception:
        games = []
    from .teams import get_team_assets as _assets
    def _abbr(x: str) -> str:
        try:
            return (_assets(str(x)).get("abbr") or "").upper()
        except Exception:
            return ""
    updated = 0
    skipped = 0
    errors = 0
    snap = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for g in games:
        try:
            st = str(g.get("gameState") or "").upper()
        except Exception:
            st = ""
        # Only capture closing for FINAL games
        if not st.startswith("FINAL"):
            skipped += 1
            continue
        try:
            h_ab = _abbr(g.get("home"))
            a_ab = _abbr(g.get("away"))
            if not h_ab or not a_ab:
                skipped += 1
                continue
            res = _capture_closing_for_game(date, h_ab, a_ab, snapshot=snap)
            if res.get("status") == "ok":
                updated += 1
            else:
                skipped += 1
        except Exception:
            errors += 1
    return {"status": "ok", "date": date, "updated": int(updated), "skipped": int(skipped), "errors": int(errors)}


def _american_from_prob(prob: float) -> Optional[int]:
    """Convert a fair probability into an American odds price (rounded to nearest 5).

    Example: p=0.6 -> decimal=1/0.6=1.666.. -> American -150
             p=0.4 -> decimal=2.5 -> American +150
    """
    try:
        import math
        p = float(prob)
        if not math.isfinite(p) or p <= 0 or p >= 1:
            return None
        dec = 1.0 / p
        if dec >= 2.0:
            val = int(round((dec - 1.0) * 100 / 5.0) * 5)
            return max(+100, val)
        else:
            val = int(round(100.0 / (dec - 1.0) / 5.0) * 5)
            return -max(100, val)
    except Exception:
        return None


@app.get("/health")
async def health_async():
    # Keep compatibility for async route; delegate to sync implementation above
    try:
        et_today = _today_ymd()
    except Exception:
        et_today = None
    pred_exists = False
    try:
        pred_path = PROC_DIR / f"predictions_{et_today}.csv" if et_today else None
        pred_exists = bool(pred_path and pred_path.exists())
    except Exception:
        pred_exists = False
    return {"status": "ok", "date_et": et_today, "predictions_today": pred_exists}


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
    # Preserve the originally requested date (may differ if auto-forward logic adjusts for future empty slates)
    requested_date = date or _today_ymd()
    date = requested_date
    try:
        print(f"[cards] requested_date={requested_date} initial date={date}")
    except Exception:
        pass
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
        df_old_global = _read_csv_fallback(PROC_DIR / f"predictions_{date}.csv")
    except Exception:
        df_old_global = pd.DataFrame()
    # Ensure models exist (Elo/config); if missing, do a quick bootstrap inline (only needed for non-settled views)
    if not settled:
        try:
            await _ensure_models(quick=True)
        except Exception:
            pass
    # Ensure we have predictions for the date; run inline if missing (unless read-only)
    pred_path = PROC_DIR / f"predictions_{date}.csv"
    read_only = _read_only(date)
    if not pred_path.exists():
        # Attempt Bovada first (skipped in read-only mode)
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if (not settled) and (not read_only):
            try:
                predict_core(date=date, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
            except Exception:
                pass
            # Fallback to Odds API if no odds captured
            if pred_path.exists():
                try:
                    tmp = _read_csv_fallback(pred_path)
                except Exception:
                    tmp = pd.DataFrame()
                if not _has_any_odds_df(tmp):
                    if (not settled) and (not read_only):
                        try:
                            predict_core(date=date, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
                        except Exception:
                            pass
        # If file still doesn't exist, at least generate predictions without odds (allowed during live to show something)
        if (not pred_path.exists()) and (not read_only):
            try:
                predict_core(date=date, source="web", odds_source="csv")
            except Exception:
                pass
    df = _read_csv_fallback(pred_path) if pred_path.exists() else pd.DataFrame()
    # Also ensure neighbor-day predictions exist so late ET games (crossing UTC midnight) can be surfaced
    if (not settled) and (not read_only):
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
    # If odds are missing, attempt to populate even during live slates (safe: only adds odds fields)
    if pred_path.exists() and not _has_any_odds_df(df) and (not settled) and (not read_only):
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Preserve any existing odds if present in old df
        try:
            df_old = _read_csv_fallback(pred_path)
        except Exception:
            df_old = pd.DataFrame()
        try:
            predict_core(date=date, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
            df = _read_csv_fallback(pred_path)
            if not df_old.empty:
                df = _merge_preserve_odds(df_old, df)
                df.to_csv(pred_path, index=False)
        except Exception:
            pass
        if not _has_any_odds_df(df):
            try:
                predict_core(date=date, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings")
                df = _read_csv_fallback(pred_path)
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
                    df_alt = _read_csv_fallback(PROC_DIR / f"predictions_{date}.csv")
                    if not df_alt.empty:
                        df = df_alt
                except Exception:
                    pass
        except Exception:
            pass
    # Only auto-forward to the next available slate for non-settled (today/future) dates.
    # For past (settled) dates, preserve the user's requested date even if there were no games.
    if df.empty and not settled:
        try:
            client = NHLWebClient()
            base = pd.to_datetime(date)
            for i in range(1, 11):
                d2 = (base + timedelta(days=i)).strftime("%Y-%m-%d")
                games = client.schedule_range(d2, d2)
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
                    snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    try:
                        predict_core(date=d2, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True)
                    except Exception:
                        pass
                    alt_path = PROC_DIR / f"predictions_{d2}.csv"
                    if alt_path.exists():
                        try:
                            df2 = _read_csv_fallback(alt_path)
                        except Exception:
                            df2 = pd.DataFrame()
                        if not df2.empty:
                            df = df2
                            note_msg = f"No games on {date}. Showing next slate on {d2}."
                            date = d2
                            break
                    if not alt_path.exists():
                        try:
                            predict_core(date=d2, source="web", odds_source="csv")
                        except Exception:
                            pass
                        if alt_path.exists():
                            try:
                                df2 = _read_csv_fallback(alt_path)
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
    if (not settled) and (not read_only):
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
                    dfn = _read_csv_fallback(p)
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
    # Ensure EV fields exist for display/recommendations: compute from probabilities and odds if missing
    if not df.empty:
        try:
            import math as _math
            from ..utils.odds import american_to_decimal, ev_unit
            def _num(v):
                if v is None:
                    return None
                try:
                    if isinstance(v, (int, float)):
                        f = float(v)
                        return f if _math.isfinite(f) else None
                    s = str(v).strip().replace(",", "")
                    if s == "":
                        return None
                    return float(s)
                except Exception:
                    return None
            def _ensure_ev_row(row: pd.Series, p_key: str, odds_key: str, ev_key: str):
                try:
                    ev_present = (ev_key in row) and (row.get(ev_key) is not None) and not (isinstance(row.get(ev_key), float) and pd.isna(row.get(ev_key)))
                    if ev_present:
                        return row
                    p = None
                    if p_key in row and pd.notna(row.get(p_key)):
                        p = float(row.get(p_key))
                        if not (0.0 <= p <= 1.0) or not _math.isfinite(p):
                            p = None
                    price = _num(row.get(odds_key)) if odds_key in row else None
                    if price is None:
                        # fallback to closing odds
                        close_map = {
                            "home_ml_odds": "close_home_ml_odds",
                            "away_ml_odds": "close_away_ml_odds",
                            "over_odds": "close_over_odds",
                            "under_odds": "close_under_odds",
                            "home_pl_-1.5_odds": "close_home_pl_-1.5_odds",
                            "away_pl_+1.5_odds": "close_away_pl_+1.5_odds",
                        }
                        ck = close_map.get(odds_key)
                        if ck and (ck in row):
                            price = _num(row.get(ck))
                    if (p is not None) and (price is not None):
                        dec = american_to_decimal(price)
                        if dec is not None and _math.isfinite(dec):
                            row[ev_key] = round(ev_unit(p, dec), 4)
                except Exception:
                    return row
                return row
            # Apply to DataFrame
            for i, r in df.iterrows():
                r = _ensure_ev_row(r, "p_home_ml", "home_ml_odds", "ev_home_ml")
                r = _ensure_ev_row(r, "p_away_ml", "away_ml_odds", "ev_away_ml")
                r = _ensure_ev_row(r, "p_over", "over_odds", "ev_over")
                r = _ensure_ev_row(r, "p_under", "under_odds", "ev_under")
                r = _ensure_ev_row(r, "p_home_pl_-1.5", "home_pl_-1.5_odds", "ev_home_pl_-1.5")
                r = _ensure_ev_row(r, "p_away_pl_+1.5", "away_pl_+1.5_odds", "ev_away_pl_+1.5")
                df.iloc[i] = r
        except Exception:
            pass
    rows = df.to_dict(orient="records") if not df.empty else []
    # Fallback/sanitization: if predictions CSV lacks projection fields (older files) or they are NaN, derive them now
    if rows:
        try:
            _pois_fb = PoissonGoals()
        except Exception:
            _pois_fb = None
        import math as _math
        for r in rows:
            # Helper to check NaN
            def _isnan(x):
                try:
                    return isinstance(x, float) and _math.isnan(x)
                except Exception:
                    return False
            # Clean up NaN text fields so Jinja doesn't render 'nan'
            for k in ("totals_pick", "ats_pick", "winner_actual", "result_total", "result_ats"):
                v = r.get(k)
                if _isnan(v):
                    r[k] = None
            # Compute projections if missing/NaN and we have probabilities
            mt = r.get("model_total")
            ph = r.get("p_home_ml")
            if (mt is None or _isnan(mt)) and ph is not None and not _isnan(ph) and _pois_fb is not None:
                try:
                    tl = r.get("total_line_used")
                    if tl is None or _isnan(tl):
                        tl = r.get("close_total_line_used")
                    tl_val = float(tl) if tl is not None and not _isnan(tl) else 6.0
                    ph_val = float(ph)
                    lam_h, lam_a = _pois_fb.lambdas_from_total_split(tl_val, ph_val)
                    r["proj_home_goals"] = round(float(lam_h), 2)
                    r["proj_away_goals"] = round(float(lam_a), 2)
                    r["model_total"] = round(float(lam_h + lam_a), 2)
                    r["model_spread"] = round(float(lam_h - lam_a), 2)
                except Exception:
                    # As a last resort, null fields if they are NaN
                    for k in ("proj_home_goals", "proj_away_goals", "model_total", "model_spread"):
                        v = r.get(k)
                        if _isnan(v):
                            r[k] = None
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
                    r_state = r.get("game_state") or g.get("gameState")
                    # If we have final scores but state not clearly final, force it
                    if (r.get("final_home_goals") is not None and r.get("final_away_goals") is not None) and (not r_state or "FINAL" not in str(r_state).upper()):
                        r_state = "FINAL"
                    r["game_state"] = r_state or "FINAL"
            except Exception:
                pass
        # Backfill outcome fields if missing (winner_actual, result_total, result_ats) using final scores.
        # Persist updates to the original predictions CSV only if we successfully compute at least one field.
        try:
            pred_csv_path = PROC_DIR / f"predictions_{date}.csv"
            df_pred = _read_csv_fallback(pred_csv_path) if pred_csv_path.exists() else pd.DataFrame()
        except Exception:
            df_pred = pd.DataFrame()
        backfilled = 0
        # Helper to look up model per-game total line for totals result; fallback order.
        for r in rows:
            try:
                fh = r.get("final_home_goals")
                fa = r.get("final_away_goals")
                if fh is None or fa is None:
                    continue
                # Skip if already populated
                # We still may need to compute correctness fields even if some present; do not early-continue yet.
                total_line = None
                for key in ("close_total_line_used", "total_line_used", "pl_line_used"):
                    v = r.get(key)
                    if v is None:
                        continue
                    s = str(v).strip()
                    if s == "":
                        continue
                    try:
                        total_line = float(s)
                        break
                    except Exception:
                        continue
                fh_i = int(fh); fa_i = int(fa)
                actual_total = fh_i + fa_i
                # winner_actual
                if not r.get("winner_actual"):
                    r["winner_actual"] = r.get("home") if fh_i > fa_i else (r.get("away") if fa_i > fh_i else "Draw")
                # result_total
                if total_line is not None and not r.get("result_total"):
                    if actual_total > total_line:
                        r["result_total"] = "Over"
                    elif actual_total < total_line:
                        r["result_total"] = "Under"
                    else:
                        r["result_total"] = "Push"
                # result_ats (puck line at -1.5 / +1.5)
                if not r.get("result_ats"):
                    diff = fh_i - fa_i
                    r["result_ats"] = "home_-1.5" if diff > 1.5 else "away_+1.5"
                # Populate actual_* convenience fields if absent
                if r.get("actual_home_goals") is None:
                    r["actual_home_goals"] = fh_i
                if r.get("actual_away_goals") is None:
                    r["actual_away_goals"] = fa_i
                if r.get("actual_total") is None:
                    r["actual_total"] = actual_total
                # winner_model (based on probabilities) if missing
                if not r.get("winner_model"):
                    try:
                        ph = float(r.get("p_home_ml")) if r.get("p_home_ml") is not None else None
                        pa = float(r.get("p_away_ml")) if r.get("p_away_ml") is not None else None
                        if ph is not None and pa is not None:
                            r["winner_model"] = r.get("home") if ph >= pa else r.get("away")
                    except Exception:
                        pass
                # winner_correct
                if r.get("winner_actual") and r.get("winner_model") and r.get("winner_correct") is None:
                    r["winner_correct"] = (r.get("winner_actual") == r.get("winner_model"))
                # total_diff (model_total - actual_total)
                if r.get("model_total") is not None and r.get("total_diff") is None:
                    try:
                        r["total_diff"] = round(float(r.get("model_total")) - float(actual_total), 2)
                    except Exception:
                        pass
                # totals_pick_correct
                if r.get("totals_pick") and r.get("result_total") and r.get("totals_pick_correct") is None:
                    if r.get("result_total") != "Push":
                        r["totals_pick_correct"] = (r.get("result_total") == r.get("totals_pick"))
                # ats_pick_correct
                if r.get("ats_pick") and r.get("result_ats") and r.get("ats_pick_correct") is None:
                    r["ats_pick_correct"] = (r.get("ats_pick") == r.get("result_ats"))
                backfilled += 1
            except Exception:
                pass
        # Persist backfill into CSV (match by home/away abbreviations for robustness)
        if backfilled and (not df_pred.empty) and (not read_only):
            def _abbr2(x: str) -> str:
                try:
                    return (get_team_assets(str(x)).get("abbr") or "").upper()
                except Exception:
                    return ""
            try:
                if {"home","away"}.issubset(df_pred.columns):
                    for r in rows:
                        hk = _abbr2(r.get("home")); ak = _abbr2(r.get("away"))
                        try:
                            mask = df_pred.apply(lambda rw: _abbr2(rw.get("home")) == hk and _abbr2(rw.get("away")) == ak, axis=1)
                        except Exception:
                            continue
                        if mask.any():
                            idx = df_pred.index[mask][0]
                            for col in ("winner_actual","result_total","result_ats","final_home_goals","final_away_goals","actual_home_goals","actual_away_goals","actual_total","winner_model","winner_correct","total_diff","totals_pick_correct","ats_pick_correct"):
                                if col not in df_pred.columns:
                                    df_pred[col] = pd.NA
                                val = r.get(col)
                                if val is not None and (pd.isna(df_pred.at[idx, col]) or df_pred.at[idx, col] in (None, "")):
                                    df_pred.at[idx, col] = val
                            # Persist normalized FINAL state if changed
                            try:
                                if "game_state" in r and r.get("game_state") and ("game_state" in df_pred.columns):
                                    cur_gs = df_pred.at[idx, "game_state"] if "game_state" in df_pred.columns else None
                                    if (cur_gs is None or str(cur_gs).strip() == "" or "FINAL" not in str(cur_gs).upper()) and "FINAL" in str(r.get("game_state")).upper():
                                        df_pred.at[idx, "game_state"] = r.get("game_state")
                            except Exception:
                                pass
                df_pred.to_csv(pred_csv_path, index=False)
                try:
                    print(f"[cards/backfill] date={date} rows_backfilled={backfilled}")
                except Exception:
                    pass
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
    # In read-only mode, do not show inferred odds by default
    allow_inferred = os.getenv("WEB_ALLOW_INFERRED_ODDS", "").strip().lower() in ("1", "true", "yes")
    inferred_map = {}
    if allow_inferred:
        try:
            inf_path = PROC_DIR / f"inferred_odds_{date}.csv"
            if inf_path.exists():
                dfi = _read_csv_fallback(inf_path)
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
            if allow_inferred and not _has(r.get("disp_home_ml_odds")):
                v = inferred_map.get((hn, an, "home_ml"))
                if _has(v):
                    r["disp_home_ml_odds"] = v
                    r["disp_home_ml_book"] = "Inferred"
            if allow_inferred and not _has(r.get("disp_away_ml_odds")):
                v = inferred_map.get((hn, an, "away_ml"))
                if _has(v):
                    r["disp_away_ml_odds"] = v
                    r["disp_away_ml_book"] = "Inferred"
            # Final fallback: infer ML odds directly from model probabilities (disabled unless allow_inferred)
            if allow_inferred and not _has(r.get("disp_home_ml_odds")):
                try:
                    ph = float(r.get("p_home_ml")) if r.get("p_home_ml") is not None else None
                except Exception:
                    ph = None
                if ph is not None:
                    am = _american_from_prob(ph)
                    if am is not None:
                        r["disp_home_ml_odds"] = am
                        r["disp_home_ml_book"] = "Inferred"
            if allow_inferred and not _has(r.get("disp_away_ml_odds")):
                try:
                    pa = float(r.get("p_away_ml")) if r.get("p_away_ml") is not None else None
                except Exception:
                    pa = None
                if pa is not None:
                    am = _american_from_prob(pa)
                    if am is not None:
                        r["disp_away_ml_odds"] = am
                        r["disp_away_ml_book"] = "Inferred"
            # Totals
            r["disp_over_odds"] = _fb(r.get("over_odds"), r.get("close_over_odds"))
            r["disp_under_odds"] = _fb(r.get("under_odds"), r.get("close_under_odds"))
            r["disp_over_book"] = _fb(r.get("over_book"), r.get("close_over_book"))
            r["disp_under_book"] = _fb(r.get("under_book"), r.get("close_under_book"))
            r["disp_total_line_used"] = _fb(r.get("total_line_used"), r.get("close_total_line_used"))
            # Inferred fallback for totals (line may remain unknown)
            if allow_inferred and not _has(r.get("disp_over_odds")):
                v = inferred_map.get((hn, an, "over"))
                if _has(v):
                    r["disp_over_odds"] = v
                    r["disp_over_book"] = "Inferred"
            if allow_inferred and not _has(r.get("disp_under_odds")):
                v = inferred_map.get((hn, an, "under"))
                if _has(v):
                    r["disp_under_odds"] = v
                    r["disp_under_book"] = "Inferred"
            # If still missing, avoid synthetic defaults in read-only mode
            if allow_inferred and not _has(r.get("disp_over_odds")) and (r.get("model_total") is not None or r.get("disp_total_line_used") is not None):
                r["disp_over_odds"] = -110
                r["disp_over_book"] = "Inferred"
            if allow_inferred and not _has(r.get("disp_under_odds")) and (r.get("model_total") is not None or r.get("disp_total_line_used") is not None):
                r["disp_under_odds"] = -110
                r["disp_under_book"] = "Inferred"
            # Puck line
            r["disp_home_pl_-1.5_odds"] = _fb(r.get("home_pl_-1.5_odds"), r.get("close_home_pl_-1.5_odds"))
            r["disp_away_pl_+1.5_odds"] = _fb(r.get("away_pl_+1.5_odds"), r.get("close_away_pl_+1.5_odds"))
            r["disp_home_pl_-1.5_book"] = _fb(r.get("home_pl_-1.5_book"), r.get("close_home_pl_-1.5_book"))
            r["disp_away_pl_+1.5_book"] = _fb(r.get("away_pl_+1.5_book"), r.get("close_away_pl_+1.5_book"))
            # Inferred fallback for puck line
            if allow_inferred and not _has(r.get("disp_home_pl_-1.5_odds")):
                v = inferred_map.get((hn, an, "home_pl_-1.5"))
                if _has(v):
                    r["disp_home_pl_-1.5_odds"] = v
                    r["disp_home_pl_-1.5_book"] = "Inferred"
            if allow_inferred and not _has(r.get("disp_away_pl_+1.5_odds")):
                v = inferred_map.get((hn, an, "away_pl_+1.5"))
                if _has(v):
                    r["disp_away_pl_+1.5_odds"] = v
                    r["disp_away_pl_+1.5_book"] = "Inferred"
            # Avoid synthetic defaults in read-only mode unless allowed
            if allow_inferred and not _has(r.get("disp_home_pl_-1.5_odds")):
                r["disp_home_pl_-1.5_odds"] = -110
                r["disp_home_pl_-1.5_book"] = "Inferred"
            if allow_inferred and not _has(r.get("disp_away_pl_+1.5_odds")):
                r["disp_away_pl_+1.5_odds"] = -110
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
    # Inline safety derivation: ensure outcome fields exist for ANY row that clearly has final scores (even if viewing a future-day slate that includes prior midnight-crossing games).
    if rows:
        try:
            any_persist_needed = False
            for r in rows:
                score_fields = []
                fh_raw = r.get("final_home_goals")
                fa_raw = r.get("final_away_goals")
                if fh_raw is None and r.get("actual_home_goals") is not None:
                    fh_raw = r.get("actual_home_goals")
                if fa_raw is None and r.get("actual_away_goals") is not None:
                    fa_raw = r.get("actual_away_goals")
                if fh_raw is None or fa_raw is None:
                    continue
                try:
                    fh_i = int(fh_raw); fa_i = int(fa_raw)
                except Exception:
                    continue
                # Force FINAL state if we have concrete scores
                gs = (r.get("game_state") or "").upper()
                if "FINAL" not in gs:
                    r["game_state"] = "FINAL"
                # Winner actual
                if not r.get("winner_actual"):
                    r["winner_actual"] = r.get("home") if fh_i > fa_i else (r.get("away") if fa_i > fh_i else "Draw")
                # Winner model
                if not r.get("winner_model"):
                    try:
                        ph = float(r.get("p_home_ml")) if r.get("p_home_ml") is not None else None
                        pa = float(r.get("p_away_ml")) if r.get("p_away_ml") is not None else None
                        if ph is not None and pa is not None:
                            r["winner_model"] = r.get("home") if ph >= pa else r.get("away")
                    except Exception:
                        pass
                if r.get("winner_correct") is None and r.get("winner_actual") and r.get("winner_model"):
                    r["winner_correct"] = (r.get("winner_actual") == r.get("winner_model"))
                # Totals logic
                total_line = None
                for key in ("close_total_line_used","total_line_used"):
                    v = r.get(key)
                    if v is None:
                        continue
                    try:
                        total_line = float(v); break
                    except Exception:
                        continue
                actual_total = fh_i + fa_i
                import math
                cur_at = r.get("actual_total")
                if (cur_at is None) or (isinstance(cur_at, float) and math.isnan(cur_at)):
                    r["actual_total"] = actual_total
                # Ensure component actual goals too
                cur_ah = r.get("actual_home_goals")
                if (cur_ah is None) or (isinstance(cur_ah, float) and math.isnan(cur_ah)):
                    r["actual_home_goals"] = fh_i
                cur_aa = r.get("actual_away_goals")
                if (cur_aa is None) or (isinstance(cur_aa, float) and math.isnan(cur_aa)):
                    r["actual_away_goals"] = fa_i
                if total_line is not None and not r.get("result_total"):
                    if actual_total > total_line:
                        r["result_total"] = "Over"
                    elif actual_total < total_line:
                        r["result_total"] = "Under"
                    else:
                        r["result_total"] = "Push"
                if r.get("totals_pick") and r.get("result_total") and r.get("totals_pick_correct") is None and r.get("result_total") != "Push":
                    r["totals_pick_correct"] = (r.get("totals_pick") == r.get("result_total"))
                # ATS puck line
                if not r.get("result_ats"):
                    diff = fh_i - fa_i
                    r["result_ats"] = "home_-1.5" if diff > 1.5 else "away_+1.5"
                if r.get("ats_pick") and r.get("result_ats") and r.get("ats_pick_correct") is None:
                    r["ats_pick_correct"] = (r.get("ats_pick") == r.get("result_ats"))
                # total_diff
                if r.get("model_total") is not None and (r.get("total_diff") is None or (isinstance(r.get("total_diff"), float) and math.isnan(r.get("total_diff")))):
                    try:
                        r["total_diff"] = round(float(r.get("model_total")) - float(actual_total), 2)
                    except Exception:
                        pass
                # Mark debug flag if any expected field still missing
                missing_keys = []
                for k in ("winner_actual","winner_model","winner_correct","result_total","result_ats","actual_total"):
                    val_chk = r.get(k)
                    missing = False
                    if val_chk is None or val_chk == "":
                        missing = True
                    else:
                        try:
                            if isinstance(val_chk, float) and math.isnan(val_chk):
                                missing = True
                        except Exception:
                            pass
                    if missing:
                        missing_keys.append(k)
                derived_any = False
                # Track if we set fields in this pass (simplistic: check keys we expect)
                for chk in ("winner_actual","winner_model","winner_correct","result_total","result_ats","totals_pick_correct","ats_pick_correct","actual_total","total_diff"):
                    if r.get(chk) is not None:
                        derived_any = True
                if derived_any:
                    any_persist_needed = True
                if missing_keys:
                    r["debug_missing_outcome"] = ",".join(missing_keys)
            # Persist back into predictions CSV if we derived anything
            if any_persist_needed:
                try:
                    pred_csv_path2 = PROC_DIR / f"predictions_{date}.csv"
                    if pred_csv_path2.exists():
                        df2 = _read_csv_fallback(pred_csv_path2)
                        if not df2.empty and {"home","away"}.issubset(df2.columns):
                            def _abbr3(x: str) -> str:
                                try:
                                    return (get_team_assets(str(x)).get("abbr") or "").upper()
                                except Exception:
                                    return ""
                            for r in rows:
                                hk = _abbr3(r.get("home")); ak = _abbr3(r.get("away"))
                                if not hk or not ak:
                                    continue
                                try:
                                    mask = df2.apply(lambda rw: _abbr3(rw.get("home")) == hk and _abbr3(rw.get("away")) == ak, axis=1)
                                except Exception:
                                    continue
                                if not mask.any():
                                    continue
                                idx = df2.index[mask][0]
                                for col in ("winner_actual","winner_model","winner_correct","result_total","result_ats","totals_pick_correct","ats_pick_correct","actual_home_goals","actual_away_goals","actual_total","total_diff","final_home_goals","final_away_goals"):
                                    if col not in df2.columns:
                                        df2[col] = pd.NA
                                    val = r.get(col)
                                    if val is not None:
                                        try:
                                            cur = df2.at[idx, col]
                                            if (isinstance(cur, float) and pd.isna(cur)) or cur in (None, ""):
                                                df2.at[idx, col] = val
                                        except Exception:
                                            pass
                            if not read_only:
                                df2.to_csv(pred_csv_path2, index=False)
                except Exception:
                    pass
        except Exception:
            pass
    template = env.get_template("cards.html")
    # Last update info for footer note
    lu = _last_update_info(date)
    html = template.render(
        date=date,
        original_date=requested_date,
        rows=rows,
        note=note_msg,
        live_now=live_now,
        settled=settled,
        last_updates=lu,
    )
    return HTMLResponse(content=html)


    


def _capture_openers_for_day(date: str) -> dict:
    """Persist first-seen 'opening' odds into predictions_{date}.csv.

    For each row, if open_* columns are missing or empty, copy current odds/book/line fields.
    Idempotent: does not overwrite existing open_* values.
    """
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return {"status": "no-file", "date": date}
    df = _read_csv_fallback(path)
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
        # Best-effort GitHub write-back for openers snapshot
        try:
            _gh_upsert_file_if_configured(path, f"web: capture openers for {date}")
        except Exception:
            pass
    return {"status": "ok", "updated": int(updated), "date": date}


    


@app.post("/api/capture-openers")
async def api_capture_openers(date: Optional[str] = Query(None)):
    date = date or _today_ymd()
    # Don't capture openers if slate is live; we want pregame numbers
    if _is_live_day(date):
        return JSONResponse({"status": "skipped-live", "date": date})
    res = _capture_openers_for_day(date)
    return JSONResponse(res, status_code=200 if res.get("status") == "ok" else 400)


## Scoreboard Stats API Cache
# Lightweight per-process cache for Stats API enrichment in scoreboard.
# Reduces repeated Stats API calls when nothing has changed for a game.
# Policy:
#  - Keyed by gamePk (int)
#  - Stores: last_fetch_ts (epoch seconds), signature tuple, cached stats fields
#  - Signature: (gameState, homeScore, awayScore, web_period)
#  - Refresh if > SCOREBOARD_STATS_MIN_REFRESH_SEC seconds since last fetch OR signature changed
#  - Purge entry when gameState starts with 'FINAL'
# Debug: pass ?debug_cache=1 to include debug metadata per game.

# Cache config constants
SCOREBOARD_STATS_MIN_REFRESH_SEC = 60  # user requested 60 second minimum

# Internal cache store
_SCOREBOARD_STATS_CACHE: dict[int, dict] = {}

@app.get("/api/scoreboard")
async def api_scoreboard(date: Optional[str] = Query(None), debug_cache: Optional[int] = Query(0)):
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
    # For LIVE games, try to enrich with linescore (with multi-endpoint fallbacks) to get precise period/clock
    try:
        now_ts = datetime.utcnow().timestamp()
        debug_mode = bool(int(debug_cache or 0))
        for r in rows:
            st = str(r.get("gameState") or "").upper()
            if any(k in st for k in ["LIVE", "IN", "PROGRESS", "CRIT"]) and r.get("gamePk"):
                try:
                    ls = client.linescore(int(r.get("gamePk")))  # now may include fallback extraction
                    if ls:
                        if ls.get("period") is not None:
                            r["period"] = ls.get("period")
                        if ls.get("clock"):
                            r["clock"] = ls.get("clock")
                            r["source_clock"] = f"web-{ls.get('source') or 'linescore'}"
                except Exception:
                    pass
                # Decide whether to call Stats API based on cache
                try:
                    game_pk = int(r.get("gamePk"))
                    sig = (
                        st,
                        r.get("homeScore"),
                        r.get("awayScore"),
                        r.get("period"),
                    )
                    entry = _SCOREBOARD_STATS_CACHE.get(game_pk)
                    should_fetch = False
                    reason = None
                    if entry is None:
                        should_fetch = True; reason = "miss"
                    else:
                        age = now_ts - entry.get("last_fetch_ts", 0)
                        if sig != entry.get("signature"):
                            should_fetch = True; reason = "signature-change"
                        elif age > SCOREBOARD_STATS_MIN_REFRESH_SEC:
                            should_fetch = True; reason = "stale"
                    if should_fetch:
                        stats_client = NHLStatsClient()
                        glf = stats_client.game_live_feed(game_pk)
                        live = (glf or {}).get("liveData", {})
                        ls2 = live.get("linescore", {})
                        inter = ls2.get("intermissionInfo", {}) if isinstance(ls2, dict) else {}
                        in_inter = bool(inter.get("inIntermission"))
                        clock2 = ls2.get("currentPeriodTimeRemaining")
                        clock_val = None
                        if isinstance(clock2, str) and clock2:
                            if clock2.strip().upper() == "END":
                                clock2 = "0:00"
                            clock_val = clock2
                        # currentPlay fallback
                        if not clock_val:
                            curp = live.get("plays", {}).get("currentPlay", {}).get("about", {})
                            clock3 = curp.get("periodTimeRemaining")
                            if isinstance(clock3, str) and clock3:
                                clock_val = clock3
                        curp = live.get("plays", {}).get("currentPlay", {}).get("about", {})
                        per2 = (ls2.get("currentPeriod") if isinstance(ls2, dict) else None) or (ls2.get("period") if isinstance(ls2, dict) else None) or curp.get("period")
                        cached_stats = {}
                        if clock_val:
                            cached_stats["clock"] = clock_val
                            cached_stats["source_clock"] = "stats"
                        if per2 is not None:
                            cached_stats["period"] = per2
                        cached_stats["intermission"] = in_inter
                        _SCOREBOARD_STATS_CACHE[game_pk] = {
                            "last_fetch_ts": now_ts,
                            "signature": sig,
                            "cached_stats": cached_stats,
                        }
                        if debug_mode:
                            cached_stats["_debug_fetch_reason"] = reason
                    else:
                        # Reuse cached stats
                        cached_stats = entry.get("cached_stats", {}) if entry else {}
                        if debug_mode and cached_stats is not None:
                            # add age and reuse reason
                            cached_stats = dict(cached_stats)  # shallow copy
                            cached_stats["_debug_cache_age"] = round(now_ts - entry.get("last_fetch_ts", 0), 2)
                            cached_stats["_debug_fetch_reason"] = "cache-hit"
                    # Merge cached stats into row
                    for k, v in (cached_stats or {}).items():
                        if k.startswith("_debug_") and not debug_mode:
                            continue
                        r[k] = v
                except Exception:
                    pass
            # Derive display period and intermission flag
            try:
                per = r.get("period")
                st2 = str(r.get("gameState") or "").upper()
                period_disp = None
                if st2.startswith("FINAL"):
                    period_disp = "Final"
                else:
                    if per is not None:
                        try:
                            p_int = int(per)
                        except Exception:
                            p_int = None
                        if p_int == 1: period_disp = "P1"
                        elif p_int == 2: period_disp = "P2"
                        elif p_int == 3: period_disp = "P3"
                        elif p_int == 4: period_disp = "OT"
                        elif p_int == 5: period_disp = "SO"
                        else: period_disp = f"P{per}"
                r["period_disp"] = period_disp
                # If stats intermission flag not set earlier, ensure boolean present
                if "intermission" not in r:
                    r["intermission"] = False
            except Exception:
                pass
    except Exception:
        pass
    # Purge cache for finished games
    try:
        done_keys = [gid for gid, e in _SCOREBOARD_STATS_CACHE.items() if any(str(g.get("gamePk")) == str(gid) and str(g.get("gameState") or "").upper().startswith("FINAL") for g in rows)]
        for k in done_keys:
            _SCOREBOARD_STATS_CACHE.pop(k, None)
    except Exception:
        pass
    # Attach a fetched_at timestamp (UTC)
    fetched_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for r in rows:
        r["fetched_at"] = fetched_at
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
    # Robust read to avoid decode/empty errors across environments
    df = _read_csv_fallback(path)
    return JSONResponse(_df_jsonsafe_records(df))


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
    market: Optional[str] = Query(None, description="Filter by market: SOG, SAVES, GOALS, ASSISTS, POINTS"),
    min_ev: float = Query(0.0, description="Minimum EV threshold for ev_over"),
    top: int = Query(50, description="Top N to return after filtering/sorting by EV desc"),
):
    path = PROC_DIR / "props_predictions.csv"
    if not path.exists():
        # Return an empty list instead of 404 so the UI can render gracefully
        return JSONResponse([], status_code=200)
    df = _read_csv_fallback(path)
    if market:
        df = df[df["market"].str.upper() == market.upper()]
    if "ev_over" in df.columns:
        df = df[df["ev_over"].astype(float) >= float(min_ev)]
        df = df.sort_values("ev_over", ascending=False)
    if top and top > 0:
        df = df.head(top)
    return JSONResponse(_df_jsonsafe_records(df))


@app.get("/api/player-props")
async def api_player_props(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    market: Optional[str] = Query(None, description="SOG,SAVES,GOALS,ASSISTS,POINTS"),
):
    """Return canonical player props lines for a date from data/props/player_props_lines/date=YYYY-MM-DD/*.parquet."""
    date = date or _today_ymd()
    try:
        base = PROC_DIR.parent / "props" / f"player_props_lines/date={date}"
        parts = []
        for name in ("bovada.parquet", "oddsapi.parquet"):
            p = base / name
            if p.exists():
                try:
                    parts.append(pd.read_parquet(p))
                except Exception:
                    pass
        if not parts:
            return JSONResponse({"date": date, "data": []})
        df = pd.concat(parts, ignore_index=True)
        if market:
            df = df[df["market"].astype(str).str.upper() == market.upper()]
        # Lightweight response
        keep = [c for c in ["date","player_name","player_id","team","market","line","over_price","under_price","book","is_current"] if c in df.columns]
        out = _df_jsonsafe_records(df[keep].rename(columns={"player_name":"player"}))
        return JSONResponse({"date": date, "data": out})
    except Exception as e:
        return JSONResponse({"date": date, "error": str(e), "data": []}, status_code=200)


@app.get("/api/props/recommendations")
async def api_props_recommendations(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    market: Optional[str] = Query(None, description="SOG,SAVES,GOALS,ASSISTS,POINTS"),
    min_ev: float = Query(0.0),
    top: int = Query(200),
):
    """Serve props recommendations for a given date. If cached CSV exists, read; else compute on the fly via CLI logic."""
    date = date or _today_ymd()
    # Respect read-only mode: if cache missing, do not compute on-demand
    read_only_ui = _read_only(date)
    rec_path = PROC_DIR / f"props_recommendations_{date}.csv"
    df = None
    if rec_path.exists():
        try:
            df = pd.read_csv(rec_path)
        except Exception:
            df = None
    if (df is None or df.empty) and (not read_only_ui):
        # Compute on the fly by invoking the same logic inline (avoid spawning a subprocess)
        try:
            from ..models.props import SkaterShotsModel, GoalieSavesModel, SkaterGoalsModel
            from ..models.props import SkaterAssistsModel, SkaterPointsModel, SkaterBlocksModel
            from ..data.collect import collect_player_game_stats
            from ..utils.io import RAW_DIR
            # Load canonical lines
            base = PROC_DIR.parent / "props" / f"player_props_lines/date={date}"
            parts = []
            for name in ("bovada.parquet", "oddsapi.parquet"):
                p = base / name
                if p.exists():
                    try:
                        parts.append(pd.read_parquet(p))
                    except Exception:
                        pass
            if not parts:
                return JSONResponse({"date": date, "data": []})
            lines = pd.concat(parts, ignore_index=True)
            # Ensure history exists
            stats_path = RAW_DIR / "player_game_stats.csv"
            if not stats_path.exists():
                try:
                    from datetime import datetime as _dt, timedelta as _td
                    start = (_dt.strptime(date, "%Y-%m-%d") - _td(days=365)).strftime("%Y-%m-%d")
                    collect_player_game_stats(start, date, source="stats")
                except Exception:
                    pass
            hist = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
            shots = SkaterShotsModel(); saves = GoalieSavesModel(); goals = SkaterGoalsModel(); assists = SkaterAssistsModel(); points = SkaterPointsModel(); blocks = SkaterBlocksModel()
            def proj_prob(m, player, ln):
                m = (m or '').upper()
                if m == 'SOG':
                    lam = shots.player_lambda(hist, player); return lam, shots.prob_over(lam, ln)
                if m == 'SAVES':
                    lam = saves.player_lambda(hist, player); return lam, saves.prob_over(lam, ln)
                if m == 'GOALS':
                    lam = goals.player_lambda(hist, player); return lam, goals.prob_over(lam, ln)
                if m == 'ASSISTS':
                    lam = assists.player_lambda(hist, player); return lam, assists.prob_over(lam, ln)
                if m == 'POINTS':
                    lam = points.player_lambda(hist, player); return lam, points.prob_over(lam, ln)
                if m == 'BLOCKS':
                    lam = blocks.player_lambda(hist, player); return lam, blocks.prob_over(lam, ln)
                return None, None
            recs = []
            for _, r in lines.iterrows():
                m = str(r.get('market') or '').upper()
                if market and m != market.upper():
                    continue
                player = r.get('player_name') or r.get('player')
                if not player:
                    continue
                try:
                    ln = float(r.get('line'))
                except Exception:
                    continue
                op = r.get('over_price'); up = r.get('under_price')
                if pd.isna(op) and pd.isna(up):
                    continue
                lam, p_over = proj_prob(m, str(player), ln)
                if lam is None or p_over is None:
                    continue
                # EV calc
                def _dec(a):
                    try:
                        a = float(a); return 1.0 + (a/100.0) if a > 0 else 1.0 + (100.0/abs(a))
                    except Exception:
                        return None
                ev_o = (p_over * (_dec(op)-1.0) - (1.0 - p_over)) if (op is not None and _dec(op) is not None) else None
                p_under = max(0.0, 1.0 - float(p_over))
                ev_u = (p_under * (_dec(up)-1.0) - (1.0 - p_under)) if (up is not None and _dec(up) is not None) else None
                side = None; price = None; ev = None
                if ev_o is not None or ev_u is not None:
                    if (ev_u is None) or (ev_o is not None and ev_o >= ev_u):
                        side = 'Over'; price = op; ev = ev_o
                    else:
                        side = 'Under'; price = up; ev = ev_u
                if ev is None or not (float(ev) >= float(min_ev)):
                    continue
                recs.append({
                    'date': date,
                    'player': player,
                    'team': r.get('team') or None,
                    'market': m,
                    'line': ln,
                    'proj': float(lam),
                    'p_over': float(p_over),
                    'over_price': op if pd.notna(op) else None,
                    'under_price': up if pd.notna(up) else None,
                    'book': r.get('book'),
                    'side': side,
                    'ev': float(ev) if ev is not None else None,
                })
            df = pd.DataFrame(recs)
            if not df.empty:
                df = df.sort_values('ev', ascending=False)
                if top and top > 0:
                    df = df.head(top)
            try:
                save_df(df, rec_path)
                try:
                    _gh_upsert_file_if_configured(rec_path, f"web: update props recommendations for {date}")
                except Exception:
                    pass
            except Exception:
                pass
        except Exception as e:
            return JSONResponse({"date": date, "error": str(e), "data": []}, status_code=200)
    # Apply API filters on cached df
    if df is None:
        # In read-only mode, serve empty if cache missing
        df = pd.read_csv(rec_path) if rec_path.exists() else pd.DataFrame()
    if market and not df.empty and 'market' in df.columns:
        df = df[df['market'].astype(str).str.upper() == market.upper()]
    try:
        if not df.empty and 'ev' in df.columns:
            df = df[df['ev'].astype(float) >= float(min_ev)].sort_values('ev', ascending=False)
        if not df.empty and top and top > 0:
            df = df.head(top)
    except Exception:
        pass
    # Serialize safely: replace Inf/NaN and return a proper JSONResponse with standard-compliant values
    if df is None or df.empty:
        rows = []
    else:
        _df = df.replace([np.inf, -np.inf], np.nan)
        # Convert NaN -> None for JSON compliance
        rows = _df.where(pd.notnull(_df), None).to_dict(orient='records')
    return JSONResponse({"date": str(date), "data": rows})


@app.get("/api/player-props-reconciliation")
async def api_player_props_reconciliation(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    refresh: int = Query(0, description="If 1, recompute instead of reading cache"),
):
    """Join canonical props lines with realized stats for the date to compare projections vs actuals."""
    date = date or _today_ymd()
    cache = PROC_DIR / f"player_props_vs_actuals_{date}.csv"
    if refresh == 0 and cache.exists():
        try:
            df = _read_csv_fallback(cache)
            return JSONResponse({"date": date, "data": _df_jsonsafe_records(df)})
        except Exception:
            pass
    # Build on the fly using existing CLI utilities
    try:
        from ..utils.io import RAW_DIR
        # Load canonical lines
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
        # Ensure stats exist for the date
        stats_path = RAW_DIR / "player_game_stats.csv"
        stats = _read_csv_fallback(stats_path) if stats_path.exists() else pd.DataFrame()
        stats['date_key'] = pd.to_datetime(stats['date'], errors='coerce').dt.strftime('%Y-%m-%d') if not stats.empty else pd.Series(dtype=str)
        stats_day = stats[stats['date_key'] == date].copy() if not stats.empty else pd.DataFrame()
        # Assemble reconciliation rows
        if lines.empty or stats_day.empty:
            save_df(pd.DataFrame(), cache)
            return JSONResponse({"date": date, "data": []})
        left = lines.rename(columns={"date":"date_key","player_name":"player"}).copy()
        keep_stats = [c for c in ['player','shots','goals','assists','saves','blocked'] if c in stats_day.columns]
        right = stats_day[['date_key'] + keep_stats]
        merged = left.merge(right, on=['date_key','player'], how='left', suffixes=('', '_act'))
        # Compute actual numeric per market
        def _act(row):
            m = str(row.get('market') or '').upper()
            if m == 'SOG': return row.get('shots')
            if m == 'GOALS': return row.get('goals')
            if m == 'SAVES': return row.get('saves')
            if m == 'ASSISTS': return row.get('assists')
            if m == 'POINTS':
                try:
                    g = float(row.get('goals') or 0); a = float(row.get('assists') or 0); return g+a
                except Exception:
                    return None
            if m == 'BLOCKS': return row.get('blocked')
            return None
        merged['actual'] = merged.apply(_act, axis=1)
        try:
            save_df(merged, cache)
            try:
                _gh_upsert_file_if_configured(cache, f"web: update props reconciliation for {date}")
            except Exception:
                pass
        except Exception:
            pass
        return JSONResponse({"date": date, "data": _df_jsonsafe_records(merged)})
    except Exception as e:
        return JSONResponse({"date": date, "error": str(e), "data": []}, status_code=200)


@app.post("/api/cron/props-collect")
async def api_cron_props_collect(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD; defaults to ET today"),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional alternative to token query param)"),
):
    """Secure endpoint to collect canonical player props lines (Parquet) for a date.

    - Writes data/props/player_props_lines/date=YYYY-MM-DD/{bovada,oddsapi}.parquet
    - Best-effort upserts resulting Parquet files to GitHub
    """
    secret = os.getenv("REFRESH_CRON_TOKEN", "")
    supplied = (token or "").strip()
    if (not supplied) and authorization:
        try:
            auth = str(authorization)
            if auth.lower().startswith("bearer "):
                supplied = auth.split(" ", 1)[1].strip()
        except Exception:
            supplied = supplied
    if not (secret and supplied and _const_time_eq(supplied, secret)):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    d = _normalize_date_param(date)
    try:
        from ..data import player_props as props_data
        base = PROC_DIR.parent / "props" / f"player_props_lines/date={d}"
        base.mkdir(parents=True, exist_ok=True)
        out = {"date": d, "written": [], "errors": []}
        # Collect from Bovada and Odds API
        for which, src in (("bovada", "bovada"), ("oddsapi", "oddsapi")):
            try:
                cfg = props_data.PropsCollectionConfig(output_root=str(PROC_DIR.parent / "props"), book=which, source=src)
                res = props_data.collect_and_write(d, roster_df=None, cfg=cfg)
                path = res.get("output_path")
                if path:
                    out["written"].append(str(path))
                    try:
                        _gh_upsert_file_if_configured(Path(path), f"web: update props lines {which} for {d}")
                    except Exception:
                        pass
            except Exception as e:
                out["errors"].append({"book": which, "error": str(e)})
        return JSONResponse({"ok": True, **out})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "date": d}, status_code=500)


@app.post("/api/cron/props-recommendations")
async def api_cron_props_recommendations(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD; defaults to ET today"),
    market: Optional[str] = Query(None, description="SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS"),
    min_ev: float = Query(0.0),
    top: int = Query(200),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional alternative to token query param)"),
):
    """Secure endpoint to compute props recommendations for a date and push CSV to GitHub.

    - If Parquet lines are missing, attempts to collect them first via props-collect
    - Writes data/processed/props_recommendations_{date}.csv and upserts to GitHub
    """
    secret = os.getenv("REFRESH_CRON_TOKEN", "")
    supplied = (token or "").strip()
    if (not supplied) and authorization:
        try:
            auth = str(authorization)
            if auth.lower().startswith("bearer "):
                supplied = auth.split(" ", 1)[1].strip()
        except Exception:
            supplied = supplied
    if not (secret and supplied and _const_time_eq(supplied, secret)):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    d = _normalize_date_param(date)
    # Ensure lines exist; if not, collect
    try:
        base = PROC_DIR.parent / "props" / f"player_props_lines/date={d}"
        need_collect = not ((base / "bovada.parquet").exists() or (base / "oddsapi.parquet").exists())
        if need_collect:
            # Best-effort
            try:
                await api_cron_props_collect(token=token, date=d)
            except Exception:
                pass
    except Exception:
        pass
    # Compute recommendations using the same logic as api_props_recommendations (compute branch)
    try:
        from ..models.props import SkaterShotsModel, GoalieSavesModel, SkaterGoalsModel
        from ..models.props import SkaterAssistsModel, SkaterPointsModel, SkaterBlocksModel
        from ..data.collect import collect_player_game_stats
        # Load lines
        parts = []
        for name in ("bovada.parquet", "oddsapi.parquet"):
            p = base / name
            if p.exists():
                try:
                    parts.append(pd.read_parquet(p))
                except Exception:
                    pass
        if not parts:
            return JSONResponse({"ok": True, "date": d, "rows": 0, "message": "no-lines"})
        lines = pd.concat(parts, ignore_index=True)
        # Ensure stats exist for projection
        stats_path = RAW_DIR / "player_game_stats.csv"
        if not stats_path.exists():
            try:
                from datetime import datetime as _dt, timedelta as _td
                start = (_dt.strptime(d, "%Y-%m-%d") - _td(days=365)).strftime("%Y-%m-%d")
                collect_player_game_stats(start, d, source="stats")
            except Exception:
                pass
        hist = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
        shots = SkaterShotsModel(); saves = GoalieSavesModel(); goals = SkaterGoalsModel(); assists = SkaterAssistsModel(); points = SkaterPointsModel(); blocks = SkaterBlocksModel()
        def proj_prob(m, player, ln):
            m = (m or '').upper()
            if m == 'SOG':
                lam = shots.player_lambda(hist, player); return lam, shots.prob_over(lam, ln)
            if m == 'SAVES':
                lam = saves.player_lambda(hist, player); return lam, saves.prob_over(lam, ln)
            if m == 'GOALS':
                lam = goals.player_lambda(hist, player); return lam, goals.prob_over(lam, ln)
            if m == 'ASSISTS':
                lam = assists.player_lambda(hist, player); return lam, assists.prob_over(lam, ln)
            if m == 'POINTS':
                lam = points.player_lambda(hist, player); return lam, points.prob_over(lam, ln)
            if m == 'BLOCKS':
                lam = blocks.player_lambda(hist, player); return lam, blocks.prob_over(lam, ln)
            return None, None
        recs = []
        for _, r in lines.iterrows():
            m = str(r.get('market') or '').upper()
            if market and m != market.upper():
                continue
            player = r.get('player_name') or r.get('player')
            if not player:
                continue
            try:
                ln = float(r.get('line'))
            except Exception:
                continue
            op = r.get('over_price'); up = r.get('under_price')
            if pd.isna(op) and pd.isna(up):
                continue
            lam, p_over = proj_prob(m, str(player), ln)
            if lam is None or p_over is None:
                continue
            # EV calc
            def _dec(a):
                try:
                    a = float(a); return 1.0 + (a/100.0) if a > 0 else 1.0 + (100.0/abs(a))
                except Exception:
                    return None
            ev_o = (p_over * (_dec(op)-1.0) - (1.0 - p_over)) if (op is not None and _dec(op) is not None) else None
            p_under = max(0.0, 1.0 - float(p_over))
            ev_u = (p_under * (_dec(up)-1.0) - (1.0 - p_under)) if (up is not None and _dec(up) is not None) else None
            side = None; price = None; ev = None
            if ev_o is not None or ev_u is not None:
                if (ev_u is None) or (ev_o is not None and ev_o >= ev_u):
                    side = 'Over'; price = op; ev = ev_o
                else:
                    side = 'Under'; price = up; ev = ev_u
            if ev is None or not (float(ev) >= float(min_ev)):
                continue
            recs.append({
                'date': d,
                'player': player,
                'team': r.get('team') or None,
                'market': m,
                'line': ln,
                'proj': float(lam),
                'p_over': float(p_over),
                'over_price': op if pd.notna(op) else None,
                'under_price': up if pd.notna(up) else None,
                'book': r.get('book'),
                'side': side,
                'ev': float(ev) if ev is not None else None,
            })
        df = pd.DataFrame(recs)
        if not df.empty:
            df = df.sort_values('ev', ascending=False)
            if top and top > 0:
                df = df.head(int(top))
        rec_path = PROC_DIR / f"props_recommendations_{d}.csv"
        try:
            save_df(df, rec_path)
            try:
                _gh_upsert_file_if_configured(rec_path, f"web: update props recommendations for {d}")
            except Exception:
                pass
        except Exception:
            pass
        return JSONResponse({"ok": True, "date": d, "rows": 0 if df is None or df.empty else int(len(df)), "path": str(rec_path)})
    except Exception as e:
        return JSONResponse({"ok": False, "date": d, "error": str(e)}, status_code=500)

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
            data = _json.loads(resp.body)
            # Ensure list payload; if dict (e.g., error), render as empty
            rows = data if isinstance(data, list) else []
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
    return JSONResponse(_df_jsonsafe_records(df))


@app.get("/api/refresh-odds")
async def api_refresh_odds(
    date: Optional[str] = Query(None),
    snapshot: Optional[str] = Query(None),
    bankroll: float = Query(0.0, description="Bankroll for Kelly sizing; 0 disables"),
    kelly_fraction_part: float = Query(0.5, description="Kelly fraction, e.g., 0.5 for half-Kelly"),
    backfill: bool = Query(False, description="If true, during live slates only fill missing odds without overwriting existing prices"),
    overwrite_prestart: bool = Query(False, description="If true, allow refresh even during live days and overwrite odds for games that have not started yet"),
):
    """Refresh odds/predictions for a date. Tries Bovada, then Odds API; ensures predictions CSV exists; recomputes recs.

    This simplified implementation replaces a previously inlined version that was accidentally disrupted.
    """
    date = date or _today_ymd()
    if not snapshot:
        snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Skip refresh during live slates unless explicitly allowed
    try:
        if _is_live_day(date) and not (backfill or overwrite_prestart):
            return JSONResponse({"status": "skipped-live", "date": date, "message": "Live games in progress; odds refresh skipped."}, status_code=200)
    except Exception:
        pass
    # Ensure base models exist
    try:
        await _ensure_models(quick=True)
    except Exception:
        pass
    # Step 1: Bovada odds
    try:
        predict_core(date=date, source="web", odds_source="bovada", snapshot=snapshot, odds_best=True, bankroll=bankroll, kelly_fraction_part=kelly_fraction_part)
    except Exception:
        pass
    # Step 2: If still no odds, try Odds API (DK preferred)
    try:
        path = PROC_DIR / f"predictions_{date}.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        df = pd.DataFrame()
    if df.empty or not any(col in df.columns and df[col].notna().any() for col in ["home_ml_odds","away_ml_odds","over_odds","under_odds"]):
        try:
            predict_core(date=date, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=False, odds_bookmaker="draftkings", bankroll=bankroll, kelly_fraction_part=kelly_fraction_part)
        except Exception:
            pass
    # Ensure predictions file exists
    try:
        path = PROC_DIR / f"predictions_{date}.csv"
        if not path.exists():
            predict_core(date=date, source="web", odds_source="csv")
    except Exception:
        pass
    # Recompute edges and recommendations (best-effort)
    try:
        await _recompute_edges_and_recommendations(date)
    except Exception:
        pass
    return JSONResponse({"status": "ok", "date": date})

@app.get("/props/recommendations")
async def props_recommendations_page(
    date: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    min_ev: float = Query(0.0),
    top: int = Query(200),
):
    date = date or _today_ymd()
    read_only_ui = _read_only(date)
    # Read directly from processed CSV to avoid JSON roundtrip issues
    rows = []
    try:
        path = PROC_DIR / f"props_recommendations_{date}.csv"
        if path.exists():
            _df = pd.read_csv(path)
            if market and 'market' in _df.columns:
                _df = _df[_df['market'].astype(str).str.upper() == market.upper()]
            if 'ev' in _df.columns:
                _df = _df[_df['ev'].astype(float) >= float(min_ev)].sort_values('ev', ascending=False)
            if top and top > 0:
                _df = _df.head(top)
            # Sanitize for template
            _df = _df.replace([np.inf, -np.inf], np.nan)
            rows = _df.where(pd.notnull(_df), None).to_dict(orient='records')
        else:
            # No cache yet; show empty and let cron/CLI populate later (avoid inline compute in read-only)
            rows = []
    except Exception:
        rows = []
    template = env.get_template("props_recommendations.html")
    html = template.render(date=date, market=market or "", min_ev=min_ev, top=top, rows=rows)
    return HTMLResponse(content=html)

@app.get("/props/reconciliation")
async def props_reconciliation_page(
    date: Optional[str] = Query(None),
    refresh: int = Query(0),
):
    date = date or _today_ymd()
    # Reuse API
    resp = await api_player_props_reconciliation(date=date, refresh=refresh)
    rows = []
    try:
        import json as _json
        js = _json.loads(resp.body)
        rows = js.get("data") or []
    except Exception:
        rows = []
    template = env.get_template("props_reconciliation.html")
    html = template.render(date=date, refresh=(refresh==1), rows=rows)
    return HTMLResponse(content=html)


@app.post("/api/cron/refresh-bovada")
async def api_cron_refresh_bovada(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD; defaults to ET today"),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional alternative to token query param)"),
):
    """Cron-friendly endpoint to backfill Bovada odds for the slate.

    - Requires REFRESH_CRON_TOKEN env var (token must match).
    - Defaults to today's ET date.
    - Runs in backfill mode: fills missing odds without overwriting existing numbers.
    """
    secret = os.getenv("REFRESH_CRON_TOKEN", "")
    # Prefer explicit token param; else, parse Bearer token from header if present
    supplied = (token or "").strip()
    if (not supplied) and authorization:
        try:
            auth = str(authorization)
            if auth.lower().startswith("bearer "):
                supplied = auth.split(" ", 1)[1].strip()
        except Exception:
            supplied = supplied
    if not (secret and supplied and _const_time_eq(supplied, secret)):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    d = _normalize_date_param(date)
    # Ensure models exist quickly; safe even if already present
    try:
        await _ensure_models(quick=True)
    except Exception:
        pass
    # Reuse existing refresh logic in backfill mode
    try:
        # Use backfill to avoid clobbering existing values and overwrite_prestart to update pre-start games during the day
        res = await api_refresh_odds(date=d, snapshot=None, bankroll=0.0, kelly_fraction_part=0.5, backfill=True, overwrite_prestart=True)
        # Normalize response in case it's a JSONResponse
        if isinstance(res, JSONResponse):
            try:
                import json as _json
                body = _json.loads(res.body)
            except Exception:
                body = {"status": "ok"}
            # After refresh, attempt to capture closing for any FINAL games (idempotent)
            try:
                clo = _capture_closing_for_day(d)
                body["closing"] = clo
            except Exception:
                pass
            try:
                body["settlement"] = _backfill_settlement_for_date(d)
            except Exception:
                pass
            return JSONResponse({"ok": True, "date": d, "result": body})
        # If non-JSONResponse path, still try closing capture
        out = {"ok": True, "date": d, "result": res}
        try:
            clo = _capture_closing_for_day(d)
            out["closing"] = clo
        except Exception:
            pass
        try:
            out["settlement"] = _backfill_settlement_for_date(d)
        except Exception:
            pass
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"ok": False, "date": d, "error": str(e)}, status_code=500)


@app.post("/api/cron/retune")
async def api_cron_retune(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional alternative to token query param)"),
):
    """Cron-friendly endpoint to run a quick model retune using yesterday's completed games (ET).

    - Requires REFRESH_CRON_TOKEN env var (token must match).
    - Updates Elo ratings, trends, and lightly blends base_mu.
    """
    secret = os.getenv("REFRESH_CRON_TOKEN", "")
    supplied = (token or "").strip()
    if (not supplied) and authorization:
        try:
            auth = str(authorization)
            if auth.lower().startswith("bearer "):
                supplied = auth.split(" ", 1)[1].strip()
        except Exception:
            supplied = supplied
    if not (secret and supplied and _const_time_eq(supplied, secret)):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    # Import here to avoid circular imports
    try:
        from nhl_betting.scripts.daily_update import quick_retune_from_yesterday as _retune
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"import-failed: {e}"}, status_code=500)
    try:
        res = _retune(verbose=False)
        return JSONResponse({"ok": True, "result": res})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/api/cron/capture-closing")
async def api_cron_capture_closing(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD; defaults to ET today"),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional alternative to token query param)"),
):
    """Cron-friendly endpoint to capture closing odds for all FINAL games on a date.

    - Requires REFRESH_CRON_TOKEN env var (token must match).
    - Safe and idempotent: fills close_* only if empty.
    """
    secret = os.getenv("REFRESH_CRON_TOKEN", "")
    supplied = (token or "").strip()
    if (not supplied) and authorization:
        try:
            auth = str(authorization)
            if auth.lower().startswith("bearer "):
                supplied = auth.split(" ", 1)[1].strip()
        except Exception:
            supplied = supplied
    if not (secret and supplied and _const_time_eq(supplied, secret)):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    d = _normalize_date_param(date)
    try:
        res = _capture_closing_for_day(d)
        out = {"ok": True, "date": d, "result": res}
        try:
            out["settlement"] = _backfill_settlement_for_date(d)
        except Exception:
            pass
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"ok": False, "date": d, "error": str(e)}, status_code=500)


@app.post("/api/debug/push-test")
async def api_debug_push_test(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional alternative to token query param)"),
):
    """Write a tiny file under data/processed and attempt to upsert it to GitHub to validate settings.

    Protected by REFRESH_CRON_TOKEN to avoid abuse.
    """
    secret = os.getenv("REFRESH_CRON_TOKEN", "")
    supplied = (token or "").strip()
    if (not supplied) and authorization:
        try:
            auth = str(authorization)
            if auth.lower().startswith("bearer "):
                supplied = auth.split(" ", 1)[1].strip()
        except Exception:
            supplied = supplied
    if not (secret and supplied and _const_time_eq(supplied, secret)):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        p = PROC_DIR / "_gh_push_test.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"push-test {now}\n")
        res = _gh_upsert_file_if_configured(p, f"web: push-test {now}")
        return JSONResponse({"ok": True, "path": str(p), "result": res})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


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


@app.get("/api/cron/config")
def api_cron_config():
    """Tiny diagnostics: tell if cron and GitHub tokens are configured (booleans only)."""
    try:
        cron_ok = bool(os.getenv("REFRESH_CRON_TOKEN", "").strip())
    except Exception:
        cron_ok = False
    try:
        gh_ok = bool(os.getenv("GITHUB_TOKEN", "").strip())
    except Exception:
        gh_ok = False
    return JSONResponse({
        "cron_token_configured": cron_ok,
        "github_token_configured": gh_ok,
    })


# === Props stats calibration (stats-only) ===
def _latest_props_calibration_file() -> Optional[Path]:
    try:
        # Look for files like props_stats_calibration_*.json in processed dir
        cand = sorted(
            [p for p in PROC_DIR.glob("props_stats_calibration_*.json") if p.is_file()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        return cand[0] if cand else None
    except Exception:
        return None


@app.get("/api/props/stats-calibration")
async def api_props_stats_calibration(
    file: Optional[str] = Query(None, description="Filename under data/processed to read; defaults to most recent props_stats_calibration_*.json"),
    market: Optional[str] = Query(None, description="Filter by market: SOG, GOALS, ASSISTS, POINTS, SAVES"),
    window: Optional[int] = Query(None, description="Rolling window (e.g., 5,10,20)"),
    fmt: str = Query("json", description="Output format: json or csv"),
):
    # Normalize potential FastAPI Query objects when invoked internally
    try:
        from fastapi import params as _params
    except Exception:
        _params = None
    def _norm(v, default=None):
        if _params and isinstance(v, _params.Query):
            return v.default if v.default is not None else default
        return v if v is not None else default
    file = _norm(file, None)
    market = _norm(market, None)
    window = _norm(window, None)
    fmt = str(_norm(fmt, "json") or "json")
    # Resolve file path
    path = None
    try:
        if file:
            cand = PROC_DIR / file
            if cand.exists():
                path = cand
        if path is None:
            path = _latest_props_calibration_file()
    except Exception:
        path = None
    if not path or not path.exists():
        return JSONResponse({"error": "no-calibration-file", "hint": "Run CLI props-stats-calibration to generate one."}, status_code=404)
    # Load JSON
    import json as _json
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
    except Exception as e:
        return JSONResponse({"error": f"read-failed: {e}"}, status_code=500)
    groups = data.get("groups", []) or []
    # Normalize and filter
    rows = []
    mkt = (str(market or "").upper().strip())
    for g in groups:
        try:
            gm = str(g.get("market") or "").upper()
            if mkt and gm != mkt:
                continue
            if window is not None and int(g.get("window")) != int(window):
                continue
            rows.append({
                "market": gm,
                "line": g.get("line"),
                "window": g.get("window"),
                "count": g.get("count"),
                "accuracy": g.get("accuracy"),
                "brier": g.get("brier"),
            })
        except Exception:
            continue
    # CSV output
    if str(fmt).lower() == "csv":
        try:
            import io, csv
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=["market","line","window","count","accuracy","brier"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
            csv_text = buf.getvalue()
            fname = f"props_stats_calibration_summary.csv"
            headers = {"Content-Disposition": f"attachment; filename={fname}"}
            return Response(content=csv_text, media_type="text/csv", headers=headers)
        except Exception as e:
            return JSONResponse({"error": f"csv-failed: {e}"}, status_code=500)
    # JSON output
    payload = {
        "file": str(path.name),
        "start": data.get("start"),
        "end": data.get("end"),
        "summary": rows,
        "available_files": [p.name for p in sorted(PROC_DIR.glob("props_stats_calibration_*.json"))],
    }
    return JSONResponse(payload)


@app.get("/props/stats-calibration")
async def props_stats_calibration_page(
    file: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    window: Optional[int] = Query(None),
):
    # Reuse API
    resp = await api_props_stats_calibration(file=file, market=market, window=window, fmt="json")
    payload = {}
    if isinstance(resp, JSONResponse):
        try:
            import json as _json
            payload = _json.loads(resp.body)
        except Exception:
            payload = {"summary": []}
    template = env.get_template("props_stats_calibration.html")
    html = template.render(
        file=payload.get("file"),
        market=(market or ""),
        window=window,
        rows=payload.get("summary", []),
        start=payload.get("start"),
        end=payload.get("end"),
        files=payload.get("available_files", []),
    )
    return HTMLResponse(content=html)


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
    read_only_ui = _read_only(date)
    # Normalize potential FastAPI Query objects when this function is invoked internally.
    try:
        from fastapi import params as _params
    except Exception:  # pragma: no cover
        _params = None
    def _norm(v, default=None):
        if _params and isinstance(v, _params.Query):
            return v.default if v.default is not None else default
        return v if v is not None else default
    markets = _norm(markets, "all")
    try:
        min_ev = float(_norm(min_ev, 0.0))
    except Exception:
        min_ev = 0.0
    try:
        bankroll = float(_norm(bankroll, 0.0))
    except Exception:
        bankroll = 0.0
    try:
        kelly_fraction_part = float(_norm(kelly_fraction_part, 0.5))
    except Exception:
        kelly_fraction_part = 0.5
    try:
        top = int(_norm(top, 20))
    except Exception:
        top = 20
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return JSONResponse({"error": "No predictions for date", "date": date}, status_code=404)
    df = pd.read_csv(path)

    # Helper to add a rec if EV present and above threshold
    recs = []
    def add_rec(row: pd.Series, market_key: str, label: str, prob_key: str, ev_key: str, edge_key: str, odds_key: str, book_key: Optional[str] = None):
        # Safe numeric extraction for odds price
        def _num(v):
            if v is None:
                return None
            try:
                if isinstance(v, (int, float)):
                    import math as _math
                    _fv = float(v)
                    return _fv if _math.isfinite(_fv) else None
                s = str(v).strip()
                if s == '':
                    return None
                import re
                if re.fullmatch(r'[a-zA-Z_\-]+', s):
                    return None
                import math as _math
                _fv2 = float(s)
                return _fv2 if _math.isfinite(_fv2) else None
            except Exception:
                return None
        # Determine price with fallbacks (use close_* when current odds missing, and -110 for totals/puckline)
        raw_price = row.get(odds_key) if odds_key in row else None
        price_val = _num(raw_price)
        if price_val is None:
            close_map = {
                "home_ml_odds": "close_home_ml_odds",
                "away_ml_odds": "close_away_ml_odds",
                "over_odds": "close_over_odds",
                "under_odds": "close_under_odds",
                "home_pl_-1.5_odds": "close_home_pl_-1.5_odds",
                "away_pl_+1.5_odds": "close_away_pl_+1.5_odds",
            }
            ck = close_map.get(odds_key)
            if ck and ck in row:
                price_val = _num(row.get(ck))
        if price_val is None and market_key in ("totals", "puckline"):
            price_val = -110.0
        # Pull probability
        prob_val = None
        try:
            if prob_key in row and pd.notna(row.get(prob_key)):
                import math as _math
                _pv = float(row.get(prob_key))
                if _math.isfinite(_pv) and 0.0 <= _pv <= 1.0:
                    prob_val = _pv
                else:
                    prob_val = None
        except Exception:
            prob_val = None
        # Determine EV: use precomputed if present; else compute from prob and price
        ev_val = None
        try:
            if ev_key in row and pd.notna(row[ev_key]):
                import math as _math
                _ev = float(row[ev_key])
                ev_val = _ev if _math.isfinite(_ev) else None
        except Exception:
            ev_val = None
        if ev_val is None and (prob_val is not None) and (price_val is not None):
            try:
                from ..utils.odds import american_to_decimal
                dec = american_to_decimal(price_val)
                # Expected ROI per $1 stake
                import math as _math
                _ev2 = prob_val * (dec - 1.0) - (1.0 - prob_val)
                ev_val = _ev2 if _math.isfinite(_ev2) else None
            except Exception:
                ev_val = None
        # If still no EV or below threshold, skip
        try:
            _ok = (ev_val is not None) and (float(ev_val) >= float(min_ev))
        except Exception:
            _ok = False
        if not _ok:
            return
        # Sanitize optional numeric fields
        import math as _math
        edge_val = None
        try:
            if edge_key in row and pd.notna(row.get(edge_key)):
                _edge = float(row.get(edge_key))
                edge_val = _edge if _math.isfinite(_edge) else None
        except Exception:
            edge_val = None
        total_line_used_val = None
        try:
            if "total_line_used" in row and pd.notna(row.get("total_line_used")):
                _tlu = float(row.get("total_line_used"))
                total_line_used_val = _tlu if _math.isfinite(_tlu) else None
        except Exception:
            total_line_used_val = None
        rec = {
            "date": row.get("date"),
            "home": row.get("home"),
            "away": row.get("away"),
            "market": market_key,
            "bet": label,
            "model_prob": prob_val,
            "ev": ev_val,
            "edge": edge_val,
            "price": price_val,
            "book": row.get(book_key) if book_key and (book_key in row) and pd.notna(row.get(book_key)) else None,
            "total_line_used": total_line_used_val,
            "stake": None,
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
        # Stake (UI no bankroll, keep None)
        recs.append(rec)

    # Market filters
    try:
        f_markets = set([m.strip().lower() for m in str(markets).split(",")]) if markets and str(markets) != "all" else {"moneyline", "totals", "puckline"}
    except Exception:
        f_markets = {"moneyline", "totals", "puckline"}

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
    # Persist snapshot for historical tracking
    try:
        cols = [
            "date","home","away","market","bet","price","model_prob","ev","edge","book","result"
        ]
        import pandas as _pd
        _df_out = _pd.DataFrame([{k: r.get(k) for k in cols} for r in recs_sorted])
        out_path = PROC_DIR / f"recommendations_{date}.csv"
        _df_out.to_csv(out_path, index=False)
        # Best-effort GitHub write-back for recommendations snapshot
        try:
            _gh_upsert_file_if_configured(out_path, f"web: update recommendations for {date}")
        except Exception:
            pass
    except Exception:
        pass
    # Ensure JSON-safe output (convert NaN/Inf to None)
    try:
        from fastapi.encoders import jsonable_encoder as _jsonable_encoder
        _safe = _jsonable_encoder(recs_sorted, exclude_none=False)
    except Exception:
        # Fallback manual cleaning
        import math as _math
        def _clean_val(v):
            try:
                if isinstance(v, float) and not _math.isfinite(v):
                    return None
            except Exception:
                pass
            return v
        _safe = [{k: _clean_val(v) for k, v in r.items()} for r in recs_sorted]
    return JSONResponse(_safe)


@app.get("/recommendations")
async def recommendations(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD"),
    min_ev: float = Query(0.0, description="Minimum EV threshold to include"),
    top: int = Query(20, description="Top N recommendations to show"),
    markets: str = Query("all", description="Comma-separated filters: moneyline,totals,puckline"),
    bankroll: float = Query(0.0, description="If > 0, show Kelly stake using provided bankroll"),
    kelly_fraction_part: float = Query(0.5, description="Kelly fraction; used only if bankroll>0"),
    high_ev: float = Query(0.08, description="EV threshold for High confidence grouping (e.g., 0.08 for 8%)"),
    med_ev: float = Query(0.04, description="EV threshold for Medium confidence grouping (e.g., 0.04 for 4%)"),
    sort_by: str = Query("ev", description="Sort key within groups: ev, edge, prob, price, bet"),
):
    date = date or _today_ymd()
    read_only_ui = _read_only(date)
    # Normalize potential Query objects (when called internally) to raw values
    try:
        from fastapi import params as _params
    except Exception:
        _params = None
    def _norm(v, default=None):
        if _params and isinstance(v, _params.Query):
            return v.default if v.default is not None else default
        return v if v is not None else default
    markets = _norm(markets, "all")
    try: min_ev = float(_norm(min_ev, 0.0))
    except Exception: min_ev = 0.0
    try: top = int(_norm(top, 20))
    except Exception: top = 20
    try: bankroll = float(_norm(bankroll, 0.0))
    except Exception: bankroll = 0.0
    try: kelly_fraction_part = float(_norm(kelly_fraction_part, 0.5))
    except Exception: kelly_fraction_part = 0.5
    try: high_ev = float(_norm(high_ev, 0.05))
    except Exception: high_ev = 0.05
    try: med_ev = float(_norm(med_ev, 0.02))
    except Exception: med_ev = 0.02
    sort_by = str(_norm(sort_by, "ev") or "ev").lower()
    # Ensure predictions exist (skip in read-only mode)
    pred_path = PROC_DIR / f"predictions_{date}.csv"
    if (not pred_path.exists()) and (not read_only_ui):
        snapshot = datetime.now(timezone.utc).replace(hour=18, minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            predict_core(date=date, source="web", odds_source="oddsapi", snapshot=snapshot, odds_best=True, bankroll=bankroll, kelly_fraction_part=kelly_fraction_part)
        except Exception:
            pass
    # For settled slates (prior to today's ET), perform a one-shot settlement backfill to populate results
    try:
        et_today = _today_ymd()
        if str(date) < str(et_today):
            # Settlement backfill updates predictions file; allowed even in read-only UI
            _backfill_settlement_for_date(date)
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
        if sb == "bet":
            order = {"moneyline": 0, "totals": 1, "puckline": 2}
            return lambda x: (
                order.get((x.get("market") or "").lower(), 99),
                str(x.get("bet") or "")
            )
        # default ev
        return lambda x: x.get("ev", -999)
    _sk = sort_key_func(sort_by)
    # For alphabetical/typed sort, use ascending; for numeric metrics keep descending
    _reverse = False if sort_by == "bet" else True
    rows_high.sort(key=_sk, reverse=_reverse)
    rows_medium.sort(key=_sk, reverse=_reverse)
    rows_low.sort(key=_sk, reverse=_reverse)
    rows_other.sort(key=_sk, reverse=_reverse)
    # Reconciliation summary (closing-based) for the same date
    recon_summary = {}
    try:
        _recon_resp = await api_reconciliation(date=date)
        if isinstance(_recon_resp, JSONResponse):
            import json as _json
            _payload = _json.loads(_recon_resp.body)
            recon_summary = _payload.get("summary", {}) or {}
    except Exception:
        recon_summary = {}
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
    # prev/next date for quick navigation
    prev_date = None
    next_date = None
    try:
        base_dt = datetime.fromisoformat(date)
        prev_date = (base_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        next_date = (base_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        pass
    html = template.render(
        date=date,
        prev_date=prev_date,
        next_date=next_date,
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
        recon_summary=recon_summary,
        last_updates=_last_update_info(date),
    )
    return HTMLResponse(content=html)


@app.get("/api/odds-coverage")
async def api_odds_coverage(date: Optional[str] = Query(None)):
    date = date or _today_ymd()
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        # In read-only mode, do not attempt to generate; return 404 so UI can handle gracefully
        return JSONResponse({"error": "No predictions for date", "date": date}, status_code=404)
    df = _read_csv_fallback(path)
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
    df = _read_csv_fallback(path)
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

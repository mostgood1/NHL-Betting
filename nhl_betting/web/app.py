from __future__ import annotations

import os, time, json
import re
import math
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional, Dict, Any
import threading
import uuid
import asyncio
from io import StringIO

import numpy as np
import pandas as pd
import requests

from ..data.odds_api import OddsAPIClient
from ..utils.io import PROC_DIR, RAW_DIR, MODEL_DIR
from .teams import get_team_assets

from fastapi import FastAPI, Request, Query, Header
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape
# App and templating setup
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)

# ----------------------------------------------------------------------------------
# In-memory cache for /props (all players) to mitigate repeated large DataFrame ->
# JSON + template render overhead on resource-constrained deploys.
# Keyed by ("props_all_html", date, TEAM, MARKET, SORT, TOP). TTL configurable.
# ----------------------------------------------------------------------------------
_CACHE: dict = {}
try:
    _CACHE_TTL = int(os.getenv("CACHED_PROPS_TTL_SECONDS", "180"))  # default 3 minutes
except Exception:
    _CACHE_TTL = 180
START_TIME = time.time()

def _cache_get(key):
    if _CACHE_TTL <= 0:
        return None
    ent = _CACHE.get(key)
    if not ent:
        return None
    if (time.time() - ent.get("ts", 0)) > _CACHE_TTL:
        _CACHE.pop(key, None)
        return None
    return ent.get("value")

def _cache_put(key, value):
    if _CACHE_TTL <= 0:
        return
    _CACHE[key] = {"value": value, "ts": time.time()}

def _compute_allowed() -> bool:
    """Return True if server-side compute fallback is allowed.

    We explicitly disable on public hosts and when PROPS_NO_COMPUTE=1.
    """
    try:
        if os.getenv('PROPS_NO_COMPUTE', '0') == '1':
            return False
    except Exception:
        pass
    # Never compute on public deploys unless explicitly overridden
    return not _is_public_host_env()

app = FastAPI()


@app.exception_handler(StarletteHTTPException)
async def _http_exc_handler(request: Request, exc: StarletteHTTPException):
    try:
        return JSONResponse(
            {
                "ok": False,
                "error": "http_exception",
                "status_code": int(getattr(exc, "status_code", 500) or 500),
                "detail": getattr(exc, "detail", None),
                "path": str(getattr(request.url, "path", "")),
                "commit": (_git_commit_hash() or "")[:12],
            },
            status_code=int(getattr(exc, "status_code", 500) or 500),
        )
    except Exception:
        return JSONResponse({"ok": False, "error": "http_exception"}, status_code=500)


@app.exception_handler(Exception)
async def _unhandled_exc_handler(request: Request, exc: Exception):
    # Last-resort guardrail: ensure callers get a JSON body rather than a blank 500.
    # Do not include stack traces in the response (Render logs will still show them).
    try:
        try:
            import traceback

            print("UNHANDLED_EXCEPTION", getattr(request.url, "path", ""))
            traceback.print_exc()
        except Exception:
            pass
        return JSONResponse(
            {
                "ok": False,
                "error": "unhandled_exception",
                "detail": str(exc),
                "path": str(getattr(request.url, "path", "")),
                "commit": (_git_commit_hash() or "")[:12],
            },
            status_code=500,
        )
    except Exception:
        return JSONResponse({"ok": False, "error": "unhandled_exception"}, status_code=500)


@app.get("/diag/healthz")
def diag_healthz():
    """Ultra-minimal readiness probe with a little file-system context."""
    try:
        return {
            "ok": True,
            "commit": (_git_commit_hash() or "")[:12],
            "cwd": os.getcwd(),
            "template_dir": str(TEMPLATE_DIR),
            "templates_exist": bool(TEMPLATE_DIR.exists()),
            "cards_only_exists": bool((TEMPLATE_DIR / "cards_only.html").exists()),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ----------------------------------------------------------------------------------
# Cards-only UI mode: treat all other HTML pages as removed.
# Keeps JSON APIs (/api/*) and stable bundles API (/v1/*).
# ----------------------------------------------------------------------------------
_UI_CARDS_ONLY = str(os.getenv("UI_CARDS_ONLY", "1")).strip().lower() in {"1", "true", "yes"}


@app.middleware("http")
async def _cards_only_ui_mw(request: Request, call_next):
    if not _UI_CARDS_ONLY:
        return await call_next(request)

    path = request.url.path or "/"

    # Allow the single page + assets + APIs + docs
    if path == "/":
        return await call_next(request)
    if path.startswith("/diag/"):
        return await call_next(request)
    if path.startswith("/static/") or path.startswith("/v1/") or path.startswith("/api/"):
        return await call_next(request)
    if path in {"/openapi.json", "/docs", "/redoc"}:
        return await call_next(request)

    # Block any other HTML-facing routes.
    accept = (request.headers.get("accept") or "").lower()
    if "text/html" in accept:
        return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)

    return await call_next(request)
# ----------------------------------------------------------------------------------
# Lightweight cron job status tracker (in-memory, best-effort)
_CRON_JOBS: Dict[str, Dict[str, Any]] = {}
_CRON_LOCK = threading.Lock()
_CRON_MAX = 100

def _cron_now_iso() -> str:
    try:
        return datetime.utcnow().isoformat()
    except Exception:
        return ""

def _cron_set(job_id: str, patch: Dict[str, Any]):
    try:
        with _CRON_LOCK:
            rec = _CRON_JOBS.get(job_id, {})
            rec.update(patch)
            rec['updated_at'] = _cron_now_iso()
            _CRON_JOBS[job_id] = rec
    except Exception:
        pass

def _cron_add(name: str, params: Dict[str, Any]) -> str:
    jid = str(uuid.uuid4())[:12]
    try:
        with _CRON_LOCK:
            _CRON_JOBS[jid] = {
                'id': jid,
                'name': name,
                'params': params or {},
                'state': 'queued',
                'created_at': _cron_now_iso(),
                'updated_at': _cron_now_iso(),
                'error': None,
                'result': None,
            }
            # trim if over max
            if len(_CRON_JOBS) > _CRON_MAX:
                # drop oldest by created_at
                try:
                    items = sorted(_CRON_JOBS.items(), key=lambda kv: kv[1].get('created_at',''))
                    for k,_ in items[: max(0, len(_CRON_JOBS) - _CRON_MAX) ]:
                        _CRON_JOBS.pop(k, None)
                except Exception:
                    pass
    except Exception:
        pass
    return jid

def _queue_cron(name: str, params: Dict[str, Any], target_fn):
    jid = _cron_add(name, params)
    def _runner():
        _cron_set(jid, {'state': 'running'})
        try:
            res = target_fn()
            # compact result if large
            compact = res
            try:
                if isinstance(res, dict):
                    compact = {k: res[k] for k in list(res.keys())[:20]}
            except Exception:
                compact = res
            _cron_set(jid, {'state': 'done', 'result': compact})
        except Exception as e:
            _cron_set(jid, {'state': 'error', 'error': str(e)})
    try:
        threading.Thread(target=_runner, daemon=True).start()
    except Exception as e:
        _cron_set(jid, {'state': 'error', 'error': f"thread_start_failed: {e}"})
    return jid

@app.get("/api/cron/status")
async def api_cron_status(job_id: Optional[str] = Query(None), name: Optional[str] = Query(None), limit: int = Query(50)):
    try:
        with _CRON_LOCK:
            vals = list(_CRON_JOBS.values())
        if job_id:
            for rec in vals:
                if rec.get('id') == job_id:
                    return JSONResponse({'ok': True, 'job': rec})
            return JSONResponse({'ok': False, 'error': 'not_found', 'job_id': job_id}, status_code=404)
        if name:
            vals = [r for r in vals if str(r.get('name') or '') == str(name)]
        try:
            vals.sort(key=lambda r: r.get('updated_at',''), reverse=True)
        except Exception:
            pass
        if limit and limit > 0:
            vals = vals[: int(limit)]
        return JSONResponse({'ok': True, 'jobs': vals, 'count': len(vals)})
    except Exception as e:
        return JSONResponse({'ok': False, 'error': str(e)}, status_code=500)

def _is_public_host_env() -> bool:
    """Heuristic to detect if we're on a public host (Render/production) vs local/test.

    We consider it public if any of these env vars are present or typical of deploys:
    - RENDER, RENDER_EXTERNAL_HOSTNAME, RENDER_SERVICE_ID
    - PORT set (common in PaaS)
    - GITHUB_ACTIONS (CI)
    """
    try:
        env = os.environ
        if env.get('RENDER') or env.get('RENDER_EXTERNAL_HOSTNAME') or env.get('RENDER_SERVICE_ID'):
            return True
        if env.get('GITHUB_ACTIONS'):
            return True
        # If a PORT is set and not the usual local ones, assume public
        port = env.get('PORT')
        if port and port not in ('8000','8010','3000','5000'):
            return True
    except Exception:
        pass
    return False

def _use_headshot_proxy() -> bool:
    """Return True if we should proxy NHL headshots via this server.

    Default: enabled locally (not public host) unless PROXY_HEADSHOTS=0.
    """
    try:
        v = os.getenv('PROXY_HEADSHOTS')
        if v is None:
            return not _is_public_host_env()
        return str(v).strip().lower() in ('1','true','yes','on')
    except Exception:
        return not _is_public_host_env()

def _nhl_season_code(d_ymd: Optional[str]) -> str:
    """Return NHL season code like '20252026' for a given YYYY-MM-DD date string.
    Uses July (7) as the season boundary.
    """
    try:
        from datetime import datetime as _dt
        if d_ymd and isinstance(d_ymd, str):
            dt = _dt.strptime(d_ymd, '%Y-%m-%d')
        else:
            dt = _dt.utcnow()
        start_year = dt.year if dt.month >= 7 else (dt.year - 1)
        return f"{start_year}{start_year+1}"
    except Exception:
        # Fallback to current UTC year pairing
        y = datetime.utcnow().year
        return f"{y}{y+1}"

"""
Primary Player Props page.

Historically this endpoint redirected to /props/all to avoid duplication during refactors.
Now it renders the same table directly by delegating to props_all_players_page so the
URL matches the NFL-Betting convention (/props) while preserving identical behavior.
"""
@app.get("/props", include_in_schema=False)
async def props_main(
    request: Request,
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    game: Optional[str] = Query(None, description="Filter by game as AWY@HOME (team abbreviations)"),
    team: Optional[str] = Query(None, description="Filter by team abbreviation"),
    market: Optional[str] = Query(None, description="Filter by market"),
    sort: Optional[str] = Query("ev_desc", description="Sort: ev_desc, ev_asc, p_over_desc, p_over_asc, lambda_desc, lambda_asc, name, team, market, line, book"),
    top: int = Query(500, description="Max rows to display before pagination"),
    min_ev: float = Query(0.0, description="Minimum EV filter (over/under best side)"),
    page: int = Query(1, description="Page number (1-based)"),
    page_size: Optional[int] = Query(None, description="Rows per page (defaults PROPS_PAGE_SIZE env or 250"),
):
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)

# Secondary explicit safeguard endpoint to validate redirect logic without colliding with /props.
@app.get("/props-safeguard", include_in_schema=False)
async def props_safeguard(date: Optional[str] = None):
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)

# Root HEAD handler: avoids 405s from HEAD probes without invoking heavy work

@app.head("/", include_in_schema=False)
async def root_head():
    """Explicit HEAD for root to prevent heavy GET invocation."""
    from fastapi import Response
    return Response(status_code=204)

@app.get("/diag/info")
async def diag_info():
    """Expose diagnostic information to debug deployment mismatches & 502 causes (non-sensitive)."""
    import sys, inspect
    try:
        app_file = inspect.getsourcefile(app.__class__)
    except Exception:
        app_file = None
    route_paths = []
    try:
        for r in app.routes:
            try:
                route_paths.append(getattr(r, 'path', None))
            except Exception:
                pass
    except Exception:
        pass
    return {
        "commit_live": (_git_commit_hash() or '')[:12],
        "routes_contains_props": [p for p in route_paths if p and 'props' in p.lower()],
        "total_routes": len(route_paths),
        "sys_path_head": sys.path[:5],
        "cwd": os.getcwd(),
        "app_file": app_file,
    }

@app.middleware("http")
async def _commit_header_mw(request, call_next):
    response = await call_next(request)
    try:
        h = (_git_commit_hash() or '')[:12]
        if h:
            response.headers['X-App-Commit'] = h
    except Exception:
        pass
    return response

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app_: FastAPI):
    # Startup phase
    try:
        print(json.dumps({"event":"startup_diag","commit": (_git_commit_hash() or '')[:12], "route_count": len(app_.routes)}))
    except Exception:
        pass
    # Model bootstrap scheduling (merged from old second startup handler)
    try:
        from ..utils.io import MODEL_DIR
        ratings_path = MODEL_DIR / "elo_ratings.json"
        cfg_path = MODEL_DIR / "config.json"
        if not (ratings_path.exists() and cfg_path.exists()):
            now = datetime.now(timezone.utc)
            end = f"{now.year}-08-01"
            start_year = now.year - 2
            start = f"{start_year}-09-01"
            async def _do_bootstrap():
                try:
                    await asyncio.to_thread(cli_fetch, start, end, "web")
                    await asyncio.to_thread(cli_train)
                except Exception:
                    pass
            asyncio.create_task(_do_bootstrap())
    except Exception:
        pass
    # On public host deploy/restart, optionally run a one-time light odds refresh + edges recompute
    try:
        if _is_public_host_env() and str(os.getenv("WEB_ON_DEPLOY_REFRESH_EDGES", "1")).strip().lower() in ("1","true","yes","on"):
            d = _today_ymd()
            async def _do_light_refresh():
                try:
                    # Best-effort inject OddsAPI odds without running models; allow prestart overwrite
                    summary = await asyncio.to_thread(_inject_oddsapi_odds_into_predictions, d, True)
                except Exception:
                    summary = None
                # Recompute only if odds changed
                try:
                    if isinstance(summary, dict) and int(summary.get("updated_fields") or 0) > 0:
                        if _recs_recompute_shared is not None:
                            # Run shared recompute in a thread (I/O/CPU bound)
                            await asyncio.to_thread(_recs_recompute_shared, d, 0.0)
                except Exception:
                    pass
                # Refresh props recommendations (function will skip if lines unchanged)
                try:
                    _ = _refresh_props_recommendations(d, min_ev=0.0, top=200)
                except Exception:
                    pass
            asyncio.create_task(_do_light_refresh())
    except Exception:
        pass
    yield
    # Shutdown phase (none currently)

# Apply lifespan to app (FastAPI allows providing lifespan in constructor, but we retrofit here)
app.router.lifespan_context = lifespan

# ----------------------------------------------------------------------------------
# Game recommendations/edges/reconciliation API moved to web/game_recs.py
# ----------------------------------------------------------------------------------
try:
    from .game_recs import router as _game_recs_router
    app.include_router(_game_recs_router)
except Exception:
    # If import fails (e.g., local dev while editing), continue without these endpoints
    pass
try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
except Exception:
    # Mounting static is best-effort (e.g., path may not exist in some deploys)
    pass


# ----------------------------------------------------------------------------------
# v1 read-only bundles API
# ----------------------------------------------------------------------------------
_V1_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

# Lightweight in-memory cache for live odds snapshots to avoid hammering OddsAPI.
_LIVE_ODDS_CACHE: dict = {}
try:
    _LIVE_ODDS_TTL_SECONDS = int(os.getenv("LIVE_ODDS_TTL_SECONDS", "30"))
except Exception:
    _LIVE_ODDS_TTL_SECONDS = 30


def _live_odds_cache_get(key: str):
    try:
        ent = _LIVE_ODDS_CACHE.get(key)
        if not ent:
            return None
        if _LIVE_ODDS_TTL_SECONDS <= 0:
            return None
        if (time.time() - float(ent.get("ts", 0.0))) > float(_LIVE_ODDS_TTL_SECONDS):
            _LIVE_ODDS_CACHE.pop(key, None)
            return None
        return ent.get("value")
    except Exception:
        return None


def _live_odds_cache_put(key: str, value: object):
    try:
        if _LIVE_ODDS_TTL_SECONDS <= 0:
            return
        _LIVE_ODDS_CACHE[key] = {"ts": time.time(), "value": value}
    except Exception:
        pass


def _norm_game_key(away: object, home: object) -> str:
    try:
        a = " ".join(str(away or "").strip().split()).lower()
        h = " ".join(str(home or "").strip().split()).lower()
        if not a or not h:
            return ""
        return f"{a} @ {h}"
    except Exception:
        return ""


def _parse_mmss_clock(clock: object) -> Optional[int]:
    """Parse a scoreboard clock like '12:34' into seconds remaining in period."""
    try:
        s = str(clock or "").strip()
        if not s or ":" not in s:
            return None
        mm, ss = s.split(":", 1)
        mm_i = int(mm)
        ss_i = int(ss)
        if mm_i < 0 or ss_i < 0 or ss_i >= 60:
            return None
        return 60 * mm_i + ss_i
    except Exception:
        return None


def _strict_json_sanitize(obj: Any) -> Any:
    """Convert obj graph to strict-JSON-safe primitives (no NaN/Inf).

    Starlette's JSONResponse uses allow_nan=False. If any NaN/Inf slips into the
    response payload, it raises "Out of range float values are not JSON compliant".
    This sanitizer converts NaN/Inf -> None and numpy scalars -> Python scalars.
    """
    try:
        import numpy as _np
    except Exception:
        _np = None
    try:
        import datetime as _dt
    except Exception:
        _dt = None

    def _san(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, dict):
            out: dict[str, Any] = {}
            for k, v in x.items():
                try:
                    ks = str(k)
                except Exception:
                    ks = "key"
                out[ks] = _san(v)
            return out
        if isinstance(x, (list, tuple)):
            return [_san(v) for v in x]

        # pandas NA / NaN
        try:
            if pd.isna(x):
                return None
        except Exception:
            pass

        # numpy scalar
        try:
            if _np is not None and isinstance(x, _np.generic):
                return _san(x.item())
        except Exception:
            pass

        # datetime-like
        try:
            if isinstance(x, pd.Timestamp):
                return x.isoformat()
        except Exception:
            pass
        try:
            if _dt is not None and isinstance(x, (_dt.datetime, _dt.date)):
                return x.isoformat()
        except Exception:
            pass

        if isinstance(x, float):
            try:
                if not math.isfinite(x):
                    return None
            except Exception:
                return None
        return x

    try:
        return _san(obj)
    except Exception:
        return obj


def _load_bundle_predictions_map(date_ymd: str) -> dict:
    """Best-effort: load bundle prediction rows keyed by normalized away@home."""
    try:
        from ..publish.daily_bundles import bundle_path, build_daily_bundle

        p = bundle_path(date_ymd, PROC_DIR)
        obj = None
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                obj = None
        if obj is None:
            try:
                obj = build_daily_bundle(date_ymd, PROC_DIR)
            except Exception:
                obj = None

        rows = (
            (obj or {}).get("data", {})
            .get("games", {})
            .get("predictions", {})
            .get("rows", [])
        )
        if not isinstance(rows, list):
            return {}
        out: dict = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            k = _norm_game_key(r.get("away"), r.get("home"))
            if k:
                out[k] = r
        return out
    except Exception:
        return {}


def _v1_odds_payload(date_ymd: str, regions: str = "us", best: bool = True, inplay: bool = False) -> dict:
    """Build odds payload object (as returned by /v1/odds/{date}) with caching."""
    d = str(date_ymd or "").strip()
    if not _V1_DATE_RE.fullmatch(d):
        return {"ok": False, "error": "invalid_date", "date": date_ymd}

    cache_key = f"v1_odds::{d}::{str(regions or 'us').lower()}::{int(bool(best))}::{int(bool(inplay))}"
    cached = _live_odds_cache_get(cache_key)
    if cached is not None:
        return cached

    asof = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    try:
        client = OddsAPIClient()
    except Exception as e:
        obj = {"ok": True, "date": d, "asof_utc": asof, "games": [], "note": str(e)}
        _live_odds_cache_put(cache_key, obj)
        return obj

    # Flat snapshot returns normalized rows for all current events.
    df = client.flat_snapshot(
        d,
        regions=str(regions or "us"),
        markets="h2h,totals,spreads",
        snapshot_iso=None,
        odds_format="american",
        bookmaker=None,
        best=bool(best),
        inplay=bool(inplay),
    )

    games: list[dict] = []
    if df is not None and not df.empty:
        try:
            df2 = df[df.get("date") == d].copy()
        except Exception:
            df2 = df.copy()
        if df2 is not None and not df2.empty:
            def _i(x):
                try:
                    return int(x)
                except Exception:
                    return None

            def _f(x):
                try:
                    if x is None:
                        return None
                    return float(x)
                except Exception:
                    return None

            for _, r in df2.iterrows():
                home = str(r.get("home") or "")
                away = str(r.get("away") or "")
                if not home or not away:
                    continue
                games.append({
                    "date": d,
                    "home": home,
                    "away": away,
                    "key": _norm_game_key(away, home),
                    "ml": {
                        "home": _i(r.get("home_ml")),
                        "away": _i(r.get("away_ml")),
                        "home_book": r.get("home_ml_book"),
                        "away_book": r.get("away_ml_book"),
                    },
                    "total": {
                        "line": _f(r.get("total_line")),
                        "over": _i(r.get("over")),
                        "under": _i(r.get("under")),
                        "over_book": r.get("over_book"),
                        "under_book": r.get("under_book"),
                    },
                    "puckline": {
                        "home_-1.5": _i(r.get("home_pl_-1.5")),
                        "away_+1.5": _i(r.get("away_pl_+1.5")),
                        "home_-1.5_book": r.get("home_pl_-1.5_book"),
                        "away_+1.5_book": r.get("away_pl_+1.5_book"),
                    },
                })

    obj = {"ok": True, "date": d, "asof_utc": asof, "games": games}
    _live_odds_cache_put(cache_key, obj)
    return obj


@app.get("/v1/manifest")
async def v1_manifest():
    try:
        from ..publish.daily_bundles import manifest_path, build_manifest

        p = manifest_path(PROC_DIR)
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    return JSONResponse({"ok": True, **obj})
            except Exception:
                pass
        obj = build_manifest(PROC_DIR)
        if not isinstance(obj, dict):
            return JSONResponse({"ok": False, "error": "invalid_manifest_format"}, status_code=500)
        return JSONResponse({"ok": True, **obj})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/v1/dates")
async def v1_dates():
    # IMPORTANT: this endpoint must never 500 because the cards-only UI depends on it.
    note = None
    dates: list[str] = []
    latest = None
    try:
        from ..publish.daily_bundles import build_manifest

        man = build_manifest(PROC_DIR)
        if isinstance(man, dict):
            dates = man.get("dates") or []
            latest = man.get("latest")
        else:
            note = "manifest_not_dict"
    except Exception as e:
        note = f"manifest_error: {e}"

    # Provide convenience trio for the UI (yesterday/today/tomorrow)
    trio: list[str] = []
    try:
        try:
            t = _today_ymd()
        except Exception:
            t = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        base = datetime.strptime(t, "%Y-%m-%d")
        trio = [
            (base - timedelta(days=1)).strftime("%Y-%m-%d"),
            base.strftime("%Y-%m-%d"),
            (base + timedelta(days=1)).strftime("%Y-%m-%d"),
        ]
    except Exception:
        trio = []

    payload: dict[str, Any] = {"ok": True, "dates": dates, "latest": latest, "trio": trio}
    if note:
        payload["note"] = note
    return JSONResponse(payload)


@app.get("/v1/bundle/{date}")
async def v1_bundle(date: str):
    try:
        d = str(date or "").strip()
        if not _V1_DATE_RE.fullmatch(d):
            return JSONResponse({"ok": False, "error": "invalid_date", "date": date}, status_code=400)

        def _enrich_predictions_team_assets(obj: dict) -> dict:
            """Best-effort: attach `home_logo`/`away_logo` to prediction rows.

            This keeps the v1 bundle schema stable while letting the cards UI
            display team logos without additional roundtrips.
            """
            try:
                rows = (
                    (obj.get("data") or {})
                    .get("games", {})
                    .get("predictions", {})
                    .get("rows", [])
                )
                if not isinstance(rows, list) or not rows:
                    return obj
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    try:
                        home = str(r.get("home") or "")
                        away = str(r.get("away") or "")
                        h = get_team_assets(home) or {}
                        a = get_team_assets(away) or {}
                        r.setdefault("home_logo", h.get("logo_dark") or h.get("logo_light") or h.get("logo"))
                        r.setdefault("away_logo", a.get("logo_dark") or a.get("logo_light") or a.get("logo"))
                    except Exception:
                        continue
            except Exception:
                return obj
            return obj

        def _enrich_predictions_schedule(obj: dict, date_ymd: str) -> dict:
            """Best-effort: attach scheduled start time to prediction rows.

            Prediction artifacts often only contain the calendar day (or no time at all).
            The UI needs a stable per-game start time for ordering.

            Data source: NHL Web API schedule_day (no API key required).
            """
            try:
                rows = (
                    (obj.get("data") or {})
                    .get("games", {})
                    .get("predictions", {})
                    .get("rows", [])
                )
                if not isinstance(rows, list) or not rows:
                    return obj

                # If rows already have all the schedule-enriched fields, don't bother.
                # Note: some persisted prediction artifacts include a start time but not gamePk;
                # we still want gamePk and a fresh game_state for robust live overlays.
                all_ok = True
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    has_time = bool(r.get("scheduled_start_utc") or r.get("start_time_utc") or r.get("commence_time_utc"))
                    has_pk = (r.get("gamePk") is not None)
                    has_state = bool(str(r.get("game_state") or "").strip())
                    has_venue = bool(str(r.get("venue") or "").strip())
                    if not (has_time and has_pk and has_state and has_venue):
                        all_ok = False
                        break
                if all_ok:
                    return obj

                # Keep this *fast*: a single HTTP call with short timeout.
                # (NHLWebClient includes retry/backoff/sleep which can slow /v1/bundle.)
                timeout = float(os.getenv("BUNDLE_SCHEDULE_TIMEOUT_SEC", "3"))
                sched: dict[str, dict] = {}
                try:
                    import requests

                    url = f"https://api-web.nhle.com/v1/schedule/{str(date_ymd)}"
                    r = requests.get(url, timeout=timeout)
                    if r.status_code == 200:
                        data = r.json() or {}
                    else:
                        data = {}
                except Exception:
                    data = {}

                def _team_name(team: dict) -> str:
                    try:
                        place = (team or {}).get("placeName", {})
                        common = (team or {}).get("commonName", {})
                        place_s = place.get("default") if isinstance(place, dict) else (place or "")
                        common_s = common.get("default") if isinstance(common, dict) else (common or "")
                        name = f"{place_s} {common_s}".strip()
                        return " ".join(name.split())
                    except Exception:
                        return ""

                try:
                    for wk in data.get("gameWeek", []) or []:
                        if wk.get("date") != str(date_ymd):
                            continue
                        for g in wk.get("games", []) or []:
                            try:
                                home = _team_name(g.get("homeTeam", {}) or {})
                                away = _team_name(g.get("awayTeam", {}) or {})
                                k = _norm_game_key(away, home)
                                if not k:
                                    continue
                                venue = None
                                try:
                                    v = g.get("venue")
                                    if isinstance(v, dict):
                                        venue = v.get("default") or v.get("name")
                                    elif isinstance(v, str):
                                        venue = v
                                except Exception:
                                    venue = None
                                sched[k] = {
                                    "scheduled_start_utc": g.get("startTimeUTC") or None,
                                    "venue": venue,
                                    "game_state": g.get("gameState"),
                                    "gamePk": g.get("id"),
                                }
                            except Exception:
                                continue
                except Exception:
                    sched = {}

                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    try:
                        k = _norm_game_key(r.get("away"), r.get("home"))
                        s = sched.get(k)
                        if not s:
                            continue
                        if s.get("scheduled_start_utc"):
                            r.setdefault("scheduled_start_utc", s.get("scheduled_start_utc"))
                        if s.get("venue"):
                            r.setdefault("venue", s.get("venue"))
                        if s.get("game_state"):
                            # Always prefer current gameState from NHL Web schedule.
                            # Predictions artifacts may contain stale states (e.g. FUT).
                            r["game_state"] = s.get("game_state")
                        if s.get("gamePk") is not None:
                            r.setdefault("gamePk", s.get("gamePk"))
                    except Exception:
                        continue
            except Exception:
                return obj
            return obj

        from ..publish.daily_bundles import bundle_path, build_daily_bundle

        p = bundle_path(d, PROC_DIR)
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(obj, dict):
                    obj = None
                # Guard against stale persisted bundles (common in read-only deploys):
                # if the persisted bundle points at older source files than we'd pick today,
                # rebuild in-memory so the UI sees sim-backed markets (TOTAL/PL) when available.
                try:
                    fresh = build_daily_bundle(d, PROC_DIR)
                except Exception:
                    fresh = None

                if isinstance(fresh, dict) and isinstance(obj, dict):
                    try:
                        if (obj.get("files") or {}) != (fresh.get("files") or {}):
                            obj = fresh
                    except Exception:
                        pass
                elif isinstance(fresh, dict) and obj is None:
                    obj = fresh

                if not isinstance(obj, dict):
                    obj = build_daily_bundle(d, PROC_DIR)

                obj = _enrich_predictions_team_assets(obj)
                obj = _enrich_predictions_schedule(obj, d)
                obj = _strict_json_sanitize(obj)
                return JSONResponse({"ok": True, **obj})
            except Exception:
                # Fall back to rebuilding in-memory
                pass

        obj = build_daily_bundle(d, PROC_DIR)
        obj = _enrich_predictions_team_assets(obj)
        obj = _enrich_predictions_schedule(obj, d)
        obj = _strict_json_sanitize(obj)
        return JSONResponse({"ok": True, **obj, "note": "bundle_not_persisted"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/v1/live/{date}")
async def v1_live(date: str):
    """Live lens (read-only) for a slate date.

    This endpoint is intended for the cards-only UI to overlay live game state
    and simple period/totals stats without mutating any artifacts.

    Data source: NHL Web API via NHLWebClient.
    """
    try:
        d = str(date or "").strip()
        if not _V1_DATE_RE.fullmatch(d):
            return JSONResponse({"ok": False, "error": "invalid_date", "date": date}, status_code=400)

        # Import locally so the endpoint doesn't depend on module import order
        # (and stays best-effort on constrained deploys).
        from ..data.nhl_api_web import NHLWebClient

        def _norm(s: object) -> str:
            try:
                return " ".join(str(s or "").strip().split()).lower()
            except Exception:
                return str(s or "").lower()

        def _maybe_float(x: object) -> Optional[float]:
            try:
                if x is None:
                    return None
                if isinstance(x, str):
                    s = x.strip()
                    if s.endswith("%"):
                        s = s[:-1]
                    if s == "":
                        return None
                    return float(s)
                return float(x)
            except Exception:
                return None

        def _maybe_int(x: object) -> Optional[int]:
            try:
                if x is None:
                    return None
                if isinstance(x, bool):
                    return int(x)
                if isinstance(x, (int, np.integer)):
                    return int(x)
                fx = _maybe_float(x)
                return int(fx) if fx is not None else None
            except Exception:
                return None

        def _pick(dct: object, *keys: str) -> object:
            if not isinstance(dct, dict):
                return None
            for k in keys:
                if k in dct:
                    return dct.get(k)
            return None

        def _extract_team_totals(team_obj: object) -> dict:
            if not isinstance(team_obj, dict):
                return {}
            stats_obj = (
                _pick(team_obj, "teamStats", "teamGameStats", "statistics", "stats", "teamStatsSummary")
                or {}
            )
            if not isinstance(stats_obj, dict):
                stats_obj = {}

            # Try to pull SOG / shots in a robust way.
            sog = _maybe_int(_pick(stats_obj, "sog", "shotsOnGoal", "shots_on_goal", "shotsOnGoalFor"))
            shots = _maybe_int(_pick(stats_obj, "shots", "shotAttempts", "shotsFor"))
            goals = _maybe_int(_pick(team_obj, "score", "goals"))

            # Faceoff win pct often exists as a percent number.
            fo = _maybe_float(_pick(stats_obj, "faceoffWinningPctg", "faceoffWinPct", "faceoffWinPctg", "faceoffPct"))

            # Powerplay as a string like "1/3" or a pct like 25.0.
            pp_conv = _pick(stats_obj, "powerPlayConversion", "powerPlayConversionPct", "powerPlay", "powerPlayPct")
            if isinstance(pp_conv, (int, float, str)):
                pp = pp_conv
            else:
                pp = None

            out = {
                "goals": goals,
                "sog": sog,
                "shots": shots,
                "faceoff_win_pct": fo,
                "pp": pp,
            }
            # Remove None values to keep payload clean.
            return {k: v for k, v in out.items() if v is not None}

        def _extract_goalie_summary(box: dict, side: str) -> list[dict]:
            try:
                pbg = box.get("playerByGameStats") or {}
                team_key = "homeTeam" if side == "home" else "awayTeam"
                t = pbg.get(team_key) or {}
                goalies = t.get("goalies") or []
                out = []
                for g in goalies if isinstance(goalies, list) else []:
                    if not isinstance(g, dict):
                        continue
                    pid = _maybe_int(_pick(g, "playerId", "player_id", "id"))
                    nm = None
                    try:
                        name_obj = g.get("name")
                        if isinstance(name_obj, dict):
                            nm = name_obj.get("default") or name_obj.get("full")
                        elif isinstance(name_obj, str):
                            nm = name_obj
                    except Exception:
                        nm = None
                    saves = _maybe_int(_pick(g, "saves"))
                    sa = _maybe_int(_pick(g, "shotsAgainst", "shots_against", "shots"))
                    sv = _maybe_float(_pick(g, "savePctg", "svPct", "savePercentage"))
                    row = {"player_id": pid, "name": nm, "saves": saves, "shots_against": sa, "sv_pct": sv}
                    row = {k: v for k, v in row.items() if v is not None and v != ""}
                    if row:
                        out.append(row)
                return out
            except Exception:
                return []

        def _extract_skaters_summary(box: dict, side: str) -> list[dict]:
            """Best-effort extraction of skater boxscore stats from NHL Web boxscore."""
            try:
                pbg = box.get("playerByGameStats") or {}
                team_key = "homeTeam" if side == "home" else "awayTeam"
                t = pbg.get(team_key) or {}
                if not isinstance(t, dict):
                    return []
                # NHL web payload can be either:
                #  - `skaters`: combined list
                #  - split lists: `forwards` + `defense` (observed)
                # Use a union (not first-match) so defensemen aren't dropped.
                skaters: list[dict] = []
                raw = t.get("skaters")
                if isinstance(raw, list) and raw:
                    skaters.extend([x for x in raw if isinstance(x, dict)])
                for key in ("forwards", "defense", "defence", "defensemen", "defencemen"):
                    arr = t.get(key)
                    if isinstance(arr, list) and arr:
                        skaters.extend([x for x in arr if isinstance(x, dict)])

                if not skaters:
                    return []

                # De-dupe while preserving order.
                seen_ids: set[int] = set()
                seen_names: set[str] = set()
                uniq: list[dict] = []
                for s in skaters:
                    pid = _maybe_int(_pick(s, "playerId", "player_id", "id"))
                    if pid is not None:
                        if pid in seen_ids:
                            continue
                        seen_ids.add(pid)
                        uniq.append(s)
                        continue
                    nm = None
                    try:
                        name_obj = s.get("name")
                        if isinstance(name_obj, dict):
                            nm = name_obj.get("default") or name_obj.get("full")
                        elif isinstance(name_obj, str):
                            nm = name_obj
                    except Exception:
                        nm = None
                    nk = str(nm or "").strip().lower()
                    if nk and nk in seen_names:
                        continue
                    if nk:
                        seen_names.add(nk)
                    uniq.append(s)

                skaters = uniq
                out: list[dict] = []
                for s in skaters:
                    if not isinstance(s, dict):
                        continue
                    pid = _maybe_int(_pick(s, "playerId", "player_id", "id"))
                    nm = None
                    try:
                        name_obj = s.get("name")
                        if isinstance(name_obj, dict):
                            nm = name_obj.get("default") or name_obj.get("full")
                        elif isinstance(name_obj, str):
                            nm = name_obj
                    except Exception:
                        nm = None
                    toi = None
                    try:
                        toi = _pick(s, "toi", "timeOnIce", "time_on_ice")
                        if isinstance(toi, dict):
                            toi = toi.get("default") or toi.get("displayValue")
                        if toi is not None:
                            toi = str(toi)
                    except Exception:
                        toi = None

                    row = {
                        "player_id": pid,
                        "name": nm,
                        "pos": _pick(s, "position", "pos"),
                        "g": _maybe_int(_pick(s, "goals", "g")),
                        "a": _maybe_int(_pick(s, "assists", "a")),
                        "p": _maybe_int(_pick(s, "points", "p")),
                        # NHL Web may use different keys; include SOG variants.
                        "s": _maybe_int(_pick(s, "shots", "s", "sog", "shotsOnGoal", "shots_on_goal")),
                        "blk": _maybe_int(_pick(s, "blockedShots", "blocked", "blocks", "blk")),
                        "hits": _maybe_int(_pick(s, "hits")),
                        "toi": toi,
                    }
                    # strip None/empty
                    row = {k: v for k, v in row.items() if v is not None and v != ""}
                    if row:
                        out.append(row)
                return out
            except Exception:
                return []

        def _extract_periods(box: dict) -> list[dict]:
            periods = box.get("periods") or box.get("periodSummary") or []
            if not isinstance(periods, list):
                return []
            out = []
            for p in periods:
                if not isinstance(p, dict):
                    continue
                pd = p.get("periodDescriptor") or {}
                per = None
                if isinstance(pd, dict):
                    per = pd.get("number") or pd.get("period")
                if per is None:
                    per = p.get("period") or p.get("currentPeriod")
                home_p = p.get("home") or p.get("homeTeam") or {}
                away_p = p.get("away") or p.get("awayTeam") or {}
                if not isinstance(home_p, dict):
                    home_p = {}
                if not isinstance(away_p, dict):
                    away_p = {}

                row = {
                    "period": _maybe_int(per) or per,
                    "home": {
                        "goals": _maybe_int(_pick(home_p, "goals", "score")),
                        "sog": _maybe_int(_pick(home_p, "sog", "shotsOnGoal", "shots_on_goal")),
                        "shots": _maybe_int(_pick(home_p, "shots", "shotAttempts")),
                    },
                    "away": {
                        "goals": _maybe_int(_pick(away_p, "goals", "score")),
                        "sog": _maybe_int(_pick(away_p, "sog", "shotsOnGoal", "shots_on_goal")),
                        "shots": _maybe_int(_pick(away_p, "shots", "shotAttempts")),
                    },
                }
                # strip None fields
                for side in ("home", "away"):
                    row[side] = {k: v for k, v in (row.get(side) or {}).items() if v is not None}
                if row.get("home") or row.get("away"):
                    out.append(row)
            return out

        def _has_any_period_goals(per_list: object) -> bool:
            try:
                if not isinstance(per_list, list):
                    return False
                for p in per_list:
                    if not isinstance(p, dict):
                        continue
                    a = p.get("away") if isinstance(p.get("away"), dict) else {}
                    h = p.get("home") if isinstance(p.get("home"), dict) else {}
                    if (a.get("goals") is not None) or (h.get("goals") is not None):
                        return True
            except Exception:
                return False
            return False

        web = NHLWebClient()

        async def _extract_periods_landing_fallback(game_pk: int) -> list[dict]:
            """Fallback per-period goals from NHL Web /landing payload.

            The landing payload includes `summary.scoring`: a list of per-period buckets
            each with a `goals` list containing `isHome` flags.
            """
            try:
                landing = await asyncio.to_thread(web._get, f"/gamecenter/{int(game_pk)}/landing", None, 1)
                summary = landing.get("summary") if isinstance(landing, dict) else None
                scoring = (summary or {}).get("scoring") if isinstance(summary, dict) else None
                if not isinstance(scoring, list):
                    return []

                out: list[dict] = []
                for bucket in scoring:
                    if not isinstance(bucket, dict):
                        continue
                    pd = bucket.get("periodDescriptor") or {}
                    per = None
                    if isinstance(pd, dict):
                        per = pd.get("number") or pd.get("period")
                    if per is None:
                        per = bucket.get("period")

                    goals = bucket.get("goals")
                    home_g = 0
                    away_g = 0
                    if isinstance(goals, list):
                        for goal in goals:
                            if not isinstance(goal, dict):
                                continue
                            ih = goal.get("isHome")
                            if ih is True:
                                home_g += 1
                            elif ih is False:
                                away_g += 1

                    row = {
                        "period": _maybe_int(per) or per,
                        "home": {"goals": home_g},
                        "away": {"goals": away_g},
                    }
                    out.append(row)
                return out
            except Exception:
                return []

        # scoreboard_day is lightweight and usually includes gamePk/state/period/clock
        games = await asyncio.to_thread(web.scoreboard_day, d)
        out_games: list[dict] = []

        for g in games or []:
            try:
                game_pk = g.get("gamePk")
                if game_pk is None:
                    continue
                game_pk_i = int(game_pk)
            except Exception:
                continue

            home = str(g.get("home") or "")
            away = str(g.get("away") or "")
            game_state = g.get("gameState")

            # Best-effort detailed stats from boxscore; safe to fail without breaking endpoint.
            box = None
            try:
                box = await asyncio.to_thread(web.boxscore, game_pk_i)
            except Exception:
                box = None

            home_obj = box.get("homeTeam") if isinstance(box, dict) else None
            away_obj = box.get("awayTeam") if isinstance(box, dict) else None

            lens = {
                "totals": {
                    "home": _extract_team_totals(home_obj),
                    "away": _extract_team_totals(away_obj),
                },
                "periods": _extract_periods(box) if isinstance(box, dict) else [],
                "players": {
                    "home": _extract_skaters_summary(box, "home") if isinstance(box, dict) else [],
                    "away": _extract_skaters_summary(box, "away") if isinstance(box, dict) else [],
                },
                "goalies": {
                    "home": _extract_goalie_summary(box, "home") if isinstance(box, dict) else [],
                    "away": _extract_goalie_summary(box, "away") if isinstance(box, dict) else [],
                },
                "xg": None,
            }

            # Backfill per-period goals for live games when NHL Web boxscore doesn't provide it.
            try:
                st = str(game_state or "").upper()
                is_live_like = ("LIVE" in st) or ("IN_PROGRESS" in st) or ("CRIT" in st)
                if is_live_like:
                    per_list = lens.get("periods") if isinstance(lens, dict) else None
                    if (not per_list) or (not _has_any_period_goals(per_list)):
                        per_web = await _extract_periods_landing_fallback(game_pk_i)
                        if per_web:
                            lens["periods"] = per_web
            except Exception:
                pass

            # Remove empty blocks
            if not lens["totals"]["home"] and not lens["totals"]["away"]:
                lens.pop("totals", None)
            if not lens.get("periods"):
                lens.pop("periods", None)
            if not (lens.get("players", {}).get("home") or lens.get("players", {}).get("away")):
                lens.pop("players", None)
            if not (lens.get("goalies", {}).get("home") or lens.get("goalies", {}).get("away")):
                lens.pop("goalies", None)

            out_games.append({
                "date": d,
                "gamePk": game_pk_i,
                "home": home,
                "away": away,
                "key": f"{_norm(away)} @ {_norm(home)}",
                "gameState": game_state,
                "period": g.get("period"),
                "clock": g.get("clock"),
                "score": {
                    "home": g.get("home_goals"),
                    "away": g.get("away_goals"),
                },
                "lens": lens,
            })

        return JSONResponse({"ok": True, "date": d, "games": out_games})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/v1/odds/{date}")
async def v1_odds(date: str, regions: str = "us", best: bool = True, inplay: bool = False):
    """Current game odds snapshot for a slate date (read-only).

    Source: OddsAPI (the-odds-api.com). Requires ODDS_API_KEY.
    Returns normalized rows keyed by away@home, suitable for overlay in cards-only UI.

    Notes
    - This endpoint is best-effort and may return ok=True with an empty games list.
    - Uses a small in-memory TTL cache to reduce external calls.
    """
    try:
        d = str(date or "").strip()
        if not _V1_DATE_RE.fullmatch(d):
            return JSONResponse({"ok": False, "error": "invalid_date", "date": date}, status_code=400)

        obj = await asyncio.to_thread(_v1_odds_payload, d, str(regions or "us"), bool(best), bool(inplay))
        if not obj.get("ok") and obj.get("error") == "invalid_date":
            return JSONResponse(obj, status_code=400)
        return JSONResponse(obj)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/v1/live-lens/{date}")
async def v1_live_lens_combined(
    date: str,
    regions: str = "us",
    best: bool = True,
    include_non_live: bool = False,
    inplay: bool = True,
    include_pbp: bool = False,
):
    """Combined live lens endpoint: live game state + live odds + simple guidance signals.

    Intended for the cards-only UI to make a single request while games are live.
    - Live state source: NHL Web API via NHLWebClient
    - Odds source: OddsAPI via OddsAPIClient
    - Guidance: lightweight, best-effort heuristics (pace/SOG/goalie)
    """
    try:
        d = str(date or "").strip()
        if not _V1_DATE_RE.fullmatch(d):
            return JSONResponse({"ok": False, "error": "invalid_date", "date": date}, status_code=400)

        asof = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        try:
            odds_timeout_sec = float(os.getenv("LIVE_LENS_ODDS_TIMEOUT_SEC", "6"))
        except Exception:
            odds_timeout_sec = 6.0

        def _is_live_state(s: object) -> bool:
            try:
                x = str(s or "").upper()
                return ("LIVE" in x) or ("IN_PROGRESS" in x) or ("CRIT" in x)
            except Exception:
                return False

        def _american_to_implied_prob(price: object) -> Optional[float]:
            try:
                if price is None:
                    return None
                p = int(price)
                if p == 0:
                    return None
                if p > 0:
                    return 100.0 / (float(p) + 100.0)
                return float(-p) / (float(-p) + 100.0)
            except Exception:
                return None

        def _poisson_cdf(k: int, mu: float) -> float:
            try:
                if mu <= 0:
                    return 1.0 if k >= 0 else 0.0
                if k < 0:
                    return 0.0
                from math import exp

                pmf = exp(-float(mu))
                s = pmf
                for i in range(0, int(k)):
                    pmf = pmf * float(mu) / float(i + 1)
                    s += pmf
                return float(max(0.0, min(1.0, s)))
            except Exception:
                return 0.0

        def _poisson_sf(k: int, mu: float) -> float:
            try:
                if k <= 0:
                    return 1.0
                return float(max(0.0, 1.0 - _poisson_cdf(int(k) - 1, float(mu))))
            except Exception:
                return 0.0

        def _sigmoid(z: float) -> float:
            try:
                z = float(z)
                if z >= 0:
                    ez = math.exp(-z)
                    return 1.0 / (1.0 + ez)
                ez = math.exp(z)
                return ez / (1.0 + ez)
            except Exception:
                return 0.5

        def _logit(p: float) -> float:
            p = float(p)
            p = max(1e-6, min(1.0 - 1e-6, p))
            return math.log(p / (1.0 - p))

        def _norm_cdf(x: float) -> float:
            try:
                # Standard normal CDF via erf
                return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))
            except Exception:
                return 0.5

        def _to_float(x: object) -> Optional[float]:
            try:
                if x is None:
                    return None
                v = float(x)
                if not math.isfinite(v):
                    return None
                return float(v)
            except Exception:
                return None

        def _to_int(x: object) -> Optional[int]:
            try:
                if x is None:
                    return None
                return int(x)
            except Exception:
                return None

        def _parse_toi_to_min(toi: object) -> Optional[float]:
            """Parse TOI strings like '12:34' or '1:02:03' into minutes."""
            try:
                s = str(toi or "").strip()
                if not s or ":" not in s:
                    return None
                parts = s.split(":")
                parts_i = [int(p) for p in parts]
                if len(parts_i) == 2:
                    mm, ss = parts_i
                    if mm < 0 or ss < 0 or ss >= 60:
                        return None
                    return float(mm) + float(ss) / 60.0
                if len(parts_i) == 3:
                    hh, mm, ss = parts_i
                    if hh < 0 or mm < 0 or ss < 0 or mm >= 60 or ss >= 60:
                        return None
                    return float(60 * hh + mm) + float(ss) / 60.0
                return None
            except Exception:
                return None

        def _prob_to_american(p: float) -> Optional[int]:
            """Breakeven American odds for probability p."""
            try:
                p = float(p)
                if not math.isfinite(p) or p <= 0.0 or p >= 1.0:
                    return None
                if p >= 0.5:
                    return int(round(-100.0 * p / (1.0 - p)))
                return int(round(100.0 * (1.0 - p) / p))
            except Exception:
                return None

        def _safe_prob(p: object) -> Optional[float]:
            v = _to_float(p)
            if v is None:
                return None
            return float(max(0.0, min(1.0, v)))

        def _parse_mmss(s: object) -> Optional[int]:
            try:
                x = str(s or "").strip()
                if not x or ":" not in x:
                    return None
                mm, ss = x.split(":", 1)
                m = int(mm)
                sec = int(ss)
                if m < 0 or sec < 0 or sec >= 60:
                    return None
                return 60 * m + sec
            except Exception:
                return None

        def _decode_situation_code(code: object) -> Optional[dict]:
            """Decode NHL Web situationCode like '1551' as awaySkaters, homeSkaters, awayGoalie, homeGoalie."""
            try:
                s = str(code or "").strip()
                if len(s) != 4 or (not s.isdigit()):
                    return None
                a_s = int(s[0]); h_s = int(s[1]); a_g = int(s[2]); h_g = int(s[3])
                return {
                    "away_skaters": a_s,
                    "home_skaters": h_s,
                    "away_goalie": a_g,
                    "home_goalie": h_g,
                }
            except Exception:
                return None

        def _poisson_pmf_array(mu: float, kmax: int) -> list[float]:
            try:
                mu = float(mu)
                if mu < 0 or (not math.isfinite(mu)):
                    mu = 0.0
                kmax = int(max(0, kmax))
                from math import exp

                p0 = exp(-mu)
                out = [p0]
                p = p0
                for k in range(1, kmax + 1):
                    p = p * mu / float(k)
                    out.append(p)
                return out
            except Exception:
                return [1.0]

        def _win_cover_probs(gd: int, mu_home: float, mu_away: float) -> dict:
            """Compute in-regulation win and +1.5/-1.5 cover probs from independent Poisson remaining goals.

            Returns keys:
              p_home_win, p_away_win, p_tie, p_home_m15, p_away_p15
            """
            try:
                mu_home = float(max(0.0, mu_home))
                mu_away = float(max(0.0, mu_away))
                mu_tot = mu_home + mu_away
                # truncate tail reasonably; cap to keep fast
                kmax = int(min(14, max(8, math.ceil(mu_tot + 6.0 * math.sqrt(max(1e-6, mu_tot))))))
                ph = _poisson_pmf_array(mu_home, kmax)
                pa = _poisson_pmf_array(mu_away, kmax)
                p_home_win = 0.0
                p_away_win = 0.0
                p_tie = 0.0
                p_home_m15 = 0.0
                p_away_p15 = 0.0
                # final diff = gd + (h - a)
                for hi, p_hi in enumerate(ph):
                    if p_hi <= 0:
                        continue
                    for ai, p_ai in enumerate(pa):
                        if p_ai <= 0:
                            continue
                        p_ = p_hi * p_ai
                        d = int(gd) + int(hi) - int(ai)
                        if d > 0:
                            p_home_win += p_
                        elif d < 0:
                            p_away_win += p_
                        else:
                            p_tie += p_
                        if d >= 2:
                            p_home_m15 += p_
                        if d <= 1:
                            p_away_p15 += p_
                # allocate ties 50/50 to approximate OT
                p_home_win = float(max(0.0, min(1.0, p_home_win + 0.5 * p_tie)))
                p_away_win = float(max(0.0, min(1.0, p_away_win + 0.5 * p_tie)))
                return {
                    "p_home_win": p_home_win,
                    "p_away_win": p_away_win,
                    "p_tie": float(max(0.0, min(1.0, p_tie))),
                    "p_home_m15": float(max(0.0, min(1.0, p_home_m15))),
                    "p_away_p15": float(max(0.0, min(1.0, p_away_p15))),
                }
            except Exception:
                return {
                    "p_home_win": None,
                    "p_away_win": None,
                    "p_tie": None,
                    "p_home_m15": None,
                    "p_away_p15": None,
                }

        def _signal(action: str, scope: str, market: str, label: str, **kwargs) -> dict:
            out = {
                "action": str(action or "WATCH").upper(),
                "scope": str(scope or "game"),
                "market": str(market or ""),
                "label": str(label or ""),
            }
            for k, v in (kwargs or {}).items():
                out[str(k)] = v
            return out

        # Pull live lens via existing endpoint logic (single source of truth).
        live_resp = await v1_live(d)
        try:
            live_obj = json.loads(getattr(live_resp, "body", b"{}").decode("utf-8"))
        except Exception:
            live_obj = None
        if not isinstance(live_obj, dict) or not live_obj.get("ok"):
            # Best-effort: keep endpoint usable even if NHL live fetch fails.
            return JSONResponse({
                "ok": True,
                "date": d,
                "asof_utc": asof,
                "regions": str(regions or "us"),
                "best": bool(best),
                "odds_asof_utc": None,
                "games": [],
                "note": "live_unavailable",
            })

        # Odds payload (cached; best-effort)
        try:
            odds_obj = await asyncio.wait_for(
                asyncio.to_thread(_v1_odds_payload, d, str(regions or "us"), bool(best), bool(inplay)),
                timeout=odds_timeout_sec,
            )
        except Exception:
            odds_obj = {"ok": False, "games": []}
        odds_map: dict[str, dict] = {}
        if isinstance(odds_obj, dict):
            for og in odds_obj.get("games") or []:
                if not isinstance(og, dict):
                    continue
                k = og.get("key") or _norm_game_key(og.get("away"), og.get("home"))
                k = str(k or "").strip().lower()
                if k:
                    odds_map[k] = og

        # Player props odds (OddsAPI; cached, best-effort)
        try:
            props_odds_obj = await asyncio.wait_for(
                asyncio.to_thread(_v1_props_odds_payload, d, str(regions or "us"), bool(best), bool(inplay)),
                timeout=odds_timeout_sec,
            )
        except Exception:
            props_odds_obj = {"ok": False, "games": []}
        props_odds_map: dict[str, dict] = {}
        if isinstance(props_odds_obj, dict):
            for pg in props_odds_obj.get("games") or []:
                if not isinstance(pg, dict):
                    continue
                k = pg.get("key") or _norm_game_key(pg.get("away"), pg.get("home"))
                k = str(k or "").strip().lower()
                if k:
                    props_odds_map[k] = pg

        # Pregame per-player lambdas for props (best-effort)
        proj_df = None
        try:
            proj_df = _read_all_players_projections(d)
        except Exception:
            proj_df = None

        def _canon_prop_market(x: object) -> str:
            s = str(x or "").strip().upper()
            if s in {"SOG", "SHOTS", "SHOT", "SHOTS_ON_GOAL", "PLAYER_SHOTS"}:
                return "SOG"
            if s in {"GOALS", "GOAL"}:
                return "GOALS"
            if s in {"ASSISTS", "ASSIST"}:
                return "ASSISTS"
            if s in {"POINTS", "POINT"}:
                return "POINTS"
            if s in {"SAVES", "SAVE"}:
                return "SAVES"
            if s in {"BLOCKS", "BLK", "BLOCKED_SHOTS"}:
                return "BLOCKS"
            return s

        def _norm_person(s: object) -> str:
            try:
                x = str(s or "").strip().lower()
                x = re.sub(r"[\.'’]", "", x)
                x = " ".join(x.split())
                return x
            except Exception:
                return str(s or "").strip().lower()

        proj_lam: dict[tuple[str, str], float] = {}
        proj_lam_team: dict[tuple[str, str, str], float] = {}
        try:
            if proj_df is not None and not proj_df.empty and {"player", "market", "proj_lambda"}.issubset(proj_df.columns):
                for _, r in proj_df.iterrows():
                    mk = _canon_prop_market(r.get("market"))
                    nm = _norm_person(r.get("player"))
                    try:
                        lam = float(r.get("proj_lambda")) if r.get("proj_lambda") is not None else None
                    except Exception:
                        lam = None
                    if not nm or not mk or lam is None or (not math.isfinite(lam)):
                        continue
                    proj_lam[(mk, nm)] = float(lam)
                    try:
                        team = str(r.get("team") or "").strip().upper() if "team" in proj_df.columns else ""
                        if team:
                            proj_lam_team[(mk, nm, team)] = float(lam)
                    except Exception:
                        pass
        except Exception:
            proj_lam = {}
            proj_lam_team = {}

        # Pregame model fields from bundle (best-effort)
        pred_map = _load_bundle_predictions_map(d)

        # Rest flags (best-effort): B2B and 3-in-4 via schedule lookback.
        # Gate behind inplay=0 to avoid extra network calls during live polling.
        played_by_day: dict[str, set[str]] = {}
        try:
            if not bool(inplay):
                from datetime import date as _date

                base_dt = datetime.fromisoformat(d).date() if isinstance(d, str) else None
                if isinstance(base_dt, _date):
                    web_fast = NHLWebClient(timeout=float(os.getenv("LIVE_LENS_SCHEDULE_TIMEOUT_SEC", "3")))

                    def _teams_played(day_str: str) -> set[str]:
                        try:
                            obj = web_fast._get(f"/schedule/{day_str}", None, 1)
                            out: set[str] = set()
                            for wk in obj.get("gameWeek", []) if isinstance(obj, dict) else []:
                                if wk.get("date") != day_str:
                                    continue
                                for gg in wk.get("games", []) or []:
                                    try:
                                        h = _team_name(gg.get("homeTeam", {}))
                                        a = _team_name(gg.get("awayTeam", {}))
                                        if h:
                                            out.add(_norm(h))
                                        if a:
                                            out.add(_norm(a))
                                    except Exception:
                                        continue
                            return out
                        except Exception:
                            return set()

                    for ddays in (1, 2, 3):
                        day_str = (base_dt - timedelta(days=ddays)).strftime("%Y-%m-%d")
                        played_by_day[day_str] = _teams_played(day_str)
        except Exception:
            played_by_day = {}

        out_games: list[dict] = []
        for g in live_obj.get("games") or []:
            if not isinstance(g, dict):
                continue
            if (not include_non_live) and (not _is_live_state(g.get("gameState"))):
                continue

            key = str(g.get("key") or _norm_game_key(g.get("away"), g.get("home")) or "").strip().lower()
            og = odds_map.get(key)
            pm = pred_map.get(key)

            # Current score
            score = g.get("score") or {}
            try:
                away_goals = int((score or {}).get("away")) if (score or {}).get("away") is not None else None
            except Exception:
                away_goals = None
            try:
                home_goals = int((score or {}).get("home")) if (score or {}).get("home") is not None else None
            except Exception:
                home_goals = None
            total_goals = (away_goals + home_goals) if (away_goals is not None and home_goals is not None) else None

            # Elapsed time (regulation only)
            try:
                period_i = int(g.get("period")) if g.get("period") is not None else None
            except Exception:
                period_i = None
            clock_sec = _parse_mmss_clock(g.get("clock"))
            elapsed_min = None
            if period_i is not None and clock_sec is not None and 1 <= period_i <= 3:
                per_len = 20 * 60
                elapsed_sec = (period_i - 1) * per_len + (per_len - clock_sec)
                if elapsed_sec >= 0:
                    elapsed_min = float(elapsed_sec) / 60.0

            # If the game looks final/off and we have a score but no clock, treat regulation as complete.
            # Gate behind include_pbp so normal live polling / UI doesn't emit end-of-game "certainty" signals.
            try:
                if bool(include_pbp):
                    st0 = str(g.get("gameState") or "").upper()
                    if elapsed_min is None and clock_sec is None and period_i == 3 and total_goals is not None and (not _is_live_state(st0)):
                        elapsed_min = 60.0
            except Exception:
                pass
            remaining_min = None
            if elapsed_min is not None:
                remaining_min = max(0.0, 60.0 - float(elapsed_min))

            # Lens-derived pace inputs
            lens = g.get("lens") or {}
            sog_total = None
            away_sog = None
            home_sog = None
            try:
                totals = lens.get("totals") if isinstance(lens, dict) else None
                if isinstance(totals, dict):
                    a = totals.get("away") or {}
                    h = totals.get("home") or {}
                    if isinstance(a, dict) and isinstance(h, dict):
                        a_sog = a.get("sog")
                        h_sog = h.get("sog")
                        if a_sog is not None and h_sog is not None:
                            away_sog = int(a_sog)
                            home_sog = int(h_sog)
                            sog_total = int(a_sog) + int(h_sog)
            except Exception:
                sog_total = None

            # If team SOG totals aren't available, derive them from per-player shots.
            try:
                if sog_total is None and isinstance(lens, dict):
                    pl = lens.get("players")
                    if isinstance(pl, dict):
                        def _sum_shots(arr: object) -> Optional[int]:
                            if not isinstance(arr, list) or not arr:
                                return None
                            s = 0
                            seen = False
                            for r in arr:
                                if not isinstance(r, dict):
                                    continue
                                if r.get("s") is None:
                                    continue
                                try:
                                    s += int(r.get("s") or 0)
                                    seen = True
                                except Exception:
                                    continue
                            return int(s) if seen else None

                        a_s = _sum_shots(pl.get("away"))
                        h_s = _sum_shots(pl.get("home"))
                        if a_s is not None and h_s is not None:
                            away_sog = int(a_s)
                            home_sog = int(h_s)
                            sog_total = int(a_s) + int(h_s)
            except Exception:
                pass

            goal_pace_60 = None
            sog_pace_60 = None
            if elapsed_min is not None and elapsed_min > 0:
                if total_goals is not None:
                    goal_pace_60 = 60.0 * (float(total_goals) / float(elapsed_min))
                if sog_total is not None:
                    sog_pace_60 = 60.0 * (float(sog_total) / float(elapsed_min))

            goalie_sv: list[float] = []
            try:
                gl = lens.get("goalies") if isinstance(lens, dict) else None

                def _pick_goalie_sv(arr: object) -> Optional[float]:
                    if not isinstance(arr, list):
                        return None
                    best_sv = None
                    best_sa = -1
                    for r in arr:
                        if not isinstance(r, dict):
                            continue
                        sv = r.get("sv_pct")
                        if sv is None:
                            continue
                        try:
                            sa = int(r.get("shots_against") or 0)
                        except Exception:
                            sa = 0
                        try:
                            sv_f = float(sv)
                        except Exception:
                            continue
                        if sa > best_sa:
                            best_sa = sa
                            best_sv = sv_f
                    return best_sv

                if isinstance(gl, dict):
                    for side in ("away", "home"):
                        sv = _pick_goalie_sv(gl.get(side))
                        if sv is not None:
                            goalie_sv.append(float(sv))
            except Exception:
                goalie_sv = []

            # Pregame model context
            model_total = None
            model_spread = None
            pre_total_line = None
            if isinstance(pm, dict):
                model_total = pm.get("model_total")
                model_spread = pm.get("model_spread")
                pre_total_line = pm.get("total_line_used")

            live_total_line = None
            total_over_price = None
            total_under_price = None
            try:
                if isinstance(og, dict):
                    tot = og.get("total") or {}
                    if isinstance(tot, dict):
                        live_total_line = tot.get("line")
                        total_over_price = tot.get("over")
                        total_under_price = tot.get("under")
            except Exception:
                live_total_line = None

            # Guidance math
            notes: list[str] = []
            mu_remaining = None
            p_over = None
            p_under = None
            p_push = None
            implied_over = _american_to_implied_prob(total_over_price)
            implied_under = _american_to_implied_prob(total_under_price)
            edge_over = None
            edge_under = None
            lean_total = "neutral"

            # Conservative pace/goalie multipliers (small adjustments only)
            pace_mult = 1.0
            # Extra live context from play-by-play: manpower (PP/EN), shot attempts, xG proxy
            pbp_ctx: dict[str, object] = {}
            pbp_error: Optional[str] = None
            try:
                st = str(g.get("gameState") or "").upper()
                want_pbp = bool(include_pbp) or _is_live_state(st)
                if want_pbp and g.get("gamePk"):
                    # Import locally so the endpoint doesn't depend on module import order.
                    from ..data.nhl_api_web import NHLWebClient
                    pbp_web = NHLWebClient(timeout=float(os.getenv("LIVE_LENS_PBP_TIMEOUT_SEC", "4")))
                    pbp = await asyncio.to_thread(pbp_web._get, f"/gamecenter/{int(g.get('gamePk'))}/play-by-play", None, 1)
                    plays = pbp.get("plays") if isinstance(pbp, dict) else None
                    hteam = pbp.get("homeTeam") if isinstance(pbp, dict) else None
                    ateam = pbp.get("awayTeam") if isinstance(pbp, dict) else None
                    home_id = hteam.get("id") if isinstance(hteam, dict) else None
                    away_id = ateam.get("id") if isinstance(ateam, dict) else None

                    # current time in absolute seconds (reg only)
                    cur_abs_sec = None
                    try:
                        if period_i is not None and clock_sec is not None and 1 <= period_i <= 3:
                            per_len = 20 * 60
                            cur_abs_sec = int((period_i - 1) * per_len + (per_len - int(clock_sec)))
                    except Exception:
                        cur_abs_sec = None

                    # Fallback: infer current time from play-by-play if clock is missing (e.g. FINAL/off games).
                    try:
                        if cur_abs_sec is None and isinstance(plays, list) and plays:
                            max_abs = None
                            for ev in plays:
                                if not isinstance(ev, dict):
                                    continue
                                pdn = (ev.get("periodDescriptor") or {}).get("number")
                                tin = _parse_mmss(ev.get("timeInPeriod"))
                                if pdn is None or tin is None:
                                    continue
                                if 1 <= int(pdn) <= 3:
                                    abs_sec = int((int(pdn) - 1) * 1200 + int(tin))
                                    if max_abs is None or abs_sec > max_abs:
                                        max_abs = abs_sec
                            cur_abs_sec = max_abs
                    except Exception:
                        pass

                    # If we inferred time from PBP, backfill elapsed/remaining for pace calculations.
                    try:
                        if elapsed_min is None and cur_abs_sec is not None:
                            elapsed_min = float(cur_abs_sec) / 60.0
                            remaining_min = max(0.0, 60.0 - float(elapsed_min))
                    except Exception:
                        pass

                    # latest situation code
                    latest = None
                    if isinstance(plays, list) and plays:
                        try:
                            latest = max([p for p in plays if isinstance(p, dict)], key=lambda x: int(x.get("sortOrder") or 0))
                        except Exception:
                            latest = plays[-1] if isinstance(plays[-1], dict) else None
                    sit = _decode_situation_code((latest or {}).get("situationCode")) if isinstance(latest, dict) else None
                    if sit:
                        pbp_ctx.update({
                            "away_skaters": sit.get("away_skaters"),
                            "home_skaters": sit.get("home_skaters"),
                            "away_goalie": sit.get("away_goalie"),
                            "home_goalie": sit.get("home_goalie"),
                        })
                        try:
                            pbp_ctx["manpower"] = f"{int(sit['away_skaters'])}v{int(sit['home_skaters'])}"
                        except Exception:
                            pass
                        try:
                            # PP team: side with more skaters
                            pp_team = None
                            if int(sit.get("away_skaters") or 0) > int(sit.get("home_skaters") or 0):
                                pp_team = "away"
                            elif int(sit.get("home_skaters") or 0) > int(sit.get("away_skaters") or 0):
                                pp_team = "home"
                            if pp_team:
                                pbp_ctx["pp_team"] = pp_team
                        except Exception:
                            pass
                        try:
                            pbp_ctx["away_empty_net"] = bool(int(sit.get("away_goalie") or 1) == 0)
                            pbp_ctx["home_empty_net"] = bool(int(sit.get("home_goalie") or 1) == 0)
                        except Exception:
                            pass

                        # Estimate PP segment remaining time by scanning back to when current manpower began.
                        try:
                            if cur_abs_sec is not None and pbp_ctx.get("pp_team") and isinstance(plays, list):
                                cur_code = str((latest or {}).get("situationCode") or "")
                                start_abs = None
                                for ev in reversed(plays):
                                    if not isinstance(ev, dict):
                                        continue
                                    if str(ev.get("situationCode") or "") != cur_code:
                                        break
                                    pdn = (ev.get("periodDescriptor") or {}).get("number")
                                    tin = _parse_mmss(ev.get("timeInPeriod"))
                                    if pdn is None or tin is None:
                                        continue
                                    if 1 <= int(pdn) <= 3:
                                        start_abs = int((int(pdn) - 1) * 1200 + int(tin))
                                if start_abs is not None:
                                    dur = int(max(0, cur_abs_sec - start_abs))
                                    # assume minor (2:00) for display; clamp
                                    pp_rem = int(max(0, 120 - dur))
                                    pbp_ctx["pp_sec_remaining_est"] = pp_rem
                        except Exception:
                            pass

                    # Shot attempts + xG proxy
                    try:
                        att_types = {"shot-on-goal", "missed-shot", "blocked-shot", "goal"}
                        home_att = away_att = 0
                        home_att_l5 = away_att_l5 = 0
                        home_xg = away_xg = 0.0
                        home_xg_l10 = away_xg_l10 = 0.0

                        def _xg_weight(dd: dict) -> float:
                            try:
                                x = dd.get("xCoord")
                                y = dd.get("yCoord")
                                if x is None or y is None:
                                    return 0.0
                                x = float(x); y = float(y)
                                # net at ~x=89, ignore orientation by taking abs(x)
                                dx = 89.0 - abs(x)
                                dy = abs(y)
                                dist = math.sqrt(max(0.0, dx * dx + dy * dy))
                                if dist <= 10:
                                    w = 0.24
                                elif dist <= 20:
                                    w = 0.12
                                elif dist <= 30:
                                    w = 0.07
                                elif dist <= 40:
                                    w = 0.04
                                else:
                                    w = 0.02
                                stp = str(dd.get("shotType") or "").lower()
                                if stp in {"tip", "tip-in", "deflected", "wrap-around"}:
                                    w *= 1.25
                                return float(max(0.0, min(0.35, w)))
                            except Exception:
                                return 0.0

                        if isinstance(plays, list) and plays and home_id is not None and away_id is not None and cur_abs_sec is not None:
                            for ev in plays:
                                if not isinstance(ev, dict):
                                    continue
                                if str(ev.get("typeDescKey") or "") not in att_types:
                                    continue
                                dd = ev.get("details") if isinstance(ev.get("details"), dict) else {}
                                tid = dd.get("eventOwnerTeamId")
                                # clock window
                                pdn = (ev.get("periodDescriptor") or {}).get("number")
                                tin = _parse_mmss(ev.get("timeInPeriod"))
                                abs_sec = None
                                if pdn is not None and tin is not None and 1 <= int(pdn) <= 3:
                                    abs_sec = int((int(pdn) - 1) * 1200 + int(tin))

                                w = _xg_weight(dd)
                                if tid == home_id:
                                    home_att += 1
                                    home_xg += w
                                    if abs_sec is not None and (cur_abs_sec - abs_sec) <= 5 * 60:
                                        home_att_l5 += 1
                                    if abs_sec is not None and (cur_abs_sec - abs_sec) <= 10 * 60:
                                        home_xg_l10 += w
                                elif tid == away_id:
                                    away_att += 1
                                    away_xg += w
                                    if abs_sec is not None and (cur_abs_sec - abs_sec) <= 5 * 60:
                                        away_att_l5 += 1
                                    if abs_sec is not None and (cur_abs_sec - abs_sec) <= 10 * 60:
                                        away_xg_l10 += w

                        pbp_ctx.update({
                            "home_att": home_att,
                            "away_att": away_att,
                            "home_att_l5": home_att_l5,
                            "away_att_l5": away_att_l5,
                            "home_xg_proxy": float(home_xg),
                            "away_xg_proxy": float(away_xg),
                            "home_xg_proxy_l10": float(home_xg_l10),
                            "away_xg_proxy_l10": float(away_xg_l10),
                        })
                    except Exception:
                        pass
            except Exception as e:
                try:
                    pbp_error = str(e)
                except Exception:
                    pbp_error = "pbp_fetch_failed"
                pbp_ctx = {}
            if sog_pace_60 is not None:
                try:
                    pace_mult = float(sog_pace_60) / 65.0
                    pace_mult = max(0.85, min(1.15, pace_mult))
                except Exception:
                    pace_mult = 1.0

            # Blend in shot-attempt pace when available (Corsi-like) for a better pace signal than SOG alone.
            try:
                if elapsed_min is not None and elapsed_min > 0 and pbp_ctx.get("home_att") is not None and pbp_ctx.get("away_att") is not None:
                    att_tot = int(pbp_ctx.get("home_att") or 0) + int(pbp_ctx.get("away_att") or 0)
                    att_pace_60 = 60.0 * (float(att_tot) / float(elapsed_min))
                    pbp_ctx["att_pace_60"] = float(att_pace_60)
                    # Typical total attempts pace ~110-120 per 60; keep conservative.
                    att_mult = float(att_pace_60) / 112.0
                    att_mult = max(0.85, min(1.15, att_mult))
                    pace_mult = float(max(0.80, min(1.20, 0.55 * float(pace_mult) + 0.45 * float(att_mult))))
            except Exception:
                pass

            # Small special-teams / empty-net adjustments used by totals + ML/PL.
            pp_bonus_home = 0.0
            pp_bonus_away = 0.0
            en_bonus_home = 0.0
            en_bonus_away = 0.0
            try:
                if pbp_ctx.get("pp_team") == "home":
                    pp_bonus_home = 0.18
                elif pbp_ctx.get("pp_team") == "away":
                    pp_bonus_away = 0.18

                rm_ = float(remaining_min) if remaining_min is not None else 0.0
                en_factor = max(0.0, min(1.0, rm_ / 3.0))
                if pbp_ctx.get("home_empty_net"):
                    en_bonus_away = 0.55 * en_factor
                if pbp_ctx.get("away_empty_net"):
                    en_bonus_home = 0.55 * en_factor
            except Exception:
                pp_bonus_home = pp_bonus_away = 0.0
                en_bonus_home = en_bonus_away = 0.0
            goalie_mult = 1.0
            if goalie_sv:
                try:
                    avg_sv = sum(goalie_sv) / float(len(goalie_sv))
                    if avg_sv <= 0.890:
                        goalie_mult = 1.05
                        notes.append("Goaltending underperforming (sv%)")
                    elif avg_sv >= 0.925:
                        goalie_mult = 0.95
                        notes.append("Goaltending strong (sv%)")
                except Exception:
                    goalie_mult = 1.0

            # Simple finishing diagnostic (helps explain extreme totals without full xG/PBP).
            try:
                em_ = _to_float(elapsed_min)
                if total_goals is not None and sog_total is not None and em_ is not None and em_ >= 10.0:
                    if float(sog_total) > 0:
                        sh = float(total_goals) / float(sog_total)
                        if sh >= 0.14:
                            notes.append(f"Shooting% high ({100.0 * sh:.1f}%)")
                        elif sh <= 0.06:
                            notes.append(f"Shooting% low ({100.0 * sh:.1f}%)")
            except Exception:
                pass

            try:
                if model_total is not None and total_goals is not None and remaining_min is not None and live_total_line is not None:
                    mt = float(model_total)
                    rm = float(remaining_min)
                    mu_remaining = max(0.0, mt * (rm / 60.0) * float(pace_mult) * float(goalie_mult))
                    # Conservative PP/EN bump (PP adds to advantaged team; EN adds to opponent scoring).
                    mu_remaining = float(mu_remaining) + float(pp_bonus_home + pp_bonus_away) * (rm / 60.0) + float(en_bonus_home + en_bonus_away)

                    line_f = float(live_total_line)
                    is_int_line = abs(line_f - round(line_f)) < 1e-9
                    if is_int_line:
                        line_i = int(round(line_f))
                        over_min_total = line_i + 1
                        under_max_total = line_i - 1
                        push_total = line_i
                    else:
                        over_min_total = int((line_f // 1) + 1)
                        under_max_total = int(line_f // 1)
                        push_total = None

                    need_over = int(over_min_total) - int(total_goals)
                    need_under = int(under_max_total) - int(total_goals)
                    p_over = 1.0 if need_over <= 0 else _poisson_sf(int(need_over), float(mu_remaining))
                    p_under = 0.0 if need_under < 0 else _poisson_cdf(int(need_under), float(mu_remaining))
                    if push_total is not None:
                        k_push = int(push_total) - int(total_goals)
                        if k_push < 0:
                            p_push = 0.0
                        else:
                            p_push = max(0.0, _poisson_cdf(k_push, float(mu_remaining)) - _poisson_cdf(k_push - 1, float(mu_remaining)))

                    if p_over is not None and implied_over is not None:
                        edge_over = float(p_over) - float(implied_over)
                    if p_under is not None and implied_under is not None:
                        edge_under = float(p_under) - float(implied_under)

                    thr = 0.03
                    if edge_over is not None and edge_under is not None:
                        if float(edge_over) >= float(edge_under) and float(edge_over) >= thr:
                            lean_total = "over"
                        elif float(edge_under) > float(edge_over) and float(edge_under) >= thr:
                            lean_total = "under"
                    elif edge_over is not None and float(edge_over) >= thr:
                        lean_total = "over"
                    elif edge_under is not None and float(edge_under) >= thr:
                        lean_total = "under"

                    if sog_pace_60 is not None:
                        try:
                            if float(sog_pace_60) >= 78.0:
                                notes.append("High shot volume pace")
                            elif float(sog_pace_60) <= 55.0:
                                notes.append("Low shot volume pace")
                        except Exception:
                            pass

                    if p_over is not None and implied_over is not None:
                        notes.append(f"Model O {p_over:.0%} vs implied {implied_over:.0%}")
                    if p_under is not None and implied_under is not None:
                        notes.append(f"Model U {p_under:.0%} vs implied {implied_under:.0%}")
            except Exception:
                pass

            # Add play-by-play context notes (manpower/EN/pressure) and apply small modifiers.
            try:
                mp = pbp_ctx.get("manpower")
                if mp:
                    pp_team = pbp_ctx.get("pp_team")
                    pp_rem = pbp_ctx.get("pp_sec_remaining_est")
                    extra = ""
                    if pp_team:
                        extra = f" {str(pp_team).upper()} PP"
                    if pp_rem is not None:
                        try:
                            mm = int(pp_rem) // 60
                            ss = int(pp_rem) % 60
                            extra += f" (est {mm}:{ss:02d})"
                        except Exception:
                            pass
                    notes.append(f"Manpower {mp}{extra}")
                # Empty net flags from situation
                if pbp_ctx.get("home_empty_net"):
                    notes.append("Home goalie pulled (empty net)")
                if pbp_ctx.get("away_empty_net"):
                    notes.append("Away goalie pulled (empty net)")
            except Exception:
                pass

            try:
                ha5 = pbp_ctx.get("home_att_l5")
                aa5 = pbp_ctx.get("away_att_l5")
                if ha5 is not None and aa5 is not None:
                    notes.append(f"Attempts last5: A {int(aa5)} / H {int(ha5)}")
            except Exception:
                pass

            try:
                hx = pbp_ctx.get("home_xg_proxy_l10")
                ax = pbp_ctx.get("away_xg_proxy_l10")
                if hx is not None and ax is not None:
                    notes.append(f"xG proxy last10: A {float(ax):.2f} / H {float(hx):.2f}")
            except Exception:
                pass

            # Rest flags (B2B / 3-in-4) -> notes
            try:
                away_nm = _norm(g.get("away"))
                home_nm = _norm(g.get("home"))
                # yesterday
                yday = None
                if played_by_day:
                    yday = sorted(list(played_by_day.keys()))[0]  # smallest is 3 days ago; we want 1 day ago
                    # safer: recompute keys by day offset
                    yday = (datetime.fromisoformat(d).date() - timedelta(days=1)).strftime("%Y-%m-%d")
                if yday and yday in played_by_day:
                    if away_nm in played_by_day.get(yday, set()):
                        notes.append("Away on B2B")
                    if home_nm in played_by_day.get(yday, set()):
                        notes.append("Home on B2B")
                # 3-in-4
                if played_by_day:
                    def _count_recent(team_norm: str) -> int:
                        c = 0
                        for day_str, teams in played_by_day.items():
                            if team_norm in (teams or set()):
                                c += 1
                        return c
                    if _count_recent(away_nm) >= 2:
                        notes.append("Away 3-in-4")
                    if _count_recent(home_nm) >= 2:
                        notes.append("Home 3-in-4")
            except Exception:
                pass

            # Line context note
            if live_total_line is not None and total_goals is not None:
                try:
                    if float(total_goals) >= float(live_total_line):
                        notes.append("Total already reached")
                    else:
                        rem = float(live_total_line) - float(total_goals)
                        notes.append(f"Needs ~{rem:.1f} more goals to reach total")
                except Exception:
                    pass

            out = dict(g)
            out["odds"] = og
            out["pregame"] = {
                "model_total": model_total,
                "model_spread": model_spread,
                "total_line_used": pre_total_line,
            }
            out["guidance"] = {
                "asof_utc": asof,
                "elapsed_min": elapsed_min,
                "remaining_min": remaining_min,
                "total_goals": total_goals,
                "sog_total": sog_total,
                "goal_pace_60": goal_pace_60,
                "sog_pace_60": sog_pace_60,
                "live_total_line": live_total_line,
                "lean_total": lean_total,
                "mu_remaining": mu_remaining,
                "p_over": p_over,
                "p_under": p_under,
                "p_push": p_push,
                "implied_over": implied_over,
                "implied_under": implied_under,
                "edge_over": edge_over,
                "edge_under": edge_under,
                "notes": notes,
            }

            # Attach play-by-play derived context for the UI.
            try:
                if pbp_ctx:
                    out["guidance"].update({
                        "manpower": pbp_ctx.get("manpower"),
                        "pp_team": pbp_ctx.get("pp_team"),
                        "pp_sec_remaining_est": pbp_ctx.get("pp_sec_remaining_est"),
                        "home_empty_net": pbp_ctx.get("home_empty_net"),
                        "away_empty_net": pbp_ctx.get("away_empty_net"),
                        "home_att": pbp_ctx.get("home_att"),
                        "away_att": pbp_ctx.get("away_att"),
                        "home_att_l5": pbp_ctx.get("home_att_l5"),
                        "away_att_l5": pbp_ctx.get("away_att_l5"),
                        "att_pace_60": pbp_ctx.get("att_pace_60"),
                        "home_xg_proxy": pbp_ctx.get("home_xg_proxy"),
                        "away_xg_proxy": pbp_ctx.get("away_xg_proxy"),
                        "home_xg_proxy_l10": pbp_ctx.get("home_xg_proxy_l10"),
                        "away_xg_proxy_l10": pbp_ctx.get("away_xg_proxy_l10"),
                    })
                if bool(include_pbp) and pbp_error:
                    out["guidance"]["pbp_error"] = str(pbp_error)[:200]
            except Exception:
                pass

            # ------------------------------------------------------------------
            # Signals: "WATCH" vs "BET" for game + key props, best-effort.
            # Notes:
            # - Game totals: use our existing Poisson edge math vs live total odds.
            # - Player shots / goalie saves: no live prop lines here, so we emit
            #   actionable "target" guidance (line + max price) from live pace.
            # ------------------------------------------------------------------
            signals: list[dict] = []

            # 1) Game total signal
            try:
                em = _to_float(elapsed_min)
                if live_total_line is not None and total_goals is not None and em is not None:
                    eo = _to_float(edge_over)
                    eu = _to_float(edge_under)
                    po = _safe_prob(p_over)
                    pu = _safe_prob(p_under)
                    line_f = _to_float(live_total_line)

                    side = None
                    edge = None
                    p_model = None
                    price = None
                    if lean_total == "over":
                        side = "OVER"
                        edge = eo
                        p_model = po
                        price = total_over_price
                    elif lean_total == "under":
                        side = "UNDER"
                        edge = eu
                        p_model = pu
                        price = total_under_price

                    if side and edge is not None and p_model is not None and line_f is not None:
                        # Gate: avoid betting too early unless edge is very strong.
                        abs_edge = abs(float(edge))
                        action = "WATCH"
                        if em >= 10.0 and abs_edge >= 0.04:
                            action = "BET"
                        if em >= 6.0 and abs_edge >= 0.06:
                            action = "BET"

                        # If total already reached, it's usually not a normal live bet.
                        try:
                            if float(total_goals) >= float(line_f):
                                action = "WATCH"
                        except Exception:
                            pass

                        p_target = max(0.01, min(0.99, float(p_model) - 0.03))
                        fair = _prob_to_american(float(p_model))
                        max_price = _prob_to_american(float(p_target))
                        signals.append(_signal(
                            action,
                            scope="game",
                            market="TOTAL",
                            label=f"Total {side} {line_f:g}",
                            side=side,
                            line=line_f,
                            price=_to_int(price),
                            p_model=float(p_model),
                            edge=float(edge),
                            fair_price_american=fair,
                            target_max_price_american=max_price,
                            elapsed_min=float(em),
                        ))
            except Exception:
                pass

            # 1b) Moneyline and puck line signals (best-effort)
            try:
                em = _to_float(elapsed_min)
                rm = _to_float(remaining_min)
                if em is not None and rm is not None and em >= 2.0:
                    # In-play odds
                    ml_home_price = None
                    ml_away_price = None
                    pl_home_m15 = None
                    pl_away_p15 = None
                    try:
                        if isinstance(og, dict):
                            ml = og.get("ml") or {}
                            if isinstance(ml, dict):
                                ml_home_price = ml.get("home")
                                ml_away_price = ml.get("away")
                            pl = og.get("puckline") or {}
                            if isinstance(pl, dict):
                                pl_home_m15 = pl.get("home_-1.5")
                                pl_away_p15 = pl.get("away_+1.5")
                    except Exception:
                        pass

                    implied_home = _american_to_implied_prob(ml_home_price)
                    implied_away = _american_to_implied_prob(ml_away_price)
                    implied_pl_home = _american_to_implied_prob(pl_home_m15)
                    implied_pl_away = _american_to_implied_prob(pl_away_p15)

                    # Pregame baseline
                    p0_home = None
                    try:
                        if isinstance(pm, dict):
                            p0_home = _safe_prob(pm.get("p_home_ml"))
                    except Exception:
                        p0_home = None
                    if p0_home is not None:
                        # Score/time adjustment (conservative heuristic), enriched with attempts/xG proxy/manpower.
                        gd = None
                        if home_goals is not None and away_goals is not None:
                            gd = int(home_goals) - int(away_goals)
                        sd = None
                        if home_sog is not None and away_sog is not None:
                            sd = int(home_sog) - int(away_sog)
                        ad = None
                        try:
                            if pbp_ctx.get("home_att") is not None and pbp_ctx.get("away_att") is not None:
                                ad = int(pbp_ctx.get("home_att") or 0) - int(pbp_ctx.get("away_att") or 0)
                        except Exception:
                            ad = None
                        xd = None
                        try:
                            if pbp_ctx.get("home_xg_proxy_l10") is not None and pbp_ctx.get("away_xg_proxy_l10") is not None:
                                xd = float(pbp_ctx.get("home_xg_proxy_l10") or 0.0) - float(pbp_ctx.get("away_xg_proxy_l10") or 0.0)
                        except Exception:
                            xd = None

                        z = _logit(float(p0_home))
                        if gd is not None:
                            # Goals matter more as the game progresses
                            w = 0.5 + 1.5 * (float(em) / 60.0)
                            z += 1.15 * float(gd) * float(w)
                        if sd is not None:
                            # Shot advantage is a mild signal
                            z += 0.03 * float(max(-20, min(20, sd)))
                        if ad is not None:
                            # Attempts advantage: mild but more stable than SOG
                            z += 0.012 * float(max(-30, min(30, ad)))
                        if xd is not None:
                            # xG proxy last10: very small influence
                            z += 0.60 * float(max(-1.5, min(1.5, xd)))
                        # Manpower advantage -> small bump
                        try:
                            if pbp_ctx.get("pp_team") == "home":
                                z += 0.25
                            elif pbp_ctx.get("pp_team") == "away":
                                z -= 0.25
                        except Exception:
                            pass

                        p_home_live = float(max(0.01, min(0.99, _sigmoid(z))))
                        p_away_live = float(max(0.01, min(0.99, 1.0 - p_home_live)))

                        # Optional: if we can compute Poisson win probs from model_total/spread, prefer that.
                        try:
                            if model_total is not None and model_spread is not None and gd is not None and rm is not None:
                                mt = float(model_total)
                                ms = float(model_spread)
                                mu_home_full = max(0.0, (mt + ms) / 2.0)
                                mu_away_full = max(0.0, (mt - ms) / 2.0)
                                mu_home_rem = mu_home_full * (float(rm) / 60.0) * float(pace_mult) * float(goalie_mult)
                                mu_away_rem = mu_away_full * (float(rm) / 60.0) * float(pace_mult) * float(goalie_mult)
                                # add mild PP bonus
                                mu_home_rem += float(pp_bonus_home) * (float(rm) / 60.0) + float(en_bonus_home)
                                mu_away_rem += float(pp_bonus_away) * (float(rm) / 60.0) + float(en_bonus_away)
                                pr = _win_cover_probs(int(gd), float(mu_home_rem), float(mu_away_rem))
                                if pr.get("p_home_win") is not None:
                                    p_home_live = float(pr.get("p_home_win"))
                                    p_away_live = float(pr.get("p_away_win"))
                                    # expose for UI/debug
                                    out["guidance"]["p_home_win"] = p_home_live
                                    out["guidance"]["p_away_win"] = p_away_live
                                    out["guidance"]["p_tie_reg"] = pr.get("p_tie")
                        except Exception:
                            pass

                        # ML signal: always emit WATCH when model is decisive;
                        # upgrade to BET only when we have odds to compute edge.
                        try:
                            ml_watch_side = None
                            ml_watch_p = None
                            if float(p_home_live) >= 0.62:
                                ml_watch_side = "HOME"
                                ml_watch_p = float(p_home_live)
                            elif float(p_away_live) >= 0.62:
                                ml_watch_side = "AWAY"
                                ml_watch_p = float(p_away_live)
                            if ml_watch_side and ml_watch_p is not None:
                                fair = _prob_to_american(float(ml_watch_p))
                                max_price = _prob_to_american(max(0.01, min(0.99, float(ml_watch_p) - 0.03)))
                                price = ml_home_price if ml_watch_side == "HOME" else ml_away_price
                                implied = implied_home if ml_watch_side == "HOME" else implied_away
                                edge = (float(ml_watch_p) - float(implied)) if implied is not None else None
                                action = "WATCH"
                                if edge is not None and float(edge) >= 0.03:
                                    if float(em) >= 8.0 and float(edge) >= 0.045:
                                        action = "BET"
                                    if float(em) >= 4.0 and float(edge) >= 0.06:
                                        action = "BET"
                                signals.append(_signal(
                                    action,
                                    scope="game",
                                    market="ML",
                                    label=f"ML {ml_watch_side}",
                                    side=ml_watch_side,
                                    price=_to_int(price),
                                    p_model=float(ml_watch_p),
                                    implied=float(implied) if implied is not None else None,
                                    edge=float(edge) if edge is not None else None,
                                    fair_price_american=fair,
                                    target_max_price_american=max_price,
                                    elapsed_min=float(em),
                                ))
                        except Exception:
                            pass

                        # Puck line cover probability via normal approx on goal differential
                        try:
                            if model_total is not None and model_spread is not None and gd is not None and rm is not None:
                                mt = float(model_total)
                                ms = float(model_spread)
                                # Convert total/spread to team means
                                mu_home_full = max(0.0, (mt + ms) / 2.0)
                                mu_away_full = max(0.0, (mt - ms) / 2.0)
                                mu_home_rem = mu_home_full * (float(rm) / 60.0) * float(pace_mult) * float(goalie_mult)
                                mu_away_rem = mu_away_full * (float(rm) / 60.0) * float(pace_mult) * float(goalie_mult)

                                # add mild PP bonus
                                mu_home_rem += float(pp_bonus_home) * (float(rm) / 60.0) + float(en_bonus_home)
                                mu_away_rem += float(pp_bonus_away) * (float(rm) / 60.0) + float(en_bonus_away)

                                # Prefer exact discrete cover probs when available
                                try:
                                    pr = _win_cover_probs(int(gd), float(mu_home_rem), float(mu_away_rem))
                                    p_home_m15 = pr.get("p_home_m15")
                                    p_away_p15 = pr.get("p_away_p15")
                                except Exception:
                                    p_home_m15 = None
                                    p_away_p15 = None

                                mu_d = float(mu_home_rem - mu_away_rem)
                                var_d = float(max(1e-6, mu_home_rem + mu_away_rem))
                                sd_d = math.sqrt(var_d)

                                # If exact discrete probs missing, fallback to normal approx.
                                if p_home_m15 is None or p_away_p15 is None:
                                    # P(home -1.5): final diff >= 2
                                    thr_home = 2.0 - float(gd)
                                    z_home = (thr_home - float(mu_d)) / float(sd_d)
                                    p_home_m15 = float(max(0.0, min(1.0, 1.0 - _norm_cdf(z_home))))
                                    # P(away +1.5): final diff <= 1
                                    thr_away = 1.0 - float(gd)
                                    z_away = (thr_away - float(mu_d)) / float(sd_d)
                                    p_away_p15 = float(max(0.0, min(1.0, _norm_cdf(z_away))))

                                # Puck line: emit WATCH when model is decisive; BET requires odds edge.
                                pl_watch_side = None
                                pl_watch_p = None
                                if float(p_away_p15) >= 0.62:
                                    pl_watch_side = "AWAY_+1.5"
                                    pl_watch_p = float(p_away_p15)
                                elif float(p_home_m15) >= 0.62:
                                    pl_watch_side = "HOME_-1.5"
                                    pl_watch_p = float(p_home_m15)

                                if pl_watch_side and pl_watch_p is not None:
                                    price = pl_away_p15 if pl_watch_side == "AWAY_+1.5" else pl_home_m15
                                    implied = implied_pl_away if pl_watch_side == "AWAY_+1.5" else implied_pl_home
                                    edge = (float(pl_watch_p) - float(implied)) if implied is not None else None
                                    action = "WATCH"
                                    if edge is not None and float(edge) >= 0.03:
                                        if float(em) >= 8.0 and float(edge) >= 0.05:
                                            action = "BET"
                                        if float(em) >= 4.0 and float(edge) >= 0.065:
                                            action = "BET"
                                    fair = _prob_to_american(float(pl_watch_p))
                                    max_price = _prob_to_american(max(0.01, min(0.99, float(pl_watch_p) - 0.03)))
                                    signals.append(_signal(
                                        action,
                                        scope="game",
                                        market="PUCKLINE",
                                        label=f"PL {pl_watch_side}",
                                        side=pl_watch_side,
                                        price=_to_int(price),
                                        p_model=float(pl_watch_p),
                                        implied=float(implied) if implied is not None else None,
                                        edge=float(edge) if edge is not None else None,
                                        fair_price_american=fair,
                                        target_max_price_american=max_price,
                                        elapsed_min=float(em),
                                    ))
                        except Exception:
                            pass
            except Exception:
                pass

            # 2) Player shots signals (pace + TOI)
            try:
                em = _to_float(elapsed_min)
                rm = _to_float(remaining_min)
                if em is not None and rm is not None and em >= 5.0 and rm > 0.0:
                    # If we have priced OddsAPI SOG props, prefer those over model-only heuristics.
                    players = None
                    try:
                        pg = props_odds_map.get(key)
                        offered = (pg or {}).get("props") if isinstance(pg, dict) else None
                        has_priced_sog = bool(isinstance(offered, dict) and isinstance(offered.get("SOG"), list) and offered.get("SOG"))
                    except Exception:
                        has_priced_sog = False

                    if not has_priced_sog:
                        if isinstance(lens, dict):
                            pl = lens.get("players")
                            if isinstance(pl, dict):
                                players = pl

                    def _player_shots_signals(arr: list, team_abbr: str):
                        out_sigs: list[dict] = []
                        for r in arr or []:
                            if not isinstance(r, dict):
                                continue
                            name = str(r.get("name") or "").strip()
                            if not name:
                                continue
                            shots = _to_int(r.get("s"))
                            if shots is None:
                                continue
                            toi_min = _parse_toi_to_min(r.get("toi"))
                            if toi_min is None or toi_min < 4.0:
                                continue
                            # Shots per minute of TOI (usage-adjusted)
                            rate = float(shots) / max(1e-6, float(toi_min))
                            if not math.isfinite(rate):
                                continue
                            # Project remaining TOI based on share so far; clamp to plausible.
                            share = float(toi_min) / max(1e-6, float(em))
                            share = max(0.05, min(0.90, share))
                            rem_toi = float(share) * float(rm)
                            rem_toi = max(0.0, min(float(rm), rem_toi))
                            mu_add = float(rate) * float(rem_toi)
                            mu_add = max(0.0, min(10.0, mu_add))

                            # Candidate common shot lines
                            best = None
                            for line in (2.5, 3.5, 4.5, 5.5, 1.5):
                                req_total = int(math.floor(float(line)) + 1)
                                need = req_total - int(shots)
                                p = 1.0 if need <= 0 else _poisson_sf(int(need), float(mu_add))
                                if not math.isfinite(p):
                                    continue
                                # Prefer higher lines when multiple are strong.
                                score = (float(p) - 0.5) * 10.0 + (float(line) * 0.1)
                                cand = {
                                    "line": float(line),
                                    "p_over": float(max(0.0, min(1.0, p))),
                                    "need": int(need),
                                    "mu_add": float(mu_add),
                                    "score": float(score),
                                }
                                if (best is None) or (cand["score"] > best["score"]):
                                    best = cand

                            if not best:
                                continue

                            p_over = float(best["p_over"])
                            # Only emit meaningful signals
                            if p_over < 0.62:
                                continue
                            action = "WATCH"  # no live OddsAPI price for this heuristic
                            fair = _prob_to_american(p_over)
                            max_price = _prob_to_american(max(0.01, min(0.99, p_over - 0.03)))

                            out_sigs.append(_signal(
                                action,
                                scope="prop",
                                market="PLAYER_SHOTS",
                                label=f"{name} shots OVER {best['line']:g}",
                                player=name,
                                team=team_abbr,
                                line=float(best["line"]),
                                p_model=float(p_over),
                                fair_price_american=fair,
                                target_max_price_american=max_price,
                                note="model_only_no_odds",
                                shots=int(shots),
                                toi_min=float(toi_min),
                                elapsed_min=float(em),
                            ))

                        # Return top few by action then p
                        def _rank(s: dict) -> tuple:
                            a = 0 if s.get("action") == "BET" else 1
                            p = _to_float(s.get("p_model")) or 0.0
                            return (a, -float(p))

                        out_sigs.sort(key=_rank)
                        return out_sigs[:4]

                    if isinstance(players, dict):
                        away_abbr = str(g.get("away") or "Away")
                        home_abbr = str(g.get("home") or "Home")
                        if isinstance(players.get("away"), list):
                            signals.extend(_player_shots_signals(players.get("away"), away_abbr))
                        if isinstance(players.get("home"), list):
                            signals.extend(_player_shots_signals(players.get("home"), home_abbr))
            except Exception:
                pass

            # 2b) Player props signals using OddsAPI lines/prices + pregame lambdas + live counts
            try:
                em = _to_float(elapsed_min)
                rm = _to_float(remaining_min)
                if em is not None and rm is not None and rm > 0.0 and em >= 3.0:
                    # Build live stat lookup (name -> stats) for this game
                    players_live = {}
                    goalies_live = {}
                    try:
                        if isinstance(lens, dict):
                            pl = lens.get("players")
                            if isinstance(pl, dict):
                                for side in ("away", "home"):
                                    arr = pl.get(side)
                                    if isinstance(arr, list):
                                        for r in arr:
                                            if not isinstance(r, dict):
                                                continue
                                            nm = _norm_person(r.get("name"))
                                            if nm:
                                                players_live[nm] = r
                            gl = lens.get("goalies")
                            if isinstance(gl, dict):
                                for side in ("away", "home"):
                                    arr = gl.get(side)
                                    if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                                        nm = _norm_person(arr[0].get("name"))
                                        if nm:
                                            goalies_live[nm] = arr[0]
                    except Exception:
                        players_live = {}
                        goalies_live = {}

                    # Pull offered prop lines for this event
                    pg = props_odds_map.get(key)
                    offered = (pg or {}).get("props") if isinstance(pg, dict) else None
                    if isinstance(offered, dict) and offered:
                        def _get_count(market: str, nm: str) -> Optional[int]:
                            try:
                                if market == "SOG":
                                    return _to_int((players_live.get(nm) or {}).get("s"))
                                if market == "GOALS":
                                    return _to_int((players_live.get(nm) or {}).get("g"))
                                if market == "ASSISTS":
                                    return _to_int((players_live.get(nm) or {}).get("a"))
                                if market == "POINTS":
                                    return _to_int((players_live.get(nm) or {}).get("p"))
                                if market == "BLOCKS":
                                    return _to_int((players_live.get(nm) or {}).get("blk"))
                                if market == "SAVES":
                                    return _to_int((goalies_live.get(nm) or {}).get("saves"))
                                return None
                            except Exception:
                                return None

                        def _usage_mult(nm: str) -> float:
                            try:
                                r = players_live.get(nm)
                                if not isinstance(r, dict):
                                    return 1.0
                                toi_min = _parse_toi_to_min(r.get("toi"))
                                if toi_min is None or toi_min <= 0.0 or em <= 0.0:
                                    return 1.0
                                share = float(toi_min) / float(em)
                                # Scale relative to a typical top-line share ~0.33, clamp hard
                                m = share / 0.33
                                return float(max(0.75, min(1.25, m)))
                            except Exception:
                                return 1.0

                        def _line_probs(cur: int, line_f: float, mu_rem: float) -> tuple[Optional[float], Optional[float], Optional[float]]:
                            """Return (p_over, p_under, p_push) for a two-way over/under line."""
                            try:
                                if mu_rem < 0:
                                    mu_rem = 0.0
                                is_int_line = abs(float(line_f) - round(float(line_f))) < 1e-9
                                if is_int_line:
                                    line_i = int(round(float(line_f)))
                                    over_min_total = line_i + 1
                                    under_max_total = line_i - 1
                                    push_total = line_i
                                else:
                                    over_min_total = int((float(line_f) // 1) + 1)
                                    under_max_total = int(float(line_f) // 1)
                                    push_total = None

                                need_over = int(over_min_total) - int(cur)
                                need_under = int(under_max_total) - int(cur)
                                p_over = 1.0 if need_over <= 0 else _poisson_sf(int(need_over), float(mu_rem))
                                p_under = 0.0 if need_under < 0 else _poisson_cdf(int(need_under), float(mu_rem))
                                p_push = None
                                if push_total is not None:
                                    k_push = int(push_total) - int(cur)
                                    if k_push < 0:
                                        p_push = 0.0
                                    else:
                                        p_push = max(0.0, _poisson_cdf(k_push, float(mu_rem)) - _poisson_cdf(k_push - 1, float(mu_rem)))
                                return (
                                    float(max(0.0, min(1.0, p_over))) if p_over is not None else None,
                                    float(max(0.0, min(1.0, p_under))) if p_under is not None else None,
                                    float(max(0.0, min(1.0, p_push))) if p_push is not None else None,
                                )
                            except Exception:
                                return (None, None, None)

                        prop_sigs: list[dict] = []
                        team_away = str(g.get("away") or "").strip().upper()
                        team_home = str(g.get("home") or "").strip().upper()

                        for mk0, rows in offered.items():
                            mk = _canon_prop_market(mk0)
                            if mk not in {"SOG", "GOALS", "ASSISTS", "POINTS"}:
                                continue
                            if not isinstance(rows, list):
                                continue
                            for rec in rows[:300]:
                                if not isinstance(rec, dict):
                                    continue
                                player = rec.get("player")
                                nm = _norm_person(player)
                                if not nm:
                                    continue
                                try:
                                    line_f = float(rec.get("line"))
                                except Exception:
                                    continue
                                over_price = rec.get("over")
                                under_price = rec.get("under")

                                cur = _get_count(mk, nm)
                                if cur is None:
                                    continue

                                # Lambda lookup (prefer team-disambiguated)
                                lam = None
                                if (mk, nm, team_away) in proj_lam_team:
                                    lam = proj_lam_team.get((mk, nm, team_away))
                                elif (mk, nm, team_home) in proj_lam_team:
                                    lam = proj_lam_team.get((mk, nm, team_home))
                                else:
                                    lam = proj_lam.get((mk, nm))
                                if lam is None:
                                    continue

                                # Convert full-game lambda to remaining-time mean (with mild multipliers)
                                mu_rem = float(lam) * (float(rm) / 60.0) * float(pace_mult) * float(goalie_mult) * float(_usage_mult(nm))
                                mu_rem = max(0.0, min(20.0, mu_rem))
                                p_over, p_under, p_push = _line_probs(int(cur), float(line_f), float(mu_rem))

                                implied_over = _american_to_implied_prob(over_price)
                                implied_under = _american_to_implied_prob(under_price)
                                edge_over = (float(p_over) - float(implied_over)) if (p_over is not None and implied_over is not None) else None
                                edge_under = (float(p_under) - float(implied_under)) if (p_under is not None and implied_under is not None) else None

                                # pick best side
                                side = None
                                p_model = None
                                implied = None
                                edge = None
                                price = None
                                if edge_over is not None and edge_under is not None:
                                    if float(edge_over) >= float(edge_under):
                                        side = "OVER"; p_model = p_over; implied = implied_over; edge = edge_over; price = over_price
                                    else:
                                        side = "UNDER"; p_model = p_under; implied = implied_under; edge = edge_under; price = under_price
                                elif edge_over is not None:
                                    side = "OVER"; p_model = p_over; implied = implied_over; edge = edge_over; price = over_price
                                elif edge_under is not None:
                                    side = "UNDER"; p_model = p_under; implied = implied_under; edge = edge_under; price = under_price

                                # If we don't have odds, skip (OddsAPI is source of truth)
                                if side is None or p_model is None or implied is None or edge is None:
                                    continue

                                # Gate actions
                                action = "WATCH"
                                if float(em) >= 10.0 and float(edge) >= 0.04:
                                    action = "BET"
                                if float(em) >= 6.0 and float(edge) >= 0.06:
                                    action = "BET"

                                fair = _prob_to_american(float(p_model))
                                max_price = _prob_to_american(max(0.01, min(0.99, float(p_model) - 0.03)))
                                prop_sigs.append(_signal(
                                    action,
                                    scope="prop",
                                    market=f"PROP_{mk}",
                                    label=f"{rec.get('player') or player} {mk} {side} {line_f:g}",
                                    player=rec.get("player") or player,
                                    line=float(line_f),
                                    side=side,
                                    price=_to_int(price),
                                    p_model=float(p_model),
                                    implied=float(implied),
                                    edge=float(edge),
                                    fair_price_american=fair,
                                    target_max_price_american=max_price,
                                    elapsed_min=float(em),
                                ))

                        # Keep a manageable number, prefer BET then highest edge
                        def _rank(s: dict) -> tuple:
                            a = 0 if str(s.get("action") or "").upper() == "BET" else 1
                            e = _to_float(s.get("edge")) or 0.0
                            return (a, -float(e))

                        prop_sigs.sort(key=_rank)
                        signals.extend(prop_sigs[:10])
            except Exception:
                pass

            # 3) Goalie saves signals (shots-against pace)
            try:
                em = _to_float(elapsed_min)
                rm = _to_float(remaining_min)
                if em is not None and rm is not None and em >= 5.0 and rm > 0.0:
                    goalies = None
                    if isinstance(lens, dict):
                        gl = lens.get("goalies")
                        if isinstance(gl, dict):
                            goalies = gl

                    def _goalie_saves_signals(arr: list, team_abbr: str):
                        out_sigs: list[dict] = []
                        if not isinstance(arr, list) or not arr:
                            return out_sigs
                        for r in arr[:1]:
                            if not isinstance(r, dict):
                                continue
                            name = str(r.get("name") or "").strip() or "Goalie"
                            saves = _to_int(r.get("saves"))
                            sa = _to_int(r.get("shots_against"))
                            if saves is None or sa is None or sa < 1:
                                continue
                            sa_rate = float(sa) / max(1e-6, float(em))
                            sa_rem = max(0.0, min(70.0, float(sa_rate) * float(rm)))
                            sv = _to_float(r.get("sv_pct"))
                            if sv is None:
                                sv = 0.91
                            sv = max(0.88, min(0.94, float(sv)))
                            mu_add = float(sa_rem) * float(sv)
                            mu_add = max(0.0, min(50.0, mu_add))

                            best = None
                            for line in (20.5, 22.5, 24.5, 26.5, 28.5, 30.5, 32.5, 34.5):
                                req_total = int(math.floor(float(line)) + 1)
                                need = req_total - int(saves)
                                p = 1.0 if need <= 0 else _poisson_sf(int(need), float(mu_add))
                                if not math.isfinite(p):
                                    continue
                                # Prefer lines near plausible final
                                score = (float(p) - 0.5) * 10.0 - abs((float(saves) + float(mu_add)) - float(line)) * 0.05
                                cand = {
                                    "line": float(line),
                                    "p_over": float(max(0.0, min(1.0, p))),
                                    "mu_add": float(mu_add),
                                    "score": float(score),
                                }
                                if (best is None) or (cand["score"] > best["score"]):
                                    best = cand

                            if not best:
                                continue
                            p_over = float(best["p_over"])
                            if p_over < 0.62:
                                continue
                            action = "WATCH"  # no live OddsAPI price for this heuristic
                            fair = _prob_to_american(p_over)
                            max_price = _prob_to_american(max(0.01, min(0.99, p_over - 0.03)))
                            out_sigs.append(_signal(
                                action,
                                scope="prop",
                                market="GOALIE_SAVES",
                                label=f"{name} saves OVER {best['line']:g}",
                                goalie=name,
                                team=team_abbr,
                                line=float(best["line"]),
                                p_model=float(p_over),
                                fair_price_american=fair,
                                target_max_price_american=max_price,
                                note="model_only_no_odds",
                                saves=int(saves),
                                shots_against=int(sa),
                                elapsed_min=float(em),
                            ))
                        return out_sigs[:1]

                    if isinstance(goalies, dict):
                        away_abbr = str(g.get("away") or "Away")
                        home_abbr = str(g.get("home") or "Home")
                        if isinstance(goalies.get("away"), list):
                            signals.extend(_goalie_saves_signals(goalies.get("away"), away_abbr))
                        if isinstance(goalies.get("home"), list):
                            signals.extend(_goalie_saves_signals(goalies.get("home"), home_abbr))
            except Exception:
                pass

            # Sort signals: BET first, then by p_model
            try:
                def _sig_rank(s: dict) -> tuple:
                    a = 0 if str(s.get("action") or "").upper() == "BET" else 1
                    p = _to_float(s.get("p_model")) or 0.0
                    return (a, -float(p))

                signals.sort(key=_sig_rank)
                signals = signals[:20]
            except Exception:
                pass

            out["signals"] = signals
            out_games.append(out)

        payload = {
            "ok": True,
            "date": d,
            "asof_utc": asof,
            "regions": str(regions or "us"),
            "best": bool(best),
            "odds_asof_utc": (odds_obj or {}).get("asof_utc") if isinstance(odds_obj, dict) else None,
            "games": out_games,
        }
        payload = _strict_json_sanitize(payload)

        # Optional: persist a compact snapshot on disk (Render disk-friendly).
        # This is best-effort and should never fail the endpoint.
        try:
            snap_dir_s = (os.getenv("NHL_LIVE_LENS_DIR") or os.getenv("LIVE_LENS_DIR") or "").strip()
            if snap_dir_s:
                try:
                    min_sec = float(os.getenv("LIVE_LENS_SNAPSHOT_MIN_SECONDS", "60") or "60")
                except Exception:
                    min_sec = 60.0

                snap_dir = Path(snap_dir_s)
                snap_dir.mkdir(parents=True, exist_ok=True)

                # Append compact JSONL for later ROI/audit tools.
                # Keep it intentionally small: no full lens ladders, just score/state + signals.
                def _compact_game(x: dict) -> dict:
                    try:
                        return {
                            "gamePk": x.get("gamePk"),
                            "key": x.get("key"),
                            "away": x.get("away"),
                            "home": x.get("home"),
                            "gameState": x.get("gameState"),
                            "period": x.get("period"),
                            "clock": x.get("clock"),
                            "score": x.get("score"),
                            "signals": x.get("signals") or [],
                        }
                    except Exception:
                        return {"signals": x.get("signals") or []}

                rec = {
                    "asof_utc": payload.get("asof_utc"),
                    "date": payload.get("date"),
                    "games": [_compact_game(g) for g in (payload.get("games") or [])],
                }

                # Only write when there is something to log.
                has_any_signal = False
                try:
                    for gg in rec.get("games") or []:
                        if gg.get("signals"):
                            has_any_signal = True
                            break
                except Exception:
                    has_any_signal = False

                # Throttle writes (per date) so polling doesn't explode disk usage.
                do_write = bool(has_any_signal)
                try:
                    now_ts = datetime.now(timezone.utc).timestamp()
                    m = getattr(app.state, "live_lens_last_snapshot_by_date", None)
                    if not isinstance(m, dict):
                        m = {}
                        setattr(app.state, "live_lens_last_snapshot_by_date", m)
                    last_ts = float(m.get(d) or 0.0)
                    if (now_ts - last_ts) < float(min_sec):
                        do_write = False
                    else:
                        m[d] = float(now_ts)
                except Exception:
                    # If throttling fails, still write (best-effort).
                    do_write = bool(has_any_signal)

                if not do_write:
                    raise RuntimeError("skip_live_lens_snapshot")

                out_jsonl = snap_dir / f"live_lens_signals_{d}.jsonl"
                with open(out_jsonl, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

                # Overwrite a "latest" pointer for quick manual inspection.
                latest_fp = snap_dir / f"live_lens_signals_{d}_latest.json"
                latest_fp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        return JSONResponse(payload)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


def _github_raw_read_csv(rel_path: str, timeout_sec: Optional[float] = None, attempts: Optional[int] = None) -> pd.DataFrame:
    """Fetch a CSV from the GitHub repo's raw content and return as DataFrame.

    rel_path should be a posix-style path like 'data/processed/props_projections_YYYY-MM-DD.csv'.
    Uses env GITHUB_REPO and GITHUB_BRANCH (defaults to mostgood1/NHL-Betting@master).

    Timeouts and retries are aggressively reduced on public hosts to avoid 502s.
    """
    try:
        repo = os.getenv("GITHUB_REPO", "mostgood1/NHL-Betting").strip() or "mostgood1/NHL-Betting"
        branch = os.getenv("GITHUB_BRANCH", "master").strip() or "master"
        # Normalize leading slashes
        rel = rel_path.lstrip("/")
        # URL-encode path components (especially for date=YYYY-MM-DD patterns)
        from urllib.parse import quote
        rel_encoded = "/".join(quote(part, safe='') for part in rel.split("/"))
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{rel_encoded}"
        # Tune network behavior to avoid tying up request workers
        if timeout_sec is None:
            timeout_sec = 2.0 if _is_public_host_env() else 7.0
        if attempts is None:
            attempts = 1 if _is_public_host_env() else 2
        last_exc = None
        for _ in range(max(1, int(attempts))):
            try:
                resp = requests.get(url, timeout=float(timeout_sec))
                if resp.status_code == 200 and resp.text:
                    try:
                        return pd.read_csv(StringIO(resp.text))
                    except Exception:
                        return pd.DataFrame()
            except Exception as e:
                last_exc = e
            # brief backoff
            try:
                import time as _t
                _t.sleep(0.2 if _is_public_host_env() else 0.4)
            except Exception:
                pass
        # On failure return empty (callers handle empty as cache miss)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _github_raw_read_parquet(rel_path: str, timeout_sec: Optional[float] = None) -> pd.DataFrame:
    """Fetch a Parquet file from the GitHub repo's raw content and return as DataFrame.

    rel_path should be a posix-style path like 'data/props/player_props_lines/date=YYYY-MM-DD/oddsapi.parquet'.
    Uses env GITHUB_REPO and GITHUB_BRANCH (defaults to mostgood1/NHL-Betting@master).
    """
    try:
        repo = os.getenv("GITHUB_REPO", "mostgood1/NHL-Betting").strip() or "mostgood1/NHL-Betting"
        branch = os.getenv("GITHUB_BRANCH", "master").strip() or "master"
        rel = rel_path.lstrip("/")
        # URL-encode path components (especially for date=YYYY-MM-DD patterns)
        from urllib.parse import quote
        rel_encoded = "/".join(quote(part, safe='') for part in rel.split("/"))
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{rel_encoded}"
        if timeout_sec is None:
            timeout_sec = 3.0 if _is_public_host_env() else 15.0
        resp = requests.get(url, timeout=float(timeout_sec))
        if resp.status_code == 200 and resp.content:
            try:
                import io as _io
                return pd.read_parquet(_io.BytesIO(resp.content))
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _gh_lookback_days(default_public: int = 2, default_local: int = 7) -> int:
    """Determine how many days back to search GitHub raw for props artifacts.

    Public hosts default to a very small lookback to avoid long serial network loops.
    Can be overridden by PROPS_GH_LOOKBACK_DAYS env var.
    """
    try:
        v = os.getenv('PROPS_GH_LOOKBACK_DAYS')
        if v is not None and str(v).strip().isdigit():
            return max(0, int(str(v).strip()))
    except Exception:
        pass
    return int(default_public if _is_public_host_env() else default_local)


# -----------------------------------------------------------------------------
# Sim artifacts loader helpers (local-first with GitHub raw fallback)
# -----------------------------------------------------------------------------
def _load_sim_df(date: str, filenames: list[str]) -> pd.DataFrame:
    """Attempt to load a sim artifact for a given date.

    Tries local `data/processed/<filename>` candidates first, then GitHub raw when
    running on public hosts. Returns an empty DataFrame if nothing is found.
    """
    try:
        # Local-first
        for fname in filenames:
            p = PROC_DIR / fname.format(date=date)
            if p.exists():
                try:
                    return pd.read_csv(p)
                except Exception:
                    pass
        # Public host fallback to GitHub raw
        if _is_public_host_env():
            for fname in filenames:
                rel = f"data/processed/{fname.format(date=date)}"
                try:
                    df = _github_raw_read_csv(rel)
                except Exception:
                    df = pd.DataFrame()
                if df is not None and not df.empty:
                    return df
    except Exception:
        pass
    return pd.DataFrame()

def _norm_str(s: Optional[str]) -> str:
    try:
        x = str(s or "").strip(); return " ".join(x.split())
    except Exception:
        return str(s or "")

def _abbr_or_none(team_name: Optional[str]) -> Optional[str]:
    try:
        a = get_team_assets(str(team_name)) or {}
        ab = str(a.get("abbr") or "").upper().strip()
        return ab or None
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Sim engine data APIs (source-of-truth for period/game/player aggregates)
# -----------------------------------------------------------------------------
@app.get("/api/sim/summary")
async def api_sim_summary(date: Optional[str] = Query(None)):
    """Report presence and counts for core sim artifacts for a date."""
    d = date or _today_ymd()
    games = _load_sim_df(d, ["sim_games_{date}.csv", "sim_games_pos_{date}.csv"])  # allow pos variant
    events = _load_sim_df(d, ["sim_events_pos_{date}.csv"])  # possession/events aggregate
    # Prefer props-aggregated boxscores; fall back to per-game player boxscores
    boxscores = _load_sim_df(d, ["props_boxscores_sim_{date}.csv", "sim_boxscores_pos_{date}.csv"])

    # Best-effort n_sims inference for UI display.
    # 1) If histogram exists, it carries explicit n_sims.
    # 2) Else infer from samples sim_idx max+1 (parquet preferred, csv fallback).
    def _infer_n_sims(date_str: str) -> Optional[int]:
        try:
            hist = PROC_DIR / f"props_boxscores_sim_hist_{date_str}.csv"
            if hist.exists() and getattr(hist.stat(), "st_size", 0) > 0:
                try:
                    hdf = pd.read_csv(hist, usecols=["n_sims"])
                    if hdf is not None and (not hdf.empty) and "n_sims" in hdf.columns:
                        v = hdf["n_sims"].dropna()
                        if len(v):
                            return int(v.iloc[0])
                except Exception:
                    pass
            parq = PROC_DIR / f"props_boxscores_sim_samples_{date_str}.parquet"
            csvp = PROC_DIR / f"props_boxscores_sim_samples_{date_str}.csv"
            if parq.exists() and getattr(parq.stat(), "st_size", 0) > 0:
                try:
                    sdf = pd.read_parquet(parq, columns=["sim_idx"])  # type: ignore
                    if sdf is not None and (not sdf.empty) and "sim_idx" in sdf.columns:
                        m = sdf["sim_idx"].max()
                        if pd.notna(m):
                            return int(m) + 1
                except Exception:
                    pass
            if csvp.exists() and getattr(csvp.stat(), "st_size", 0) > 0:
                try:
                    sdf = pd.read_csv(csvp, usecols=["sim_idx"])
                    if sdf is not None and (not sdf.empty) and "sim_idx" in sdf.columns:
                        m = sdf["sim_idx"].max()
                        if pd.notna(m):
                            return int(m) + 1
                except Exception:
                    pass
        except Exception:
            return None
        return None

    n_sims = _infer_n_sims(d)
    def _summary(df: pd.DataFrame) -> dict:
        try:
            return {"exists": (df is not None and not df.empty), "count": int(len(df))}
        except Exception:
            return {"exists": False, "count": 0}
    return JSONResponse({
        "ok": True,
        "date": d,
        "n_sims": n_sims,
        "games": _summary(games),
        "events": _summary(events),
        "boxscores": _summary(boxscores),
    })

@app.get("/api/sim/games")
async def api_sim_games(
    date: Optional[str] = Query(None),
    game: Optional[str] = Query(None, description="Filter by AWY@HOME abbreviations"),
    home: Optional[str] = Query(None),
    away: Optional[str] = Query(None),
):
    """Return simulated game-level aggregates for a date (home/away teams, totals, odds)."""
    d = date or _today_ymd()
    df = _load_sim_df(d, ["sim_games_{date}.csv", "sim_games_pos_{date}.csv"])  # allow pos variant
    if df is None or df.empty:
        return JSONResponse({"ok": True, "date": d, "rows": [], "count": 0})
    # Normalize filters
    h_ab = str(home or "").upper().strip() or None
    a_ab = str(away or "").upper().strip() or None
    g_tok = str(game or "").upper().strip()
    if g_tok and "@" in g_tok:
        try:
            a_tok, h_tok = [t.strip() for t in g_tok.split("@", 1)]
            h_ab = h_ab or (h_tok if h_tok else None)
            a_ab = a_ab or (a_tok if a_tok else None)
        except Exception:
            pass
    # Attempt to harmonize team fields to abbreviations for robust filtering
    for col in ("home", "away"):
        try:
            df[col + "_abbr"] = df[col].map(_abbr_or_none)
        except Exception:
            pass
    try:
        if h_ab:
            df = df[df.get("home_abbr").astype(str).str.upper() == h_ab]
        if a_ab:
            df = df[df.get("away_abbr").astype(str).str.upper() == a_ab]
    except Exception:
        pass
    try:
        df = df.sort_values(by=["home_abbr", "away_abbr"]).reset_index(drop=True)
    except Exception:
        pass
    return JSONResponse({"ok": True, "date": d, "count": int(len(df)), "rows": df.to_dict(orient="records")})

@app.get("/api/sim/events")
async def api_sim_events(
    date: Optional[str] = Query(None),
    game: Optional[str] = Query(None),
    home: Optional[str] = Query(None),
    away: Optional[str] = Query(None),
):
    """Return simulated possession/events aggregates per game (EV/PP/PK shots, penalties, goals)."""
    d = date or _today_ymd()
    df = _load_sim_df(d, ["sim_events_pos_{date}.csv"])  # possession/events aggregate
    if df is None or df.empty:
        return JSONResponse({"ok": True, "date": d, "rows": [], "count": 0})
    h_ab = str(home or "").upper().strip() or None
    a_ab = str(away or "").upper().strip() or None
    g_tok = str(game or "").upper().strip()
    if g_tok and "@" in g_tok:
        try:
            a_tok, h_tok = [t.strip() for t in g_tok.split("@", 1)]
            h_ab = h_ab or (h_tok if h_tok else None)
            a_ab = a_ab or (a_tok if a_tok else None)
        except Exception:
            pass
    for col in ("home", "away"):
        try:
            df[col + "_abbr"] = df[col].map(_abbr_or_none)
        except Exception:
            pass
    try:
        if h_ab:
            df = df[df.get("home_abbr").astype(str).str.upper() == h_ab]
        if a_ab:
            df = df[df.get("away_abbr").astype(str).str.upper() == a_ab]
    except Exception:
        pass
    try:
        df = df.sort_values(by=["home_abbr", "away_abbr"]).reset_index(drop=True)
    except Exception:
        pass
    return JSONResponse({"ok": True, "date": d, "count": int(len(df)), "rows": df.to_dict(orient="records")})

@app.get("/api/sim/boxscores")
async def api_sim_boxscores(
    date: Optional[str] = Query(None),
    game: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    player: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    period: Optional[int] = Query(None),
    dressed: Optional[bool] = Query(None, description="When present, filter is_dressed==1/0 if column exists"),
    top: Optional[int] = Query(None),
):
    """Return simulated player boxscores for a date.

    Prefers `props_boxscores_sim_{date}.csv` for aggregated props-facing outputs,
    falling back to `sim_boxscores_pos_{date}.csv` if needed.
    """
    d = date or _today_ymd()
    df = _load_sim_df(d, ["props_boxscores_sim_{date}.csv", "sim_boxscores_pos_{date}.csv"])
    if df is None or df.empty:
        return JSONResponse({"ok": True, "date": d, "rows": [], "count": 0})

    # Best-effort: enrich with player position from roster snapshot when missing.
    # Sim artifacts often omit position; we can join on player_id using processed roster snapshots.
    try:
        has_pos = ("pos" in df.columns) or ("position" in df.columns) or ("player_position" in df.columns)
        if (not has_pos) and ("player_id" in df.columns):
            snap = PROC_DIR / f"roster_snapshot_{d}.csv"
            if snap.exists() and getattr(snap.stat(), "st_size", 0) > 0:
                rdf = pd.read_csv(snap)
                if rdf is not None and (not rdf.empty) and {"player_id", "position"}.issubset(rdf.columns):
                    try:
                        m = dict(zip(rdf["player_id"].astype(int), rdf["position"].astype(str)))
                        df["pos"] = df["player_id"].astype(int).map(m)
                    except Exception:
                        pass
    except Exception:
        pass
    # Optional filters
    try:
        if market:
            df = df[df.get("market").astype(str).str.upper() == str(market).upper()]
    except Exception:
        pass
    try:
        if period is not None and "period" in df.columns:
            df = df[df.get("period") == int(period)]
    except Exception:
        pass
    try:
        if dressed is not None and "is_dressed" in df.columns:
            df = df[df.get("is_dressed") == (1 if bool(dressed) else 0)]
    except Exception:
        pass
    try:
        if team:
            df = df[df.get("team").astype(str).str.upper() == str(team).upper()]
    except Exception:
        pass
    try:
        if player:
            q = _norm_str(player).lower()
            df = df[df.get("player").astype(str).str.lower().str.contains(q)]
    except Exception:
        pass
    # Game filter via AWY@HOME using mapped abbreviations on events/games if present
    g_tok = str(game or "").upper().strip()
    if g_tok and "@" in g_tok:
        try:
            a_tok, h_tok = [t.strip() for t in g_tok.split("@", 1)]
        except Exception:
            a_tok = None; h_tok = None
        # If game columns exist, use them; else try to infer via team abbr columns
        try:
            if {"home","away"}.issubset(df.columns):
                df["home_abbr"] = df["home"].map(_abbr_or_none)
                df["away_abbr"] = df["away"].map(_abbr_or_none)
                if h_tok:
                    df = df[df.get("home_abbr").astype(str).str.upper() == h_tok]
                if a_tok:
                    df = df[df.get("away_abbr").astype(str).str.upper() == a_tok]
            else:
                # Fallback: filter by team abbreviation presence
                if h_tok:
                    df = df[df.get("team").astype(str).str.upper() == h_tok]
        except Exception:
            pass
    # Sort for consistency and truncate
    try:
        sort_cols = [c for c in ["team", "market", "player"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(by=sort_cols).reset_index(drop=True)
    except Exception:
        pass
    if top is not None and top > 0:
        df = df.head(int(top))
    return JSONResponse({"ok": True, "date": d, "count": int(len(df)), "rows": df.to_dict(orient="records")})


def _compute_props_projections(date: str, market: Optional[str] = None) -> pd.DataFrame:
    """Build player props projections using canonical lines, preferring sim-derived lambdas.

    Source-of-truth for `proj_lambda` is props_projections_all_{date}.csv when present
    (derived from play-level sim boxscores). Falls back to historical models otherwise.

    Returns DataFrame with columns:
    [market, player, team, line, over_price, under_price, proj_lambda, p_over, ev_over, book]
    Sorted by ev_over desc then p_over desc.
    """
    try:
        base = PROC_DIR.parent / "props" / f"player_props_lines/date={date}"
        parts = []
        # Prefer parquet, but fall back to CSV if parquet is unavailable
        for name in ("oddsapi.parquet", "bovada.parquet"):
            p = base / name
            if p.exists():
                try:
                    parts.append(pd.read_parquet(p))
                except Exception:
                    pass
        if not parts:
            for name in ("oddsapi.csv",):
                p = base / name
                if p.exists():
                    try:
                        parts.append(pd.read_csv(p))
                    except Exception:
                        pass
        lines = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    except Exception:
        lines = pd.DataFrame()
    if lines is None or lines.empty:
        return pd.DataFrame()
    if market:
        try:
            lines = lines[lines["market"].astype(str).str.upper() == str(market).upper()]
        except Exception:
            pass
    # Prefer sim-derived projections for lambda
    try:
        sim_proj_path = PROC_DIR / f"props_projections_all_{date}.csv"
        sim_proj = pd.read_csv(sim_proj_path) if sim_proj_path.exists() else pd.DataFrame()
    except Exception:
        sim_proj = pd.DataFrame()
    # Build mapping from (norm_player, market) -> lambda, optionally disambiguated by team
    def _norm_name(s: str) -> str:
        try:
            x = str(s or "").strip(); return " ".join(x.split()).lower()
        except Exception:
            return str(s or "").lower()
    sim_map: dict[tuple[str, str], float] = {}
    sim_map_team: dict[tuple[str, str, str], float] = {}
    if sim_proj is not None and not sim_proj.empty and {'player','market','proj_lambda'}.issubset(sim_proj.columns):
        try:
            for _, r in sim_proj.iterrows():
                nm = _norm_name(r.get('player'))
                mk = str(r.get('market') or '').upper()
                lam = r.get('proj_lambda')
                try:
                    lam = float(lam) if lam is not None else None
                except Exception:
                    lam = None
                if lam is None:
                    continue
                sim_map[(nm, mk)] = lam
                team = str(r.get('team') or '').strip().upper() if 'team' in sim_proj.columns else None
                if team:
                    sim_map_team[(nm, mk, team)] = lam
        except Exception:
            sim_map = {}
            sim_map_team = {}
    # Historical per-player stats for fallback lambda estimation
    try:
        stats_path = RAW_DIR / "player_game_stats.csv"
        hist = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    except Exception:
        hist = pd.DataFrame()
    # Harmonize historical names to full names using roster snapshots to improve matching
    try:
        if hist is not None and not hist.empty and 'player' in hist.columns:
            import re, ast
            # Build last-name -> list of full names from roster snapshots
            roster = _rosters.build_all_team_roster_snapshots()
            last_to_full = {}
            if roster is not None and not roster.empty and 'full_name' in roster.columns:
                for nm in roster['full_name'].dropna().astype(str).unique().tolist():
                    parts = nm.strip().split(' ')
                    if len(parts) >= 2:
                        last = parts[-1].lower()
                        last_to_full.setdefault(last, set()).add(nm)
            def _extract_default(s: str):
                if isinstance(s, str) and s.strip().startswith('{'):
                    try:
                        d = ast.literal_eval(s)
                        if isinstance(d, dict):
                            v = d.get('default') or d.get('name') or ''
                            if isinstance(v, str):
                                return v
                    except Exception:
                        return s
                return s
            def _fix(n: str) -> str:
                n = _extract_default(n)
                m = re.match(r"^([A-Za-z])[\.]?\s+([A-Za-z\-']+)$", str(n).strip())
                if m:
                    ini = m.group(1).lower(); last = m.group(2).lower()
                    cands = list(last_to_full.get(last, []))
                    if len(cands) == 1:
                        return cands[0]
                    for c in cands:
                        first = c.split(' ')[0]
                        if first and first[0].lower() == ini:
                            return c
                return str(n)
            hist['player'] = hist['player'].astype(str).map(_fix)
    except Exception:
        pass
    from ..models.props import (
        PropsConfig,
        SkaterShotsModel, GoalieSavesModel, SkaterGoalsModel,
        SkaterAssistsModel, SkaterPointsModel, SkaterBlocksModel,
    )
    from ..props.utils import compute_props_lam_scale_mean
    from ..data.nhl_api_web import NHLWebClient as _Web
    from ..web.teams import get_team_assets as _assets
    shots = SkaterShotsModel(); saves = GoalieSavesModel(); goals = SkaterGoalsModel()
    assists = SkaterAssistsModel(); points = SkaterPointsModel(); blocks = SkaterBlocksModel()

    # Optional player-level guardrails: clamp sim-derived lambdas toward recent/season baselines.
    # This is intentionally conservative and configurable via env vars.
    try:
        _guard_on = str(os.getenv("PROPS_LAMBDA_GUARDRAILS", "1")).strip().lower() in {"1", "true", "yes"}
    except Exception:
        _guard_on = True
    try:
        _guard_min_ratio = float(os.getenv("PROPS_LAMBDA_GUARD_MIN_RATIO", "0.65"))
    except Exception:
        _guard_min_ratio = 0.65
    try:
        _guard_max_ratio = float(os.getenv("PROPS_LAMBDA_GUARD_MAX_RATIO", "1.60"))
    except Exception:
        _guard_max_ratio = 1.60
    try:
        _guard_w_recent = float(os.getenv("PROPS_LAMBDA_GUARD_W_RECENT", "0.55"))
        _guard_w_season = float(os.getenv("PROPS_LAMBDA_GUARD_W_SEASON", "0.30"))
        _guard_w_career = float(os.getenv("PROPS_LAMBDA_GUARD_W_CAREER", "0.15"))
    except Exception:
        _guard_w_recent, _guard_w_season, _guard_w_career = 0.55, 0.30, 0.15
    _w_sum = max(1e-9, (_guard_w_recent + _guard_w_season + _guard_w_career))
    _guard_w_recent /= _w_sum
    _guard_w_season /= _w_sum
    _guard_w_career /= _w_sum

    # Baseline model variants (reuse the same implementations, with different windows/weights)
    _cfg_recent = PropsConfig(
        window=int(os.getenv("PROPS_LAMBDA_GUARD_RECENT_WINDOW", "10")),
        recency_alpha=float(os.getenv("PROPS_LAMBDA_GUARD_RECENT_ALPHA", "0.3")),
    )
    _cfg_season = PropsConfig(
        window=int(os.getenv("PROPS_LAMBDA_GUARD_SEASON_WINDOW", "82")),
        recency_alpha=0.0,
    )
    _cfg_career = PropsConfig(
        window=int(os.getenv("PROPS_LAMBDA_GUARD_CAREER_WINDOW", "164")),
        recency_alpha=0.0,
    )

    _shots_recent = SkaterShotsModel(_cfg_recent)
    _shots_season = SkaterShotsModel(_cfg_season)
    _shots_career = SkaterShotsModel(_cfg_career)
    _goals_recent = SkaterGoalsModel(_cfg_recent)
    _goals_season = SkaterGoalsModel(_cfg_season)
    _goals_career = SkaterGoalsModel(_cfg_career)
    _assists_recent = SkaterAssistsModel(_cfg_recent)
    _assists_season = SkaterAssistsModel(_cfg_season)
    _assists_career = SkaterAssistsModel(_cfg_career)
    _points_recent = SkaterPointsModel(_cfg_recent)
    _points_season = SkaterPointsModel(_cfg_season)
    _points_career = SkaterPointsModel(_cfg_career)
    _blocks_recent = SkaterBlocksModel(_cfg_recent)
    _blocks_season = SkaterBlocksModel(_cfg_season)
    _blocks_career = SkaterBlocksModel(_cfg_career)
    _saves_recent = GoalieSavesModel(_cfg_recent)
    _saves_season = GoalieSavesModel(_cfg_season)
    _saves_career = GoalieSavesModel(_cfg_career)

    # Season-filtered historical frame (best-effort). Used only for baseline computation.
    _hist_season = None
    try:
        if hist is not None and not hist.empty and 'date' in hist.columns:
            _dt = pd.to_datetime(hist.get('date'), errors='coerce', utc=True)
            try:
                _dt_et = _dt.dt.tz_convert('America/New_York')
            except Exception:
                _dt_et = _dt
            _d0 = pd.to_datetime(str(date), errors='coerce')
            if _d0 is not None and pd.notna(_d0):
                y = int(_d0.year)
                m = int(_d0.month)
                season_start_year = y if m >= 7 else (y - 1)
                season_start = pd.Timestamp(year=season_start_year, month=7, day=1)
                season_end = pd.Timestamp(_d0.date())
                _mask = (_dt_et.dt.tz_localize(None) >= season_start) & (_dt_et.dt.tz_localize(None) <= season_end)
                _hist_season = hist.loc[_mask].copy()
    except Exception:
        _hist_season = None

    _baseline_cache: dict[tuple[str, str], Optional[float]] = {}

    def _baseline_lambda(market_key: str, player_name: str) -> Optional[float]:
        if not _guard_on:
            return None
        mk = str(market_key or '').upper().strip()
        nm = _norm_name(player_name)
        key = (mk, nm)
        if key in _baseline_cache:
            return _baseline_cache[key]
        try:
            # Recent baseline uses the full hist frame (it already tails a window)
            if mk == 'SOG':
                r = float(_shots_recent.player_lambda(hist, player_name))
                s = float(_shots_season.player_lambda(_hist_season if _hist_season is not None else hist, player_name))
                c = float(_shots_career.player_lambda(hist, player_name))
            elif mk == 'GOALS':
                r = float(_goals_recent.player_lambda(hist, player_name))
                s = float(_goals_season.player_lambda(_hist_season if _hist_season is not None else hist, player_name))
                c = float(_goals_career.player_lambda(hist, player_name))
            elif mk == 'ASSISTS':
                r = float(_assists_recent.player_lambda(hist, player_name))
                s = float(_assists_season.player_lambda(_hist_season if _hist_season is not None else hist, player_name))
                c = float(_assists_career.player_lambda(hist, player_name))
            elif mk == 'POINTS':
                r = float(_points_recent.player_lambda(hist, player_name))
                s = float(_points_season.player_lambda(_hist_season if _hist_season is not None else hist, player_name))
                c = float(_points_career.player_lambda(hist, player_name))
            elif mk == 'BLOCKS':
                r = float(_blocks_recent.player_lambda(hist, player_name))
                s = float(_blocks_season.player_lambda(_hist_season if _hist_season is not None else hist, player_name))
                c = float(_blocks_career.player_lambda(hist, player_name))
            elif mk == 'SAVES':
                r = float(_saves_recent.player_lambda(hist, player_name))
                s = float(_saves_season.player_lambda(_hist_season if _hist_season is not None else hist, player_name))
                c = float(_saves_career.player_lambda(hist, player_name))
            else:
                _baseline_cache[key] = None
                return None
            base = (_guard_w_recent * r) + (_guard_w_season * s) + (_guard_w_career * c)
            if not (base is not None and np.isfinite(base) and base > 0):
                base = None
            _baseline_cache[key] = base
            return base
        except Exception:
            _baseline_cache[key] = None
            return None

    def _apply_guardrails(market_key: str, player_name: str, lam_value: Optional[float]) -> Optional[float]:
        if not _guard_on:
            return lam_value
        try:
            if lam_value is None:
                return None
            lam_f = float(lam_value)
            if not np.isfinite(lam_f) or lam_f <= 0:
                return lam_value
            base = _baseline_lambda(market_key, player_name)
            if base is None:
                return lam_value
            lo = float(base) * float(_guard_min_ratio)
            hi = float(base) * float(_guard_max_ratio)
            if hi < lo:
                lo, hi = hi, lo
            return float(min(max(lam_f, lo), hi))
        except Exception:
            return lam_value
    # Load team features used for scaling
    import numpy as _np, json
    xg_path = PROC_DIR / "team_xg_latest.csv"
    xg_map = {}
    if xg_path.exists():
        try:
            _xg = pd.read_csv(xg_path)
            if not _xg.empty and {"abbr","xgf60"}.issubset(_xg.columns):
                xg_map = {str(r.abbr).upper(): float(r.xgf60) for _, r in _xg.iterrows()}
        except Exception:
            xg_map = {}
    league_xg = float(_np.mean(list(xg_map.values()))) if xg_map else 2.6
    pen_path = PROC_DIR / "team_penalty_rates.json"
    pen_comm = {}
    if pen_path.exists():
        try:
            pen_comm = json.loads(pen_path.read_text(encoding="utf-8"))
        except Exception:
            pen_comm = {}
    league_pen = float(_np.mean([float(v.get("committed_per60", 0.0)) for v in pen_comm.values()])) if pen_comm else 3.0
    # Goalie form (sv% L10)
    from datetime import date as _date
    gf_today = PROC_DIR / f"goalie_form_{_date.today().strftime('%Y-%m-%d')}.csv"
    gf_map = {}
    if gf_today.exists():
        try:
            _gf = pd.read_csv(gf_today)
            if not _gf.empty and {"team","sv_pct_l10"}.issubset(_gf.columns):
                gf_map = {str(r.team).upper(): float(r.sv_pct_l10) for _, r in _gf.iterrows()}
        except Exception:
            gf_map = {}
    league_sv = float(_np.mean(list(gf_map.values()))) if gf_map else 0.905
    # Possession events-derived PP fractions (optional)
    opp_pp_frac_map: dict[str, float] = {}
    team_pp_frac_map: dict[str, float] = {}
    league_pp_frac = 0.18
    try:
        ev_path = PROC_DIR / f"sim_events_pos_{date}.csv"
        if ev_path.exists():
            ev = pd.read_csv(ev_path)
            def _abbr(n: str | None) -> str | None:
                try:
                    a = _assets(str(n)) or {}
                    return str(a.get("abbr") or "").upper() or None
                except Exception:
                    return None
            for _, r in ev.iterrows():
                h = str(r.get("home") or ""); a = str(r.get("away") or "")
                h_ab = _abbr(h); a_ab = _abbr(a)
                if not h_ab or not a_ab:
                    continue
                sh_home_total = float(r.get("shots_ev_home", 0)) + float(r.get("shots_pp_home", 0)) + float(r.get("shots_pk_home", 0))
                sh_home_pp = float(r.get("shots_pp_home", 0))
                team_pp_home = (sh_home_pp / sh_home_total) if sh_home_total > 0 else None
                if team_pp_home is not None:
                    team_pp_frac_map[h_ab] = float(team_pp_home)
                sh_away_total = float(r.get("shots_ev_away", 0)) + float(r.get("shots_pp_away", 0)) + float(r.get("shots_pk_away", 0))
                sh_away_pp = float(r.get("shots_pp_away", 0))
                team_pp_away = (sh_away_pp / sh_away_total) if sh_away_total > 0 else None
                if team_pp_away is not None:
                    team_pp_frac_map[a_ab] = float(team_pp_away)
                opp_pp_frac_home = (sh_away_pp / sh_away_total) if sh_away_total > 0 else None
                if opp_pp_frac_home is not None:
                    opp_pp_frac_map[h_ab] = float(opp_pp_frac_home)
                opp_pp_frac_away = (sh_home_pp / sh_home_total) if sh_home_total > 0 else None
                if opp_pp_frac_away is not None:
                    opp_pp_frac_map[a_ab] = float(opp_pp_frac_away)
            vals = [v for v in team_pp_frac_map.values() if v is not None]
            if vals:
                league_pp_frac = float(_np.mean(vals))
    except Exception:
        opp_pp_frac_map = {}
        team_pp_frac_map = {}
    # Opponent mapping via schedule
    abbr_to_opp: dict[str, str] = {}
    try:
        web = _Web(); sched = web.schedule_day(date)
        games=[]
        for name in ("oddsapi.parquet", "bovada.parquet"):
            h=str(getattr(g, "home", "")); a=str(getattr(g, "away", ""))
            ha = (_assets(h) or {}).get("abbr"); aa = (_assets(a) or {}).get("abbr")
            ha = str(ha or "").upper(); aa = str(aa or "").upper()
            if ha and aa:
                abbr_to_opp[ha]=aa; abbr_to_opp[aa]=ha
    except Exception:
        abbr_to_opp = {}
    def proj_prob(m, player, ln, team_abbr: str | None, opp_abbr: str | None):
        m = (m or '').upper()
        # Normalize name for sim-map lookup
        p_norm = _norm_name(player)
        # Try team-specific sim lambda first, then name-only sim lambda, else model fallback
        if m == 'SOG':
            lam = sim_map_team.get((p_norm, m, str(team_abbr or ''))) or sim_map.get((p_norm, m))
            if lam is None:
                lam = shots.player_lambda(hist, player)
            lam = _apply_guardrails(m, player, lam)
            scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
            lam_eff = (lam or 0.0) * float(scale)
            return lam, shots.prob_over(lam_eff, ln), lam_eff
        if m == 'SAVES':
            lam = sim_map_team.get((p_norm, m, str(team_abbr or ''))) or sim_map.get((p_norm, m))
            if lam is None:
                lam = saves.player_lambda(hist, player)
            lam = _apply_guardrails(m, player, lam)
            scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
            lam_eff = (lam or 0.0) * float(scale)
            return lam, saves.prob_over(lam_eff, ln), lam_eff
        if m == 'GOALS':
            lam = sim_map_team.get((p_norm, m, str(team_abbr or ''))) or sim_map.get((p_norm, m))
            if lam is None:
                lam = goals.player_lambda(hist, player)
            lam = _apply_guardrails(m, player, lam)
            scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
            lam_eff = (lam or 0.0) * float(scale)
            return lam, goals.prob_over(lam_eff, ln), lam_eff
        if m == 'ASSISTS':
            lam = sim_map_team.get((p_norm, m, str(team_abbr or ''))) or sim_map.get((p_norm, m))
            if lam is None:
                lam = assists.player_lambda(hist, player)
            lam = _apply_guardrails(m, player, lam)
            scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
            lam_eff = (lam or 0.0) * float(scale)
            return lam, assists.prob_over(lam_eff, ln), lam_eff
        if m == 'POINTS':
            lam = sim_map_team.get((p_norm, m, str(team_abbr or ''))) or sim_map.get((p_norm, m))
            if lam is None:
                lam = points.player_lambda(hist, player)
            lam = _apply_guardrails(m, player, lam)
            scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
            lam_eff = (lam or 0.0) * float(scale)
            return lam, points.prob_over(lam_eff, ln), lam_eff
        if m == 'BLOCKS':
            lam = sim_map_team.get((p_norm, m, str(team_abbr or ''))) or sim_map.get((p_norm, m))
            if lam is None:
                lam = blocks.player_lambda(hist, player)
            lam = _apply_guardrails(m, player, lam)
            scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
            lam_eff = (lam or 0.0) * float(scale)
            return lam, blocks.prob_over(lam_eff, ln), lam_eff
        return None, None
    def _dec(a):
        try:
            a = float(a); return 1.0 + (a/100.0) if a > 0 else 1.0 + (100.0/abs(a))
        except Exception:
            return None
    out = []
    for _, r in lines.iterrows():
        player = r.get('player_name') or r.get('player')
        if not player:
            continue
        m = str(r.get('market') or '').upper()
        try:
            ln = float(r.get('line'))
        except Exception:
            ln = None
        lam, p_over, lam_eff = (None, None, None)
        if (ln is not None):
            team_abbr = (str(r.get('team') or '').strip().upper() or None)
            opp_abbr = abbr_to_opp.get(team_abbr) if team_abbr else None
            lam, p_over, lam_eff = proj_prob(m, str(player), ln, team_abbr, opp_abbr)
        over_price = r.get('over_price') if pd.notna(r.get('over_price')) else None
        ev_over = None
        if (p_over is not None) and (over_price is not None):
            dec = _dec(over_price)
            if dec is not None:
                ev_over = float(p_over) * (dec - 1.0) - (1.0 - float(p_over))
        out.append({
            'market': m,
            'player': player,
            'team': r.get('team') or None,
            'line': ln,
            'over_price': over_price,
            'under_price': r.get('under_price') if pd.notna(r.get('under_price')) else None,
            'proj_lambda': float(lam) if lam is not None else None,
            'proj_lambda_eff': float(lam_eff) if lam_eff is not None else None,
            'p_over': float(p_over) if p_over is not None else None,
            'ev_over': float(ev_over) if ev_over is not None else None,
            'book': r.get('book'),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        try:
            if 'ev_over' in df.columns and df['ev_over'].notna().any():
                df = df.sort_values('ev_over', ascending=False)
            elif 'p_over' in df.columns and df['p_over'].notna().any():
                df = df.sort_values('p_over', ascending=False)
        except Exception:
            pass
    return df


def _v1_props_odds_payload(date_ymd: str, regions: str = "us", best: bool = True, inplay: bool = True) -> dict:
    """Build a lightweight player-props odds payload (OddsAPI) with caching.

    Intended usage: live overlay for the cards-only UI (Live Lens).

    Notes
    - OddsAPI rejects unknown market keys with 422. We restrict to core keys that
      our existing pipeline already uses successfully.
    - Returns a normalized structure keyed by away@home and canonical markets.
    """
    d = str(date_ymd or "").strip()
    if not _V1_DATE_RE.fullmatch(d):
        return {"ok": False, "error": "invalid_date", "date": date_ymd}

    cache_key = f"v1_props_odds::{d}::{str(regions or 'us').lower()}::{int(bool(best))}::{int(bool(inplay))}"
    cached = _live_odds_cache_get(cache_key)
    if cached is not None:
        return cached

    asof = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    # Core markets that we already use in data pipeline
    markets = [
        "player_points",
        "player_assists",
        "player_goals",
        "player_shots_on_goal",
    ]

    # Map OddsAPI market keys to canonical market labels used elsewhere in this repo
    m_map = {
        "player_points": "POINTS",
        "player_assists": "ASSISTS",
        "player_goals": "GOALS",
        "player_shots_on_goal": "SOG",
        # alternate key variants sometimes appear
        "player_points_alternate": "POINTS",
        "player_assists_alternate": "ASSISTS",
        "player_goals_alternate": "GOALS",
        "player_shots_on_goal_alternate": "SOG",
    }

    def _norm_person(s: object) -> str:
        try:
            x = str(s or "").strip().lower()
            x = re.sub(r"[\.'’]", "", x)
            x = " ".join(x.split())
            return x
        except Exception:
            return str(s or "").strip().lower()

    def _best_price(existing: Optional[tuple], price: object, book: str) -> Optional[tuple]:
        """Pick the best price for bettor (max decimal odds)."""
        try:
            if price is None:
                return existing
            p = int(price)
            from ..utils.odds import american_to_decimal

            dec = float(american_to_decimal(float(p)))
            if existing is None:
                return (p, str(book or ""), dec)
            cur_dec = float(existing[2])
            if dec > cur_dec:
                return (p, str(book or ""), dec)
            return existing
        except Exception:
            return existing

    # Build event list for the ET slate day (extended window like our pipeline)
    def _utc_window_for_et_date(d_ymd: str) -> tuple[Optional[str], Optional[str]]:
        try:
            tz_et = ZoneInfo("America/New_York")
            d0 = datetime.strptime(str(d_ymd), "%Y-%m-%d").replace(tzinfo=tz_et)
            # Expanded window to catch late west coast games on the ET slate
            utc_start = d0.astimezone(timezone.utc) - timedelta(hours=5)
            utc_end = utc_start + timedelta(hours=33)
            start = utc_start.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            end = utc_end.replace(microsecond=0).isoformat().replace("+00:00", "Z")
            return start, end
        except Exception:
            return None, None

    games_out: list[dict] = []
    try:
        client = OddsAPIClient(rate_limit_per_sec=10.0)
    except Exception as e:
        obj = {"ok": True, "date": d, "asof_utc": asof, "games": [], "note": str(e)}
        _live_odds_cache_put(cache_key, obj)
        return obj

    sport_keys = ["icehockey_nhl", "icehockey_nhl_preseason"]
    start_iso, end_iso = _utc_window_for_et_date(d)

    for sk in sport_keys:
        try:
            # For player props we always use per-event current odds.
            evs, _ = client.list_events(sk, commence_from_iso=start_iso, commence_to_iso=end_iso)
            if not isinstance(evs, list) or not evs:
                continue

            # per-game accumulator: (key, market, player, line) -> best over/under
            per_game: dict[str, dict] = {}

            for ev in evs:
                try:
                    eid = str((ev or {}).get("id") or "").strip()
                    if not eid:
                        continue
                    eo, _ = client.event_odds(
                        sport=sk,
                        event_id=eid,
                        markets=",".join(markets),
                        regions=str(regions or "us"),
                        bookmakers=None,  # allow all; we'll aggregate best
                        odds_format="american",
                        date_format="iso",
                    )
                    if not isinstance(eo, dict) or not eo.get("bookmakers"):
                        continue
                    home = eo.get("home_team")
                    away = eo.get("away_team")
                    if not home or not away:
                        continue
                    gkey = _norm_game_key(away, home)
                    if not gkey:
                        continue

                    if gkey not in per_game:
                        per_game[gkey] = {
                            "date": d,
                            "home": str(home),
                            "away": str(away),
                            "key": gkey,
                            "props": {},  # canonical market -> player -> line -> sides
                        }

                    for bk in eo.get("bookmakers") or []:
                        bkey = str((bk or {}).get("key") or "").strip() or "oddsapi"
                        for m in (bk or {}).get("markets") or []:
                            mk = str((m or {}).get("key") or "").strip()
                            canon = m_map.get(mk)
                            if not canon:
                                continue
                            outs = (m or {}).get("outcomes") or []
                            for oc in outs:
                                if not isinstance(oc, dict):
                                    continue
                                side = str((oc.get("name") or "")).strip().upper()
                                if side not in {"OVER", "UNDER"}:
                                    continue
                                player = (
                                    oc.get("description")
                                    or oc.get("participant")
                                    or oc.get("player_name")
                                    or oc.get("player")
                                    or ""
                                )
                                player = str(player or "").strip()
                                if not player:
                                    continue
                                try:
                                    line = float(oc.get("point")) if oc.get("point") is not None else None
                                except Exception:
                                    line = None
                                if line is None:
                                    continue
                                price = oc.get("price")
                                try:
                                    price_i = int(price) if price is not None else None
                                except Exception:
                                    price_i = None
                                if price_i is None:
                                    continue

                                nm = _norm_person(player)
                                game_obj = per_game[gkey]
                                props = game_obj["props"]
                                if canon not in props:
                                    props[canon] = {}
                                if nm not in props[canon]:
                                    props[canon][nm] = {}
                                line_key = float(line)
                                if line_key not in props[canon][nm]:
                                    props[canon][nm][line_key] = {
                                        "player": player,
                                        "line": float(line_key),
                                        "over": None,
                                        "under": None,
                                        "over_book": None,
                                        "under_book": None,
                                    }
                                rec = props[canon][nm][line_key]
                                # Keep best-of-all (by decimal odds)
                                if side == "OVER":
                                    cur = (rec.get("over"), rec.get("over_book"), -1.0) if rec.get("over") is not None else None
                                    chosen = _best_price(cur, price_i, bkey)
                                    if chosen is not None:
                                        rec["over"] = int(chosen[0])
                                        rec["over_book"] = str(chosen[1])
                                else:
                                    cur = (rec.get("under"), rec.get("under_book"), -1.0) if rec.get("under") is not None else None
                                    chosen = _best_price(cur, price_i, bkey)
                                    if chosen is not None:
                                        rec["under"] = int(chosen[0])
                                        rec["under_book"] = str(chosen[1])
                except Exception:
                    continue

            # Flatten per_game into output list (limit size)
            for gkey, obj in per_game.items():
                try:
                    # Convert nested dicts to lists for JSON payload
                    props_out: dict[str, list[dict]] = {}
                    for canon, by_player in (obj.get("props") or {}).items():
                        rows: list[dict] = []
                        for _, by_line in (by_player or {}).items():
                            for _, rec in (by_line or {}).items():
                                if not isinstance(rec, dict):
                                    continue
                                # Require at least one side
                                if rec.get("over") is None and rec.get("under") is None:
                                    continue
                                rows.append(rec)
                        # Keep top N rows per market to avoid huge payloads
                        rows = rows[:250]
                        props_out[str(canon)] = rows
                    out_obj = dict(obj)
                    out_obj["props"] = props_out
                    games_out.append(out_obj)
                except Exception:
                    continue

            break  # got something for this sport key
        except Exception:
            continue

    obj = {"ok": True, "date": d, "asof_utc": asof, "markets": markets, "games": games_out}
    obj = _strict_json_sanitize(obj)
    _live_odds_cache_put(cache_key, obj)
    return obj
    


def _today_ymd() -> str:
    """Return today's date in US/Eastern to align the slate with 'tonight'."""
    try:
        et = ZoneInfo("America/New_York")
        return datetime.now(et).strftime("%Y-%m-%d")
    except Exception:
        # If the IANA timezone database isn't available (common in minimal Linux images),
        # prefer a fixed-offset Eastern fallback over UTC to avoid date rollovers during
        # evening ET (which breaks live slate lookups).
        try:
            et_fixed = timezone(timedelta(hours=-5))
            return datetime.now(et_fixed).strftime("%Y-%m-%d")
        except Exception:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _git_commit_hash() -> Optional[str]:
    """Return short git commit hash if repository metadata is present.

    In some deployment environments (Docker image without .git) this will return None.
    """
    try:
        root = Path(__file__).resolve().parents[2]
        head_file = root / '.git' / 'HEAD'
        if not head_file.exists():
            return None
        head_content = head_file.read_text().strip()
        if head_content.startswith('ref:'):
            ref_path = head_content.split(' ', 1)[1].strip()
            ref_file = root / '.git' / ref_path
            if ref_file.exists():
                return ref_file.read_text().strip()[:12]
        return head_content[:12]
    except Exception:
        return None


def _normalize_date_param(d: Optional[str]) -> str:
    """Normalize 'today'/'yesterday' to ET YYYY-MM-DD; pass-through other values."""
    if not d:
        return _today_ymd()
    s = str(d).strip().lower()
    try:
        et = ZoneInfo("America/New_York")
    except Exception:
        et = timezone(timedelta(hours=-5))
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


def _looks_like_synthetic_props(df: pd.DataFrame) -> bool:
    """Heuristic: detect our tiny test/synthetic props frames (Test Player A/B/C/D).

    Returns True if the frame appears to be the 4-row synthetic sample or similar.
    We check for any of:
      - player values starting with "Test Player"
      - team values among {AAA, BBB, CCC, DDD} with a very small row count
      - market set limited to {Shots, Goals, Assists, Points} with tiny row count
    """
    try:
        if df is None or df.empty:
            return False
        # Locate columns case-insensitively
        def _col(name: str):
            ln = name.lower()
            for c in df.columns:
                if str(c).lower() == ln:
                    return c
            return None
        pc = _col('player'); tc = _col('team'); mc = _col('market')
        n = len(df)
        if pc is not None:
            try:
                if any(str(x).startswith('Test Player') for x in df[pc].astype(str).head(min(20, n))):
                    return True
            except Exception:
                pass
        if tc is not None and n <= 20:
            try:
                teams = {str(x).upper() for x in df[tc].dropna().astype(str).unique().tolist()}
                if teams.issubset({'AAA','BBB','CCC','DDD'}) and len(teams) > 0:
                    return True
            except Exception:
                pass
        if mc is not None and n <= 12:
            try:
                mk = {str(x).strip().upper() for x in df[mc].dropna().astype(str).unique().tolist()}
                # Allow common canonical names used in the synthetic rows
                if mk.issubset({'SHOTS','GOALS','ASSISTS','POINTS'}) and len(mk) > 0:
                    return True
            except Exception:
                pass
        return False
    except Exception:
        return False


def _artifact_info_for_date(d: str) -> dict:
    """Summarize key artifacts for a given ET date with exists/rows/mtime.

    Includes predictions, edges, props recommendations, props projections (per-player and ALL),
    and canonical props lines parquet presence by book.
    """
    info: dict[str, Any] = {"date": d}
    try:
        def _rows_csv(p: Path):
            try:
                if not p.exists():
                    return None
                df = _read_csv_fallback(p)
                return 0 if df is None or df.empty else int(len(df))
            except Exception:
                return None
        def _rows_parquet(p: Path):
            try:
                if not p.exists():
                    return None
                df = pd.read_parquet(p)
                return 0 if df is None or df.empty else int(len(df))
            except Exception:
                return None
        # Predictions / edges
        pred = PROC_DIR / f"predictions_{d}.csv"
        edges = PROC_DIR / f"edges_{d}.csv"
        info["predictions"] = {"exists": pred.exists(), "rows": _rows_csv(pred), "mtime": _file_mtime_iso(pred)}
        info["edges"] = {"exists": edges.exists(), "rows": _rows_csv(edges), "mtime": _file_mtime_iso(edges)}
        # Recommendations and props projections
        rec = PROC_DIR / f"props_recommendations_{d}.csv"
        proj = PROC_DIR / f"props_projections_{d}.csv"
        proj_all = PROC_DIR / f"props_projections_all_{d}.csv"
        info["props_recommendations"] = {"exists": rec.exists(), "rows": _rows_csv(rec), "mtime": _file_mtime_iso(rec)}
        info["props_projections"] = {"exists": proj.exists(), "rows": _rows_csv(proj), "mtime": _file_mtime_iso(proj)}
        info["props_projections_all"] = {"exists": proj_all.exists(), "rows": _rows_csv(proj_all), "mtime": _file_mtime_iso(proj_all)}
        # Canonical lines parquet by book
        lines_base = PROC_DIR.parent / "props" / f"player_props_lines/date={d}"
        books = {
            "oddsapi": lines_base / "oddsapi.parquet",
            "bovada": lines_base / "bovada.parquet",
        }
        info["props_lines"] = {
            "path": str(lines_base),
            "exists": lines_base.exists(),
            "books": {bk: {"exists": p.exists(), "rows": _rows_parquet(p), "mtime": _file_mtime_iso(p)} for bk, p in books.items()},
        }
    except Exception:
        pass
    return info


_ROSTER_CACHE = None
def _get_roster_snapshot():
    global _ROSTER_CACHE
    if _ROSTER_CACHE is not None:
        return _ROSTER_CACHE
    try:
        _ROSTER_CACHE = _rosters.build_all_team_roster_snapshots()
    except Exception:
        _ROSTER_CACHE = None
    return _ROSTER_CACHE

def _clean_player_display_name(name: str) -> str:
    """Normalize player name strings that may be dict-like (e.g., "{'default': 'A. Last'}").

    Additionally attempts to expand initials to full names using roster snapshots when available.
    """
    try:
        import ast, re
        s = name
        if isinstance(s, str) and s.strip().startswith('{'):
            try:
                d = ast.literal_eval(s)
                if isinstance(d, dict):
                    v = d.get('default') or d.get('name') or s
                    if isinstance(v, str):
                        s = v
            except Exception:
                pass
        # Fast modes: skip disambiguation altogether
        if os.getenv('FAST_PROPS_TEST','0') == '1' or os.getenv('PROPS_FORCE_SYNTHETIC','0') == '1':
            return str(s)
        # Expand formats like "A. Last" when roster can disambiguate
        m = re.match(r"^([A-Za-z])[\.]?\s+([A-Za-z\-']+)$", str(s).strip())
        if m:
            ini = m.group(1).lower(); last = m.group(2).lower()
            roster = _get_roster_snapshot()
            if roster is not None and not roster.empty and 'full_name' in roster.columns:
                cands = [fn for fn in roster['full_name'].dropna().astype(str).unique().tolist() if fn.lower().endswith(' ' + last)]
                if len(cands) == 1:
                    return cands[0]
                for c in cands:
                    first = c.split(' ')[0]
                    if first and first[0].lower() == ini:
                        return c
        return str(s)
    except Exception:
        return str(name)


def _read_all_players_projections(date: str) -> pd.DataFrame:
    """Read data/processed/props_projections_all_{date}.csv locally or via GitHub raw.

    In fast synthetic modes we skip reading to force compute path's synthetic return.
    """
    # Never enable synthetic short-circuits on public hosts
    if (os.getenv('FAST_PROPS_TEST','0') == '1' or os.getenv('PROPS_FORCE_SYNTHETIC','0') == '1') and not _is_public_host_env():
        return None
    p = PROC_DIR / f"props_projections_all_{date}.csv"
    if p.exists():
        try:
            return _read_csv_fallback(p)
        except Exception:
            pass
    # GitHub fallback
    gh = _github_raw_read_csv(f"data/processed/props_projections_all_{date}.csv")
    try:
        if gh is not None and not gh.empty and _looks_like_synthetic_props(gh):
            # Treat synthetic placeholder as missing to trigger compute or deeper fallback
            return pd.DataFrame()
    except Exception:
        pass
    return gh


def _compute_all_players_projections(date: str) -> pd.DataFrame:
    """Compute model-only projections for all rostered players on the slate for the date.

    Mirrors CLI behavior; avoids external NHL Stats API when unavailable by using historical enrichment.
    """
    t_global_start = time.perf_counter()
    verbose = os.getenv('PROPS_VERBOSE','0') == '1'
    def _v(msg: str):
        if verbose:
            try:
                print(f"[props_compute][{date}] {msg}")
            except Exception:
                pass
    # Synthetic short circuit flags
    fast_flag = os.getenv('FAST_PROPS_TEST','0') == '1'
    force_synth = os.getenv('PROPS_FORCE_SYNTHETIC','0') == '1'
    no_compute = os.getenv('PROPS_NO_COMPUTE','0') == '1'
    # Never serve synthetic data when running on public hosts
    if (fast_flag or force_synth) and not _is_public_host_env():
        _v("FAST_PROPS_TEST or PROPS_FORCE_SYNTHETIC enabled -> returning synthetic frame")
        try:
            df_synth = pd.DataFrame([
                {"player":"Test Player A","team":"AAA","market":"Shots","proj_lambda":2.1},
                {"player":"Test Player B","team":"BBB","market":"Goals","proj_lambda":0.4},
                {"player":"Test Player C","team":"CCC","market":"Assists","proj_lambda":0.7},
                {"player":"Test Player D","team":"DDD","market":"Points","proj_lambda":1.2},
            ])
            return df_synth
        except Exception:
            return pd.DataFrame()
    if no_compute:
        _v("PROPS_NO_COMPUTE=1 set -> skipping compute and returning empty frame")
        return pd.DataFrame()
    _v("Beginning compute pipeline (may involve IO)")
    # Ensure stats history exists (best effort)
    try:
        from ..data.collect import collect_player_game_stats as _collect_stats
        start = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
        stats_path = RAW_DIR / "player_game_stats.csv"
        need = (not stats_path.exists()) or (stats_path.stat().st_size == 0)
        if need:
            try:
                _collect_stats(start, date, source="web")
            except Exception:
                _collect_stats(start, date, source="stats")
        try:
            hist = _read_csv_fallback(stats_path)
        except Exception:
            hist = pd.DataFrame()
    except Exception:
        hist = pd.DataFrame()
    # Slate teams via Web API
    try:
        web = NHLWebClient()
        games = web.schedule_day(date)
    except Exception:
        games = []
    slate_names = set()
    for g in games or []:
        slate_names.add(str(g.home))
        slate_names.add(str(g.away))
    slate_abbrs = set()
    for nm in slate_names:
        ab = (get_team_assets(str(nm)).get('abbr') or '').upper()
        if ab:
            slate_abbrs.add(ab)
    # Try live roster; fallback to historical enrichment
    roster_df = pd.DataFrame()
    try:
        from ..data.rosters import list_teams as _list_teams, fetch_current_roster as _fetch
        teams = _list_teams()
        name_to_id = { str(t.get('name') or '').strip().lower(): int(t.get('id')) for t in teams }
        id_to_abbr = { int(t.get('id')): str(t.get('abbreviation') or '').upper() for t in teams }
        rows = []
        for nm in sorted(slate_names):
            tid = name_to_id.get(str(nm).strip().lower())
            if not tid:
                continue
            try:
                players = _fetch(tid)
            except Exception:
                players = []
            for p in players:
                rows.append({ 'player_id': p.player_id, 'player': p.full_name, 'position': p.position, 'team': id_to_abbr.get(tid) })
        roster_df = pd.DataFrame(rows)
    except Exception:
        roster_df = pd.DataFrame()
    if roster_df is None or roster_df.empty:
        # Historical enrichment
        try:
            from ..data import player_props as _pp
            enrich = _pp._build_roster_enrichment()
        except Exception:
            enrich = pd.DataFrame()
        if enrich is None or enrich.empty:
            return pd.DataFrame(columns=["date","player","team","position","market","proj_lambda"])
        def _to_abbr(x):
            try:
                a = get_team_assets(str(x)).get('abbr')
                return str(a).upper() if a else None
            except Exception:
                return None
        enrich = enrich.copy()
        enrich['team_abbr'] = enrich['team'].map(_to_abbr)
        # Infer position from historical if available
        pos_map = {}
        try:
            if hist is not None and not hist.empty and {'player','primary_position'}.issubset(hist.columns):
                tmp = hist.dropna(subset=['player']).copy()
                tmp['player'] = tmp['player'].astype(str)
                last_pos = tmp.dropna(subset=['primary_position']).groupby('player')['primary_position'].last()
                pos_map = {k: v for k, v in last_pos.items() if isinstance(k, str)}
        except Exception:
            pos_map = {}
        rows = []
        for _, rr in enrich.iterrows():
            ab = rr.get('team_abbr')
            if slate_abbrs and (not ab or ab not in slate_abbrs):
                continue
            nm = rr.get('full_name')
            pos_raw = pos_map.get(str(nm), '')
            pos = 'G' if str(pos_raw).upper().startswith('G') else ('D' if str(pos_raw).upper().startswith('D') else 'F')
            rows.append({'player_id': rr.get('player_id'), 'player': nm, 'position': pos, 'team': ab})
        roster_df = pd.DataFrame(rows)
    if roster_df is None or roster_df.empty:
        return pd.DataFrame(columns=["date","player","team","position","market","proj_lambda"])
    # Models
    shots = _SkaterShotsModel(); saves = _GoalieSavesModel(); goals = _SkaterGoalsModel(); assists = _SkaterAssistsModel(); points = _SkaterPointsModel(); blocks = _SkaterBlocksModel()
    out_rows = []
    for _, r in roster_df.iterrows():
        player = _clean_player_display_name(str(r.get('player') or ''))
        pos = str(r.get('position') or '').upper()
        team = r.get('team')
        if not player:
            continue
        try:
            if pos == 'G':
                lam = saves.player_lambda(hist, player)
                out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'SAVES', 'proj_lambda': float(lam) if lam is not None else None})
            else:
                lam = shots.player_lambda(hist, player); out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'SOG', 'proj_lambda': float(lam) if lam is not None else None})
                lam = goals.player_lambda(hist, player); out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'GOALS', 'proj_lambda': float(lam) if lam is not None else None})
                lam = assists.player_lambda(hist, player); out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'ASSISTS', 'proj_lambda': float(lam) if lam is not None else None})
                lam = points.player_lambda(hist, player); out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'POINTS', 'proj_lambda': float(lam) if lam is not None else None})
                lam = blocks.player_lambda(hist, player); out_rows.append({'date': date, 'player': player, 'team': team, 'position': pos, 'market': 'BLOCKS', 'proj_lambda': float(lam) if lam is not None else None})
        except Exception:
            continue
    df = pd.DataFrame(out_rows)
    if not df.empty:
        try:
            df = df.sort_values(['team','position','player','market'])
        except Exception:
            pass
    return df


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
        # Prefer sim-native predictions when available
        pred_sim_p = PROC_DIR / f"predictions_sim_{date}.csv"
        pred_p = pred_sim_p if pred_sim_p.exists() else (PROC_DIR / f"predictions_{date}.csv")
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

    - Reads predictions_sim_{date}.csv if present, else predictions_{date}.csv
    - Ensures EV columns exist (if odds present). If missing, recompute from p_* and *_odds
    - Writes edges_{date}.csv (long format) and recommendations_{date}.csv (top-N style via API)
    """
    try:
        # Prefer sim-native predictions file
        pred_sim = PROC_DIR / f"predictions_sim_{date}.csv"
        pred_path = pred_sim if pred_sim.exists() else (PROC_DIR / f"predictions_{date}.csv")
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
                # market-specific default odds if still missing
                if price is None:
                    if odds_key in ("f10_yes_odds", "f10_no_odds"):
                        price = -150.0
                    elif odds_key in (
                        "over_odds", "under_odds",
                        "home_pl_-1.5_odds", "away_pl_+1.5_odds",
                        "p1_over_odds", "p1_under_odds",
                        "p2_over_odds", "p2_under_odds",
                        "p3_over_odds", "p3_under_odds",
                    ):
                        price = -110.0
                if (p is not None) and (price is not None):
                    dec = american_to_decimal(price)
                    if dec is not None and _math.isfinite(dec):
                        # For totals with integer lines, account for push probability in EV
                        p_push = 0.0
                        try:
                            if prob_key in ("p_over", "p_under"):
                                tl = row.get("total_line_used") if "total_line_used" in row else None
                                if tl is None:
                                    tl = row.get("close_total_line_used") if "close_total_line_used" in row else None
                                mt = row.get("model_total") if "model_total" in row else None
                                if tl is not None and mt is not None:
                                    tl_f = float(tl); mt_f = float(mt)
                                    # consider integer line within small epsilon
                                    if _math.isfinite(tl_f) and _math.isfinite(mt_f) and abs(tl_f - round(tl_f)) < 1e-9:
                                        k = int(round(tl_f))
                                        # Poisson PMF at k with mean mt_f
                                        # p(k) = e^-mu * mu^k / k!
                                        from math import exp, factorial
                                        p_push = float(exp(-mt_f) * (mt_f ** k) / factorial(k)) if k >= 0 else 0.0
                        except Exception:
                            p_push = 0.0
                        # Win prob is p; loss prob excludes push if applicable
                        p_loss = max(0.0, 1.0 - float(p) - float(p_push)) if prob_key in ("p_over", "p_under") else max(0.0, 1.0 - float(p))
                        row[ev_key] = round(float(p) * (dec - 1.0) - p_loss, 4)
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
                                # First 10 minutes Yes/No
                                ("p_f10_yes", "f10_yes_odds"): ("p_f10_no", "f10_no_odds"),
                                ("p_f10_no", "f10_no_odds"): ("p_f10_yes", "f10_yes_odds"),
                                # Period totals Over/Under pairs
                                ("p1_over_prob", "p1_over_odds"): ("p1_under_prob", "p1_under_odds"),
                                ("p1_under_prob", "p1_under_odds"): ("p1_over_prob", "p1_over_odds"),
                                ("p2_over_prob", "p2_over_odds"): ("p2_under_prob", "p2_under_odds"),
                                ("p2_under_prob", "p2_under_odds"): ("p2_over_prob", "p2_over_odds"),
                                ("p3_over_prob", "p3_over_odds"): ("p3_under_prob", "p3_under_odds"),
                                ("p3_under_prob", "p3_under_odds"): ("p3_over_prob", "p3_over_odds"),
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
        # Helpers for Poisson PMF/CDF for integer goals
        from math import exp, factorial, floor
        def _pois_pmf(mu: float, k: int) -> float:
            try:
                if k < 0 or mu < 0 or not _math.isfinite(mu):
                    return 0.0
                return float(exp(-mu) * (mu ** k) / factorial(k))
            except Exception:
                return 0.0
        def _pois_cdf(mu: float, k: int) -> float:
            try:
                if k < 0:
                    return 0.0
                s = 0.0
                for i in range(0, k + 1):
                    s += _pois_pmf(mu, i)
                return float(min(1.0, max(0.0, s)))
            except Exception:
                return 0.0

        # Apply to rows
        if not df.empty:
            for i, r in df.iterrows():
                # First 10 minutes Yes/No probabilities
                try:
                    # Prefer existing p_f10_yes if present (computed upstream by shared core)
                    if "p_f10_yes" in r and pd.notna(r.get("p_f10_yes")):
                        try:
                            py = float(r.get("p_f10_yes"))
                            if 0.0 <= py <= 1.0 and _math.isfinite(py):
                                r["p_f10_yes"] = py
                                r["p_f10_no"] = 1.0 - py
                                raise StopIteration  # skip further fallback computation
                        except Exception:
                            pass
                    # Team-driven estimate from period1 projections with calibrated factor
                    p_yes = None
                    try:
                        h1 = float(r.get("period1_home_proj")) if pd.notna(r.get("period1_home_proj")) else None
                        a1 = float(r.get("period1_away_proj")) if pd.notna(r.get("period1_away_proj")) else None
                    except Exception:
                        h1 = a1 = None
                    if h1 is not None and a1 is not None and (h1 >= 0 and a1 >= 0):
                        # Resolve factor: env > first10_eval.json (p1_scale) > model_calibration.json > default 0.55
                        def _clamp(v: float, lo: float = 0.35, hi: float = 0.7) -> float:
                            try:
                                return max(lo, min(hi, float(v)))
                            except Exception:
                                return v
                        f = None
                        # 1) Environment override
                        try:
                            ev = os.getenv("F10_EARLY_FACTOR")
                            if ev is not None:
                                f = float(ev)
                        except Exception:
                            f = None
                        # 2) Data-driven evaluation file
                        if f is None or (not _math.isfinite(f)) or f <= 0:
                            try:
                                import json as _json
                                _eval_path = PROC_DIR / "first10_eval.json"
                                if _eval_path.exists():
                                    _obj = _json.loads(_eval_path.read_text(encoding="utf-8"))
                                    _p1_scale = _obj.get("p1_scale")
                                    if _p1_scale is not None:
                                        f = float(_p1_scale)
                            except Exception:
                                f = None
                        # 3) Model calibration fallback
                        if f is None or (not _math.isfinite(f)) or f <= 0:
                            try:
                                import json as _json
                                _cal_path = PROC_DIR / "model_calibration.json"
                                if _cal_path.exists():
                                    _obj = _json.loads(_cal_path.read_text(encoding="utf-8"))
                                    _f = _obj.get("f10_early_factor")
                                    if _f is not None:
                                        f = float(_f)
                            except Exception:
                                f = None
                        # 4) Sensible default
                        if f is None or (not _math.isfinite(f)) or f <= 0:
                            f = 0.55
                        # Clamp to avoid extremes from overfitting
                        f = _clamp(f)
                        lam10 = f * (float(h1) + float(a1))
                        if _math.isfinite(lam10) and lam10 >= 0:
                            p_yes = 1.0 - exp(-lam10)
                    # Fallback to legacy single-prob/proj fields if needed
                    if p_yes is None:
                        if "first_10min_prob" in r and pd.notna(r.get("first_10min_prob")):
                            p_yes = float(r.get("first_10min_prob"))
                        elif "first_10min_proj" in r and pd.notna(r.get("first_10min_proj")):
                            lam10 = float(r.get("first_10min_proj"))
                            if _math.isfinite(lam10) and lam10 >= 0:
                                p_yes = 1.0 - exp(-lam10)
                    if p_yes is not None:
                        p_yes = max(0.0, min(1.0, float(p_yes)))
                        r["p_f10_yes"] = p_yes
                        r["p_f10_no"] = 1.0 - p_yes
                except Exception:
                    pass

                # Period totals probabilities for P1..P3 if period projections and lines available
                for pn in (1, 2, 3):
                    try:
                        hkey = f"period{pn}_home_proj"; akey = f"period{pn}_away_proj"
                        lkey = f"p{pn}_total_line"
                        if (hkey in r and akey in r and lkey in r and pd.notna(r.get(hkey)) and pd.notna(r.get(akey)) and pd.notna(r.get(lkey))):
                            mu = float(r.get(hkey)) + float(r.get(akey))
                            ln = float(r.get(lkey))
                            if not (_math.isfinite(mu) and _math.isfinite(ln)):
                                raise ValueError
                            # Half lines vs integer lines
                            if abs(ln - round(ln)) < 1e-9:
                                k = int(round(ln))
                                p_push = _pois_pmf(mu, k)
                                p_under = _pois_cdf(mu, k - 1)
                                p_over = max(0.0, 1.0 - _pois_cdf(mu, k))
                            else:
                                k = floor(ln)
                                p_push = 0.0
                                p_under = _pois_cdf(mu, k)
                                p_over = max(0.0, 1.0 - p_under)
                            r[f"p{pn}_over_prob"] = max(0.0, min(1.0, float(p_over)))
                            r[f"p{pn}_under_prob"] = max(0.0, min(1.0, float(p_under)))
                            # store push probability for UI if useful
                            r[f"p{pn}_push_prob"] = max(0.0, min(1.0, float(p_push)))
                    except Exception:
                        continue

                r = _ensure_ev(r, "p_home_ml", "home_ml_odds", "ev_home_ml", "edge_home_ml")
                r = _ensure_ev(r, "p_away_ml", "away_ml_odds", "ev_away_ml", "edge_away_ml")
                r = _ensure_ev(r, "p_over", "over_odds", "ev_over", "edge_over")
                r = _ensure_ev(r, "p_under", "under_odds", "ev_under", "edge_under")
                r = _ensure_ev(r, "p_home_pl_-1.5", "home_pl_-1.5_odds", "ev_home_pl_-1.5", "edge_home_pl_-1.5")
                r = _ensure_ev(r, "p_away_pl_+1.5", "away_pl_+1.5_odds", "ev_away_pl_+1.5", "edge_away_pl_+1.5")
                # First 10 minutes EVs
                r = _ensure_ev(r, "p_f10_yes", "f10_yes_odds", "ev_f10_yes", "edge_f10_yes")
                r = _ensure_ev(r, "p_f10_no", "f10_no_odds", "ev_f10_no", "edge_f10_no")
                # Period totals EVs
                r = _ensure_ev(r, "p1_over_prob", "p1_over_odds", "ev_p1_over", None)
                r = _ensure_ev(r, "p1_under_prob", "p1_under_odds", "ev_p1_under", None)
                r = _ensure_ev(r, "p2_over_prob", "p2_over_odds", "ev_p2_over", None)
                r = _ensure_ev(r, "p2_under_prob", "p2_under_odds", "ev_p2_under", None)
                r = _ensure_ev(r, "p3_over_prob", "p3_over_odds", "ev_p3_over", None)
                r = _ensure_ev(r, "p3_under_prob", "p3_under_odds", "ev_p3_under", None)
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
    # Cache PBP per gamePk to avoid repeated fetches
    pbp_cache: dict[int, dict] = {}
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
            game_pk = None
            if g:
                game_pk = g.get("gamePk")
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
            # First-10 and Period totals results via play-by-play (if we have gamePk)
            try:
                if game_pk is not None:
                    if game_pk not in pbp_cache:
                        try:
                            pbp_cache[game_pk] = client.play_by_play(int(game_pk))
                        except Exception:
                            pbp_cache[game_pk] = {}
                    pbp = pbp_cache.get(game_pk) or {}
                    plays = pbp.get("plays") if isinstance(pbp, dict) else None
                    # Count goals per period and detect first 10 minutes goal
                    goals_by_period = {1: 0, 2: 0, 3: 0}
                    first10_yes = False
                    if isinstance(plays, list):
                        for p_ in plays:
                            try:
                                tkey = (str(p_.get("typeDescKey") or p_.get("type")) or "").lower()
                                # Only count actual scoring events; exclude 'shot-on-goal' etc.
                                if tkey != "goal":
                                    continue
                                # period number
                                per = None
                                try:
                                    pdsc = p_.get("periodDescriptor") or {}
                                    per = int(pdsc.get("number") or pdsc.get("period") or p_.get("period") or 0)
                                except Exception:
                                    per = int(p_.get("period") or 0)
                                if per in goals_by_period:
                                    goals_by_period[per] += 1
                                # clock fields: timeRemaining or timeInPeriod
                                tr = p_.get("timeRemaining") or p_.get("timeInPeriod") or p_.get("time")
                                mm = ss = None
                                if isinstance(tr, str) and ":" in tr:
                                    parts = tr.split(":")
                                    try:
                                        mm = int(parts[0]); ss = int(parts[1])
                                    except Exception:
                                        mm, ss = None, None
                                # Determine if goal occurred in first 10 minutes of P1
                                if per == 1 and (mm is not None and ss is not None):
                                    secs = mm * 60 + ss
                                    # Respect field semantics; if ambiguous 'time' is present, assume elapsed-time clock
                                    try:
                                        if p_.get("timeRemaining") is not None:
                                            if secs >= 600:
                                                first10_yes = True
                                        elif p_.get("timeInPeriod") is not None:
                                            if secs <= 600:
                                                first10_yes = True
                                        else:
                                            if secs <= 600:
                                                first10_yes = True
                                    except Exception:
                                        pass
                            except Exception:
                                continue
                    # Write first-10 result if missing
                    try:
                        if pd.isna(r.get("result_first10")) or not r.get("result_first10"):
                            df.at[i, "result_first10"] = "Yes" if first10_yes else "No"
                    except Exception:
                        pass
                    # Period totals results for P1..P3 if lines exist
                    for pn in (1, 2, 3):
                        try:
                            lkey1 = f"close_p{pn}_total_line"; lkey2 = f"p{pn}_total_line"
                            ln = _num(r.get(lkey1)) if lkey1 in r else None
                            if ln is None:
                                ln = _num(r.get(lkey2)) if lkey2 in r else None
                            if ln is None:
                                continue
                            actual_p = goals_by_period.get(pn)
                            if actual_p is None:
                                continue
                            res_key = f"result_p{pn}_total"
                            if pd.isna(r.get(res_key)) or not r.get(res_key):
                                if abs(ln - round(ln)) < 1e-9 and int(round(ln)) == int(actual_p):
                                    df.at[i, res_key] = "Push"
                                elif float(actual_p) > float(ln):
                                    df.at[i, res_key] = "Over"
                                else:
                                    df.at[i, res_key] = "Under"
                        except Exception:
                            continue
            except Exception:
                pass
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
        if os.getenv("WEB_DISABLE_GH_UPSERT", "").strip() in ("1","true","yes"):
            return {"skipped": True, "reason": "disabled_by_env"}
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


def _safe_rows_count_csv(path: Path) -> int:
    try:
        df = _read_csv_fallback(path)
        return 0 if df is None or df.empty else int(len(df))
    except Exception:
        return 0


def _safe_rows_count_parquet(path: Path) -> int:
    try:
        if path.exists():
            df = pd.read_parquet(path)
            return 0 if df is None or df.empty else int(len(df))
    except Exception:
        return 0
    return 0


def _gh_upsert_file_if_better_or_same(path: Path, message: str, rel_hint: Optional[str] = None) -> dict:
    """Upsert to GitHub only if content changed AND does not regress materially.

    For CSVs, regression = fewer rows than existing remote (or local cache as fallback).
    For Parquet lines, regression = fewer rows as well.
    """
    try:
        if os.getenv("WEB_DISABLE_GH_UPSERT", "").strip() in ("1","true","yes"):
            return {"skipped": True, "reason": "disabled_by_env"}
        token = os.getenv("GITHUB_TOKEN", "").strip()
        repo = os.getenv("GITHUB_REPO", "").strip()
        branch = os.getenv("GITHUB_BRANCH", "master").strip() or "master"
        if not token or not repo:
            return {"skipped": True, "reason": "missing_token_or_repo"}
        # Determine rel path
        rel_path = rel_hint
        if not rel_path:
            rel_path = str(path).replace("\\", "/")
            try:
                parts = rel_path.split("/")
                if "data" in parts:
                    idx = parts.index("data")
                    rel_path = "/".join(parts[idx-0:])
            except Exception:
                pass
        # Compute local new rows
        new_rows = 0
        is_csv = rel_path.lower().endswith(".csv")
        is_parquet = rel_path.lower().endswith(".parquet")
        if is_csv:
            new_rows = _safe_rows_count_csv(path)
        elif is_parquet:
            new_rows = _safe_rows_count_parquet(path)
        # Read remote for comparison
        old_rows = 0
        try:
            if is_csv:
                df_remote = _github_raw_read_csv(rel_path)
                if df_remote is not None and not df_remote.empty:
                    old_rows = int(len(df_remote))
            elif is_parquet:
                df_remote = _github_raw_read_parquet(rel_path)
                if df_remote is not None and not df_remote.empty:
                    old_rows = int(len(df_remote))
        except Exception:
            old_rows = 0
        # If we would regress in row count, skip to preserve earlier, richer data
        if (old_rows > 0) and (new_rows > 0) and (new_rows < old_rows):
            return {"skipped": True, "reason": "regression_rows", "old": old_rows, "new": new_rows, "path": rel_path}
        # Otherwise, perform standard upsert
        return _gh_upsert_file_if_configured(path, message)
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
            import datetime as _dt, math as _math
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
                        fv = float(v)
                        return fv if _math.isfinite(fv) else None
                except Exception:
                    pass
                # Handle native float infinities/NaN
                if isinstance(v, float):
                    return v if _math.isfinite(v) else None
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

def _json_sanitize(obj):
    """Recursively sanitize Python objects for strict JSON: remove NaN/Inf, coerce numpy/pandas scalars, and isoformat datetimes."""
    try:
        import numpy as _np, math as _math, datetime as _dt
        if obj is None:
            return None
        # Scalars
        if isinstance(obj, (int, str, bool)):
            return obj
        if isinstance(obj, float):
            return obj if _math.isfinite(obj) else None
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            f = float(obj)
            return f if _math.isfinite(f) else None
        if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
            try:
                return obj.date().isoformat() if isinstance(obj, pd.Timestamp) else obj.isoformat()
            except Exception:
                return str(obj)
        # Containers
        if isinstance(obj, dict):
            return {k: _json_sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [ _json_sanitize(v) for v in list(obj) ]
        # Pandas NA
        try:
            if isinstance(obj, float) and pd.isna(obj):
                return None
        except Exception:
            pass
        # Fallback to string for any exotic types
        return obj
    except Exception:
        return obj

def _df_hard_json_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Elementwise clean a DataFrame: replace NaN/Inf with None and coerce numpy scalars.

    This is defensive for sources like Parquet that may carry exotic dtypes.
    """
    import math as _math
    import numpy as _np
    import datetime as _dt
    def _c(v):
        try:
            if v is None:
                return None
            if isinstance(v, float):
                return v if _math.isfinite(v) else None
            if isinstance(v, (_np.floating,)):
                fv = float(v); return fv if _math.isfinite(fv) else None
            if isinstance(v, (_np.integer,)):
                return int(v)
            if isinstance(v, (pd.Timestamp, _dt.datetime, _dt.date)):
                try:
                    return v.date().isoformat() if isinstance(v, pd.Timestamp) else v.isoformat()
                except Exception:
                    return str(v)
            # Pandas NA
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            return v
        except Exception:
            return v
    try:
        return df.applymap(_c)
    except Exception:
        # Fallback per-column map
        for c in df.columns:
            try:
                df[c] = df[c].map(_c)
            except Exception:
                pass
        return df


@app.get("/health")
def health():
    """Ultra-light health endpoint.

    Intentionally avoids touching large data/model code paths. Returns current ET date and
    whether today's predictions CSV exists (best-effort). Fast and safe for load balancers.
    """
    et_today = None
    try:
        et_today = _today_ymd()
    except Exception:
        pass
    pred_exists = False
    try:
        if et_today:
            pred_exists = (PROC_DIR / f"predictions_{et_today}.csv").exists()
    except Exception:
        pass
    return {"status": "ok", "date_et": et_today, "predictions_today": bool(pred_exists)}


@app.get("/health/render")
def health_render():
    """Render cards template with an empty slate to validate template compiles."""
    try:
        et_today = _today_ymd()
    except Exception:
        et_today = ""
    try:
        template = env.get_template("cards_only.html")
        html = template.render()
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


## Removed duplicate async /health route to avoid double registration & confusion.


## (Replaced by lifespan handler above) Removed deprecated @app.on_event startup hooks.


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
    # Cards-only UI: serve a single static page that pulls from the read-only /v1 bundles API.
    # This intentionally avoids server-side compute/odds fetching in response to user page loads.
    try:
        template = env.get_template("cards_only.html")
        return HTMLResponse(content=template.render())
    except Exception as e:
        h = (_git_commit_hash() or "")[:12]
        return PlainTextResponse(f"cards UI unavailable: {e} (commit {h})", status_code=200)
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
            # Compute projections if missing/NaN
            mt = r.get("model_total")
            # 1) If team projections exist but model_total is missing, derive it directly
            try:
                phg = r.get("proj_home_goals"); pag = r.get("proj_away_goals")
                if (mt is None or _isnan(mt)) and phg is not None and pag is not None and not _isnan(phg) and not _isnan(pag):
                    r["model_total"] = round(float(phg) + float(pag), 2)
                    r["model_spread"] = round(float(phg) - float(pag), 2)
                    mt = r["model_total"]
            except Exception:
                pass
            # 2) Otherwise, if we have ML probability and a totals line, back out Poisson lambdas
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
        # If sim summary exists for this ET date, override projections with sim-derived values before probabilities
        try:
            sim_path = PROC_DIR / f"sim_summary_{date}.csv"
            sim_df = _read_csv_fallback(sim_path) if sim_path.exists() else pd.DataFrame()
        except Exception:
            sim_df = pd.DataFrame()
        sim_idx = {}
        try:
            def _abbr3(x: str) -> str:
                try:
                    return (get_team_assets(str(x)).get("abbr") or "").upper()
                except Exception:
                    return ""
            if sim_df is not None and not sim_df.empty:
                for _, sr in sim_df.iterrows():
                    try:
                        hk = _abbr3(sr.get("home")); ak = _abbr3(sr.get("away"))
                        if hk and ak:
                            sim_idx[(hk, ak)] = sr
                    except Exception:
                        pass
            # Apply to rows
            for r in rows:
                try:
                    hk = _abbr3(r.get("home")); ak = _abbr3(r.get("away"))
                    sr = sim_idx.get((hk, ak)) or sim_idx.get((ak, hk))
                    if not sr:
                        continue
                    # Override lambdas and period splits from sim means
                    for k in ("proj_home_goals","proj_away_goals","model_total","model_spread",
                              "period1_home_proj","period2_home_proj","period3_home_proj",
                              "period1_away_proj","period2_away_proj","period3_away_proj",
                              "first_10min_proj","p_f10_yes"):
                        if k in sr:
                            r[k] = sr.get(k)
                    # Optionally carry over sim ML probabilities (will still be blended below)
                    if sr.get("p_home_ml_sim") is not None:
                        r["p_home_ml"] = float(sr.get("p_home_ml_sim"))
                        r["p_away_ml"] = float(max(0.0, 1.0 - float(sr.get("p_home_ml_sim"))))
                    # If sim provided p_over/p_under at a known line, attach as hints
                    if sr.get("p_over_sim") is not None and (r.get("total_line_used") or r.get("close_total_line_used") or sr.get("total_line_used") is not None):
                        r["p_over_hint_sim"] = float(sr.get("p_over_sim"))
                        r["p_under_hint_sim"] = float(sr.get("p_under_sim")) if sr.get("p_under_sim") is not None else None
                        if r.get("total_line_used") is None and sr.get("total_line_used") is not None:
                            r["total_line_used"] = sr.get("total_line_used")
                except Exception:
                    pass
        except Exception:
            pass
        # After projection fallback (and optional sim overrides), compute Dixon–Coles probabilities and apply market anchoring
        try:
            import os
            from ..utils.odds import american_to_decimal, decimal_to_implied_prob, remove_vig_two_way
        except Exception:
            os = None
        # Resolve config (model-driven): read model_calibration.json, then env, else defaults
        dc_rho = -0.05
        anchor_w_ml = 0.25
        anchor_w_totals = 0.25
        totals_temp = 1.0  # temperature scaling for totals (pushed toward 0.5 when >1)
        # JSON first (authoritative)
        try:
            import json as _json
            if 'PROC_DIR' in globals():
                _cal_path = PROC_DIR / "model_calibration.json"
                if _cal_path.exists():
                    _obj = _json.loads(_cal_path.read_text(encoding="utf-8"))
                    if _obj.get("dc_rho") is not None:
                        dc_rho = float(_obj.get("dc_rho"))
                    # per-market weights with fallback to generic key
                    if _obj.get("market_anchor_w_ml") is not None:
                        anchor_w_ml = float(_obj.get("market_anchor_w_ml"))
                    elif _obj.get("market_anchor_w") is not None:
                        anchor_w_ml = float(_obj.get("market_anchor_w"))
                    if _obj.get("market_anchor_w_totals") is not None:
                        anchor_w_totals = float(_obj.get("market_anchor_w_totals"))
                    elif _obj.get("market_anchor_w") is not None:
                        anchor_w_totals = float(_obj.get("market_anchor_w"))
                    if _obj.get("totals_temp") is not None:
                        totals_temp = float(_obj.get("totals_temp"))
        except Exception:
            pass
        # Env second (dev overrides)
        try:
            if os is not None:
                er = os.getenv("DC_RHO")
                if er is not None:
                    dc_rho = float(er)
                ew_ml = os.getenv("MARKET_ANCHOR_W_ML")
                ew_to = os.getenv("MARKET_ANCHOR_W_TOTALS")
                ew_generic = os.getenv("MARKET_ANCHOR_W")
                if ew_ml is not None:
                    anchor_w_ml = float(ew_ml)
                elif ew_generic is not None:
                    anchor_w_ml = float(ew_generic)
                if ew_to is not None:
                    anchor_w_totals = float(ew_to)
                elif ew_generic is not None:
                    anchor_w_totals = float(ew_generic)
                et = os.getenv("TOTALS_TEMP")
                if et is not None:
                    totals_temp = float(et)
        except Exception:
            pass
        # Sanitize
        try:
            dc_rho = max(-0.2, min(0.2, float(dc_rho)))
        except Exception:
            dc_rho = -0.05
        try:
            anchor_w_ml = max(0.0, min(1.0, float(anchor_w_ml)))
        except Exception:
            anchor_w_ml = 0.25
        try:
            anchor_w_totals = max(0.0, min(1.0, float(anchor_w_totals)))
        except Exception:
            anchor_w_totals = 0.25
        try:
            totals_temp = max(0.5, min(3.0, float(totals_temp)))
        except Exception:
            totals_temp = 1.0

        # Helpers for EV recompute
        def _num2(v):
            try:
                if v is None:
                    return None
                if isinstance(v, (int, float)):
                    fv = float(v)
                    return fv if _math.isfinite(fv) else None
                s = str(v).strip().replace(",", "")
                if s == "":
                    return None
                return float(s)
            except Exception:
                return None
        def _ev_from_prob(p: float, american_odds: float, p_push: float = 0.0) -> float | None:
            try:
                if p is None or american_odds is None:
                    return None
                dec = american_to_decimal(float(american_odds))
                if dec is None or not _math.isfinite(dec):
                    return None
                p_win = float(p)
                if not (0.0 <= p_win <= 1.0):
                    return None
                p_loss = max(0.0, 1.0 - p_win - float(p_push))
                return round(p_win * (dec - 1.0) - p_loss, 4)
            except Exception:
                return None

        # Compute for each row
        for r in rows:
            try:
                # Establish lambdas if available
                lam_h = lam_a = None
                phg = r.get("proj_home_goals"); pag = r.get("proj_away_goals")
                if phg is not None and pag is not None:
                    lam_h = float(phg); lam_a = float(pag)
                else:
                    # derive from total line and ML prob if possible
                    tl = r.get("total_line_used") or r.get("close_total_line_used")
                    ph = r.get("p_home_ml")
                    if _pois_fb is not None and tl is not None and ph is not None:
                        lam_h, lam_a = _pois_fb.lambdas_from_total_split(float(tl), float(ph))
                if lam_h is None or lam_a is None or not (_math.isfinite(lam_h) and _math.isfinite(lam_a)):
                    continue
                # Use DC to get market probabilities
                tl = r.get("total_line_used") or r.get("close_total_line_used")
                tl_val = float(tl) if tl is not None and _math.isfinite(_num2(tl)) else 6.0
                probs = dc_market_probs(lam_h, lam_a, total_line=tl_val, rho=dc_rho, max_goals=10)
                # Raw model probabilities
                r["p_home_ml_model"] = float(probs["home_ml"]) if "home_ml" in probs else None
                r["p_away_ml_model"] = float(probs["away_ml"]) if "away_ml" in probs else None
                r["p_over_model"] = float(probs["over"]) if "over" in probs else None
                r["p_under_model"] = float(probs["under"]) if "under" in probs else None
                r["p_home_pl_-1.5_model"] = float(probs["home_puckline_-1.5"]) if "home_puckline_-1.5" in probs else None
                r["p_away_pl_+1.5_model"] = float(probs["away_puckline_+1.5"]) if "away_puckline_+1.5" in probs else None
                # Market-implied (no-vig) where available
                # Moneyline
                nv_pair = implied_pair_from_american(r.get("home_ml_odds"), r.get("away_ml_odds")) or \
                          implied_pair_from_american(r.get("close_home_ml_odds"), r.get("close_away_ml_odds"))
                p_mkt_home = nv_pair[0] if nv_pair else None
                p_mkt_away = nv_pair[1] if nv_pair else None
                # Totals
                nv_tot = implied_pair_from_two_sided(r.get("over_odds"), r.get("under_odds")) or \
                         implied_pair_from_two_sided(r.get("close_over_odds"), r.get("close_under_odds"))
                p_mkt_over = nv_tot[0] if nv_tot else None
                p_mkt_under = nv_tot[1] if nv_tot else None
                # Puckline
                nv_pl = implied_pair_from_two_sided(r.get("home_pl_-1.5_odds"), r.get("away_pl_+1.5_odds")) or \
                        implied_pair_from_two_sided(r.get("close_home_pl_-1.5_odds"), r.get("close_away_pl_+1.5_odds"))
                p_mkt_hpl = nv_pl[0] if nv_pl else None
                p_mkt_apl = nv_pl[1] if nv_pl else None
                # Blend toward market (per-market weights)
                def _set_prob(key: str, p_model: float, p_market: float | None, w_market: float):
                    try:
                        p = blend_probability(p_model, p_market, w_market=w_market)
                        r[key] = float(max(0.0, min(1.0, p)))
                    except Exception:
                        r[key] = p_model
                if r.get("p_home_ml_model") is not None:
                    _set_prob("p_home_ml", r["p_home_ml_model"], p_mkt_home, anchor_w_ml)
                if r.get("p_away_ml_model") is not None:
                    _set_prob("p_away_ml", r["p_away_ml_model"], p_mkt_away, anchor_w_ml)
                if r.get("p_over_model") is not None:
                    _set_prob("p_over", r["p_over_model"], p_mkt_over, anchor_w_totals)
                if r.get("p_under_model") is not None:
                    _set_prob("p_under", r["p_under_model"], p_mkt_under, anchor_w_totals)
                if r.get("p_home_pl_-1.5_model") is not None:
                    _set_prob("p_home_pl_-1.5", r["p_home_pl_-1.5_model"], p_mkt_hpl, anchor_w_ml)
                if r.get("p_away_pl_+1.5_model") is not None:
                    _set_prob("p_away_pl_+1.5", r["p_away_pl_+1.5_model"], p_mkt_apl, anchor_w_ml)

                # Optional totals temperature scaling around 0.5 (logit-space)
                try:
                    if totals_temp and abs(totals_temp - 1.0) > 1e-6:
                        def _sigmoid(x: float) -> float:
                            try:
                                return 1.0 / (1.0 + _math.exp(-x))
                            except Exception:
                                return 0.5
                        def _logit(p: float) -> float:
                            p = min(max(p, 1e-9), 1 - 1e-9)
                            return _math.log(p / (1.0 - p))
                        # For integer lines, scale conditional on non-push probability mass
                        if tl_val is not None and abs(tl_val - round(tl_val)) < 1e-9:
                            S = max(1e-9, 1.0 - float(p_push))
                            po = float(r.get("p_over")) if r.get("p_over") is not None else None
                            pu = float(r.get("p_under")) if r.get("p_under") is not None else None
                            if po is not None and pu is not None and S > 0:
                                po_c = min(max(po / S, 1e-9), 1 - 1e-9)
                                lo = _logit(po_c) / float(totals_temp)
                                po_adj_c = _sigmoid(lo)
                                po_adj = float(po_adj_c * S)
                                r["p_over"] = po_adj
                                r["p_under"] = max(0.0, S - po_adj)
                        else:
                            po = float(r.get("p_over")) if r.get("p_over") is not None else None
                            if po is not None:
                                lo = _logit(po) / float(totals_temp)
                                r["p_over"] = _sigmoid(lo)
                                r["p_under"] = max(0.0, 1.0 - float(r["p_over"]))
                except Exception:
                    pass

                # Recompute EVs using updated probabilities when odds are present
                # Totals push for integer lines
                p_push = 0.0
                try:
                    if tl_val is not None and abs(tl_val - round(tl_val)) < 1e-9:
                        k = int(round(tl_val))
                        from math import exp, factorial
                        mu_tot = float(lam_h + lam_a)
                        p_push = float(exp(-mu_tot) * (mu_tot ** k) / factorial(k)) if k >= 0 else 0.0
                except Exception:
                    p_push = 0.0
                # ML
                r["ev_home_ml"] = _ev_from_prob(r.get("p_home_ml"), _num2(r.get("home_ml_odds") or r.get("close_home_ml_odds")))
                r["ev_away_ml"] = _ev_from_prob(r.get("p_away_ml"), _num2(r.get("away_ml_odds") or r.get("close_away_ml_odds")))
                # Totals
                r["ev_over"] = _ev_from_prob(r.get("p_over"), _num2(r.get("over_odds") or r.get("close_over_odds")), p_push)
                r["ev_under"] = _ev_from_prob(r.get("p_under"), _num2(r.get("under_odds") or r.get("close_under_odds")), p_push)
                # Puckline
                r["ev_home_pl_-1.5"] = _ev_from_prob(r.get("p_home_pl_-1.5"), _num2(r.get("home_pl_-1.5_odds") or r.get("close_home_pl_-1.5_odds")))
                r["ev_away_pl_+1.5"] = _ev_from_prob(r.get("p_away_pl_+1.5"), _num2(r.get("away_pl_+1.5_odds") or r.get("close_away_pl_+1.5_odds")))
            except Exception:
                continue
        # Derive simple period-by-period projections and First-10 indicator when base lambdas exist
        try:
            import math as _m
            w = [0.32, 0.33, 0.35]
            s = sum(w) or 1.0
            w = [x / s for x in w]
            for r in rows:
                try:
                    phg = r.get("proj_home_goals"); pag = r.get("proj_away_goals")
                    if phg is not None and pag is not None:
                        try:
                            h = float(phg); a = float(pag)
                            r.setdefault("period1_home_proj", round(h * w[0], 2))
                            r.setdefault("period2_home_proj", round(h * w[1], 2))
                            r.setdefault("period3_home_proj", round(h * w[2], 2))
                            r.setdefault("period1_away_proj", round(a * w[0], 2))
                            r.setdefault("period2_away_proj", round(a * w[1], 2))
                            r.setdefault("period3_away_proj", round(a * w[2], 2))
                            # First 10 min expected goals ~ total_lambda * (10/60)
                            lam10 = (h + a) * (10.0 / 60.0)
                            r.setdefault("first_10min_proj", round(lam10, 3))
                            try:
                                p_yes = 1.0 - _m.exp(-lam10)
                                r.setdefault("p_f10_yes", round(p_yes, 4))
                            except Exception:
                                pass
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
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
        # Compute model picks for ML, Totals, and Puck Line
        try:
            # Moneyline model pick
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
        try:
            # Totals model pick
            po = float(r.get("p_over")) if r.get("p_over") is not None else None
            pu = float(r.get("p_under")) if r.get("p_under") is not None else None
            if po is not None and pu is not None:
                if po >= pu:
                    r["model_pick_total"] = "Over"
                    r["model_pick_total_prob"] = po
                else:
                    r["model_pick_total"] = "Under"
                    r["model_pick_total_prob"] = pu
        except Exception:
            pass
        try:
            # Puck line model pick (-1.5 / +1.5)
            php = float(r.get("p_home_pl_-1.5")) if r.get("p_home_pl_-1.5") is not None else None
            pap = float(r.get("p_away_pl_+1.5")) if r.get("p_away_pl_+1.5") is not None else None
            if php is not None and pap is not None:
                if php >= pap:
                    r["model_pick_pl"] = "Home -1.5"
                    r["model_pick_pl_prob"] = php
                else:
                    r["model_pick_pl"] = "Away +1.5"
                    r["model_pick_pl_prob"] = pap
        except Exception:
            pass
        # Candidates aligned to model picks only (and EV must be positive to consider)
        cands = []
        ev_h = _to_float(r.get("ev_home_ml")); ev_a = _to_float(r.get("ev_away_ml"))
        if r.get("model_pick") == "Home ML" and ev_h is not None and ev_h > 0:
            cands.append({"market": "moneyline", "bet": "home_ml", "label": "Home ML", "ev": ev_h, "odds": r.get("home_ml_odds"), "book": r.get("home_ml_book")})
        if r.get("model_pick") == "Away ML" and ev_a is not None and ev_a > 0:
            cands.append({"market": "moneyline", "bet": "away_ml", "label": "Away ML", "ev": ev_a, "odds": r.get("away_ml_odds"), "book": r.get("away_ml_book")})
        ev_o = _to_float(r.get("ev_over")); ev_u = _to_float(r.get("ev_under"))
        if r.get("model_pick_total") == "Over" and ev_o is not None and ev_o > 0:
            cands.append({"market": "totals", "bet": "over", "label": "Over", "ev": ev_o, "odds": r.get("over_odds"), "book": r.get("over_book")})
        if r.get("model_pick_total") == "Under" and ev_u is not None and ev_u > 0:
            cands.append({"market": "totals", "bet": "under", "label": "Under", "ev": ev_u, "odds": r.get("under_odds"), "book": r.get("under_book")})
        ev_hpl = _to_float(r.get("ev_home_pl_-1.5")); ev_apl = _to_float(r.get("ev_away_pl_+1.5"))
        if r.get("model_pick_pl") == "Home -1.5" and ev_hpl is not None and ev_hpl > 0:
            cands.append({"market": "puckline", "bet": "home_pl_-1.5", "label": "Home -1.5", "ev": ev_hpl, "odds": r.get("home_pl_-1.5_odds"), "book": r.get("home_pl_-1.5_book")})
        if r.get("model_pick_pl") == "Away +1.5" and ev_apl is not None and ev_apl > 0:
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
        else:
            # No aligned +EV recommendation
            for k in ("rec_market","rec_bet","rec_label","rec_ev","rec_odds","rec_book","rec_result","rec_success","rec_confidence"):
                r[k] = None
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
            # Estimate push probability for totals if we have an integer line and a model total
            try:
                p_push = None
                tl = r.get("disp_total_line_used")
                mt = r.get("model_total")
                if tl is not None and mt is not None:
                    tl_f = float(tl); mt_f = float(mt)
                    if math.isfinite(tl_f) and math.isfinite(mt_f) and abs(tl_f - round(tl_f)) < 1e-9:
                        k = int(round(tl_f))
                        from math import exp, factorial
                        p_push = float(exp(-mt_f) * (mt_f ** k) / factorial(k)) if k >= 0 else 0.0
                r["p_push"] = p_push
            except Exception:
                r["p_push"] = r.get("p_push") if r.get("p_push") is not None else None
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
@app.get("/api/cards/debug-game")
async def api_cards_debug_game(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    home: Optional[str] = Query(None, description="Home team name"),
    away: Optional[str] = Query(None, description="Away team name"),
    home_abbr: Optional[str] = Query(None, description="Home team abbreviation"),
    away_abbr: Optional[str] = Query(None, description="Away team abbreviation"),
    sample: int = Query(5, description="Sample size for boxscore lists"),
):
    """Debug endpoint showing how the cards view assembles data for a game.

    Returns predictions row, props recommendations (top per team when available),
    and simulated boxscores joined by team for a specific game on an ET date.
    """
    try:
        d = date or _today_ymd()
    except Exception:
        d = date
    # Resolve abbreviations from names when not provided
    try:
        if home_abbr is None and home:
            home_abbr = (get_team_assets(str(home)).get("abbr") or "").upper()
        if away_abbr is None and away:
            away_abbr = (get_team_assets(str(away)).get("abbr") or "").upper()
    except Exception:
        pass
    def _abbr(x: str) -> str:
        try:
            return (get_team_assets(str(x)).get("abbr") or "").upper()
        except Exception:
            return ""
    # Load predictions for date and neighbors, then filter to ET bucket
    def _load_predictions_et_bucket(d_ymd: str) -> pd.DataFrame:
        try:
            base = _read_csv_fallback(PROC_DIR / f"predictions_{d_ymd}.csv")
        except Exception:
            base = pd.DataFrame()
        frames = []
        if base is not None and not base.empty:
            frames.append(base)
        try:
            base_dt = datetime.fromisoformat(d_ymd)
            for d_nei in [ (base_dt - timedelta(days=1)).strftime("%Y-%m-%d"), (base_dt + timedelta(days=1)).strftime("%Y-%m-%d") ]:
                p = PROC_DIR / f"predictions_{d_nei}.csv"
                if p.exists():
                    df_n = _read_csv_fallback(p)
                    if df_n is not None and not df_n.empty:
                        frames.append(df_n)
        except Exception:
            pass
        if not frames:
            return pd.DataFrame()
        dfall = pd.concat(frames, ignore_index=True)
        try:
            et_dates = dfall.apply(lambda r: _iso_to_et_date(r.get("date") if pd.notna(r.get("date")) else r.get("gameDate")), axis=1)
            dfall = dfall[et_dates == d_ymd]
        except Exception:
            pass
        # Drop potential duplicates
        try:
            if {"home","away"}.issubset(dfall.columns):
                dfall = dfall.drop_duplicates(subset=["home","away"], keep="first")
        except Exception:
            pass
        return dfall
    df_pred = _load_predictions_et_bucket(d or _today_ymd())
    # Find the predictions row by abbr first, then fallback to names
    pred_row = None
    try:
        if not df_pred.empty:
            for _, r in df_pred.iterrows():
                try:
                    h_ab = _abbr(r.get("home")); a_ab = _abbr(r.get("away"))
                    if home_abbr and away_abbr and h_ab == (home_abbr or "").upper() and a_ab == (away_abbr or "").upper():
                        pred_row = r; break
                except Exception:
                    pass
            if pred_row is None and home and away:
                for _, r in df_pred.iterrows():
                    try:
                        if str(r.get("home")).strip() == str(home).strip() and str(r.get("away")).strip() == str(away).strip():
                            pred_row = r; break
                    except Exception:
                        pass
    except Exception:
        pred_row = None
    # Load props recommendations and map by team
    recs_by_team = {}
    try:
        rec_path = PROC_DIR / f"props_recommendations_{d}.csv"
        df_recs = _read_csv_fallback(rec_path) if rec_path.exists() else pd.DataFrame()
        if df_recs is not None and not df_recs.empty:
            dfc = df_recs.copy()
            dfc["team_norm"] = dfc.get("team", "").astype(str).str.upper().str.strip()
            # Keep limited fields
            keep = [c for c in ["player","team_norm","market","line","side","ev","p_over","over_price","under_price","book"] if c in dfc.columns]
            dfc = dfc[keep]
            try:
                dfc = dfc.sort_values("ev", ascending=False)
            except Exception:
                pass
            for _, rr in dfc.iterrows():
                tm = str(rr.get("team_norm") or "").upper()
                if not tm:
                    continue
                obj = {
                    "player": rr.get("player"),
                    "team": tm,
                    "market": rr.get("market"),
                    "line": rr.get("line"),
                    "side": rr.get("side"),
                    "ev": float(rr.get("ev")) if pd.notna(rr.get("ev")) else None,
                    "p_over": float(rr.get("p_over")) if pd.notna(rr.get("p_over")) else None,
                    "book": rr.get("book"),
                    "price": (rr.get("over_price") if str(rr.get("side") or "").upper()=="OVER" else rr.get("under_price")),
                }
                recs_by_team.setdefault(tm, []).append(obj)
    except Exception:
        recs_by_team = {}
    # Load sim boxscores and split by team
    box_home = []; box_away = []
    try:
        box_path = PROC_DIR / f"props_boxscores_sim_{d}.csv"
        df_box = _read_csv_fallback(box_path) if box_path.exists() else pd.DataFrame()
        if df_box is not None and not df_box.empty:
            db = df_box.copy()
            if "period" in db.columns:
                db = db[db["period"].fillna(0).astype(int) == 0]
            # Filter by game context if available
            if home and away and {"game_home","game_away"}.issubset(set(db.columns)):
                try:
                    db = db[(db["game_home"].astype(str)==str(home)) & (db["game_away"].astype(str)==str(away))]
                except Exception:
                    pass
            # Map team field to abbreviations for matching
            if "team" in db.columns:
                try:
                    db["team_abbr"] = db["team"].apply(lambda v: (get_team_assets(str(v)).get("abbr") or "").upper())
                except Exception:
                    db["team_abbr"] = ""
            # Resolve target abbreviations
            h_ab = (home_abbr or (home and _abbr(home)) or "").upper()
            a_ab = (away_abbr or (away and _abbr(away)) or "").upper()
            hb = db[db.get("team_abbr", pd.Series([])).astype(str).str.upper() == h_ab] if h_ab else db.head(0)
            ab = db[db.get("team_abbr", pd.Series([])).astype(str).str.upper() == a_ab] if a_ab else db.head(0)
            def _fmt_row(rr: pd.Series) -> dict:
                def _f(k):
                    try:
                        v = rr.get(k)
                        return float(v) if v is not None and pd.notna(v) else None
                    except Exception:
                        return None
                return {
                    "player": rr.get("player") or rr.get("player_id"),
                    "shots": _f("shots"),
                    "goals": _f("goals"),
                    "assists": _f("assists"),
                    "points": _f("points"),
                    "blocks": _f("blocks"),
                    "saves": _f("saves"),
                    "toi_sec": _f("toi_sec"),
                }
            box_home = [ _fmt_row(rr) for _, rr in hb.sort_values(["toi_sec","points","shots","saves"], ascending=[False, False, False, False]).head(sample).iterrows() ]
            box_away = [ _fmt_row(rr) for _, rr in ab.sort_values(["toi_sec","points","shots","saves"], ascending=[False, False, False, False]).head(sample).iterrows() ]
    except Exception:
        box_home = []; box_away = []
    # Assemble result
    try:
        pred_summary = None
        if pred_row is not None:
            pr = pred_row.to_dict()
            # keep a compact subset of key fields
            keys = [
                "home","away","date","home_abbr","away_abbr","model_total","model_spread",
                "p_home_ml","p_away_ml","p_over","p_under","total_line_used","close_total_line_used",
                "home_ml_odds","away_ml_odds","over_odds","under_odds","game_state",
                "final_home_goals","final_away_goals"
            ]
            pred_summary = {k: pr.get(k) for k in keys if k in pr}
            # attach abbrs
            try:
                pred_summary["home_abbr"] = _abbr(pr.get("home"))
                pred_summary["away_abbr"] = _abbr(pr.get("away"))
            except Exception:
                pass
    except Exception:
        pred_summary = None
    # Props recs samples per team
    rec_home = recs_by_team.get((home_abbr or "").upper(), [])[:3]
    rec_away = recs_by_team.get((away_abbr or "").upper(), [])[:3]
    payload = {
        "date": d,
        "home": home,
        "away": away,
        "home_abbr": (home_abbr or ""),
        "away_abbr": (away_abbr or ""),
        "predictions_row": pred_summary,
        "props_top_home": rec_home,
        "props_top_away": rec_away,
        "box_home_count": int(len(box_home)),
        "box_away_count": int(len(box_away)),
        "box_home_sample": box_home,
        "box_away_sample": box_away,
    }
    try:
        payload = _json_sanitize(payload)
    except Exception:
        pass
    return JSONResponse(payload)


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
                        # Extract per-period goals for line score table (best-effort across structures)
                        home_per = []
                        away_per = []
                        try:
                            periods_obj = ls2.get("periods") if isinstance(ls2, dict) else None
                            if isinstance(periods_obj, list):
                                for p in periods_obj:
                                    try:
                                        # Common shapes: {'num':1,'home':2,'away':1} or nested team dicts
                                        hv = None; av = None
                                        if isinstance(p.get("home"), dict):
                                            hv = p.get("home", {}).get("goals") or p.get("home", {}).get("score")
                                        else:
                                            hv = p.get("home")
                                        if isinstance(p.get("away"), dict):
                                            av = p.get("away", {}).get("goals") or p.get("away", {}).get("score")
                                        else:
                                            av = p.get("away")
                                        # Fallback generic keys
                                        if hv is None:
                                            hv = p.get("homeGoals") or p.get("homeScore")
                                        if av is None:
                                            av = p.get("awayGoals") or p.get("awayScore")
                                        # Coerce to int if numeric
                                        try:
                                            hv = int(hv) if hv is not None else None
                                        except Exception:
                                            hv = None
                                        try:
                                            av = int(av) if av is not None else None
                                        except Exception:
                                            av = None
                                        if hv is not None:
                                            home_per.append(hv)
                                        else:
                                            home_per.append(None)
                                        if av is not None:
                                            away_per.append(av)
                                        else:
                                            away_per.append(None)
                                    except Exception:
                                        home_per.append(None); away_per.append(None)
                        except Exception:
                            home_per = []; away_per = []
                        cached_stats = {}
                        if clock_val:
                            cached_stats["clock"] = clock_val
                            cached_stats["source_clock"] = "stats"
                        if per2 is not None:
                            cached_stats["period"] = per2
                        if home_per or away_per:
                            cached_stats["period_goals_home"] = home_per
                            cached_stats["period_goals_away"] = away_per
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

@app.get("/api/props/health")
async def api_props_health(date: Optional[str] = Query(None)):
    """Diagnostics for props data availability for a given date.

    Reports existence and row counts for projections/recommendations CSVs and presence of raw props lines parquet files.
    """
    d = date or _today_ymd()
    proj_path = PROC_DIR / f"props_projections_{d}.csv"
    rec_path = PROC_DIR / f"props_recommendations_{d}.csv"
    lines_dir = PROC_DIR.parent / "props" / f"player_props_lines/date={d}"
    out = {
        "date": d,
        "projections_csv": {"exists": proj_path.exists(), "rows": None},
        "recommendations_csv": {"exists": rec_path.exists(), "rows": None},
        "lines": {
            "path": str(lines_dir),
            "exists": lines_dir.exists(),
            "files": [],
        },
    }
    try:
        if proj_path.exists():
            dfp = _read_csv_fallback(proj_path)
            out["projections_csv"]["rows"] = 0 if dfp is None or dfp.empty else int(len(dfp))
    except Exception:
        pass
    try:
        if rec_path.exists():
            dfr = _read_csv_fallback(rec_path)
            out["recommendations_csv"]["rows"] = 0 if dfr is None or dfr.empty else int(len(dfr))
    except Exception:
        pass
    try:
        if lines_dir.exists():
            files = []
            for p in sorted(lines_dir.glob("*.parquet")):
                files.append(p.name)
            out["lines"]["files"] = files
    except Exception:
        pass
    return JSONResponse(out)


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
    # Prefer sim-native predictions when available
    sim_path = PROC_DIR / f"predictions_sim_{date}.csv"
    path = sim_path if sim_path.exists() else (PROC_DIR / f"predictions_{date}.csv")
    if not path.exists():
        return JSONResponse({"error": "No predictions for date", "date": date}, status_code=404)
    # Robust read to avoid decode/empty errors across environments
    df = _read_csv_fallback(path)
    return JSONResponse(_df_jsonsafe_records(df))


@app.get("/api/debug/odds-match")
async def api_debug_odds_match(date: Optional[str] = Query(None)):
    """Debug endpoint: for each game on date, show how OddsAPI odds would match and what prices were found."""
    date = date or _today_ymd()
    path = PROC_DIR / f"predictions_{date}.csv"
    if not path.exists():
        return JSONResponse({"error": "No predictions for date", "date": date}, status_code=404)
    df = pd.read_csv(path)
    if df.empty:
        return JSONResponse({"error": "Empty predictions file", "date": date}, status_code=400)
    # Fetch fresh OddsAPI odds
    try:
        client = OddsAPIClient()
        events, _ = client.list_events("icehockey_nhl")
        records = []
        for ev in events or []:
            data, _ = client.event_odds(
                sport="icehockey_nhl",
                event_id=str(ev.get("id")),
                markets="h2h,totals,spreads",
                regions="us",
                bookmakers="draftkings",
                odds_format="american",
            )
            # Pick first bookmaker
            bks = data.get("bookmakers", []) if isinstance(data, dict) else []
            if not bks:
                continue
            book = bks[0]
            mkts = book.get("markets", []) or []
            # Extract
            row = {"home": ev.get("home_team"), "away": ev.get("away_team")}
            for m in mkts:
                key = m.get("key")
                if key == "h2h":
                    for oc in m.get("outcomes", []) or []:
                        nm = str(oc.get("name") or "")
                        if nm == row["home"]:
                            row["home_ml"] = oc.get("price")
                        elif nm == row["away"]:
                            row["away_ml"] = oc.get("price")
                elif key == "totals":
                    for oc in m.get("outcomes", []) or []:
                        if oc.get("name") == "Over":
                            row["over"] = oc.get("price")
                            row["total_line"] = oc.get("point")
                        elif oc.get("name") == "Under":
                            row["under"] = oc.get("price")
                elif key == "spreads":
                    for oc in m.get("outcomes", []) or []:
                        pt = oc.get("point")
                        if pt == -1.5 and oc.get("name") == row["home"]:
                            row["home_pl_-1.5"] = oc.get("price")
                        if pt == 1.5 and oc.get("name") == row["away"]:
                            row["away_pl_+1.5"] = oc.get("price")
            records.append(row)
        odds = pd.DataFrame.from_records(records)
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
        # Prefer parquet (smaller/faster), but fall back to tracked CSV artifacts on Render.
        for name in ("oddsapi.parquet", "bovada.parquet"):
            p = base / name
            if p.exists():
                try:
                    parts.append(pd.read_parquet(p))
                except Exception:
                    pass
        if not parts:
            for name in ("oddsapi.csv", "bovada.csv"):
                p = base / name
                if p.exists():
                    try:
                        parts.append(pd.read_csv(p))
                    except Exception:
                        pass
        if not parts:
            return JSONResponse({"date": date, "data": []})
        df = pd.concat(parts, ignore_index=True)
        if market:
            df = df[df["market"].astype(str).str.upper() == market.upper()]
        # Lightweight response
        keep = [c for c in ["date","player_name","player_id","team","market","line","over_price","under_price","book","is_current"] if c in df.columns]
        df_out = df[keep].rename(columns={"player_name":"player"})
        df_out = _df_hard_json_clean(df_out)
        out = _df_jsonsafe_records(df_out)
        # Extra belt-and-suspenders: sanitize the output recursively
        safe_payload = _json_sanitize({"date": date, "data": out})
        # Pre-encode JSON to guarantee compliance and avoid internal dumps errors
        try:
            import json as _json
            body = _json.dumps(safe_payload, allow_nan=False)
        except Exception:
            # As a last resort, stringify everything
            def _stringify(o):
                try:
                    return str(o)
                except Exception:
                    return None
            safe_str = _json_sanitize({"date": str(date), "data": [{k: _stringify(v) for k, v in row.items()} for row in (out or [])]})
            body = _json.dumps(safe_str, allow_nan=False)
        return Response(content=body, media_type="application/json")
    except Exception as e:
        return JSONResponse({"date": date, "error": str(e), "data": []}, status_code=200)


@app.get("/api/props/recommendations")
async def api_props_recommendations(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    market: Optional[str] = Query(None, description="SOG,SAVES,GOALS,ASSISTS,POINTS"),
    min_ev: float = Query(0.0),
    top: int = Query(200),
    fmt: Optional[str] = Query(None, description="Optional: 'text' to return plain text for debugging"),
):
    """Serve props recommendations for a given date. If cached CSV exists, read; else compute on the fly via CLI logic."""
    try:
        date = date or _today_ymd()
        # Respect read-only mode: if cache missing, do not compute on-demand
        read_only_ui = _read_only(date)
        rec_path = PROC_DIR / f"props_recommendations_{date}.csv"
        df = None
        if rec_path.exists():
            try:
                # Robust read to handle encoding/empty quirks consistently
                df = _read_csv_fallback(rec_path)
            except Exception:
                df = None
        if (df is None or df.empty) and (not read_only_ui):
            # Compute on the fly by invoking the same logic inline (avoid spawning a subprocess)
            try:
                from ..models.props import SkaterShotsModel, GoalieSavesModel, SkaterGoalsModel
                from ..models.props import SkaterAssistsModel, SkaterPointsModel, SkaterBlocksModel
                from ..data.collect import collect_player_game_stats
                from ..utils.io import RAW_DIR
                from ..props.utils import compute_props_lam_scale_mean
                from ..data.nhl_api_web import NHLWebClient as _Web
                from ..web.teams import get_team_assets as _assets
                # Load canonical lines
                base = PROC_DIR.parent / "props" / f"player_props_lines/date={date}"
                parts = []
                for name in ("oddsapi.parquet", "bovada.parquet"):
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
                # Load team features similar to CLI for scaling
                import numpy as _np, json
                xg_path = PROC_DIR / "team_xg_latest.csv"
                xg_map = {}
                if xg_path.exists():
                    try:
                        _xg = pd.read_csv(xg_path)
                        if not _xg.empty and {"abbr","xgf60"}.issubset(_xg.columns):
                            xg_map = {str(r.abbr).upper(): float(r.xgf60) for _, r in _xg.iterrows()}
                    except Exception:
                        xg_map = {}
                league_xg = float(_np.mean(list(xg_map.values()))) if xg_map else 2.6
                pen_path = PROC_DIR / "team_penalty_rates.json"
                pen_comm = {}
                if pen_path.exists():
                    try:
                        pen_comm = json.loads(pen_path.read_text(encoding="utf-8"))
                    except Exception:
                        pen_comm = {}
                league_pen = float(_np.mean([float(v.get("committed_per60", 0.0)) for v in pen_comm.values()])) if pen_comm else 3.0
                # Goalie form (sv% L10)
                from datetime import date as _date
                gf_today = PROC_DIR / f"goalie_form_{_date.today().strftime('%Y-%m-%d')}.csv"
                gf_map = {}
                if gf_today.exists():
                    try:
                        _gf = pd.read_csv(gf_today)
                        if not _gf.empty and {"team","sv_pct_l10"}.issubset(_gf.columns):
                            gf_map = {str(r.team).upper(): float(r.sv_pct_l10) for _, r in _gf.iterrows()}
                    except Exception:
                        gf_map = {}
                league_sv = float(_np.mean(list(gf_map.values()))) if gf_map else 0.905
                # Possession events-derived PP fractions (optional)
                opp_pp_frac_map: dict[str, float] = {}
                team_pp_frac_map: dict[str, float] = {}
                league_pp_frac = 0.18
                try:
                    ev_path = PROC_DIR / f"sim_events_pos_{date}.csv"
                    if ev_path.exists():
                        ev = pd.read_csv(ev_path)
                        def _abbr(n: str | None) -> str | None:
                            try:
                                a = _assets(str(n)) or {}
                                return str(a.get("abbr") or "").upper() or None
                            except Exception:
                                return None
                        for _, r in ev.iterrows():
                            h = str(r.get("home") or ""); a = str(r.get("away") or "")
                            h_ab = _abbr(h); a_ab = _abbr(a)
                            if not h_ab or not a_ab:
                                continue
                            sh_home_total = float(r.get("shots_ev_home", 0)) + float(r.get("shots_pp_home", 0)) + float(r.get("shots_pk_home", 0))
                            sh_home_pp = float(r.get("shots_pp_home", 0))
                            team_pp_home = (sh_home_pp / sh_home_total) if sh_home_total > 0 else None
                            if team_pp_home is not None:
                                team_pp_frac_map[h_ab] = float(team_pp_home)
                            sh_away_total = float(r.get("shots_ev_away", 0)) + float(r.get("shots_pp_away", 0)) + float(r.get("shots_pk_away", 0))
                            sh_away_pp = float(r.get("shots_pp_away", 0))
                            team_pp_away = (sh_away_pp / sh_away_total) if sh_away_total > 0 else None
                            if team_pp_away is not None:
                                team_pp_frac_map[a_ab] = float(team_pp_away)
                            opp_pp_frac_home = (sh_away_pp / sh_away_total) if sh_away_total > 0 else None
                            if opp_pp_frac_home is not None:
                                opp_pp_frac_map[h_ab] = float(opp_pp_frac_home)
                            opp_pp_frac_away = (sh_home_pp / sh_home_total) if sh_home_total > 0 else None
                            if opp_pp_frac_away is not None:
                                opp_pp_frac_map[a_ab] = float(opp_pp_frac_away)
                        vals = [v for v in team_pp_frac_map.values() if v is not None]
                        if vals:
                            league_pp_frac = float(_np.mean(vals))
                except Exception:
                    opp_pp_frac_map = {}
                    team_pp_frac_map = {}
                # Opponent mapping via schedule
                abbr_to_opp: dict[str, str] = {}
                try:
                    web = _Web(); sched = web.schedule_day(date)
                    games=[]
                    for g in sched:
                        h=str(getattr(g, "home", "")); a=str(getattr(g, "away", ""))
                        ha = (_assets(h) or {}).get("abbr"); aa = (_assets(a) or {}).get("abbr")
                        ha = str(ha or "").upper(); aa = str(aa or "").upper()
                        if ha and aa:
                            abbr_to_opp[ha]=aa; abbr_to_opp[aa]=ha
                except Exception:
                    abbr_to_opp = {}
                def proj_prob(m, player, ln, team_abbr: str | None, opp_abbr: str | None):
                    m = (m or '').upper()
                    if m == 'SOG':
                        lam = shots.player_lambda(hist, player)
                        scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                            league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                            league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                            opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                            props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
                        lam_eff = (lam or 0.0) * float(scale)
                        return lam, shots.prob_over(lam_eff, ln)
                    if m == 'SAVES':
                        lam = saves.player_lambda(hist, player)
                        scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                            league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                            league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                            opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                            props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
                        lam_eff = (lam or 0.0) * float(scale)
                        return lam, saves.prob_over(lam_eff, ln)
                    if m == 'GOALS':
                        lam = goals.player_lambda(hist, player)
                        scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                            league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                            league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                            opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                            props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
                        lam_eff = (lam or 0.0) * float(scale)
                        return lam, goals.prob_over(lam_eff, ln)
                    if m == 'ASSISTS':
                        lam = assists.player_lambda(hist, player)
                        scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                            league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                            league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                            opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                            props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
                        lam_eff = (lam or 0.0) * float(scale)
                        return lam, assists.prob_over(lam_eff, ln)
                    if m == 'POINTS':
                        lam = points.player_lambda(hist, player)
                        scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                            league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                            league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                            opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                            props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
                        lam_eff = (lam or 0.0) * float(scale)
                        return lam, points.prob_over(lam_eff, ln)
                    if m == 'BLOCKS':
                        lam = blocks.player_lambda(hist, player)
                        scale = compute_props_lam_scale_mean(m, team_abbr, opp_abbr,
                            league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
                            league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
                            opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
                            props_xg_gamma=0.02, props_penalty_gamma=0.06, props_goalie_form_gamma=0.02, props_strength_gamma=0.04)
                        lam_eff = (lam or 0.0) * float(scale)
                        return lam, blocks.prob_over(lam_eff, ln)
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
                    team_abbr = (str(r.get('team') or '').strip().upper() or None)
                    opp_abbr = abbr_to_opp.get(team_abbr) if team_abbr else None
                    lam, p_over = proj_prob(m, str(player), ln, team_abbr, opp_abbr)
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
        # Optionally return plain text for debugging serialization issues
        if fmt and str(fmt).lower() == 'text':
            if df is None or df.empty:
                return PlainTextResponse(f"date={date}\nrows=0\n")
            # Show a few top rows as tab-separated plaintext
            try:
                head = df.head(min(10, len(df)))
                return PlainTextResponse("date=" + str(date) + "\n" + head.to_csv(index=False))
            except Exception as e:
                return PlainTextResponse(f"date={date}\nerror={e}")
        # Serialize safely using shared helper to avoid numpy/NaN/Inf issues
        try:
            rows = [] if (df is None or df.empty) else _df_jsonsafe_records(df)
            import json as _json
            body = _json.dumps({"date": str(date), "data": rows}, allow_nan=False)
            return Response(content=body, media_type="application/json")
        except Exception as e:
            # As a last resort, return a structured error without raising 500
            return JSONResponse({"date": str(date), "error": str(e), "data": []}, status_code=200)
    except Exception as e:
        # Avoid 500s: include error string in a plain JSON payload
        try:
            return JSONResponse({"date": str(date) if date else None, "error": str(e), "data": []}, status_code=200)
        except Exception:
            return PlainTextResponse(f"date={date}\nerror={e}")


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
        for name in ("oddsapi.parquet", "bovada.parquet"):
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
    async_run: bool = Query(False, description="If true, queue work in background and return 202 immediately"),
):
    """Secure endpoint to collect canonical player props lines (Parquet) for a date.

    - Writes data/props/player_props_lines/date=YYYY-MM-DD/(oddsapi|bovada).parquet
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
    def _collect_lines_for_date(_d: str) -> Dict[str, Any]:
        from ..data import player_props as props_data
        base = PROC_DIR.parent / "props" / f"player_props_lines/date={_d}"
        base.mkdir(parents=True, exist_ok=True)
        out: Dict[str, Any] = {"date": _d, "written": [], "errors": []}
        # Per-source timeout (seconds) to prevent indefinite hangs
        try:
            step_timeout = int(os.getenv('PROPS_STEP_TIMEOUT_SEC', '90'))
        except Exception:
            step_timeout = 90
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout
        for which, src in (("oddsapi", "oddsapi"), ("bovada", "bovada")):
            try:
                cfg = props_data.PropsCollectionConfig(output_root=str(PROC_DIR.parent / "props"), book=which, source=src)
                try:
                    roster_df = _props_data._build_roster_enrichment()
                except Exception:
                    roster_df = None
                # Run the collection in a tiny thread with timeout so it can't hang forever
                def _do_collect():
                    return props_data.collect_and_write(_d, roster_df=roster_df, cfg=cfg)
                try:
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(_do_collect)
                        res = fut.result(timeout=step_timeout)
                except _FutTimeout:
                    out["errors"].append({"book": which, "error": f"timeout_after_{step_timeout}s"})
                    continue
                path = res.get("output_path")
                if path:
                    out["written"].append(str(path))
                    try:
                        rel = str(Path(path)).replace("\\", "/")
                        try:
                            parts = rel.split("/")
                            if "data" in parts:
                                idx = parts.index("data")
                                rel = "/".join(parts[idx:])
                        except Exception:
                            pass
                        _gh_upsert_file_if_better_or_same(Path(path), f"web: update props lines {which} for {_d}", rel_hint=rel)
                    except Exception:
                        pass
            except Exception as e:
                out["errors"].append({"book": which, "error": str(e)})
        return out
    try:
        if async_run:
            job_id = _queue_cron('props-collect', {'date': d}, lambda: _collect_lines_for_date(d))
            return JSONResponse({"ok": True, "date": d, "queued": True, "mode": "async", "job_id": job_id}, status_code=202)
        out = _collect_lines_for_date(d)
        return JSONResponse({"ok": True, **out})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "date": d}, status_code=500)


@app.post("/api/cron/props-projections")
async def api_cron_props_projections(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD; defaults to ET today"),
    market: Optional[str] = Query(None, description="Optional market filter: SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS"),
    top: int = Query(0, description="If >0 keep top N rows by EV/P(Over) before writing"),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional alternative"),
    async_run: bool = Query(False, description="If true, queue work in background and return 202 immediately"),
):
    """Secure endpoint to compute and persist props projections CSV for a date.

    Writes data/processed/props_projections_{date}.csv and upserts to GitHub.
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
    def _compute_and_write(_d: str) -> Dict[str, Any]:
        df = _compute_props_projections(_d, market=market)
        if df is not None and not df.empty and top and top > 0:
            df = df.head(int(top))
        out_path = PROC_DIR / f"props_projections_{_d}.csv"
        save_df(df, out_path)
        try:
            hist_path = PROC_DIR / 'props_projections_history.csv'
            h = df.copy() if df is not None else pd.DataFrame()
            if 'date' not in (h.columns if isinstance(h, pd.DataFrame) else []):
                try:
                    h['date'] = _d
                except Exception:
                    pass
            if isinstance(h, pd.DataFrame) and not h.empty:
                if 'proj' in h.columns and 'proj_lambda' not in h.columns:
                    try: h.rename(columns={'proj':'proj_lambda'}, inplace=True)
                    except Exception: pass
                keep = [c for c in ['date','player','team','position','market','proj_lambda','proj_lambda_eff','p_over','ev_over'] if c in h.columns]
                if keep:
                    h = h[keep]
                if hist_path.exists():
                    try:
                        cur = pd.read_csv(hist_path)
                        comb = pd.concat([cur, h], ignore_index=True)
                        subset_keys = [k for k in ['date','player','market'] if k in comb.columns]
                        if subset_keys:
                            comb.sort_values(subset_keys, ascending=[False]*len(subset_keys), inplace=True)
                            comb.drop_duplicates(subset=subset_keys, keep='first', inplace=True)
                        comb.to_csv(hist_path, index=False)
                    except Exception:
                        try: h.to_csv(hist_path, index=False)
                        except Exception: pass
                else:
                    try: h.to_csv(hist_path, index=False)
                    except Exception: pass
        except Exception:
            pass
        try:
            _gh_upsert_file_if_better_or_same(out_path, f"web: update props projections for {_d}")
        except Exception:
            pass
        return {"rows": 0 if df is None or df.empty else int(len(df)), "path": str(out_path)}
    try:
        if async_run:
            job_id = _queue_cron('props-projections', {'date': d, 'market': market, 'top': top}, lambda: _compute_and_write(d))
            return JSONResponse({"ok": True, "date": d, "queued": True, "mode": "async", "job_id": job_id}, status_code=202)
        res = _compute_and_write(d)
        return JSONResponse({"ok": True, "date": d, **res})
    except Exception as e:
        return JSONResponse({"ok": False, "date": d, "error": str(e)}, status_code=500)


@app.get("/api/props/projections")
async def api_props_projections(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    market: Optional[str] = Query(None, description="SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS"),
    top: int = Query(0, description="If >0, return top N rows"),
):
    """Serve props projections for a given date from cached CSV if present; compute on-the-fly otherwise.

    The cached file is props_projections_{date}.csv under data/processed.
    """
    d = _normalize_date_param(date)
    cache = PROC_DIR / f"props_projections_{d}.csv"
    df = None
    if cache.exists():
        try:
            df = _read_csv_fallback(cache)
        except Exception:
            df = None
    if (df is None) or (df is not None and df.empty):
        # Try GitHub raw fallback (read-only env)
        try:
            df = _github_raw_read_csv(f"data/processed/props_projections_{d}.csv")
        except Exception:
            df = df
    if df is None:
        # Compute quickly and do not write (UI may be in read-only mode)
        try:
            df = _compute_props_projections(d, market=market)
        except Exception:
            df = pd.DataFrame()
    if df is None or df.empty:
        return JSONResponse({"date": d, "data": []})
    if market and 'market' in df.columns:
        df = df[df['market'].astype(str).str.upper() == market.upper()]
    if top and top > 0:
        df = df.head(int(top))
    return JSONResponse({"date": d, "data": _df_jsonsafe_records(df)})


@app.post("/api/cron/props-recommendations")
async def api_cron_props_recommendations(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD; defaults to ET today"),
    market: Optional[str] = Query(None, description="SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS"),
    min_ev: float = Query(0.0),
    top: int = Query(200),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional alternative to token query param)"),
    async_run: bool = Query(False, description="If true, queue work in background and return 202 immediately"),
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
    def _compute_recommendations_for_date(_d: str) -> Dict[str, Any]:
        # Ensure lines exist; if not, collect
        try:
            try:
                step_timeout = int(os.getenv('PROPS_STEP_TIMEOUT_SEC', '90'))
            except Exception:
                step_timeout = 90
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout
            base_local = PROC_DIR.parent / "props" / f"player_props_lines/date={_d}"
            need_collect_local = not (((base_local / "oddsapi.parquet").exists()) or ((base_local / "bovada.parquet").exists()))
            if need_collect_local:
                try:
                    # Call the internal helper used by props-collect
                    from ..data import player_props as props_data
                    base_local.mkdir(parents=True, exist_ok=True)
                    for which, src in (("oddsapi", "oddsapi"), ("bovada", "bovada")):
                        try:
                            cfg = props_data.PropsCollectionConfig(output_root=str(PROC_DIR.parent / "props"), book=which, source=src)
                            try:
                                roster_df_local = _props_data._build_roster_enrichment()
                            except Exception:
                                roster_df_local = None
                            # Timeout-guard the collection
                            def _do_collect_local():
                                return props_data.collect_and_write(_d, roster_df=roster_df_local, cfg=cfg)
                            try:
                                with ThreadPoolExecutor(max_workers=1) as ex:
                                    fut = ex.submit(_do_collect_local)
                                    res_local = fut.result(timeout=step_timeout)
                            except _FutTimeout:
                                res_local = {"output_path": None}
                            path_local = res_local.get("output_path")
                            if path_local:
                                try:
                                    rel_local = str(Path(path_local)).replace("\\", "/")
                                    try:
                                        parts_local = rel_local.split("/")
                                        if "data" in parts_local:
                                            idx_local = parts_local.index("data")
                                            rel_local = "/".join(parts_local[idx_local:])
                                    except Exception:
                                        pass
                                    _gh_upsert_file_if_better_or_same(Path(path_local), f"web: update props lines {which} for {_d}", rel_hint=rel_local)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass
        except Exception:
            pass
        # Compute recommendations using the same logic as api_props_recommendations (compute branch)
        from ..models.props import SkaterShotsModel, GoalieSavesModel, SkaterGoalsModel
        from ..models.props import SkaterAssistsModel, SkaterPointsModel, SkaterBlocksModel
        from ..data.collect import collect_player_game_stats
        # Load lines
        parts = []
        base = PROC_DIR.parent / "props" / f"player_props_lines/date={_d}"
        for name in ("oddsapi.parquet", "bovada.parquet"):
            p = base / name
            if p.exists():
                try:
                    parts.append(pd.read_parquet(p))
                except Exception:
                    pass
        if not parts:
            return {"rows": 0, "message": "no-lines"}
        lines = pd.concat(parts, ignore_index=True)
        # Ensure stats exist for projection
        stats_path = RAW_DIR / "player_game_stats.csv"
        if not stats_path.exists():
            try:
                from datetime import datetime as _dt, timedelta as _td
                start = (_dt.strptime(_d, "%Y-%m-%d") - _td(days=365)).strftime("%Y-%m-%d")
                # Guard the stats backfill with a timeout as well
                def _do_stats():
                    collect_player_game_stats(start, _d, source="stats")
                try:
                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout  # reuse names if not present
                except Exception:
                    pass
                try:
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        fut = ex.submit(_do_stats)
                        fut.result(timeout=step_timeout)
                except Exception:
                    pass
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
        rec_path = PROC_DIR / f"props_recommendations_{_d}.csv"
        try:
            save_df(df, rec_path)
            try:
                _gh_upsert_file_if_better_or_same(rec_path, f"web: update props recommendations for {_d}")
            except Exception:
                pass
        except Exception:
            pass
        return {"rows": 0 if df is None or df.empty else int(len(df)), "path": str(rec_path)}
    try:
        if async_run:
            # Run with an overall timeout so we never hang indefinitely
            def _run_recs_with_timeout():
                try:
                    timeout_s = int(os.getenv('PROPS_RECS_TIMEOUT_SEC', '180'))
                except Exception:
                    timeout_s = 180
                res_holder: Dict[str, Any] = {}
                err_holder: Dict[str, Any] = {}
                def _inner():
                    try:
                        res_holder['res'] = _compute_recommendations_for_date(d)
                    except Exception as e:
                        err_holder['err'] = str(e)
                th = threading.Thread(target=_inner, daemon=True)
                th.start()
                th.join(timeout=timeout_s)
                if th.is_alive():
                    # Timed out: write an empty CSV so downstream health reflects presence
                    try:
                        rec_path = PROC_DIR / f"props_recommendations_{d}.csv"
                        save_df(pd.DataFrame(), rec_path)
                    except Exception:
                        pass
                    return {"rows": 0, "path": str(PROC_DIR / f"props_recommendations_{d}.csv"), "message": "timeout"}
                if 'err' in err_holder:
                    raise Exception(err_holder['err'])
                return res_holder.get('res', {"rows": 0, "path": str(PROC_DIR / f"props_recommendations_{d}.csv"), "message": "no-result"})
            job_id = _queue_cron('props-recommendations', {'date': d, 'market': market, 'min_ev': min_ev, 'top': top}, _run_recs_with_timeout)
            return JSONResponse({"ok": True, "date": d, "queued": True, "mode": "async", "job_id": job_id}, status_code=202)
        res = _compute_recommendations_for_date(d)
        return JSONResponse({"ok": True, "date": d, **res})
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

@app.get('/health/props')
async def health_props(date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET); defaults to today")):
    """Enhanced health probe for props data availability & cache stats.

    Adds row counts (best-effort) and cache metrics for /props/all HTML cache entries.
    """
    d = _normalize_date_param(date) if date else _today_ymd()
    proj_path = PROC_DIR / f"props_projections_all_{d}.csv"
    rec_path = PROC_DIR / f"props_recommendations_{d}.csv"
    def _mtime(p: Path):
        try:
            if p.exists():
                import datetime as _dt
                return _dt.datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        except Exception:
            return None
        return None
    def _rows(p: Path):
        try:
            if p.exists():
                df = _read_csv_fallback(p)
                if df is not None and not df.empty:
                    return int(len(df))
        except Exception:
            return None
        return 0 if p.exists() else None
    # Cache stats (only entries for props HTML)
    try:
        cache_keys = [k for k in _CACHE.keys() if isinstance(k, tuple) and k and k[0] == 'props_all_html']
        cache_entries = len(cache_keys)
    except Exception:
        cache_entries = None
        cache_keys = []
    return JSONResponse({
        "date": d,
        "projections_all_present": proj_path.exists(),
        "recommendations_present": rec_path.exists(),
        "projections_all_mtime": _mtime(proj_path),
        "recommendations_mtime": _mtime(rec_path),
        "projections_all_rows": _rows(proj_path),
        "recommendations_rows": _rows(rec_path),
        "projections_all_synthetic_like": (lambda: (lambda _df: (_looks_like_synthetic_props(_df) if _df is not None and not _df.empty else False))(_read_csv_fallback(proj_path)))(),
        # Lines presence for the date
    "lines_present": (lambda: ( (PROC_DIR.parent/"props"/f"player_props_lines/date={d}").exists() and (PROC_DIR.parent/"props"/f"player_props_lines/date={d}/oddsapi.parquet").exists() ))(),
    "lines_books": (lambda: [name for name,path in ( ("oddsapi", PROC_DIR.parent/"props"/f"player_props_lines/date={d}/oddsapi.parquet"), ) if path.exists() ])(),
        "fast_mode": os.getenv('FAST_PROPS_TEST','0') == '1',
        "force_synthetic": os.getenv('PROPS_FORCE_SYNTHETIC','0') == '1',
        "no_compute": os.getenv('PROPS_NO_COMPUTE','0') == '1',
        "commit": _git_commit_hash(),
        "cache_entries": cache_entries,
        "cache_ttl_sec": _CACHE_TTL,
    })


@app.get("/api/cron/overview")
async def api_cron_overview(date: Optional[str] = Query(None), window: int = Query(1)):
    """Summarize artifacts for date, previous day, and next day plus last cron jobs.

    window controls how many neighbor days to include on each side (default 1 => D-1, D, D+1).
    """
    d = _normalize_date_param(date)
    try:
        base = datetime.strptime(d, "%Y-%m-%d")
    except Exception:
        base = datetime.strptime(_today_ymd(), "%Y-%m-%d")
    days = []
    try:
        w = int(window) if window is not None else 1
    except Exception:
        w = 1
    for off in range(-w, w + 1):
        days.append((base + timedelta(days=off)).strftime("%Y-%m-%d"))
    artifacts = {di: _artifact_info_for_date(di) for di in days}
    # Sample last N jobs
    try:
        with _CRON_LOCK:
            jobs = list(_CRON_JOBS.values())
        jobs.sort(key=lambda r: r.get('updated_at',''), reverse=True)
        jobs = jobs[:50]
    except Exception:
        jobs = []
    return JSONResponse({"date": d, "days": days, "artifacts": artifacts, "jobs": jobs, "commit": _git_commit_hash()})


@app.get("/cron", include_in_schema=False)
async def cron_dashboard(date: Optional[str] = Query(None), window: int = Query(1)):
    """HTML dashboard summarizing artifacts and recent cron runs for quick verification."""
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)

@app.get('/api/version')
async def api_version():
    """Version & build diagnostics.

    Includes short/long commit, route count, uptime, and timestamp.
    """
    commit_full = _git_commit_hash()
    short = (commit_full or '')[:12] if commit_full else None
    try:
        uptime = round(time.time() - START_TIME, 2)
    except Exception:
        uptime = None
    return {
        "commit": commit_full,
        "commit_short": short,
        "routes": len(app.routes),
        "uptime_seconds": uptime,
        "generated_at": datetime.utcnow().isoformat(),
    }

@app.get('/api/routes')
async def api_routes():
    """List registered route paths & names (deprecated: prefer /diag/info)."""
    out = []
    try:
        for r in app.routes:
            try:
                out.append({
                    'path': getattr(r, 'path', None),
                    'name': getattr(r, 'name', None),
                    'methods': sorted(list(getattr(r, 'methods', []) or [])),
                })
            except Exception:
                pass
    except Exception:
        out = []
    commit_val = (_git_commit_hash() or '')[:12]
    return {"commit": commit_val, "count": len(out), "routes": out[:200], "deprecated": True}

@app.get("/props/all", include_in_schema=False)
async def props_all_players_page(
    request: Request,
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    game: Optional[str] = Query(None, description="Filter by game as AWY@HOME (team abbreviations)"),
    team: Optional[str] = Query(None, description="Filter by team abbreviation"),
    market: Optional[str] = Query(None, description="Filter by market"),
    sort: Optional[str] = Query("name", description="Sort by: name, team, market, lambda_desc, lambda_asc"),
    top: int = Query(2000, description="Max rows to display"),
    min_ev: float = Query(0.0, description="Minimum EV filter (over side)"),
    nocache: int = Query(0, description="Bypass in-memory cache (1 = yes)"),
    page: int = Query(1, description="Page number (1-based)"),
    page_size: Optional[int] = Query(None, description="Rows per page (server-side pagination); defaults to PROPS_PAGE_SIZE env or 250"),
    source: Optional[str] = Query(None, description="Data source: merged (default) or recs for recommendations only"),
):
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)

@app.get('/api/props/all.json')
async def api_props_all_players(
    request: Request,
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    team: Optional[str] = Query(None, description="Filter by team abbreviation"),
    market: Optional[str] = Query(None, description="Filter by market"),
    sort: Optional[str] = Query("name", description="Sort by: name, team, market, lambda_desc, lambda_asc"),
    page: int = Query(1, description="Page number (1-based)"),
    page_size: Optional[int] = Query(None, description="Rows per page (defaults PROPS_PAGE_SIZE env or 250)"),
    top: int = Query(0, description="Optional max rows before pagination (0 = no cap aside from PROPS_MAX_ROWS)"),
):
    graceful = os.getenv('PROPS_GRACEFUL_ERRORS','1') != '0'
    try:
        return await _api_props_all_players_impl(request, date, team, market, sort, page, page_size, top)
    except Exception as e:
        import traceback, json as _json
        tb = traceback.format_exc()
        try:
            print(_json.dumps({"event":"api_props_all_error","error":str(e)}))
        except Exception:
            pass
        if not graceful:
            raise
        return JSONResponse({"error":"props_api_failed","detail":str(e),"trace":tb[:4000]}, status_code=200)

async def _api_props_all_players_impl(request: Request, date, team, market, sort, page, page_size, top):
    """JSON API for all-player model-only projections with server-side pagination.

    Returns metadata: total_rows (raw), filtered_rows (after filters & top/env cap), page, page_size, total_pages.
    """
    d = _normalize_date_param(date)
    try:
        if os.getenv('FAST_PROPS_TEST','0') == '1':
            default_ps = 10
        else:
            default_ps = int(os.getenv('PROPS_PAGE_SIZE', '250'))
    except Exception:
        default_ps = 250
    if not page_size or page_size <= 0:
        page_size = default_ps
    if page <= 0:
        page = 1
    df = _read_all_players_projections(d)
    src_path = PROC_DIR / f"props_projections_all_{d}.csv"
    if (df is None or df.empty):
        # On public hosts, do NOT compute on-demand to avoid cold-start timeouts; serve empty.
        if _is_public_host_env():
            try:
                # Also attempt one more GitHub raw read in case of transient
                df = _github_raw_read_csv(f"data/processed/props_projections_all_{d}.csv")
            except Exception:
                df = df
        if df is None or df.empty:
            if _is_public_host_env():
                return JSONResponse({"date": d, "data": [], "total_rows": 0, "filtered_rows": 0, "page": 1, "page_size": page_size, "total_pages": 0})
            # Local/dev: compute and backfill cache file
            df = _compute_all_players_projections(d)
            try:
                if df is not None and not df.empty and not src_path.exists():
                    save_df(df, src_path)
            except Exception:
                pass
    total_rows = 0 if df is None or df.empty else len(df)
    if df is None or df.empty:
        return JSONResponse({"date": d, "data": [], "total_rows": 0, "filtered_rows": 0, "page": 1, "page_size": page_size, "total_pages": 0})
    # Filters
    try:
        df['player'] = df['player'].astype(str).map(_clean_player_display_name)
    except Exception:
        pass
    if team:
        try:
            df = df[df['team'].astype(str).str.upper() == str(team).upper()]
        except Exception:
            pass
    if market:
        try:
            df = df[df['market'].astype(str).str.upper() == str(market).upper()]
        except Exception:
            pass
    key = (sort or 'name').lower(); ascending = True
    if key in ('lambda_desc','lambda_asc'):
        col = 'proj_lambda'; ascending = (key == 'lambda_asc')
    elif key == 'market':
        col = 'market'
    elif key == 'team':
        col = 'team'
    else:
        col = 'player'
    if col in df.columns:
        try:
            df = df.sort_values(by=[col], ascending=ascending, na_position='last')
        except Exception:
            pass
    # Top cap (pre-pagination) + env cap
    try:
        env_cap = int(os.getenv('PROPS_MAX_ROWS', '0'))
    except Exception:
        env_cap = 0
    effective_top = int(top) if (top and top > 0) else None
    if env_cap and (effective_top is None or env_cap < effective_top):
        effective_top = env_cap
    if effective_top:
        df = df.head(effective_top)
    filtered_rows = len(df)
    total_pages = max(1, (filtered_rows + page_size - 1) // page_size) if filtered_rows else 0
    if page > total_pages and total_pages > 0:
        page = total_pages
    if total_pages == 0:
        page = 1
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    df_page = df.iloc[start_idx:end_idx]
    rows = _df_jsonsafe_records(df_page)
    # ETag / Last-Modified support for lightweight polling
    try:
        import hashlib, os as _os, email.utils as eut, time as _time
        file_mtime = None
        if src_path.exists():
            try:
                file_mtime = int(src_path.stat().st_mtime)
            except Exception:
                file_mtime = None
        etag_basis = f"{d}|{team}|{market}|{sort}|{page}|{page_size}|{total_rows}|{filtered_rows}|{effective_top or ''}|{file_mtime or ''}".encode('utf-8')
        etag = hashlib.md5(etag_basis).hexdigest()  # nosec B324 (non-cryptographic, fine for cache)
        inm = request.headers.get('if-none-match')
        if inm and inm.strip('"') == etag:
            # Not modified
            headers = {"ETag": f'"{etag}"'}
            if file_mtime:
                headers["Last-Modified"] = eut.formatdate(file_mtime, usegmt=True)
            headers["Cache-Control"] = "public, max-age=60"
            return Response(status_code=304, headers=headers)
        payload = {
            "date": d,
            "data": rows,
            "total_rows": total_rows,
            "filtered_rows": filtered_rows,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "generated_at": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        }
        body = json.dumps(payload, ensure_ascii=False)
        headers = {"ETag": f'"{etag}"', "Cache-Control": "public, max-age=60"}
        if file_mtime:
            import email.utils as _eut
            headers["Last-Modified"] = _eut.formatdate(file_mtime, usegmt=True)
        return Response(content=body, media_type="application/json", headers=headers)
    except Exception:
        return JSONResponse({
            "date": d,
            "data": rows,
            "total_rows": total_rows,
            "filtered_rows": filtered_rows,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
        })

@app.get('/api/props/recommendations/history.json')
async def api_props_recommendations_history_json(
    date: Optional[str] = Query(None, description="Anchor date (inclusive); defaults to today"),
    days: int = Query(30, description="Lookback window in days"),
    market: Optional[str] = Query(None),
    player: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    limit: int = Query(1000, description="Max rows to return after filtering"),
):
    d = _normalize_date_param(date)
    try:
        base_date = datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return JSONResponse({"error": "bad date"}, status_code=400)

    hist_path = PROC_DIR / "props_recommendations_history.csv"
    if not hist_path.exists():
        return JSONResponse({"date": d, "data": [], "total_rows": 0})

    try:
        df = pd.read_csv(hist_path)
    except Exception:
        return JSONResponse({"date": d, "data": [], "total_rows": 0})

    if df is None or df.empty:
        return JSONResponse({"date": d, "data": [], "total_rows": 0})

    # Normalize columns (historical files have had a few schema iterations)
    if "proj" in df.columns and "proj_lambda" not in df.columns:
        try:
            df = df.rename(columns={"proj": "proj_lambda"})
        except Exception:
            pass
    if "ev" in df.columns and "ev_over" not in df.columns:
        try:
            df = df.rename(columns={"ev": "ev_over"})
        except Exception:
            pass

    # Date window filter (inclusive)
    lookback_days = max(0, int(days))
    start_date = base_date - timedelta(days=lookback_days)
    end_date = base_date
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
            df = df[df["date"].notna()]
            df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]
        except Exception:
            pass

    # Filters
    market_u = str(market or "").strip().upper()
    if market_u and market_u not in ("ALL", "ALL MARKETS", "ANY") and "market" in df.columns:
        try:
            df = df[df["market"].astype(str).str.upper() == market_u]
        except Exception:
            pass
    if player and "player" in df.columns:
        try:
            df = df[df["player"].astype(str).str.casefold() == player.casefold()]
        except Exception:
            pass
    if team and "team" in df.columns:
        try:
            df = df[df["team"].astype(str).str.upper() == team.upper()]
        except Exception:
            pass

    total_rows = int(len(df))
    if total_rows == 0:
        return JSONResponse({
            "date": d,
            "lookback_days": lookback_days,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "data": [],
            "total_rows": 0,
        })

    # Default sort: newest date then EV desc
    try:
        sort_cols = []
        ascending = []
        if "date" in df.columns:
            sort_cols.append("date")
            ascending.append(False)
        ev_col = "ev_over" if "ev_over" in df.columns else ("ev" if "ev" in df.columns else None)
        if ev_col:
            sort_cols.append(ev_col)
            ascending.append(False)
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=ascending)
    except Exception:
        pass

    if limit and limit > 0:
        df = df.head(int(limit))

    rows = _df_jsonsafe_records(df)
    return JSONResponse({
        "date": d,
        "lookback_days": lookback_days,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "data": rows,
        "total_rows": total_rows,
    })

@app.get('/api/env-flags')
async def api_env_flags():
    """Diagnostic, non-sensitive environment feature flags."""
    flags = {
        'WEB_READ_ONLY_PREDICTIONS': bool(os.getenv('WEB_READ_ONLY_PREDICTIONS')),
        'WEB_DISABLE_ODDS_FETCH': bool(os.getenv('WEB_DISABLE_ODDS_FETCH')),
        'CACHED_PROPS_TTL_SECONDS': _CACHE_TTL,
    }
    return JSONResponse({'flags': flags})


@app.get('/api/oddsapi/status')
async def api_oddsapi_status():
    """Validate OddsAPI configuration on the running host (no secrets returned).

    This endpoint makes a small, real request to The Odds API to confirm the
    ODDS_API_KEY is present and accepted (e.g., not missing/expired).
    """
    configured = bool(str(os.getenv('ODDS_API_KEY', '')).strip())
    if not configured:
        return JSONResponse({'configured': False, 'ok': False, 'error': 'missing_odds_api_key'})
    try:
        client = OddsAPIClient()
        events, hdrs = client.list_events('icehockey_nhl')
        hdrs = hdrs or {}
        return JSONResponse({
            'configured': True,
            'ok': True,
            'event_count': int(len(events or [])),
            'requests': {
                'remaining': hdrs.get('x-requests-remaining'),
                'used': hdrs.get('x-requests-used'),
                'last': hdrs.get('x-requests-last'),
            },
        })
    except Exception as e:
        return JSONResponse({'configured': True, 'ok': False, 'error': str(e)})

@app.get('/api/ping')
def api_ping():
    """Ultra-fast liveness check (lighter than /health)."""
    return {"pong": True}

@app.get('/api/diag/perf')
def api_diag_perf():
    """Runtime diagnostics: uptime, cache stats, selected env flags.

    Avoids heavy data loads; safe to call frequently.
    """
    now = time.time()
    uptime = now - START_TIME
    try:
        cache_entries = len(_CACHE)
    except Exception:
        cache_entries = None
    flags = {
        'WEB_READ_ONLY_PREDICTIONS': bool(os.getenv('WEB_READ_ONLY_PREDICTIONS')),
        'WEB_DISABLE_ODDS_FETCH': bool(os.getenv('WEB_DISABLE_ODDS_FETCH')),
        'PROPS_MAX_ROWS': os.getenv('PROPS_MAX_ROWS'),
        'CACHED_PROPS_TTL_SECONDS': _CACHE_TTL,
    }
    return {
        'status': 'ok',
        'uptime_seconds': round(uptime, 2),
        'cache_entries': cache_entries,
        'env': flags,
    }


@app.get("/props/all.csv", include_in_schema=False)
async def props_all_players_csv(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
    min_ev: float = Query(0.0, description="Minimum EV (ev column) to include"),
):
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)


@app.post("/api/cron/props-all")
async def cron_props_all(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD; defaults to ET today"),
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    authorization: Optional[str] = Header(default=None, description="Authorization: Bearer <token> header (alternative to token query param)"),
    async_run: bool = Query(False, description="If true, queue work in background and return 202 immediately"),
):
    d = _normalize_date_param(date)
    # Auth: align with other cron endpoints using REFRESH_CRON_TOKEN
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
        return PlainTextResponse("Unauthorized", status_code=401)
    def _compute_all_and_write(_d: str) -> Dict[str, Any]:
        df = _compute_all_players_projections(_d)
        out_path = PROC_DIR / f"props_projections_all_{_d}.csv"
        if df is None or df.empty:
            save_df(pd.DataFrame(), out_path)
            return {"rows": 0, "github": None}
        save_df(df, out_path)
        try:
            res_local = _gh_upsert_file_if_better_or_same(out_path, f"web: update props projections ALL for {_d}")
        except Exception:
            res_local = None
        return {"rows": int(len(df)), "github": res_local}
    try:
        if async_run:
            # Run with a max duration budget; if it exceeds, write an empty CSV so health can reflect presence
            def _run_all_with_timeout():
                timeout_s = 0
                try:
                    timeout_s = int(os.getenv('PROPS_ALL_TIMEOUT_SEC', '120'))
                except Exception:
                    timeout_s = 120
                res_holder: Dict[str, Any] = {}
                err_holder: Dict[str, Any] = {}
                def _inner():
                    try:
                        res_holder['res'] = _compute_all_and_write(d)
                    except Exception as e:
                        err_holder['err'] = str(e)
                th = threading.Thread(target=_inner, daemon=True)
                th.start()
                th.join(timeout=timeout_s)
                if th.is_alive():
                    # Timed out; write empty to mark presence and return
                    try:
                        out_path = PROC_DIR / f"props_projections_all_{d}.csv"
                        save_df(pd.DataFrame(), out_path)
                    except Exception:
                        pass
                    return {"rows": 0, "github": None, "message": "timeout"}
                if 'err' in err_holder:
                    raise Exception(err_holder['err'])
                return res_holder.get('res', {"rows": 0, "github": None, "message": "no-result"})
            job_id = _queue_cron('props-all', {'date': d}, _run_all_with_timeout)
            return JSONResponse({"ok": True, "date": d, "queued": True, "mode": "async", "job_id": job_id}, status_code=202)
        res = _compute_all_and_write(d)
        return JSONResponse({"ok": True, "date": d, **res})
    except Exception as e:
        return JSONResponse({"ok": False, "date": d, "error": str(e)}, status_code=500)

@app.get("/props/players", include_in_schema=False)
async def props_players_page(
    market: Optional[str] = Query(None, description="Filter by market: SOG, SAVES, GOALS, ASSISTS, POINTS"),
    min_ev: float = Query(0.0, description="Minimum EV threshold for ev_over"),
    top: int = Query(50, description="Top N to display"),
):
    """Player props table (moved from /props)."""
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)

@app.get("/props/teams", include_in_schema=False)
async def props_teams_page(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
):
    """Team-level projections grid for the slate (moved under /props/teams)."""
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)


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
    """Refresh odds/predictions for a date using The Odds API (no Bovada), then recompute recommendations.

    Ensures predictions CSV exists; runs predict_core with odds_source=oddsapi; then recomputes edges/recs.
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
    # Step: Use Odds API (prefer a single bookmaker for stability)
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


@app.get("/api/recompute-only")
async def api_recompute_only(
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD (ET)"),
):
    """Recompute EV/edges/recommendations from existing predictions without fetching odds or running models.

    - Reads predictions_{date}.csv (must already exist locally or be present via GitHub cache in UI paths)
    - Recomputes EV and edges from current odds/close_* columns
    - Regenerates recommendations_{date}.csv
    """
    d = date or _today_ymd()
    try:
        await _recompute_edges_and_recommendations(d)
        return JSONResponse({"status": "ok", "date": d})
    except Exception as e:
        return JSONResponse({"status": "error", "date": d, "error": str(e)}, status_code=500)


def _inject_bovada_odds_into_predictions(date: str, backfill: bool = False, skip_started: bool = True) -> Dict[str, Any]:
    """Deprecated: Bovada odds injection removed (use _inject_oddsapi_odds_into_predictions instead)."""
    return {"status": "removed", "date": date}


def _inject_oddsapi_odds_into_predictions(date: str, backfill: bool = False, skip_started: bool = True, bookmaker: str = "draftkings") -> Dict[str, Any]:
    """Fetch The Odds API odds and inject into predictions_{date}.csv without running models.

    Uses current event odds for markets h2h, totals, and spreads. Prefer a single bookmaker (default DraftKings)
    for stability; could be extended to best-of-all. Returns a small summary with counts.
    """
    pred_path = PROC_DIR / f"predictions_{date}.csv"
    if not pred_path.exists():
        return {"status": "no-predictions", "date": date}
    df = _read_csv_fallback(pred_path)
    if df is None or df.empty:
        return {"status": "empty", "date": date}
    try:
        client = OddsAPIClient()
    except Exception as e:
        return {"status": "no-oddsapi", "date": date, "error": str(e)}

    import re, unicodedata
    def norm_team(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.lower()
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s

    from_zone = ZoneInfo("America/New_York")
    # Compute UTC window for the slate date in ET
    try:
        d0_et = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=from_zone)
        d1_et = d0_et + timedelta(days=1)
        start_iso = d0_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_iso = d1_et.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        start_iso = end_iso = None

    # List current events and fetch odds for each
    try:
        events, _ = client.list_events("icehockey_nhl", commence_from_iso=start_iso, commence_to_iso=end_iso)
    except Exception:
        events = []

    records = []
    for ev in events or []:
        try:
            eid = str(ev.get("id"))
            # Filter to ET date match just in case
            commence = ev.get("commence_time")
            dkey = None
            try:
                dt_utc = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
                dkey = dt_utc.astimezone(from_zone).strftime("%Y-%m-%d")
            except Exception:
                pass
            if dkey and dkey != date:
                continue
            data, _ = client.event_odds(
                sport="icehockey_nhl",
                event_id=eid,
                markets="h2h,totals,spreads",
                regions="us",
                bookmakers=bookmaker,
                odds_format="american",
            )
            bks = data.get("bookmakers", []) if isinstance(data, dict) else []
            if not bks:
                continue
            # Pick the requested bookmaker entry if present
            book = next((b for b in bks if b.get("key") == bookmaker), bks[0])
            markets = book.get("markets", [])
            # Extract prices similar to data.odds_api._extract_prices_from_markets
            def _extract_prices(markets):
                out = {}
                m_h2h = next((m for m in markets if m.get("key") == "h2h"), None)
                if m_h2h:
                    for oc in m_h2h.get("outcomes", []):
                        nm = str(oc.get("name"))
                        out[f"ml::{nm}"] = oc.get("price")
                m_tot = next((m for m in markets if m.get("key") == "totals"), None)
                if m_tot:
                    pts = None
                    for oc in m_tot.get("outcomes", []):
                        if oc.get("name") in ("Over", "Under"):
                            if pts is None:
                                pts = oc.get("point")
                            out[f"tot::{oc.get('name')}"] = oc.get("price")
                            out["tot::point"] = pts
                m_spr = next((m for m in markets if m.get("key") == "spreads"), None)
                if m_spr:
                    for oc in m_spr.get("outcomes", []):
                        try:
                            pt = float(oc.get("point"))
                        except Exception:
                            continue
                        if abs(pt) == 1.5:
                            out[f"pl::{oc.get('name')}::{pt}"] = oc.get("price")
                return out
            prices = _extract_prices(markets)
            row = {
                "home": ev.get("home_team"),
                "away": ev.get("away_team"),
                "home_ml": prices.get(f"ml::{ev.get('home_team')}") ,
                "away_ml": prices.get(f"ml::{ev.get('away_team')}") ,
                "over": prices.get("tot::Over"),
                "under": prices.get("tot::Under"),
                "total_line": prices.get("tot::point"),
                "home_pl_-1.5": prices.get(f"pl::{ev.get('home_team')}::-1.5"),
                "away_pl_+1.5": prices.get(f"pl::{ev.get('away_team')}::1.5"),
                "home_ml_book": book.get("key"),
                "away_ml_book": book.get("key"),
                "over_book": book.get("key"),
                "under_book": book.get("key"),
                "home_pl_-1.5_book": book.get("key"),
                "away_pl_+1.5_book": book.get("key"),
            }
            records.append(row)
        except Exception:
            continue
    if not records:
        return {"status": "no-odds", "date": date}

    odds = pd.DataFrame.from_records(records)
    odds["home_norm"] = odds["home"].apply(norm_team)
    odds["away_norm"] = odds["away"].apply(norm_team)

    # Prepare predictions frame for matching and update
    updated_rows = 0
    updated_fields = 0
    df = df.copy()
    df["home_norm"] = df["home"].apply(norm_team)
    df["away_norm"] = df["away"].apply(norm_team)

    # Optionally skip started games (based on date only here; deeper start-time aware logic omitted)
    # We keep simple date gating; predictions typically align to slate date.

    for idx, r in df.iterrows():
        m = odds[(odds["home_norm"] == r.get("home_norm")) & (odds["away_norm"] == r.get("away_norm"))]
        if m.empty:
            # try reversed
            m = odds[(odds["home_norm"] == r.get("away_norm")) & (odds["away_norm"] == r.get("home_norm"))]
        if m.empty:
            continue
        o = m.iloc[0]
        before = updated_fields
        def set_val(dst, val):
            nonlocal updated_fields
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return
            cur = df.at[idx, dst] if dst in df.columns else None
            if backfill:
                if cur is None or (isinstance(cur, float) and pd.isna(cur)):
                    df.at[idx, dst] = val
                    updated_fields += 1
            else:
                if str(cur) != str(val):
                    df.at[idx, dst] = val
                    updated_fields += 1
        for col, val in [
            ("home_ml_odds", o.get("home_ml")),
            ("away_ml_odds", o.get("away_ml")),
            ("over_odds", o.get("over")),
            ("under_odds", o.get("under")),
            ("total_line_used", o.get("total_line")),
            ("home_pl_-1.5_odds", o.get("home_pl_-1.5")),
            ("away_pl_+1.5_odds", o.get("away_pl_+1.5")),
            ("home_ml_book", o.get("home_ml_book")),
            ("away_ml_book", o.get("away_ml_book")),
            ("over_book", o.get("over_book")),
            ("under_book", o.get("under_book")),
            ("home_pl_-1.5_book", o.get("home_pl_-1.5_book")),
            ("away_pl_+1.5_book", o.get("away_pl_+1.5_book")),
        ]:
            set_val(col, val)
        if updated_fields > before:
            updated_rows += 1
    # Persist if any updates
    if updated_fields > 0:
        df.to_csv(pred_path, index=False)
        try:
            _gh_upsert_file_if_configured(pred_path, f"web: update predictions with fresh OddsAPI odds for {date}")
        except Exception:
            pass
    return {"status": "ok", "date": date, "updated_rows": int(updated_rows), "updated_fields": int(updated_fields)}


@app.get("/api/refresh-odds-light")
async def api_refresh_odds_light(
    date: Optional[str] = Query(None),
    backfill: bool = Query(False, description="If true, only fill missing odds; do not overwrite existing prices"),
    overwrite_prestart: bool = Query(False, description="If true, allow refresh even during live slates"),
):
    """Lightweight odds refresh that DOES NOT run models.

    - Ensures predictions_{date}.csv exists locally (uses GitHub fallback if available)
    - Fetches The Odds API odds and injects into predictions file
    - Recomputes EV/edges/recommendations from existing model probabilities ONLY if odds changed
    """
    d = date or _today_ymd()
    # Skip refresh during live slates unless explicitly allowed
    try:
        if _is_live_day(d) and not overwrite_prestart:
            return JSONResponse({"status": "skipped-live", "date": d, "message": "Live games in progress; light odds refresh skipped."}, status_code=200)
    except Exception:
        pass
    summary = _inject_oddsapi_odds_into_predictions(d, backfill=backfill)
    # Recompute only when odds actually changed
    try:
        if isinstance(summary, dict) and int(summary.get("updated_fields") or 0) > 0:
            await _recompute_edges_and_recommendations(d)
    except Exception:
        pass
    return JSONResponse({"status": "ok", **(summary or { }), "date": d})


# ---------------- Lightweight props recommendations refresh (read-only compute) -----------------
def _refresh_props_recommendations(date: str, min_ev: float = 0.0, top: int = 200) -> dict:
    """Recompute props_recommendations_{date}.csv from canonical lines + precomputed lambdas.

    - Reads data/props/player_props_lines/date=YYYY-MM-DD/oddsapi.parquet (local first; GH raw fallback)
    - Reads data/processed/props_projections_all_{date}.csv (local first; GH raw fallback)
    - Computes EV vectorized (no history scans), writes CSV, upserts to GitHub
    """
    import numpy as _np
    from scipy.stats import poisson as _poisson
    d = _normalize_date_param(date)
    # Load canonical lines (ONLY OddsAPI), prefer local; GH fallback allowed in cron
    base = PROC_DIR.parent / "props" / f"player_props_lines/date={d}"
    parts = []
    local_line_files = []
    for fname in ("oddsapi.parquet", "bovada.parquet"):
        p = base / fname
        if p.exists():
            local_line_files.append(p)
            try:
                parts.append(pd.read_parquet(p, engine="pyarrow"))
            except Exception:
                pass
        else:
            try:
                ghp = _github_raw_read_parquet(f"data/props/player_props_lines/date={d}/{fname}")
                if ghp is not None and not ghp.empty:
                    parts.append(ghp)
            except Exception:
                pass
    if not parts:
        return {"ok": False, "date": d, "reason": "no_lines"}
    lines = pd.concat(parts, ignore_index=True)
    # Change detection: if local recommendations exist and line files are not newer, skip recompute
    try:
        rec_path = PROC_DIR / f"props_recommendations_{d}.csv"
        if local_line_files and rec_path.exists():
            import os as _os
            mx = max(_os.path.getmtime(str(p)) for p in local_line_files)
            rec_m = _os.path.getmtime(str(rec_path))
            if rec_m >= mx:
                try:
                    old = _read_csv_fallback(rec_path)
                    rows = int(len(old)) if old is not None and not old.empty else 0
                except Exception:
                    rows = 0
                return {"ok": True, "date": d, "rows": rows, "skipped": True, "reason": "unchanged-lines"}
    except Exception:
        pass
    # Load projections_all
    proj = None
    try:
        local = PROC_DIR / f"props_projections_all_{d}.csv"
        if local.exists():
            proj = _read_csv_fallback(local)
        if (proj is None or proj.empty):
            proj = _github_raw_read_csv(f"data/processed/props_projections_all_{d}.csv")
    except Exception:
        proj = proj
    if proj is None or proj.empty or not {"player","market","proj_lambda"}.issubset(set(proj.columns)):
        return {"ok": False, "date": d, "reason": "no_proj_all"}
    # Lambda map
    def _norm_name(x: str) -> str:
        try:
            s = str(x or "").strip(); return " ".join(s.split())
        except Exception:
            return str(x)
    lam_map = {}
    tmp = proj.dropna(subset=["player","market","proj_lambda"]).copy()
    tmp["player_norm"] = tmp["player"].astype(str).map(_norm_name).str.lower()
    tmp["market_u"] = tmp["market"].astype(str).str.upper()
    for _, rr in tmp.iterrows():
        try:
            lam_map[(rr.get("player_norm"), rr.get("market_u"))] = float(rr.get("proj_lambda"))
        except Exception:
            continue
    # Prepare working frame
    cols = [c for c in ["market","player_name","player","team","line","over_price","under_price","book"] if c in lines.columns]
    work = lines[cols].copy()
    work["market"] = work.get("market").astype(str).str.upper()
    work["player_display"] = work.apply(lambda r: (r.get("player_name") or r.get("player") or ""), axis=1).astype(str).map(_norm_name)
    work["player_norm"] = work["player_display"].str.lower()
    work["line_num"] = pd.to_numeric(work.get("line"), errors="coerce")
    work = work.loc[work["line_num"].notna()].copy()
    # Merge lambdas
    ldf = pd.DataFrame([{"player_norm": k[0], "market": k[1], "proj_lambda": v} for k, v in lam_map.items()])
    merged = work.merge(ldf, on=["player_norm","market"], how="left")
    vec_mask = merged["proj_lambda"].notna()
    p_over_vec = pd.Series(_np.nan, index=merged.index)
    for mkt in ["SOG","SAVES","GOALS","ASSISTS","POINTS","BLOCKS"]:
        sel = vec_mask & (merged["market"] == mkt)
        if sel.any():
            lam_arr = merged.loc[sel, "proj_lambda"].astype(float).values
            line_arr = _np.floor(merged.loc[sel, "line_num"].astype(float).values + 1e-9).astype(int)
            p_over_vec.loc[sel] = _poisson.sf(line_arr, mu=lam_arr)
    def _american_to_decimal_series(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce"); pos = s[s > 0]; neg = s[s <= 0]
        out = pd.Series(_np.nan, index=s.index)
        out.loc[pos.index] = 1.0 + (pos / 100.0)
        out.loc[neg.index] = 1.0 + (100.0 / _np.abs(neg))
        return out
    dec_over = _american_to_decimal_series(merged.get("over_price"))
    dec_under = _american_to_decimal_series(merged.get("under_price"))
    p_over_s = pd.to_numeric(p_over_vec, errors="coerce")
    ev_over_s = p_over_s * (dec_over - 1.0) - (1.0 - p_over_s)
    p_under_s = (1.0 - p_over_s).clip(lower=0.0, upper=1.0)
    ev_under_s = p_under_s * (dec_under - 1.0) - (1.0 - p_under_s)
    over_better = (ev_under_s.isna()) | (~ev_over_s.isna() & (ev_over_s >= ev_under_s))
    chosen_side = _np.where(over_better, "Over", "Under")
    chosen_ev = _np.where(over_better, ev_over_s, ev_under_s)
    out = merged.copy()
    out["side"] = chosen_side
    out["ev"] = pd.to_numeric(chosen_ev, errors="coerce")
    out = out[out["ev"].notna() & (out["ev"].astype(float) >= float(min_ev))]
    out = out.assign(
        date=d,
        player=lambda df: df["player_display"],
        market=lambda df: df["market"],
        line=lambda df: df["line_num"],
        proj=lambda df: df["proj_lambda"].astype(float).round(3),
        p_over=lambda df: p_over_s.astype(float).round(4),
        over_price=lambda df: df.get("over_price"),
        under_price=lambda df: df.get("under_price"),
        book=lambda df: df.get("book"),
        team=lambda df: df.get("team"),
    )[["date","player","team","market","line","proj","p_over","over_price","under_price","book","side","ev"]]
    if not out.empty:
        out = out.sort_values("ev", ascending=False).head(int(top))
        if "proj" in out.columns and "proj_lambda" not in out.columns:
            out["proj_lambda"] = out["proj"]
        if "ev" in out.columns and "ev_over" not in out.columns:
            out["ev_over"] = out["ev"]
    path = PROC_DIR / f"props_recommendations_{d}.csv"
    save_df(out, path)
    try:
        _gh_upsert_file_if_better_or_same(path, f"web: update props_recommendations for {d}")
    except Exception:
        pass
    return {"ok": True, "date": d, "rows": int(len(out))}


@app.post("/api/cron/props-recs-refresh")
async def api_cron_props_recs_refresh(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD; defaults to ET today"),
    min_ev: float = Query(0.0),
    top: int = Query(200),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional alternative)"),
):
    d = _normalize_date_param(date)
    try:
        want = os.getenv("REFRESH_CRON_TOKEN", "").strip()
        got = (authorization or "").replace("Bearer ", "").strip() or (token or "").strip()
        if want and (got != want):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    except Exception:
        pass
    res = _refresh_props_recommendations(d, min_ev=min_ev, top=top)
    return JSONResponse(res)


@app.post("/api/cron/light-refresh")
async def api_cron_light_refresh(
    token: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    min_ev: float = Query(0.0),
    top: int = Query(200),
    do_edges: int = Query(1, description="Also recompute team edges if 1 (default 1)"),
    authorization: Optional[str] = Header(None),
):
    d = _normalize_date_param(date)
    try:
        want = os.getenv("REFRESH_CRON_TOKEN", "").strip()
        got = (authorization or "").replace("Bearer ", "").strip() or (token or "").strip()
        if want and (got != want):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    except Exception:
        pass
    out = {"date": d}
    # Update team odds from OddsAPI into predictions (best-effort)
    try:
        out["odds"] = _inject_oddsapi_odds_into_predictions(d, backfill=True, skip_started=True)
    except Exception as e:
        out["odds"] = {"status": "error", "error": str(e)}
    # Recompute edges/recommendations for team markets only if odds changed and requested
    if int(do_edges) == 1:
        try:
            if isinstance(out.get("odds"), dict) and int(out["odds"].get("updated_fields") or 0) > 0:
                await _recompute_edges_and_recommendations(d)
                out["edges"] = {"ok": True}
            else:
                out["edges"] = {"ok": True, "skipped": True, "reason": "unchanged-odds"}
        except Exception as e:
            out["edges"] = {"ok": False, "error": str(e)}
    # Refresh player props recommendations from canonical lines
    out["props"] = _refresh_props_recommendations(d, min_ev=min_ev, top=top)
    return JSONResponse({"ok": True, **out})

@app.get("/props/recommendations", include_in_schema=False)
async def props_recommendations_page(
    request: Request,  # Add Request object to inspect raw request
    date: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    min_ev: float = Query(0.0),
    # Raise default top to 500 so more players appear by default
    top: int = Query(500),
    sortBy: Optional[str] = Query("ev_desc"),
    side: Optional[str] = Query("both"),
    team: Optional[str] = Query(None, description="Filter by team abbreviation for cards"),
    game: Optional[str] = Query(None, description="Filter by game as AWY@HOME for cards"),
    # When all=1 we bypass any top slicing
    all: Optional[int] = Query(0),
    debug: Optional[int] = Query(0, description="If 1, include debug comment with photo/team mapping counts"),
    # New: grid/table view toggle (NBA-parity)
    view: Optional[str] = Query(None, description="If 'grid', render a table view with pagination instead of cards"),
    # Grid-only controls (kept optional):
    sort: Optional[str] = Query(None, description="Grid sort: ev_desc, ev_asc, p_over_desc, p_over_asc, lambda_desc, lambda_asc, name, team, market, line, book"),
    page: int = Query(1, description="Grid page number (1-based)"),
    page_size: Optional[int] = Query(None, description="Grid rows per page (defaults PROPS_PAGE_SIZE)"),
    dedupe: Optional[int] = Query(1, description="When 1 (default), show one row per player/team/market/line choosing best EV then best price"),
):
    """Card-style props recommendations page.

    Reads cached CSV for the date; if missing, falls back to GitHub raw CSV. Groups rows by player+market
    into cards with ladders and attaches team assets. Supports basic filtering/sorting.
    """
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)

# Friendly alias for common misspelling: /props/recomendations -> cards view
@app.get("/props/recomendations", include_in_schema=False)
async def props_recommendations_alias(
    request: Request,
    date: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    min_ev: float = Query(0.0),
    top: int = Query(500),
    sortBy: Optional[str] = Query("ev_desc"),
    side: Optional[str] = Query("both"),
    team: Optional[str] = Query(None),
    game: Optional[str] = Query(None),
    all: Optional[int] = Query(0),
    debug: Optional[int] = Query(0),
):
    # Delegate to the canonical cards view (no grid)
    return await props_recommendations_page(
        request=request,
        date=date,
        market=market,
        min_ev=min_ev,
        top=top,
        sortBy=sortBy,
        side=side,
        team=team,
        game=game,
        all=all,
        debug=debug,
    )

@app.get('/img/headshot/{player_id}.jpg')
async def proxy_headshot(player_id: str):
    """Proxy NHL headshots to avoid hotlink restrictions in local testing."""
    try:
        pid = int(str(player_id).split('.')[0])
    except Exception:
        return Response(status_code=400)
    import httpx, os
    cms_url = f"https://cms.nhl.bamgrid.com/images/headshots/current/168x168/{pid}.jpg"
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8'
    }
    # Local disk cache: data/models/headshots/{pid}.jpg
    try:
        cache_dir = _MODEL_DIR / 'headshots'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{pid}.jpg"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            with open(cache_path, 'rb') as f:
                return Response(content=f.read(), media_type='image/jpeg', headers={'Cache-Control': 'public, max-age=86400'})
    except Exception:
        cache_path = None
    # First try direct CMS (bamgrid)
    try:
        async with httpx.AsyncClient(timeout=6.0, headers=headers) as client:
            r = await client.get(cms_url)
            if r.status_code == 200 and r.content:
                try:
                    if cache_path:
                        with open(cache_path, 'wb') as f:
                            f.write(r.content)
                except Exception:
                    pass
                return Response(content=r.content, media_type='image/jpeg', headers={'Cache-Control': 'public, max-age=86400'})
            # If direct fails, fall back to external image proxy (DNS-resilient)
    except Exception:
        r = None
    # Second try alternate host (bamcontent)
    try:
        alt_url = f"https://nhl.bamcontent.com/images/headshots/current/168x168/{pid}.jpg"
        async with httpx.AsyncClient(timeout=6.0, headers=headers) as client:
            r_alt = await client.get(alt_url)
            if r_alt.status_code == 200 and r_alt.content:
                try:
                    if cache_path:
                        with open(cache_path, 'wb') as f:
                            f.write(r_alt.content)
                except Exception:
                    pass
                return Response(content=r_alt.content, media_type='image/jpeg', headers={'Cache-Control': 'public, max-age=86400'})
    except Exception:
        pass
    # External proxy fallbacks (browser-safe, avoid local DNS on cms host)
    try:
        # Try wsrv.nl with explicit SSL indicator
        proxy_url = f"https://images.weserv.nl/?url=ssl:cms.nhl.bamgrid.com/images/headshots/current/168x168/{pid}.jpg"
        async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
            r2 = await client.get(proxy_url)
            if r2.status_code == 200 and r2.content:
                try:
                    if cache_path:
                        with open(cache_path, 'wb') as f:
                            f.write(r2.content)
                except Exception:
                    pass
                return Response(content=r2.content, media_type='image/jpeg', headers={'Cache-Control': 'public, max-age=86400'})
            # Try statically.io mirror
            stat_url = f"https://cdn.statically.io/img/cms.nhl.bamgrid.com/images/headshots/current/168x168/{pid}.jpg?quality=85&format=jpg"
            r3 = await client.get(stat_url)
            if r3.status_code == 200 and r3.content:
                try:
                    if cache_path:
                        with open(cache_path, 'wb') as f:
                            f.write(r3.content)
                except Exception:
                    pass
                return Response(content=r3.content, media_type='image/jpeg', headers={'Cache-Control': 'public, max-age=86400'})
            return Response(status_code=r3.status_code if 'r3' in locals() else r2.status_code)
    except Exception:
        # As a last resort, return a generated SVG placeholder so <img> doesn't break
        svg = """
<svg xmlns='http://www.w3.org/2000/svg' width='168' height='168'>
  <rect width='168' height='168' fill='#E5E7EB'/>
  <circle cx='84' cy='60' r='34' fill='#9CA3AF'/>
  <rect x='34' y='104' width='100' height='44' rx='22' fill='#9CA3AF'/>
</svg>"""
        return Response(content=svg, media_type='image/svg+xml', headers={'Cache-Control': 'public, max-age=600'})

@app.get("/api/props/recommendations.json")
async def props_recommendations_json(
    date: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    min_ev: float = Query(0.0),
    side: Optional[str] = Query("both"),
):
    d = date or _today_ymd()
    df = pd.DataFrame()
    try:
        p = PROC_DIR / f"props_recommendations_{d}.csv"
        if p.exists():
            df = _read_csv_fallback(p)
        if (df is None or df.empty):
            gh = _github_raw_read_csv(f"data/processed/props_recommendations_{d}.csv")
            if gh is not None and not gh.empty:
                df = gh
    except Exception:
        df = pd.DataFrame()
    # Normalize columns if CLI wrote ev_over/proj_lambda
    try:
        if df is not None and not df.empty:
            cols = set(df.columns)
            if ('ev' not in cols) and ('ev_over' in cols):
                try:
                    df['ev'] = pd.to_numeric(df['ev_over'], errors='coerce')
                except Exception:
                    df['ev'] = df['ev_over']
            if ('proj' not in cols) and ('proj_lambda' in cols):
                try:
                    df['proj'] = pd.to_numeric(df['proj_lambda'], errors='coerce')
                except Exception:
                    df['proj'] = df['proj_lambda']
    except Exception:
        pass
    if df is None or df.empty:
        return JSONResponse({"date": d, "rows": 0, "total_rows": 0, "data": []})
    try:
        market_u = str(market or '').strip().upper()
        if market_u and market_u not in ('ALL', 'ALL MARKETS', 'ANY') and 'market' in df.columns:
            df = df[df['market'].astype(str).str.upper() == market_u]
        if 'ev' in df.columns:
            df['ev'] = pd.to_numeric(df['ev'], errors='coerce')
            df = df[df['ev'] >= float(min_ev)]
        if side and side.lower() in ("over","under") and 'side' in df.columns:
            df = df[df['side'].astype(str).str.lower() == side.lower()]
    except Exception:
        pass
    # Trim to safe size
    try:
        n = int(os.getenv('PROPS_MAX_ROWS','8000'))
        if len(df) > n:
            df = df.head(n)
    except Exception:
        pass
    try:
        recs = df.to_dict(orient='records')
    except Exception:
        recs = []
    return JSONResponse({"date": d, "rows": len(recs), "total_rows": len(recs), "data": recs})

@app.get('/props/debug/photos', response_class=PlainTextResponse)
def debug_props_photos(date: str = Query(default='today'), include_stats_sample: int = Query(0, ge=0, le=50)):
    """Diagnostic CSV of photo enrichment mapping for props recommendations.

    Columns: player_name,has_photo,photo_url,player_id_in_lines,teams_in_lines,stats_player_id,abbrev_candidate_pid
    include_stats_sample optionally appends up to N abbreviated stats name entries not present in lines.
    """
    import io, csv
    from ..utils.io import RAW_DIR as _RAW
    if date == 'today':
        date = _today_ymd()
    base = PROC_DIR.parent / 'props' / f'player_props_lines/date={date}'
    dfs = []
    for name in ('oddsapi.parquet','oddsapi.csv'):
        p = base / name
        if p.exists():
            try:
                dfs.append(pd.read_parquet(p) if p.suffix=='.parquet' else _read_csv_fallback(p))
            except Exception:
                continue
    if not dfs:
        return PlainTextResponse('no canonical lines found for date')
    lines = pd.concat(dfs, ignore_index=True)
    def _norm_name(x: str) -> str:
        try: return ' '.join(str(x or '').split())
        except Exception: return str(x)
    agg = {}
    for _, r in lines.iterrows():
        pname = _norm_name(r.get('player_name') or r.get('player'))
        if not pname: continue
        ent = agg.setdefault(pname, {'player_ids': set(), 'teams': set()})
        pid = r.get('player_id')
        if pd.notna(pid):
            try: ent['player_ids'].add(str(int(pid)))
            except Exception: pass
        t = r.get('team')
        if t and str(t).strip():
            ent['teams'].add(str(t).strip())
    stats_ids = {}
    abbrev_forms = {}
    try:
        sp = _RAW / 'player_game_stats.csv'
        if sp.exists():
            s = _read_csv_fallback(sp)
            if not s.empty and {'player','player_id'}.issubset(s.columns):
                s = s.dropna(subset=['player'])
                def _unwrap(v):
                    try:
                        vs = str(v)
                        if vs.startswith('{') and 'default' in vs:
                            import ast as _ast
                            obj = _ast.literal_eval(vs)
                            if isinstance(obj, dict):
                                dv = obj.get('default')
                                if isinstance(dv, str) and dv.strip():
                                    return dv.strip()
                        return vs
                    except Exception:
                        return str(v)
                s['player_clean'] = s['player'].map(_unwrap)
                try:
                    s['_d'] = pd.to_datetime(s['date'], errors='coerce')
                    s = s.sort_values('_d')
                except Exception:
                    pass
                last_ids = s.dropna(subset=['player_id']).groupby('player_clean')['player_id'].last()
                for nm, pid in last_ids.items():
                    stats_ids[_norm_name(nm)] = str(int(pid))
                for nm in list(stats_ids.keys()):
                    parts = nm.split()
                    if len(parts) >= 2:
                        ini = parts[0][0].upper(); last = parts[-1]
                        abbrev_forms[f'{ini}. {last}'] = stats_ids[nm]
    except Exception:
        pass
    def _url(pid: str):
        return f'https://cms.nhl.bamgrid.com/images/headshots/current/168x168/{pid}.jpg'
    out_rows = []
    for pname, meta in sorted(agg.items()):
        pids = meta['player_ids']
        pid_any = next(iter(pids)) if pids else ''
        photo_pid = ''
        if pid_any:
            photo_pid = pid_any
        elif pname in stats_ids:
            photo_pid = stats_ids[pname]
        else:
            parts = pname.split()
            if len(parts) >= 2:
                ini = parts[0][0].upper(); last = parts[-1]
                key_abbrev = f'{ini}. {last}'
                if key_abbrev in stats_ids:
                    photo_pid = stats_ids[key_abbrev]
        out_rows.append({
            'player_name': pname,
            'has_photo': bool(photo_pid),
            'photo_url': _url(photo_pid) if photo_pid else '',
            'player_id_in_lines': ';'.join(sorted(pids)) if pids else '',
            'teams_in_lines': ';'.join(sorted(meta['teams'])) if meta['teams'] else '',
            'stats_player_id': stats_ids.get(pname, ''),
            'abbrev_candidate_pid': stats_ids.get(f"{pname.split()[0][0].upper()}. {pname.split()[-1]}", ''),
        })
    if include_stats_sample > 0:
        added = 0
        for abbr, pid in abbrev_forms.items():
            if added >= include_stats_sample: break
            if not any(r['player_name'] == abbr for r in out_rows):
                out_rows.append({
                    'player_name': abbr,
                    'has_photo': True,
                    'photo_url': _url(pid),
                    'player_id_in_lines': '',
                    'teams_in_lines': '',
                    'stats_player_id': pid,
                    'abbrev_candidate_pid': pid,
                })
                added += 1
    buf = io.StringIO()
    if out_rows:
        w = csv.DictWriter(buf, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows: w.writerow(r)
    else:
        buf.write('no_rows')
    return PlainTextResponse(buf.getvalue(), media_type='text/plain')


@app.post("/api/cron/props-full")
async def api_cron_props_full(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD; defaults to ET today"),
    min_ev: float = Query(0.0, description="Minimum EV threshold for recommendations"),
    top: int = Query(200, description="Top N recommendations to keep"),
    market: Optional[str] = Query(None, description="Optional market filter for recs: SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS"),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional alternative to token query param)"),
    async_run: bool = Query(False, description="If true, queue work in background and return 202 immediately"),
):
    """Secure endpoint to run the full props pipeline for a date:

    1) Collect canonical lines from The Odds API (with roster enrichment)
    2) Compute props projections CSV
    3) Compute props recommendations CSV

    Best-effort upserts artifacts to GitHub.
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
    # If async mode requested, queue the full pipeline and return immediately
    if async_run:
        d_local = d
        def _run_full():
            # Collect lines for configured sources; ensure output dir exists
            from ..data import player_props as props_data
            base = PROC_DIR.parent / "props" / f"player_props_lines/date={d_local}"
            base.mkdir(parents=True, exist_ok=True)
            step_timeout = 90
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout
            for which, src in (("oddsapi", "oddsapi"),):
                try:
                    cfg = props_data.PropsCollectionConfig(output_root=str(PROC_DIR.parent / "props"), book=which, source=src)
                    try:
                        roster_df = _props_data._build_roster_enrichment()
                    except Exception:
                        roster_df = None
                    # Timeout-guard
                    def _do_collect_full():
                        return props_data.collect_and_write(d_local, roster_df=roster_df, cfg=cfg)
                    try:
                        with ThreadPoolExecutor(max_workers=1) as ex:
                            fut = ex.submit(_do_collect_full)
                            res = fut.result(timeout=step_timeout)
                    except _FutTimeout:
                        res = {"output_path": None}
                    path = res.get("output_path")
                    if path:
                        try:
                            rel = str(Path(path)).replace("\\", "/")
                            parts = rel.split("/")
                            if "data" in parts:
                                rel = "/".join(parts[parts.index("data"):])
                            _gh_upsert_file_if_better_or_same(Path(path), f"web: update props lines {which} for {d_local}", rel_hint=rel)
                        except Exception:
                            pass
                except Exception:
                    pass
            # Projections
            dfp = _compute_props_projections(d_local, market=market)
            out_path = PROC_DIR / f"props_projections_{d_local}.csv"
            try:
                save_df(dfp, out_path)
                try:
                    _gh_upsert_file_if_better_or_same(out_path, f"web: update props projections for {d_local}")
                except Exception:
                    pass
            except Exception:
                pass
            # Recommendations
            try:
                # Load lines if exist
                parts = []
                for name in ("oddsapi.parquet", "bovada.parquet"):
                    p = base / name
                    if p.exists():
                        try:
                            parts.append(pd.read_parquet(p))
                        except Exception:
                            pass
                rec_df = pd.DataFrame()
                if parts:
                    lines = pd.concat(parts, ignore_index=True)
                    # Ensure stats
                    stats_path = RAW_DIR / "player_game_stats.csv"
                    if not stats_path.exists():
                        try:
                            from datetime import datetime as _dt, timedelta as _td
                            start = (_dt.strptime(d_local, "%Y-%m-%d") - _td(days=365)).strftime("%Y-%m-%d")
                            from ..data.collect import collect_player_game_stats
                            # Timeout-guard stats backfill
                            def _do_stats_full():
                                collect_player_game_stats(start, d_local, source="stats")
                            try:
                                with ThreadPoolExecutor(max_workers=1) as ex:
                                    fut = ex.submit(_do_stats_full)
                                    fut.result(timeout=step_timeout)
                            except Exception:
                                pass
                        except Exception:
                            pass
                    hist = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
                    shots = _SkaterShotsModel(); saves = _GoalieSavesModel(); goals = _SkaterGoalsModel(); assists = _SkaterAssistsModel(); points = _SkaterPointsModel(); blocks = _SkaterBlocksModel()
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
                        if market and m != (market or '').upper():
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
                            'date': d_local,
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
                    rec_df = pd.DataFrame(recs)
                    if not rec_df.empty:
                        rec_df = rec_df.sort_values('ev', ascending=False)
                        if top and top > 0:
                            rec_df = rec_df.head(int(top))
                    rec_path = PROC_DIR / f"props_recommendations_{d_local}.csv"
                    try:
                        save_df(rec_df, rec_path)
                        try:
                            _gh_upsert_file_if_better_or_same(rec_path, f"web: update props recommendations for {d_local}")
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass
            return {"ok": True, "date": d_local}
        job_id = _queue_cron('props-full', {'date': d, 'min_ev': min_ev, 'top': top, 'market': market}, _run_full)
        return JSONResponse({"ok": True, "date": d, "queued": True, "mode": "async", "job_id": job_id}, status_code=202)

    # Step 1: collect lines (reusing props-collect logic)
    try:
        res_collect = await api_cron_props_collect(token=token, date=d)
        # Normalize JSON
        if isinstance(res_collect, JSONResponse):
            import json as _json
            try:
                res_collect = _json.loads(res_collect.body)
            except Exception:
                res_collect = {"ok": True}
    except Exception as e:
        res_collect = {"ok": False, "error": str(e)}
    # Step 2: projections
    try:
        res_proj = await api_cron_props_projections(token=token, date=d, market=market, top=0)
        if isinstance(res_proj, JSONResponse):
            import json as _json
            try:
                res_proj = _json.loads(res_proj.body)
            except Exception:
                res_proj = {"ok": True}
    except Exception as e:
        res_proj = {"ok": False, "error": str(e)}
    # Step 3: recommendations
    try:
        res_recs = await api_cron_props_recommendations(token=token, date=d, market=market, min_ev=min_ev, top=top)
        if isinstance(res_recs, JSONResponse):
            import json as _json
            try:
                res_recs = _json.loads(res_recs.body)
            except Exception:
                res_recs = {"ok": True}
    except Exception as e:
        res_recs = {"ok": False, "error": str(e)}
    return JSONResponse({
        "ok": True,
        "date": d,
        "collect": res_collect,
        "projections": res_proj,
        "recommendations": res_recs,
    })

@app.post("/api/cron/props-range")
async def api_cron_props_range(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD (inclusive); if provided without end, single date"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD (inclusive)"),
    back: int = Query(0, description="How many days back from today (ET) to include if start not provided"),
    ahead: int = Query(0, description="How many future days from today (ET) to include if start not provided"),
    mode: str = Query("full", description="Which pipeline steps to run: full|collect|projections|recommendations"),
    min_ev: float = Query(0.0, description="Minimum EV threshold for recommendations (when mode includes recommendations)"),
    top: int = Query(200, description="Top N recommendations to keep (when mode includes recommendations)"),
    market: Optional[str] = Query(None, description="Optional market filter passed to projections/recommendations"),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (alternative to token query param)"),
):
    """Batch props pipeline over a date window.

    Usage patterns:
      - Explicit range: provide start & end.
      - Relative window: omit start/end, use back & ahead around today (ET).

    mode behaviors:
      * full: runs collect + projections + recommendations (same as props-full per date)
      * collect: only line collection
      * projections: only projections (assumes lines exist or computes from model only)
      * recommendations: only recommendations (requires projections file; will attempt to compute if missing)
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
    # Build date list
    dates: list[str] = []
    try:
        if start:
            sd = datetime.strptime(start, "%Y-%m-%d").date()
            if end:
                ed = datetime.strptime(end, "%Y-%m-%d").date()
            else:
                ed = sd
            if ed < sd:
                sd, ed = ed, sd
            cur = sd
            while cur <= ed:
                dates.append(cur.strftime("%Y-%m-%d"))
                cur += timedelta(days=1)
        else:
            # Use existing helper _today_ymd (renders ET-based YMD elsewhere) to anchor base date
            base = datetime.strptime(_today_ymd(), "%Y-%m-%d").date()
            for i in range(back, -1, -1):
                if i == 0:
                    dates.append(base.strftime("%Y-%m-%d"))
                else:
                    dates.append((base - timedelta(days=i)).strftime("%Y-%m-%d"))
            for j in range(1, ahead + 1):
                dates.append((base + timedelta(days=j)).strftime("%Y-%m-%d"))
        # Dedup preserve order
        seen = set(); ordered = []
        for d in dates:
            if d not in seen:
                seen.add(d); ordered.append(d)
        dates = ordered
    except Exception as e:
        return JSONResponse({"error": f"date_range_parse_failed: {e}"}, status_code=400)
    mode_lc = (mode or "full").lower()
    if mode_lc not in {"full","collect","projections","recommendations"}:
        return JSONResponse({"error": f"invalid mode '{mode}'"}, status_code=400)
    out: dict[str, dict] = {}
    for d in dates:
        # For consistency pass token along to internal callables
        try:
            if mode_lc == "full":
                res = await api_cron_props_full(token=supplied, date=d, min_ev=min_ev, top=top, market=market)
                if isinstance(res, JSONResponse):
                    import json as _json
                    try:
                        res = _json.loads(res.body)
                    except Exception:
                        res = {"ok": True}
                out[d] = res
            elif mode_lc == "collect":
                res = await api_cron_props_collect(token=supplied, date=d)
                if isinstance(res, JSONResponse):
                    import json as _json
                    try:
                        res = _json.loads(res.body)
                    except Exception:
                        res = {"ok": True}
                out[d] = {"collect": res}
            elif mode_lc == "projections":
                res = await api_cron_props_projections(token=supplied, date=d, market=market, top=0)
                if isinstance(res, JSONResponse):
                    import json as _json
                    try:
                        res = _json.loads(res.body)
                    except Exception:
                        res = {"ok": True}
                out[d] = {"projections": res}
            else:  # recommendations
                res = await api_cron_props_recommendations(token=supplied, date=d, market=market, min_ev=min_ev, top=top)
                if isinstance(res, JSONResponse):
                    import json as _json
                    try:
                        res = _json.loads(res.body)
                    except Exception:
                        res = {"ok": True}
                out[d] = {"recommendations": res}
        except Exception as e:
            out[d] = {"error": str(e)}
    return JSONResponse({"ok": True, "mode": mode_lc, "dates": dates, "results": out})

@app.get("/props/reconciliation", include_in_schema=False)
async def props_reconciliation_page(
    date: Optional[str] = Query(None),
    refresh: int = Query(0),
):
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)

@app.post("/api/cron/props-fast")
async def api_cron_props_fast(
    token: Optional[str] = Query(None, description="Bearer token; must match REFRESH_CRON_TOKEN env var"),
    date: Optional[str] = Query(None, description="Slate date YYYY-MM-DD; defaults to ET today"),
    min_ev: float = Query(0.0, description="Minimum EV threshold for recommendations"),
    top: int = Query(400, description="Top N recommendations to keep"),
    market: Optional[str] = Query("", description="Optional market filter for recs: SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS"),
    authorization: Optional[str] = Header(None, description="Authorization: Bearer <token> header (optional)"),
):
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
    # Execute fast pipeline
    try:
        from ..cli import props_fast as _props_fast
        if hasattr(_props_fast, 'callback') and callable(getattr(_props_fast, 'callback')):
            _props_fast.callback(date=d, min_ev=min_ev, top=top, market=(market or ""))
        else:
            _props_fast(date=d, min_ev=min_ev, top=top, market=(market or ""))
    except Exception as e:
        return JSONResponse({"ok": False, "date": d, "error": str(e)}, status_code=500)
    return JSONResponse({"ok": True, "date": d})


    # (Bovada refresh endpoint removed; use /api/cron/light-refresh and /api/cron/props-collect instead.)

# ---------------- Normalization & new props endpoints (module level) -----------------
def _normalize_props_row_dict(r: dict) -> dict:
    out = dict(r)
    if 'proj' in out and 'proj_lambda' not in out:
        out['proj_lambda'] = out.get('proj')
    if 'ev' in out and 'ev_over' not in out:
        out['ev_over'] = out.get('ev')
    if 'price' in out and 'over_price' not in out and str(out.get('side','OVER')).upper() == 'OVER':
        out['over_price'] = out.get('price')
    return out

@app.get('/api/props/lines.json')
async def api_props_lines_json(
    date: Optional[str] = Query(None, description='Slate date YYYY-MM-DD (ET)'),
    market: Optional[str] = Query(None),
    player: Optional[str] = Query(None),
):
    d = _normalize_date_param(date)
    base = PROC_DIR.parent / 'props' / f'player_props_lines/date={d}'
    parts = []
    for fname in ('oddsapi.parquet',):
        p = base / fname
        if p.exists():
            try:
                parts.append(pd.read_parquet(p))
            except Exception:
                pass
    if not parts:
        return JSONResponse({"date": d, "data": [], "total_rows": 0})
    df = pd.concat(parts, ignore_index=True)
    if market and 'market' in df.columns:
        try: df = df[df['market'].astype(str).str.upper() == market.upper()]
        except Exception: pass
    if player and 'player_name' in df.columns:
        try: df = df[df['player_name'].astype(str).str.lower() == player.lower()]
        except Exception: pass
    keep = [c for c in ['date','player_name','player_id','team','market','line','over_price','under_price','book','first_seen_at','last_seen_at','is_current'] if c in df.columns]
    if df.empty:
        df = df[keep]
    rows = _df_jsonsafe_records(df.rename(columns={'player_name':'player'}))
    rows = [_normalize_props_row_dict(r) for r in rows]
    return JSONResponse({"date": d, "data": rows, "total_rows": len(rows)})

@app.get('/api/props/projections/history.json')
async def api_props_projections_history_json(
    date: Optional[str] = Query(None, description="Anchor date (inclusive); defaults to today"),
    days: int = Query(30, description="Lookback days"),
    market: Optional[str] = Query(None),
    player: Optional[str] = Query(None),
    position: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    limit: int = Query(2000, description="Max rows after filtering"),
):
    d = _normalize_date_param(date)
    hist_path = PROC_DIR / 'props_projections_history.csv'
    if not hist_path.exists():
        return JSONResponse({"date": d, "data": [], "total_rows": 0})
    try:
        df = pd.read_csv(hist_path)
    except Exception:
        return JSONResponse({"date": d, "data": [], "total_rows": 0})
    if df.empty:
        return JSONResponse({"date": d, "data": [], "total_rows": 0})
    try:
        df['date'] = pd.to_datetime(df['date']).dt.date
        anchor = datetime.strptime(d, '%Y-%m-%d').date()
        start = anchor - timedelta(days=max(0, days))
        df = df[(df['date'] >= start) & (df['date'] <= anchor)]
    except Exception:
        pass
    if market and 'market' in df.columns:
        try: df = df[df['market'].astype(str).str.upper() == market.upper()]
        except Exception: pass
    if player and 'player' in df.columns:
        try: df = df[df['player'].astype(str).str.lower() == player.lower()]
        except Exception: pass
    if position and 'position' in df.columns:
        try: df = df[df['position'].astype(str).str.upper() == position.upper()]
        except Exception: pass
    if team and 'team' in df.columns:
        try: df = df[df['team'].astype(str).str.upper() == team.upper()]
        except Exception: pass
    total_rows = len(df)
    try:
        sort_cols = ['date']
        asc = [False]
        if 'proj_lambda' in df.columns:
            sort_cols.append('proj_lambda'); asc.append(False)
        df.sort_values(sort_cols, ascending=asc, inplace=True)
    except Exception:
        pass
    if limit and limit > 0:
        df = df.head(limit)
    # Force all numeric columns to be finite (replace inf / -inf with NaN which later becomes None)
    try:
        import numpy as _np
        num_cols = df.select_dtypes(include=[np.number, 'float', 'int']).columns if hasattr(df, 'select_dtypes') else []
        for _c in num_cols:
            try:
                col = df[_c]
                mask = ~_np.isfinite(col.astype(float))
                if mask.any():
                    df.loc[mask, _c] = _np.nan
            except Exception:
                pass
    except Exception:
        pass
    # Ensure categorical columns that may be all-null are treated as object so NaN -> None
    for _cat in ['team','position']:
        if _cat in df.columns:
            try:
                if df[_cat].isna().all():
                    df[_cat] = df[_cat].astype(object)
            except Exception:
                pass
    rows = _df_jsonsafe_records(df)
    # Extra defensive pass: ensure no non-finite floats slipped through
    try:
        import math as _math
        for _r in rows:
            for _k, _v in list(_r.items()):
                if isinstance(_v, float) and not _math.isfinite(_v):
                    _r[_k] = None
    except Exception:
        pass
    try:
        payload = {"date": d, "data": rows, "returned_rows": len(rows), "total_rows": total_rows, "lookback_days": days}
        if '_json_sanitize' in globals():
            payload = _json_sanitize(payload)
        return JSONResponse(payload)
    except Exception:
        return JSONResponse({"date": d, "data": [], "returned_rows": 0, "total_rows": total_rows, "lookback_days": days})

    @app.get('/api/props/ladders.json')
    async def api_props_ladders_json(
        date: Optional[str] = Query(None),
        market: Optional[str] = Query('SOG'),
        min_levels: int = Query(2),
    ):
        d = _normalize_date_param(date)
        base = PROC_DIR.parent / 'props' / f'player_props_lines/date={d}'
        parts = []
        for fname in ('oddsapi.parquet',):
            p = base / fname
            if p.exists():
                try:
                    parts.append(pd.read_parquet(p))
                except Exception:
                    pass
        if not parts:
            return JSONResponse({"date": d, "market": market, "ladders": [], "total": 0})
        df = pd.concat(parts, ignore_index=True)
        # Basic normalization
        if 'market' in df.columns:
            try: df['market'] = df['market'].astype(str).str.upper()
            except Exception: pass
        if market:
            try: df = df[df['market'] == market.upper()]
            except Exception: pass
        # Choose current lines only if is_current flag exists
        if 'is_current' in df.columns:
            try: df = df[df['is_current'] == True]
            except Exception: pass
        # Build ladders per player+market
        ladders = []
        player_col = 'player_name' if 'player_name' in df.columns else ('player' if 'player' in df.columns else None)
        if not player_col or df.empty:
            return JSONResponse({"date": d, "market": market, "ladders": [], "total": 0})
        group_cols = [player_col, 'market'] if 'market' in df.columns else [player_col]
        for (grp_player, grp_market), g in df.groupby(group_cols):
            try:
                g = g.copy()
                # Keep relevant columns
                keep = [c for c in ['line','over_price','under_price','book','first_seen_at','last_seen_at'] if c in g.columns]
                g = g[keep + ([] if 'line' in keep else [])]
                # Drop null lines
                if 'line' in g.columns:
                    g = g[pd.notna(g['line'])]
                if g.empty:
                    continue
                # Sort ascending by line numeric then price
                if 'line' in g.columns:
                    try: g['line'] = g['line'].astype(float)
                    except Exception: pass
                    try: g.sort_values(['line'], inplace=True)
                    except Exception: pass
                levels = []
                for _, r in g.iterrows():
                    levels.append({
                        'line': r.get('line'),
                        'over_price': r.get('over_price'),
                        'under_price': r.get('under_price'),
                        'book': r.get('book'),
                        'first_seen_at': r.get('first_seen_at'),
                        'last_seen_at': r.get('last_seen_at'),
                    })
                if len(levels) < int(min_levels):
                    continue
                ladders.append({
                    'player': grp_player,
                    'market': grp_market if isinstance(grp_market, str) else market.upper() if market else grp_market,
                    'level_count': len(levels),
                    'levels': levels,
                })
            except Exception:
                continue
        return JSONResponse({"date": d, "market": market, "ladders": ladders, "total": len(ladders)})
        if market and 'market' in df.columns:
            try: df = df[df['market'].astype(str).str.upper() == market.upper()]
            except Exception: pass
        if 'side' in df.columns:
            try: df_over = df[df['side'].astype(str).str.upper() == 'OVER']
            except Exception: df_over = df
        else:
            df_over = df
        ladders = []
        try:
            if 'player_name' in df_over.columns and 'line' in df_over.columns:
                g = df_over.groupby(['player_name','market'])
                for (player_name, mkt), sub in g:
                    try:
                        levels = sorted([lv for lv in sub['line'].dropna().unique().tolist() if lv is not None])
                    except Exception:
                        levels = []
                    if len(levels) >= min_levels:
                        level_rows = []
                        for L in levels:
                            cand = sub[sub['line'] == L]
                            price = None
                            try:
                                if 'odds' in cand.columns:
                                    price = cand['odds'].max()
                                elif 'over_price' in cand.columns:
                                    price = cand['over_price'].max()
                            except Exception:
                                pass
                            level_rows.append({'line': L, 'best_over_price': price})
                        ladders.append({'player': player_name, 'market': mkt, 'levels': level_rows, 'levels_count': len(levels)})
        except Exception:
            ladders = []
        return JSONResponse({"date": d, "market": market, "ladders": ladders, "total": len(ladders)})


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


@app.get("/props/stats-calibration", include_in_schema=False)
async def props_stats_calibration_page(
    file: Optional[str] = Query(None),
    market: Optional[str] = Query(None),
    window: Optional[int] = Query(None),
):
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)


@app.get("/api/recommendations")
async def api_recommendations(
    date: Optional[str] = Query(None),
    min_ev: float = Query(0.0, description="Minimum EV threshold to include"),
    top: int = Query(20, description="Top N recommendations to return"),
    markets: str = Query("all", description="Comma-separated filters: moneyline,totals,puckline,first10,periods"),
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
        # Apply market-specific default odds when none present
        if price_val is None:
            if market_key == "first10":
                price_val = -150.0
            elif market_key in ("totals", "puckline", "periods"):
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
        # Optional Kelly stake if bankroll provided
        try:
            if bankroll and float(bankroll) > 0 and (prob_val is not None) and (price_val is not None):
                from ..utils.odds import american_to_decimal, kelly_stake
                dec = american_to_decimal(float(price_val))
                st = kelly_stake(float(prob_val), float(dec), float(bankroll), float(kelly_fraction_part))
                rec["stake"] = round(float(st), 2)
        except Exception:
            pass
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
        f_markets = set([m.strip().lower() for m in str(markets).split(",")]) if markets and str(markets) != "all" else {"moneyline", "totals", "puckline", "first10", "periods"}
    except Exception:
        f_markets = {"moneyline", "totals", "puckline", "first10", "periods"}

    for _, r in df.iterrows():
        # Moneyline (only recommend model-favored side)
        if "moneyline" in f_markets:
            try:
                ph = float(r.get("p_home_ml")) if pd.notna(r.get("p_home_ml")) else None
                pa = float(r.get("p_away_ml")) if pd.notna(r.get("p_away_ml")) else None
            except Exception:
                ph, pa = None, None
            if ph is not None and pa is not None:
                if ph >= pa:
                    add_rec(r, "moneyline", "home_ml", "p_home_ml", "ev_home_ml", "edge_home_ml", "home_ml_odds", "home_ml_book")
                else:
                    add_rec(r, "moneyline", "away_ml", "p_away_ml", "ev_away_ml", "edge_away_ml", "away_ml_odds", "away_ml_book")
            else:
                # If probabilities invalid, fall back to both (rare)
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
        # First 10 minutes (Yes/No)
        if "first10" in f_markets:
            add_rec(r, "first10", "f10_yes", "p_f10_yes", "ev_f10_yes", "edge_f10_yes", "f10_yes_odds", None)
            add_rec(r, "first10", "f10_no", "p_f10_no", "ev_f10_no", "edge_f10_no", "f10_no_odds", None)
        # Period totals (P1..P3 Over/Under)
        if "periods" in f_markets:
            for pn in (1, 2, 3):
                add_rec(r, "periods", f"p{pn}_over", f"p{pn}_over_prob", f"ev_p{pn}_over", None, f"p{pn}_over_odds", None)
                add_rec(r, "periods", f"p{pn}_under", f"p{pn}_under_prob", f"ev_p{pn}_under", None, f"p{pn}_under_odds", None)

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


@app.get("/recommendations", include_in_schema=False)
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
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)


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
    top: int = Query(200),
):
    """Compare model predictions vs recorded results to compute a simple PnL summary.

    Uses predictions_{date}.csv. Totals/puckline results are read from result_total/result_ats when present.
    Moneyline results are included only if price/result fields exist.
    """
    # Normalize potential FastAPI Query objects when invoked internally
    try:
        from fastapi import params as _params
    except Exception:
        _params = None
    def _norm(v, default=None):
        if _params and isinstance(v, _params.Query):
            return v.default if v.default is not None else default
        return v if v is not None else default
    d = _norm(date, _today_ymd())
    try:
        bankroll = float(_norm(bankroll, 1000.0) or 1000.0)
    except Exception:
        bankroll = 1000.0
    try:
        flat_stake = float(_norm(flat_stake, 100.0) or 100.0)
    except Exception:
        flat_stake = 100.0
    try:
        top = int(_norm(top, 200) or 200)
    except Exception:
        top = 200
    path = PROC_DIR / f"predictions_{d}.csv"
    if not path.exists():
        return JSONResponse({"summary": {"date": d, "picks": 0, "decided": 0, "wins": 0, "losses": 0, "pushes": 0, "staked": 0.0, "pnl": 0.0, "roi": None}, "rows": []})

    try:
        df = pd.read_csv(path)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    picks: list[dict] = []

    def add_pick(r, market: str, bet: str, ev_key: str, price_key: str, result_field: Optional[str]):
        ev = r.get(ev_key)
        try:
            evf = float(ev) if ev is not None and not (isinstance(ev, float) and pd.isna(ev)) else None
        except Exception:
            evf = None
        if evf is None:
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
        price = r.get(close_key) if close_key else None
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
            "winner_actual": r.get("winner_actual"),
            "actual_total": r.get("actual_total"),
            "total_line_used": r.get("close_total_line_used") if "close_total_line_used" in r else (r.get("total_line_used") if "total_line_used" in r else None),
            "final_home_goals": r.get("final_home_goals"),
            "final_away_goals": r.get("final_away_goals"),
            # period results
            "result_first10": r.get("result_first10"),
            "result_p1_total": r.get("result_p1_total"),
            "result_p2_total": r.get("result_p2_total"),
            "result_p3_total": r.get("result_p3_total"),
            "p1_total_line": r.get("p1_total_line"),
            "p2_total_line": r.get("p2_total_line"),
            "p3_total_line": r.get("p3_total_line"),
        })

    for _, r in df.iterrows():
        add_pick(r, "moneyline", "home_ml", "ev_home_ml", "home_ml_odds", None)
        add_pick(r, "moneyline", "away_ml", "ev_away_ml", "away_ml_odds", None)
        add_pick(r, "totals", "over", "ev_over", "over_odds", "result_total")
        add_pick(r, "totals", "under", "ev_under", "under_odds", "result_total")
        add_pick(r, "puckline", "home_pl_-1.5", "ev_home_pl_-1.5", "home_pl_-1.5_odds", "result_ats")
        add_pick(r, "puckline", "away_pl_+1.5", "ev_away_pl_+1.5", "away_pl_+1.5_odds", "result_ats")
        # First-10 and Period totals (if EVs exist)
        add_pick(r, "first10", "f10_yes", "ev_f10_yes", "f10_yes_odds", "result_first10")
        add_pick(r, "first10", "f10_no", "ev_f10_no", "f10_no_odds", "result_first10")
        add_pick(r, "periods", "p1_over", "ev_p1_over", "p1_over_odds", "result_p1_total")
        add_pick(r, "periods", "p1_under", "ev_p1_under", "p1_under_odds", "result_p1_total")
        add_pick(r, "periods", "p2_over", "ev_p2_over", "p2_over_odds", "result_p2_total")
        add_pick(r, "periods", "p2_under", "ev_p2_under", "p2_under_odds", "result_p2_total")
        add_pick(r, "periods", "p3_over", "ev_p3_over", "p3_over_odds", "result_p3_total")
        add_pick(r, "periods", "p3_under", "ev_p3_under", "p3_under_odds", "result_p3_total")

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
    for p in picks[: max(top, 0) if top else len(picks)]:
        stake = flat_stake
        dec = american_to_decimal_local(p.get("price")) if p.get("price") is not None else None
        # Determine result mapping to Win/Loss/Push
        rl = None
        try:
            mkt = (p.get("market") or "").lower()
            bet = (p.get("bet") or "").lower()
            if mkt == "moneyline":
                wa = p.get("winner_actual")
                if isinstance(wa, str):
                    if bet == "home_ml":
                        rl = "win" if wa == p.get("home") else "loss"
                    elif bet == "away_ml":
                        rl = "win" if wa == p.get("away") else "loss"
            elif mkt == "totals":
                # Use actual_total and total_line_used if available; else fall back to result string
                at = p.get("actual_total")
                tl = p.get("total_line_used")
                if at is not None and tl is not None:
                    try:
                        atf = float(at); tlf = float(tl)
                        if abs(tlf - round(tlf)) < 1e-9 and int(round(tlf)) == int(atf):
                            rl = "push"
                        elif bet == "over":
                            rl = "win" if atf > tlf else "loss"
                        elif bet == "under":
                            rl = "win" if atf < tlf else "loss"
                    except Exception:
                        pass
                if rl is None and isinstance(p.get("result"), str):
                    want = "over" if bet == "over" else "under"
                    rlow = p.get("result").lower()
                    if rlow == "push":
                        rl = "push"
                    elif rlow == want:
                        rl = "win"
                    else:
                        rl = "loss"
            elif mkt == "puckline":
                try:
                    fh = float(p.get("final_home_goals")); fa = float(p.get("final_away_goals"))
                    diff = fh - fa
                    if bet == "home_pl_-1.5":
                        rl = "win" if diff > 1.5 else "loss"
                    elif bet == "away_pl_+1.5":
                        rl = "win" if diff < -1.5 else ("loss" if diff >= -1.5 else None)
                except Exception:
                    pass
            elif mkt == "first10":
                rf = (p.get("result_first10") or "").lower()
                if rf in ("yes", "no"):
                    if bet == "f10_yes":
                        rl = "win" if rf == "yes" else "loss"
                    elif bet == "f10_no":
                        rl = "win" if rf == "no" else "loss"
            elif mkt == "periods":
                # Determine period from bet name (p1_over, p2_under, etc.)
                import re
                m = re.match(r"p([123])_(over|under)", bet)
                if m:
                    pn = m.group(1); side = m.group(2)
                    res_key = f"result_p{pn}_total"
                    ln_key = f"p{pn}_total_line"
                    rf = p.get(res_key)
                    if isinstance(rf, str):
                        rlow = rf.lower()
                        if rlow == "push":
                            rl = "push"
                        elif rlow == side:
                            rl = "win"
                        else:
                            rl = "loss"
        except Exception:
            rl = None

        res = p.get("result")
        # Compute PnL if decided
        if isinstance(rl, str):
            if rl == "push":
                pushes += 1
                rows.append({**p, "stake": stake, "payout": 0.0})
            elif rl == "win":
                wins += 1
                if dec:
                    pnl += stake * (dec - 1.0)
                staked += stake
                decided += 1
                rows.append({**p, "stake": stake, "payout": (stake * (dec - 1.0)) if dec else None})
            elif rl == "loss":
                losses += 1
                pnl -= stake
                staked += stake
                decided += 1
                rows.append({**p, "stake": stake, "payout": -stake})
            else:
                rows.append({**p, "stake": stake, "payout": None})
        else:
            rows.append({**p, "stake": stake, "payout": None})

    summary = {
        "date": d,
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

@app.get("/props/recommendations.csv", include_in_schema=False)
def props_recommendations_csv(date: Optional[str] = Query(None)):
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)


@app.get("/props/projections.csv", include_in_schema=False)
def props_projections_csv(date: Optional[str] = Query(None)):
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)


@app.get("/odds-coverage", include_in_schema=False)
async def odds_coverage(date: Optional[str] = Query(None)):
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)

# -----------------------------------------------------------------------------
# FIRST-10 evaluation summary (JSON + HTML)
# -----------------------------------------------------------------------------

def _read_first10_eval_local() -> Dict[str, Any]:
    try:
        p = PROC_DIR / "first10_eval.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        # Fallback: attempt repo path (when running from project root)
        p2 = Path("data/processed/first10_eval.json")
        if p2.exists():
            with open(p2, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

@app.get("/api/first10-eval")
async def api_first10_eval():
    """Return the compact first-10 evaluation JSON if available."""
    data = _read_first10_eval_local()
    if not data:
        # Optional: try GitHub raw if on public host
        try:
            if _is_public_host_env():
                url = os.getenv(
                    "FIRST10_EVAL_URL",
                    "https://raw.githubusercontent.com/mostgood1/NHL-Betting/master/data/processed/first10_eval.json",
                )
                r = requests.get(url, timeout=2.0)
                if r.status_code == 200 and r.text:
                    try:
                        data = json.loads(r.text)
                    except Exception:
                        data = {}
        except Exception:
            data = {}
    if not data:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return JSONResponse(data)

@app.get("/first10-eval", include_in_schema=False)
async def first10_eval_page():
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)


@app.get("/reconciliation", include_in_schema=False)
async def reconciliation(date: Optional[str] = Query(None)):
    return PlainTextResponse("cards-only UI: this page has been removed", status_code=404)

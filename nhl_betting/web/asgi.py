from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional
from urllib.parse import parse_qs

import json

# Minimal ASGI wrapper. No FastAPI here to keep cold-start as small as possible.

_HEAVY_APP = None  # type: Optional[object]
_HEAVY_LOCK = threading.Lock()
_HEAVY_LOADING = False
_HEAVY_LIFESPAN_CM = None  # type: Optional[object]
_HEAVY_LIFESPAN_ENTERED = False


# Lightweight live endpoints caching (avoid heavy app imports)
_SCOREBOARD_CACHE: dict[str, dict] = {}
_LIVE_LENS_CACHE: dict[str, dict] = {}

# Heavy live-lens proxy circuit-breaker (avoid hammering heavy route if it's failing)
_HEAVY_LIVE_LENS_FAILS = 0
_HEAVY_LIVE_LENS_COOLDOWN_UNTIL = 0.0


async def _try_heavy_http(
    scope,
    receive,
    send,
    *,
    timeout_sec: float,
) -> bool:
    """Attempt to serve this request via the heavy FastAPI app.

    Captures the heavy app's ASGI messages; only replays them if the response
    is a successful (2xx) HTTP response. Returns True if heavy was used.
    """
    if _HEAVY_APP is None:
        return False

    captured: list[dict] = []

    async def _send_cap(message: dict):
        captured.append(message)

    try:
        await asyncio.wait_for(_HEAVY_APP(scope, receive, _send_cap), timeout=float(timeout_sec))
    except Exception:
        return False

    start = None
    for m in captured:
        if m.get("type") == "http.response.start":
            start = m
            break
    if not isinstance(start, dict):
        return False
    try:
        status = int(start.get("status") or 0)
    except Exception:
        status = 0
    if status < 200 or status >= 300:
        return False

    try:
        for m in captured:
            await send(m)
        return True
    except Exception:
        return False


def _is_ymd(s: str) -> bool:
    try:
        if not s or len(s) != 10:
            return False
        y, m, d = s.split("-")
        return len(y) == 4 and len(m) == 2 and len(d) == 2 and y.isdigit() and m.isdigit() and d.isdigit()
    except Exception:
        return False


def _today_ymd_utc() -> str:
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d")


def _is_live_state(st: str) -> bool:
    s = (st or "").strip().upper()
    return any(k in s for k in ("LIVE", "IN_PROGRESS", "IN PROGRESS", "IN-PROGRESS", "INTERMISSION", "CRIT", "OT")) and not s.startswith("FINAL")


def _period_disp(game_state: str, period: object, intermission: bool = False) -> str:
    st = (game_state or "").strip().upper()
    if st.startswith("FINAL") or st in {"OFF", "FINAL", "FINAL_OT", "FINAL_SO"}:
        return "Final"
    if st and not _is_live_state(st):
        return ""
    try:
        p = int(period) if period is not None and str(period).strip() != "" else None
    except Exception:
        p = None
    if p is None:
        return "INT" if bool(intermission) else ""
    if p == 1:
        base = "1st"
    elif p == 2:
        base = "2nd"
    elif p == 3:
        base = "3rd"
    elif p >= 4:
        base = "OT"
    else:
        base = ""
    if bool(intermission) and base:
        return f"{base} INT"
    return base


def _normalize_scoreboard_rows(date: str, rows: list[dict]) -> list[dict]:
    # Keep response flexible for the frontend normalizer.
    from .teams import get_team_assets

    out: list[dict] = []
    for r in rows or []:
        try:
            away = str(r.get("away") or "").strip()
            home = str(r.get("home") or "").strip()
            away_assets = get_team_assets(away)
            home_assets = get_team_assets(home)
            away_abbr = (away_assets.get("abbr") or "").upper()
            home_abbr = (home_assets.get("abbr") or "").upper()

            away_score = r.get("awayScore")
            if away_score is None:
                away_score = r.get("away_goals")
            home_score = r.get("homeScore")
            if home_score is None:
                home_score = r.get("home_goals")
            try:
                away_score = int(away_score) if away_score is not None else None
            except Exception:
                away_score = None
            try:
                home_score = int(home_score) if home_score is not None else None
            except Exception:
                home_score = None

            game_state = r.get("gameState") or r.get("game_state") or r.get("state") or ""
            period = r.get("period")
            clock = r.get("clock")
            intermission = bool(r.get("intermission"))
            if intermission:
                clock = None
            per_disp = r.get("period_disp") or _period_disp(str(game_state), period, intermission=intermission)

            game_pk = r.get("gamePk")
            if game_pk is None:
                game_pk = r.get("game_pk")
            try:
                game_pk_int = int(game_pk) if game_pk is not None and str(game_pk).strip() != "" else None
            except Exception:
                game_pk_int = None

            out.append({
                "gamePk": game_pk_int,
                "gameDate": r.get("gameDate") or (date + "T00:00:00Z"),
                "away": away,
                "home": home,
                "away_abbr": away_abbr,
                "home_abbr": home_abbr,
                "awayScore": away_score,
                "homeScore": home_score,
                "away_score": away_score,
                "home_score": home_score,
                "away_goals": away_score,
                "home_goals": home_score,
                "score": {"away": away_score, "home": home_score},
                "gameState": str(game_state),
                "period": period,
                "period_disp": per_disp,
                "intermission": intermission,
                "clock": clock,
            })
        except Exception:
            continue
    return out


def _scoreboard_fetch_sync(date: str) -> list[dict]:
    from ..data.nhl_api_web import NHLWebClient

    # Tight timeout to avoid proxy-level 502s under load
    client = NHLWebClient(rate_limit_per_sec=50.0, timeout=6.0)
    rows = client.scoreboard_day(date)
    # Best-effort: for live games, fetch precise linescore/intermission state.
    for r in rows:
        try:
            st = str(r.get("gameState") or "")
            if not _is_live_state(st):
                continue
            if not r.get("gamePk"):
                continue
            ls = client.linescore(int(r.get("gamePk")))
            if ls:
                if ls.get("intermission") is not None:
                    r["intermission"] = bool(ls.get("intermission"))
                    if bool(ls.get("intermission")):
                        r["clock"] = None
                if ls.get("clock"):
                    r["clock"] = ls.get("clock")
                if ls.get("period") is not None:
                    r["period"] = ls.get("period")
        except Exception:
            continue
    return _normalize_scoreboard_rows(date, rows)


async def _get_scoreboard(date: str, *, ttl_sec: float = 6.0, stale_sec: float = 120.0) -> list[dict]:
    # Cache key includes date only; UI already cache-busts with _=timestamp.
    now = time.time()
    entry = _SCOREBOARD_CACHE.get(date)
    if entry:
        age = now - float(entry.get("ts", 0.0))
        if age <= ttl_sec:
            return entry.get("data") or []
    try:
        data = await asyncio.to_thread(_scoreboard_fetch_sync, date)
        _SCOREBOARD_CACHE[date] = {"ts": now, "data": data}
        return data
    except Exception:
        # Serve stale cache on failure if available
        if entry:
            age = now - float(entry.get("ts", 0.0))
            if age <= stale_sec:
                return entry.get("data") or []
        return []


async def _get_live_lens(date: str, *, inplay: bool, include_non_live: bool) -> dict:
    # Minimal live-lens: derived from scoreboard only.
    cache_key = f"{date}|inplay={1 if inplay else 0}|incl={1 if include_non_live else 0}"
    now = time.time()
    ttl_sec = 6.0 if inplay else 30.0
    stale_sec = 120.0
    entry = _LIVE_LENS_CACHE.get(cache_key)
    if entry:
        age = now - float(entry.get("ts", 0.0))
        if age <= ttl_sec:
            return entry.get("data") or {"ok": True, "date": date, "games": []}
    try:
        sb = await _get_scoreboard(date, ttl_sec=ttl_sec, stale_sec=stale_sec)
        games = []
        for r in sb:
            st = str(r.get("gameState") or "")
            if (not include_non_live) and (not _is_live_state(st)):
                continue
            away = r.get("away")
            home = r.get("home")
            away_abbr = r.get("away_abbr") or ""
            home_abbr = r.get("home_abbr") or ""
            key = None
            if away_abbr and home_abbr:
                key = f"{away_abbr} @ {home_abbr}"
            else:
                key = f"{away} @ {home}"
            games.append({
                "ok": True,
                "key": key,
                "gamePk": r.get("gamePk"),
                "away": away,
                "home": home,
                "away_abbr": away_abbr,
                "home_abbr": home_abbr,
                "gameState": st,
                "period": r.get("period"),
                "period_disp": r.get("period_disp"),
                "intermission": r.get("intermission"),
                "clock": r.get("clock"),
                "score": r.get("score"),
                "total_goals": None,
            })
        payload = {"ok": True, "date": date, "games": games, "source": "asgi-lite"}
        _LIVE_LENS_CACHE[cache_key] = {"ts": now, "data": payload}
        return payload
    except Exception:
        if entry:
            age = now - float(entry.get("ts", 0.0))
            if age <= stale_sec:
                return entry.get("data") or {"ok": True, "date": date, "games": []}
        return {"ok": True, "date": date, "games": [], "source": "asgi-lite"}


def _load_heavy_sync():
    global _HEAVY_APP
    from . import app as _heavy_module
    _HEAVY_APP = _heavy_module.app
    try:
        print("[asgi] heavy app loaded")
    except Exception:
        pass


def ensure_heavy_loaded(background: bool = True) -> bool:
    global _HEAVY_LOADING, _HEAVY_APP
    if _HEAVY_APP is not None:
        return True
    with _HEAVY_LOCK:
        if _HEAVY_APP is not None:
            return True
        if _HEAVY_LOADING:
            return False
        _HEAVY_LOADING = True
        if background:
            def _runner():
                try:
                    _load_heavy_sync()
                finally:
                    global _HEAVY_LOADING
                    _HEAVY_LOADING = False
            threading.Thread(target=_runner, daemon=True).start()
            return False
        else:
            try:
                _load_heavy_sync()
                return True
            finally:
                _HEAVY_LOADING = False


async def _enter_heavy_lifespan() -> bool:
    global _HEAVY_LIFESPAN_CM, _HEAVY_LIFESPAN_ENTERED
    if _HEAVY_LIFESPAN_ENTERED:
        return True
    ensure_heavy_loaded(background=False)
    if _HEAVY_APP is None:
        return False
    try:
        router = getattr(_HEAVY_APP, "router", None)
        cm_factory = getattr(router, "lifespan_context", None)
        if callable(cm_factory):
            _HEAVY_LIFESPAN_CM = cm_factory(_HEAVY_APP)
            await _HEAVY_LIFESPAN_CM.__aenter__()
        _HEAVY_LIFESPAN_ENTERED = True
        try:
            print("[asgi] heavy lifespan entered")
        except Exception:
            pass
        return True
    except Exception:
        _HEAVY_LIFESPAN_CM = None
        _HEAVY_LIFESPAN_ENTERED = False
        raise


async def _exit_heavy_lifespan() -> None:
    global _HEAVY_LIFESPAN_CM, _HEAVY_LIFESPAN_ENTERED
    cm = _HEAVY_LIFESPAN_CM
    entered = _HEAVY_LIFESPAN_ENTERED
    _HEAVY_LIFESPAN_CM = None
    _HEAVY_LIFESPAN_ENTERED = False
    if not entered:
        return
    try:
        if cm is not None:
            await cm.__aexit__(None, None, None)
        try:
            print("[asgi] heavy lifespan exited")
        except Exception:
            pass
    except Exception:
        pass


class WrapperASGI:
    async def __call__(self, scope, receive, send):
        scope_type = scope.get("type")
        if scope_type != "http":
            # Minimal lifespan protocol support to avoid warnings from servers
            if scope_type == "lifespan":
                try:
                    while True:
                        message = await receive()
                        mtype = message.get("type")
                        if mtype == "lifespan.startup":
                            try:
                                ok = await _enter_heavy_lifespan()
                                if not ok:
                                    await send({"type": "lifespan.startup.failed", "message": "heavy app failed to load"})
                                    return
                            except Exception as e:
                                await send({"type": "lifespan.startup.failed", "message": str(e)})
                                return
                            await send({"type": "lifespan.startup.complete"})
                        elif mtype == "lifespan.shutdown":
                            try:
                                await _exit_heavy_lifespan()
                            except Exception:
                                pass
                            await send({"type": "lifespan.shutdown.complete"})
                            return
                except Exception:
                    # On error, signal failure to the server
                    try:
                        await send({"type": "lifespan.startup.failed", "message": "lifespan error"})
                    except Exception:
                        pass
                    return
            # For other non-http, try heavy if available, else 503
            if _HEAVY_APP is not None:
                return await _HEAVY_APP(scope, receive, send)
            return await self._send_text(send, 503, b"Service warming up")

        path = scope.get("path") or "/"
        method = scope.get("method") or "GET"
        # Handle health without heavy imports
        if path == "/health":
            body = json.dumps({
                "status": "ok",
                "heavy_ready": bool(_HEAVY_APP is not None),
                "heavy_lifespan": bool(_HEAVY_LIFESPAN_ENTERED),
            }).encode("utf-8")
            return await self._send_json(send, 200, body)
        if path == "/ready":
            # Readiness endpoint separate from health (no side-effects)
            body = json.dumps({"ready": bool(_HEAVY_APP is not None and _HEAVY_LIFESPAN_ENTERED)}).encode("utf-8")
            return await self._send_json(send, 200, body)
        if path == "/warmup":
            # Trigger heavy load + lifespan explicitly.
            try:
                await _enter_heavy_lifespan()
            except Exception:
                try:
                    ensure_heavy_loaded(background=True)
                except Exception:
                    pass
            body = json.dumps({
                "ok": True,
                "heavy_ready": bool(_HEAVY_APP is not None),
                "heavy_lifespan": bool(_HEAVY_LIFESPAN_ENTERED),
            }).encode("utf-8")
            return await self._send_json(send, 200, body)
        if path == "/" and method == "HEAD":
            return await self._send_empty(send, 204)

        # Ultra-light live scoreboard (no heavy app required)
        if path == "/api/scoreboard":
            try:
                raw_qs = scope.get("query_string") or b""
                q = parse_qs(raw_qs.decode("utf-8"), keep_blank_values=False)
                date = (q.get("date", [""])[0] or "").strip() or None
                if not date:
                    date = _today_ymd_utc()
                if not _is_ymd(date):
                    date = _today_ymd_utc()
                rows = await _get_scoreboard(date)
                body = json.dumps(rows, ensure_ascii=False).encode("utf-8")
                return await self._send_json(send, 200, body)
            except Exception:
                return await self._send_json(send, 200, b"[]")

        # Heavy-first live-lens endpoint. This endpoint carries guidance/signals and should
        # bootstrap the heavy app if needed instead of silently degrading when possible.
        if path.startswith("/v1/live-lens/"):
            try:
                parts = [p for p in path.split("/") if p]
                date = parts[-1] if parts else ""
                if not _is_ymd(date):
                    date = _today_ymd_utc()
                raw_qs = scope.get("query_string") or b""
                q = parse_qs(raw_qs.decode("utf-8"), keep_blank_values=False)
                inplay = bool(int((q.get("inplay", ["1"])[0] or "1")))
                include_non_live = bool(int((q.get("include_non_live", ["0"])[0] or "0")))
                include_pbp = bool(int((q.get("include_pbp", ["0"])[0] or "0")))

                global _HEAVY_LIVE_LENS_FAILS, _HEAVY_LIVE_LENS_COOLDOWN_UNTIL
                try:
                    if _HEAVY_APP is None or not _HEAVY_LIFESPAN_ENTERED:
                        await _enter_heavy_lifespan()
                except Exception:
                    try:
                        ensure_heavy_loaded(background=True)
                    except Exception:
                        pass

                if _HEAVY_APP is not None:
                    heavy_timeout = 45.0 if include_pbp else (20.0 if inplay else 30.0)
                    used = await _try_heavy_http(scope, receive, send, timeout_sec=heavy_timeout)
                    if used:
                        _HEAVY_LIVE_LENS_FAILS = 0
                        _HEAVY_LIVE_LENS_COOLDOWN_UNTIL = 0.0
                        return
                    _HEAVY_LIVE_LENS_FAILS = int(_HEAVY_LIVE_LENS_FAILS or 0) + 1
                    _HEAVY_LIVE_LENS_COOLDOWN_UNTIL = time.time() + 15.0

                payload = await _get_live_lens(date, inplay=inplay, include_non_live=include_non_live)
                try:
                    if isinstance(payload, dict):
                        payload.setdefault("warning", "heavy_live_lens_unavailable")
                        payload["heavy_ready"] = bool(_HEAVY_APP is not None)
                        payload["heavy_lifespan"] = bool(_HEAVY_LIFESPAN_ENTERED)
                except Exception:
                    pass
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                return await self._send_json(send, 200, body)
            except Exception:
                return await self._send_json(send, 200, b"{\"ok\":true,\"date\":\"\",\"games\":[]}")

        # Ultra-light live scoreboard / lite live view.
        if path.startswith("/v1/live/"):
            try:
                parts = [p for p in path.split("/") if p]
                date = parts[-1] if parts else ""
                if not _is_ymd(date):
                    date = _today_ymd_utc()
                raw_qs = scope.get("query_string") or b""
                q = parse_qs(raw_qs.decode("utf-8"), keep_blank_values=False)
                inplay = bool(int((q.get("inplay", ["1"])[0] or "1")))
                include_non_live = bool(int((q.get("include_non_live", ["0"])[0] or "0")))

                now = time.time()
                if _HEAVY_APP is not None and now >= float(_HEAVY_LIVE_LENS_COOLDOWN_UNTIL or 0.0):
                    heavy_timeout = 12.0 if inplay else 20.0
                    used = await _try_heavy_http(scope, receive, send, timeout_sec=heavy_timeout)
                    if used:
                        _HEAVY_LIVE_LENS_FAILS = 0
                        _HEAVY_LIVE_LENS_COOLDOWN_UNTIL = 0.0
                        return
                    _HEAVY_LIVE_LENS_FAILS = int(_HEAVY_LIVE_LENS_FAILS or 0) + 1
                    cooldown = min(120.0, 5.0 * (2.0 ** min(int(_HEAVY_LIVE_LENS_FAILS), 4)))
                    _HEAVY_LIVE_LENS_COOLDOWN_UNTIL = now + float(cooldown)
                elif _HEAVY_APP is None:
                    try:
                        ensure_heavy_loaded(background=True)
                    except Exception:
                        pass

                payload = await _get_live_lens(date, inplay=inplay, include_non_live=include_non_live)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                return await self._send_json(send, 200, body)
            except Exception:
                return await self._send_json(send, 200, b"{\"ok\":true,\"date\":\"\",\"games\":[]}")

        # Ultra-light CSV-backed JSON for props projections (no heavy app required)
        if path == "/api/props/all.json":
            try:
                # Parse query params
                raw_qs = scope.get("query_string") or b""
                q = parse_qs(raw_qs.decode("utf-8"), keep_blank_values=False)
                date = (q.get("date", [""])[0] or "").strip() or None
                team = (q.get("team", [""])[0] or "").strip()
                market = (q.get("market", [""])[0] or "").strip()
                try:
                    page = int((q.get("page", ["1"])[0] or "1"))
                except Exception:
                    page = 1
                try:
                    page_size = int((q.get("page_size", ["250"])[0] or "250"))
                except Exception:
                    page_size = 250
                try:
                    top = int((q.get("top", ["0"])[0] or "0"))
                except Exception:
                    top = 0
                # Enforce page size cap from env
                try:
                    import os as _os
                    cap_ps = int((_os.getenv("PROPS_PAGE_SIZE") or "0"))
                    if cap_ps and page_size > cap_ps:
                        page_size = cap_ps
                except Exception:
                    pass
                # Default date fallback (UTC today)
                if not date:
                    from datetime import datetime
                    date = datetime.utcnow().strftime("%Y-%m-%d")
                # Local-file only: avoid outbound network during warmup to prevent 502s
                import csv as _csv
                import pathlib
                rel = f"data/processed/props_projections_all_{date}.csv"
                rows = []
                try:
                    # repo root = nhl_betting (parents[1])'s parent => parents[2]
                    p = pathlib.Path(__file__).resolve().parents[2] / rel
                    if p.exists():
                        with p.open("r", encoding="utf-8") as fh:
                            reader = _csv.DictReader(fh)
                            rows = list(reader)
                except Exception:
                    rows = []
                total_rows = len(rows)
                # Filter
                if team:
                    tu = team.upper()
                    rows = [r for r in rows if (r.get("team") or "").upper() == tu]
                if market:
                    mu = market.upper()
                    rows = [r for r in rows if (r.get("market") or "").upper() == mu]
                # Top cap
                if top and top > 0:
                    try:
                        import os as _os
                        cap_top = int((_os.getenv("PROPS_MAX_ROWS") or "0"))
                        if cap_top > 0 and top > cap_top:
                            top = cap_top
                    except Exception:
                        pass
                    rows = rows[: top]
                filtered_rows = len(rows)
                # Pagination
                if page <= 0:
                    page = 1
                if page_size <= 0:
                    page_size = 250
                total_pages = (filtered_rows + page_size - 1) // page_size if filtered_rows else 0
                if total_pages == 0:
                    page = 1
                elif page > total_pages:
                    page = total_pages
                start = (page - 1) * page_size
                end = start + page_size
                page_rows = rows[start:end]
                payload = {
                    "date": date,
                    "data": page_rows,
                    "total_rows": total_rows,
                    "filtered_rows": filtered_rows,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": total_pages,
                }
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                return await self._send_json(send, 200, body)
            except Exception:
                # Always return a well-formed empty payload on failure (never 5xx)
                body = b'{"date":"","data":[],"total_rows":0,"filtered_rows":0,"page":1,"page_size":250,"total_pages":0}'
                return await self._send_json(send, 200, body)

        # Lightweight CSV-backed JSON for props recommendations (no heavy app required)
        if path == "/api/props/recommendations.json":
            try:
                raw_qs = scope.get("query_string") or b""
                q = parse_qs(raw_qs.decode("utf-8"), keep_blank_values=False)
                date = (q.get("date", [""])[0] or "").strip() or None
                market = (q.get("market", [""])[0] or "").strip()
                team = (q.get("team", [""])[0] or "").strip()
                try:
                    min_ev = float((q.get("min_ev", ["0"]) [0] or "0"))
                except Exception:
                    min_ev = 0.0
                sort = (q.get("sort", ["ev_desc"]) [0] or "ev_desc").strip().lower()
                try:
                    page = int((q.get("page", ["1"]) [0] or "1"))
                except Exception:
                    page = 1
                try:
                    page_size = int((q.get("page_size", ["250"]) [0] or "250"))
                except Exception:
                    page_size = 250
                try:
                    top = int((q.get("top", ["0"]) [0] or "0"))
                except Exception:
                    top = 0
                # Enforce page size cap from env
                try:
                    import os as _os
                    cap_ps = int((_os.getenv("PROPS_PAGE_SIZE") or "0"))
                    if cap_ps and page_size > cap_ps:
                        page_size = cap_ps
                except Exception:
                    pass
                if not date:
                    from datetime import datetime
                    date = datetime.utcnow().strftime("%Y-%m-%d")

                import csv as _csv
                import pathlib
                rel = f"data/processed/props_recommendations_{date}.csv"
                rows = []
                try:
                    p = pathlib.Path(__file__).resolve().parents[2] / rel
                    if p.exists():
                        with p.open("r", encoding="utf-8") as fh:
                            reader = _csv.DictReader(fh)
                            rows = list(reader)
                except Exception:
                    rows = []
                total_rows = len(rows)
                # Normalize types and filter
                def _flt(x, default=None):
                    try:
                        return float(x)
                    except Exception:
                        return default
                if market:
                    mu = market.upper()
                    rows = [r for r in rows if (r.get("market") or "").upper() == mu]
                if team:
                    tu = team.upper()
                    rows = [r for r in rows if (r.get("team") or "").upper() == tu]
                if min_ev and min_ev > 0:
                    rows = [r for r in rows if _flt(r.get("ev"), -1e9) >= min_ev]
                # Sort
                key = None; asc = False
                if sort == "ev_desc": key = "ev"; asc = False
                elif sort == "ev_asc": key = "ev"; asc = True
                elif sort == "p_over_desc": key = "p_over"; asc = False
                elif sort == "p_over_asc": key = "p_over"; asc = True
                elif sort == "name": key = "player"; asc = True
                elif sort == "team": key = "team"; asc = True
                elif sort == "market": key = "market"; asc = True
                if key:
                    try:
                        if key in ("ev","p_over"):
                            rows.sort(key=lambda r: _flt(r.get(key), float("nan")), reverse=not asc)
                        else:
                            rows.sort(key=lambda r: (r.get(key) or ""), reverse=not asc)
                    except Exception:
                        pass
                # Top cap
                if top and top > 0:
                    try:
                        import os as _os
                        cap_top = int((_os.getenv("PROPS_MAX_ROWS") or "0"))
                        if cap_top > 0 and top > cap_top:
                            top = cap_top
                    except Exception:
                        pass
                    rows = rows[: top]
                filtered_rows = len(rows)
                # Pagination
                if page <= 0: page = 1
                if page_size <= 0: page_size = 250
                total_pages = (filtered_rows + page_size - 1) // page_size if filtered_rows else 0
                if total_pages == 0:
                    page = 1
                elif page > total_pages:
                    page = total_pages
                start = (page - 1) * page_size
                end = start + page_size
                page_rows = rows[start:end]
                payload = {
                    "date": date,
                    "data": page_rows,
                    "total_rows": total_rows,
                    "filtered_rows": filtered_rows,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": total_pages,
                }
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                return await self._send_json(send, 200, body)
            except Exception:
                body = b'{"date":"","data":[],"total_rows":0,"filtered_rows":0,"page":1,"page_size":250,"total_pages":0}'
                return await self._send_json(send, 200, body)

        # Proxy others
        if _HEAVY_APP is None:
            # Kick off heavy load in background; never block health/readiness here
            try:
                ensure_heavy_loaded(background=True)
            except Exception:
                pass
            # Fast, friendly warmup response without 5xx to avoid upstream 502s
            if method in ("HEAD", "OPTIONS"):
                return await self._send_empty(send, 204)
            warm_html = (
                b"<html><head><meta http-equiv=\"refresh\" content=\"2;url=/\" /></head>"
                b"<body><p>Service warming up. This will auto-retry in 2s...</p>"
                b"<p>If not redirected, <a href=\"/\">click here</a>.</p></body></html>"
            )
            return await self._send_html(send, 200, warm_html)

        return await _HEAVY_APP(scope, receive, send)

    async def _send_empty(self, send, status):
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _send_text(self, send, status, body: bytes, headers: Optional[dict] = None):
        hdrs = [(b"content-type", b"text/plain; charset=utf-8")]
        if headers:
            hdrs.extend(list(headers.items()))
        await send({"type": "http.response.start", "status": status, "headers": hdrs})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _send_html(self, send, status, body: bytes, headers: Optional[dict] = None):
        hdrs = [(b"content-type", b"text/html; charset=utf-8")]
        if headers:
            hdrs.extend(list(headers.items()))
        await send({"type": "http.response.start", "status": status, "headers": hdrs})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _send_json(self, send, status, body: bytes):
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": body, "more_body": False})


app = WrapperASGI()

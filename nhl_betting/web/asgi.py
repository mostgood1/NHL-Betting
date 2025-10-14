from __future__ import annotations

import threading
from typing import Optional
from urllib.parse import parse_qs

import json

# Minimal ASGI wrapper. No FastAPI here to keep cold-start as small as possible.

_HEAVY_APP = None  # type: Optional[object]
_HEAVY_LOCK = threading.Lock()
_HEAVY_LOADING = False


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


class WrapperASGI:
    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            # For non-http, try heavy if available, else 503
            if _HEAVY_APP is not None:
                return await _HEAVY_APP(scope, receive, send)
            return await self._send_text(send, 503, b"Service warming up")

        path = scope.get("path") or "/"
        method = scope.get("method") or "GET"
        # Handle health without heavy imports
        if path == "/health":
            body = b'{"status":"ok","heavy_ready":%s}' % (b"true" if _HEAVY_APP is not None else b"false")
            return await self._send_json(send, 200, body)
        if path == "/ready":
            # Readiness endpoint separate from health (no side-effects)
            body = b'{"ready":%s}' % (b"true" if _HEAVY_APP is not None else b"false")
            return await self._send_json(send, 200, body)
        if path == "/warmup":
            # Trigger background warmup explicitly
            ensure_heavy_loaded(background=True)
            body = b'{"ok":true,"heavy_ready":%s}' % (b"true" if _HEAVY_APP is not None else b"false")
            return await self._send_json(send, 200, body)
        if path == "/" and method == "HEAD":
            return await self._send_empty(send, 204)

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

        # Proxy others
        if _HEAVY_APP is None:
            ensure_heavy_loaded(background=True)
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

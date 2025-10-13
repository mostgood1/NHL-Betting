from __future__ import annotations

import os
import threading
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse

# Lightweight wrapper app that stays fast on cold start.
# It lazily loads the heavy FastAPI app (which imports numpy/pandas, etc.)
# and proxies all non-health requests to it when ready.

app = FastAPI()

_HEAVY_APP = None  # type: Optional[object]
_HEAVY_LOCK = threading.Lock()
_HEAVY_LOADING = False


def _load_heavy_sync():
    global _HEAVY_APP
    # Import inside function to defer heavy module import cost
    from . import app as _heavy_module  # noqa: F401
    _HEAVY_APP = _heavy_module.app


def ensure_heavy_loaded(background: bool = True) -> bool:
    """Ensure the heavy app is imported. If background=True, spawn a thread."""
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
                    # Allow subsequent calls even if failed
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


@app.get("/health")
async def health():
    return {"status": "ok", "heavy_ready": bool(_HEAVY_APP is not None)}


@app.get("/warmup")
async def warmup(async_run: int = 1):
    """Trigger heavy app import; async by default."""
    ready = ensure_heavy_loaded(background=bool(async_run))
    return {"ok": True, "heavy_ready": bool(_HEAVY_APP is not None), "started": not ready}


@app.head("/")
async def head_root():
    return PlainTextResponse("", status_code=204)


@app.get("/")
async def root():
    if _HEAVY_APP is None:
        # Minimal landing while warming
        return HTMLResponse(
            """
            <html><head><title>NHL Betting</title></head>
            <body>
              <h3>Service starting…</h3>
              <p>Warming up. Please <a href=\"/props/all\">retry</a> in a few seconds.</p>
            </body></html>
            """,
            status_code=200,
        )
    # Delegate to heavy root
    async def _call(scope, receive, send):
        await _HEAVY_APP(scope, receive, send)
    return await _call


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])  # catchall
async def proxy_all(request: Request, full_path: str):
    # Keep health/warmup on wrapper
    if full_path in ("health", "warmup"):
        return JSONResponse({"ok": True, "note": "handled by wrapper"})

    if _HEAVY_APP is None:
        # Kick off background load and return a quick response to avoid timeouts
        ensure_heavy_loaded(background=True)
        # For HEAD/OPTIONS keep it empty
        if request.method in ("HEAD", "OPTIONS"):
            return PlainTextResponse("", status_code=204)
        return HTMLResponse(
            "<html><body><p>Warming up app, please retry in a few seconds…</p></body></html>",
            status_code=503,
            headers={"Retry-After": "2"},
        )

    # Call the heavy app as ASGI
    async def _call(scope, receive, send):
        await _HEAVY_APP(scope, receive, send)
    return await _call

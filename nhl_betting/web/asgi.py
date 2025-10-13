from __future__ import annotations

import threading
from typing import Optional

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
        if path == "/warmup":
            ensure_heavy_loaded(background=True)
            body = b'{"ok":true,"heavy_ready":%s}' % (b"true" if _HEAVY_APP is not None else b"false")
            return await self._send_json(send, 200, body)
        if path == "/" and method == "HEAD":
            return await self._send_empty(send, 204)

        # Proxy others
        if _HEAVY_APP is None:
            ensure_heavy_loaded(background=True)
            # Fast, friendly warmup response
            if method in ("HEAD", "OPTIONS"):
                return await self._send_empty(send, 204)
            return await self._send_html(send, 503, b"<html><body><p>Warming up... retry shortly.</p></body></html>", {b"Retry-After": b"2"})

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

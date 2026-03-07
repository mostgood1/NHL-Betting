import asyncio
import json
from contextlib import asynccontextmanager

import nhl_betting.web.asgi as asgi_mod
from nhl_betting.data.nhl_api_web import NHLWebClient


def test_wrapper_enters_heavy_lifespan(monkeypatch):
    events = []

    class FakeRouter:
        @asynccontextmanager
        async def lifespan_context(self, app):
            events.append("enter")
            yield
            events.append("exit")

    class FakeApp:
        router = FakeRouter()

    monkeypatch.setattr(asgi_mod, "_HEAVY_APP", FakeApp(), raising=False)
    monkeypatch.setattr(asgi_mod, "ensure_heavy_loaded", lambda background=False: True, raising=True)
    monkeypatch.setattr(asgi_mod, "_HEAVY_LIFESPAN_CM", None, raising=False)
    monkeypatch.setattr(asgi_mod, "_HEAVY_LIFESPAN_ENTERED", False, raising=False)

    async def _run():
        await asgi_mod._enter_heavy_lifespan()
        assert events == ["enter"]
        assert asgi_mod._HEAVY_LIFESPAN_ENTERED is True

        await asgi_mod._exit_heavy_lifespan()
        assert events == ["enter", "exit"]
        assert asgi_mod._HEAVY_LIFESPAN_ENTERED is False

    asyncio.run(_run())


async def _collect_http(app, path: str, query_string: bytes = b""):
    sent = []

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": query_string,
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }
    await app(scope, _receive, _send)
    start = next(m for m in sent if m.get("type") == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m.get("type") == "http.response.body")
    return int(start.get("status") or 0), body


def test_ready_requires_heavy_lifespan(monkeypatch):
    monkeypatch.setattr(asgi_mod, "_HEAVY_APP", object(), raising=False)
    monkeypatch.setattr(asgi_mod, "_HEAVY_LIFESPAN_ENTERED", False, raising=False)

    status, body = asyncio.run(_collect_http(asgi_mod.WrapperASGI(), "/ready"))
    assert status == 200
    obj = json.loads(body.decode("utf-8"))
    assert obj["ready"] is False


def test_live_lens_prefers_heavy_and_skips_lite(monkeypatch):
    monkeypatch.setattr(asgi_mod, "_HEAVY_APP", object(), raising=False)
    monkeypatch.setattr(asgi_mod, "_HEAVY_LIFESPAN_ENTERED", True, raising=False)
    monkeypatch.setattr(asgi_mod, "_HEAVY_LIVE_LENS_FAILS", 0, raising=False)
    monkeypatch.setattr(asgi_mod, "_HEAVY_LIVE_LENS_COOLDOWN_UNTIL", 0.0, raising=False)

    calls = {"heavy": 0, "lite": 0}

    async def _fake_try_heavy_http(scope, receive, send, *, timeout_sec: float):
        calls["heavy"] += 1
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"ok":true,"source":"heavy"}', "more_body": False})
        return True

    async def _fake_get_live_lens(date: str, *, inplay: bool, include_non_live: bool):
        calls["lite"] += 1
        return {"ok": True, "date": date, "games": [], "source": "asgi-lite"}

    monkeypatch.setattr(asgi_mod, "_try_heavy_http", _fake_try_heavy_http, raising=True)
    monkeypatch.setattr(asgi_mod, "_get_live_lens", _fake_get_live_lens, raising=True)

    status, body = asyncio.run(
        _collect_http(
            asgi_mod.WrapperASGI(),
            "/v1/live-lens/2099-01-01",
            b"inplay=1&include_non_live=1&include_pbp=1",
        )
    )

    assert status == 200
    obj = json.loads(body.decode("utf-8"))
    assert obj["source"] == "heavy"
    assert calls["heavy"] == 1
    assert calls["lite"] == 0


def test_lite_scoreboard_marks_web_intermission(monkeypatch):
    def _scoreboard_day(self, date: str):
        assert date == "2099-01-03"
        return [
            {
                "gamePk": 789,
                "gameDate": f"{date}T00:00:00Z",
                "away": "Florida Panthers",
                "home": "Detroit Red Wings",
                "away_goals": 1,
                "home_goals": 1,
                "gameState": "LIVE",
                "period": 2,
                "clock": "14:06",
            }
        ]

    def _linescore(self, gamePk: int):
        assert int(gamePk) == 789
        return {"period": 2, "clock": None, "source": "boxscore", "intermission": True}

    monkeypatch.setattr(NHLWebClient, "scoreboard_day", _scoreboard_day, raising=True)
    monkeypatch.setattr(NHLWebClient, "linescore", _linescore, raising=True)
    monkeypatch.setattr(asgi_mod, "_SCOREBOARD_CACHE", {}, raising=False)

    status, body = asyncio.run(
        _collect_http(asgi_mod.WrapperASGI(), "/api/scoreboard", b"date=2099-01-03")
    )

    assert status == 200
    rows = json.loads(body.decode("utf-8"))
    assert isinstance(rows, list) and len(rows) == 1
    row = rows[0]
    assert row["gamePk"] == 789
    assert row["intermission"] is True
    assert row["clock"] is None
    assert row["period_disp"] == "2nd INT"

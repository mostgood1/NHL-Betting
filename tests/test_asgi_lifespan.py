import asyncio
from contextlib import asynccontextmanager

import nhl_betting.web.asgi as asgi_mod


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

    asyncio.run(asgi_mod._enter_heavy_lifespan())
    assert events == ["enter"]
    assert asgi_mod._HEAVY_LIFESPAN_ENTERED is True

    asyncio.run(asgi_mod._exit_heavy_lifespan())
    assert events == ["enter", "exit"]
    assert asgi_mod._HEAVY_LIFESPAN_ENTERED is False

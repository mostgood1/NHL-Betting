from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import nhl_betting.web.app as web_app


client = TestClient(web_app.app)


def test_live_lens_tick_requires_token(monkeypatch):
    monkeypatch.setenv("REFRESH_CRON_TOKEN", "TICK_TOKEN")
    setattr(web_app.app.state, "live_lens_tick_last_run_by_key", {})

    resp = client.get("/api/cron/live-lens-tick")

    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_live_lens_tick_default_date_uses_previous_day_before_cutoff(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 3, 17, 1, 30, tzinfo=ZoneInfo("America/New_York"))
            if tz is None:
                return base.replace(tzinfo=None)
            return base.astimezone(tz)

    monkeypatch.setattr(web_app, "datetime", FixedDateTime, raising=True)

    assert web_app._resolve_live_lens_tick_date(None, cutoff_hour_et=6) == "2026-03-16"
    assert web_app._resolve_live_lens_tick_date("today", cutoff_hour_et=6) == "2026-03-17"


def test_live_lens_tick_endpoint_throttles_duplicate_calls(monkeypatch):
    monkeypatch.setenv("REFRESH_CRON_TOKEN", "TICK_TOKEN")
    monkeypatch.setattr(web_app.app.state, "live_lens_tick_last_run_by_key", {})

    calls = []

    async def fake_v1_live_lens_combined(request, date, regions, best, include_non_live, inplay, include_pbp):
        calls.append(
            {
                "path": request.url.path,
                "date": date,
                "regions": regions,
                "best": best,
                "include_non_live": include_non_live,
                "inplay": inplay,
                "include_pbp": include_pbp,
            }
        )
        return JSONResponse(
            {
                "ok": True,
                "date": date,
                "asof_utc": "2026-03-16T23:10:00+00:00",
                "odds_asof_utc": "2026-03-16T23:09:00+00:00",
                "games": [
                    {"gameState": "LIVE", "signals": [{"market": "total"}]},
                    {"gameState": "OFF", "signals": []},
                ],
            }
        )

    monkeypatch.setattr(web_app, "v1_live_lens_combined", fake_v1_live_lens_combined, raising=True)
    monkeypatch.setattr(
        web_app,
        "_resolve_live_lens_tick_date",
        lambda date, cutoff_hour_et=6: "2026-03-16",
        raising=True,
    )

    headers = {"Authorization": "Bearer TICK_TOKEN"}
    resp1 = client.get("/api/cron/live-lens-tick?min_interval_sec=3600", headers=headers)
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert body1["ok"] is True
    assert body1["skipped"] is False
    assert body1["date"] == "2026-03-16"
    assert body1["games"] == 2
    assert body1["live_games"] == 1
    assert body1["final_games"] == 1
    assert body1["signal_games"] == 1
    assert body1["signals"] == 1
    assert len(calls) == 1
    assert calls[0]["path"] == "/v1/live-lens/2026-03-16"

    resp2 = client.get("/api/cron/live-lens-tick?min_interval_sec=3600", headers=headers)
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["ok"] is True
    assert body2["skipped"] is True
    assert body2["reason"] == "min_interval"
    assert len(calls) == 1

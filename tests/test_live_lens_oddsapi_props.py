import pandas as pd
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import nhl_betting.web.app as web_app


@pytest.mark.parametrize("side", ["OVER", "UNDER"])
def test_live_lens_prop_signals_use_oddsapi_prices(monkeypatch: pytest.MonkeyPatch, side: str):
    date = "2099-01-01"
    game_key = "mtl @ bos"

    async def fake_live_payload(d: str):
        assert d == date
        return {
            "ok": True,
            "date": date,
            "asof_utc": "2099-01-01T00:00:00Z",
            "games": [
                {
                    "date": date,
                    "gamePk": 1,
                    "home": "BOS",
                    "away": "MTL",
                    "key": game_key,
                    "gameState": "LIVE",
                    "period": 2,
                    "clock": "10:00",
                    "score": {"home": 1, "away": 1},
                    "lens": {
                        "totals": {"home": {"sog": 10}, "away": {"sog": 12}},
                        "players": {
                            "home": [
                                {
                                    "name": "Test Skater",
                                    "s": 2,
                                    "g": 0,
                                    "a": 0,
                                    "p": 0,
                                    "toi": "10:00",
                                }
                            ],
                            "away": [],
                        },
                        "goalies": {
                            "home": [{"name": "Test Goalie", "saves": 11, "shots_against": 12, "sv_pct": 0.917}],
                            "away": [{"name": "Away Goalie", "saves": 9, "shots_against": 10, "sv_pct": 0.9}],
                        },
                    },
                }
            ],
        }

    def fake_v1_odds_payload(*_args, **_kwargs):
        return {
            "ok": True,
            "asof_utc": "2099-01-01T00:00:00Z",
            "games": [
                {
                    "date": date,
                    "home": "BOS",
                    "away": "MTL",
                    "key": game_key,
                    "h2h": {"home": -115, "away": 105},
                    "totals": {"line": 5.5, "over": -110, "under": -110},
                    "spreads": {"home": -110, "away": -110, "line": 1.5},
                }
            ],
        }

    def fake_v1_props_odds_payload(*_args, **_kwargs):
        # Provide both sides so the code can select best edge.
        return {
            "ok": True,
            "date": date,
            "asof_utc": "2099-01-01T00:00:00Z",
            "games": [
                {
                    "date": date,
                    "home": "BOS",
                    "away": "MTL",
                    "key": game_key,
                    "props": {
                        "SOG": [
                            {
                                "player": "Test Skater",
                                "line": 3.5,
                                "over": 120 if side == "OVER" else -110,
                                "under": -140 if side == "OVER" else 120,
                            }
                        ]
                    },
                }
            ],
        }

    def fake_read_all_players_projections(d: str):
        assert d == date
        # Full-game lambda; remaining-time scaling happens in the endpoint.
        return pd.DataFrame(
            [
                {
                    "date": date,
                    "player": "Test Skater",
                    "team": "BOS",
                    "position": "F",
                    "market": "SOG",
                    "proj_lambda": 4.2,
                }
            ]
        )

    monkeypatch.setattr(web_app, "_v1_live_payload", fake_live_payload, raising=True)
    monkeypatch.setattr(web_app, "_v1_odds_payload", fake_v1_odds_payload, raising=True)
    monkeypatch.setattr(web_app, "_v1_props_odds_payload", fake_v1_props_odds_payload, raising=True)
    monkeypatch.setattr(web_app, "_read_all_players_projections", fake_read_all_players_projections, raising=True)
    monkeypatch.setattr(web_app, "_load_bundle_predictions_map", lambda _d: {}, raising=True)

    # Avoid cross-test cache contamination (live-lens payload is cached in-memory).
    try:
        web_app._LIVE_LENS_CACHE.clear()
    except Exception:
        pass

    client = TestClient(web_app.app)
    resp = client.get(f"/v1/live-lens/{date}?regions=us&best=1&include_non_live=1&inplay=1")
    assert resp.status_code == 200
    js = resp.json()
    assert js.get("ok") is True
    assert js.get("games")

    g0 = js["games"][0]
    sigs = g0.get("signals") or []

    # We expect at least one priced prop signal.
    priced = [s for s in sigs if s.get("market") == "PROP_SOG" and s.get("player") == "Test Skater"]
    assert priced, "Expected PROP_SOG signal for Test Skater"

    s0 = priced[0]
    assert s0.get("price") is not None
    assert s0.get("implied") is not None
    assert s0.get("edge") is not None
    assert isinstance(s0.get("driver_tags"), list)
    assert isinstance(s0.get("driver_meta"), dict)

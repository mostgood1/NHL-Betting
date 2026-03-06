import pandas as pd
from fastapi.testclient import TestClient

import nhl_betting.web.app as web_app
from nhl_betting.data.nhl_api_web import NHLWebClient
from nhl_betting.web.app import app


client = TestClient(app)


def test_v1_live_lens_smoke(monkeypatch):
    # Lightweight schedule scoreboard payload (NHLWebClient.scoreboard_day)
    def _scoreboard_day(self, date: str):
        return [
            {
                "gamePk": 123,
                "gameDate": f"{date}T00:00:00Z",
                "home": "Boston Bruins",
                "away": "Montreal Canadiens",
                "home_goals": 1,
                "away_goals": 2,
                "gameState": "LIVE",
                "period": 2,
                "clock": "10:00",
            }
        ]

    # Minimal-ish boxscore payload; endpoint should parse best-effort.
    def _boxscore(self, gamePk: int):
        assert int(gamePk) == 123
        return {
            "homeTeam": {
                "score": 1,
                "teamStats": {
                    "sog": 21,
                    "faceoffWinningPctg": 52.3,
                    "powerPlayConversion": "1/3",
                },
            },
            "awayTeam": {
                "score": 2,
                "teamStats": {
                    "sog": 18,
                    "faceoffWinningPctg": 47.7,
                    "powerPlayConversion": "0/2",
                },
            },
            "periods": [
                {
                    "periodDescriptor": {"number": 1},
                    "home": {"goals": 0, "sog": 10},
                    "away": {"goals": 1, "sog": 8},
                },
                {
                    "periodDescriptor": {"number": 2},
                    "home": {"goals": 1, "sog": 11},
                    "away": {"goals": 1, "sog": 10},
                },
            ],
            "playerByGameStats": {
                "homeTeam": {
                    "goalies": [
                        {
                            "name": {"default": "Home Goalie"},
                            "saves": 16,
                            "shotsAgainst": 18,
                            "savePctg": 0.8889,
                        }
                    ]
                },
                "awayTeam": {
                    "goalies": [
                        {
                            "name": {"default": "Away Goalie"},
                            "saves": 20,
                            "shotsAgainst": 21,
                            "savePctg": 0.9524,
                        }
                    ]
                },
            },
        }

    monkeypatch.setattr(NHLWebClient, "scoreboard_day", _scoreboard_day, raising=True)
    monkeypatch.setattr(NHLWebClient, "boxscore", _boxscore, raising=True)

    # Clear module-level cache so we can validate ETag/304 behavior deterministically.
    import nhl_betting.web.app as web_app
    try:
        web_app._LIVE_LENS_CACHE.clear()
    except Exception:
        pass

    r = client.get("/v1/live/2099-01-01")
    assert r.status_code == 200
    etag = r.headers.get("etag")
    assert isinstance(etag, str) and len(etag) > 0
    assert "cache-control" in {k.lower() for k in r.headers.keys()}
    assert "Accept-Encoding" in str(r.headers.get("vary") or "")

    r2 = client.get("/v1/live/2099-01-01", headers={"If-None-Match": etag})
    assert r2.status_code == 304

    obj = r.json()
    assert obj.get("ok") is True
    assert obj.get("date") == "2099-01-01"
    games = obj.get("games")
    assert isinstance(games, list)
    assert len(games) == 1

    g0 = games[0]
    assert g0["gamePk"] == 123
    assert g0["home"] == "Boston Bruins"
    assert g0["away"] == "Montreal Canadiens"
    assert "lens" in g0

    lens = g0["lens"]
    assert "periods" in lens
    assert len(lens["periods"]) == 2
    assert lens["periods"][0]["period"] == 1
    assert lens["periods"][0]["home"]["sog"] == 10
    assert lens["periods"][0]["away"]["sog"] == 8


def test_v1_live_lens_late_empty_net_projection(monkeypatch):
    date = "2099-01-03"
    game_key = "nyr @ bos"

    async def fake_live_payload(d: str):
        assert d == date
        return {
            "ok": True,
            "date": date,
            "asof_utc": "2099-01-03T00:00:00Z",
            "games": [
                {
                    "date": date,
                    "gamePk": 99,
                    "home": "BOS",
                    "away": "NYR",
                    "key": game_key,
                    "gameState": "LIVE",
                    "period": 3,
                    "clock": "02:00",
                    "score": {"home": 3, "away": 2},
                    "lens": {
                        "totals": {"home": {"sog": 27}, "away": {"sog": 31}},
                        "periods": [
                            {"period": 1, "home": {"goals": 1, "sog": 10}, "away": {"goals": 1, "sog": 11}},
                            {"period": 2, "home": {"goals": 1, "sog": 9}, "away": {"goals": 1, "sog": 10}},
                            {"period": 3, "home": {"goals": 1, "sog": 8}, "away": {"goals": 0, "sog": 10}},
                        ],
                        "players": {"home": [], "away": []},
                        "goalies": {
                            "home": [{"name": "Home Goalie", "saves": 29, "shots_against": 31, "sv_pct": 0.935}],
                            "away": [{"name": "Away Goalie", "saves": 24, "shots_against": 27, "sv_pct": 0.889}],
                        },
                    },
                }
            ],
        }

    def fake_v1_odds_payload(*_args, **_kwargs):
        return {
            "ok": True,
            "asof_utc": "2099-01-03T00:00:00Z",
            "date": date,
            "games": [
                {
                    "date": date,
                    "home": "BOS",
                    "away": "NYR",
                    "key": game_key,
                    "ml": {"home": -260, "away": 220},
                    "total": {"line": 6.5, "over": -125, "under": 105},
                    "puckline": {"home_-1.5": 125, "away_+1.5": -145},
                    "h2h": {"home": -260, "away": 220},
                    "totals": {"line": 6.5, "over": -125, "under": 105},
                    "spreads": {"home": 125, "away": -145, "line": 1.5},
                }
            ],
        }

    def fake_v1_props_odds_payload(*_args, **_kwargs):
        return {
            "ok": True,
            "date": date,
            "asof_utc": "2099-01-03T00:00:00Z",
            "games": [{"date": date, "home": "BOS", "away": "NYR", "key": game_key, "props": {}}],
        }

    def fake_read_all_players_projections(d: str):
        assert d == date
        return pd.DataFrame(columns=["date", "player", "team", "position", "market", "proj_lambda"])

    def fake_bundle_predictions_map(d: str):
        assert d == date
        return {
            game_key: {
                "model_total": 5.4,
                "model_spread": 0.3,
                "p_home_ml": 0.54,
                "total_line_used": 5.5,
            }
        }

    def fake_pbp_get(self, path: str, params: object, retries: int):
        assert "/play-by-play" in str(path)
        return {
            "homeTeam": {"id": 1},
            "awayTeam": {"id": 2},
            "plays": [
                {"sortOrder": 1, "periodDescriptor": {"number": 3}, "timeInPeriod": "15:20", "typeDescKey": "faceoff", "details": {"eventOwnerTeamId": 2}, "situationCode": "5151"},
                {"sortOrder": 2, "periodDescriptor": {"number": 3}, "timeInPeriod": "16:10", "typeDescKey": "shot-on-goal", "details": {"eventOwnerTeamId": 2, "xCoord": 73, "yCoord": 10}, "situationCode": "5151"},
                {"sortOrder": 3, "periodDescriptor": {"number": 3}, "timeInPeriod": "16:45", "typeDescKey": "missed-shot", "details": {"eventOwnerTeamId": 2, "xCoord": 78, "yCoord": 12}, "situationCode": "5151"},
                {"sortOrder": 4, "periodDescriptor": {"number": 3}, "timeInPeriod": "17:15", "typeDescKey": "blocked-shot", "details": {"eventOwnerTeamId": 2, "xCoord": 80, "yCoord": 6}, "situationCode": "5151"},
                {"sortOrder": 5, "periodDescriptor": {"number": 3}, "timeInPeriod": "17:40", "typeDescKey": "shot-on-goal", "details": {"eventOwnerTeamId": 1, "xCoord": 18, "yCoord": 2}, "situationCode": "6501"},
                {"sortOrder": 6, "periodDescriptor": {"number": 3}, "timeInPeriod": "18:00", "typeDescKey": "shot-on-goal", "details": {"eventOwnerTeamId": 2, "xCoord": 76, "yCoord": 9}, "situationCode": "6501"},
            ],
        }

    monkeypatch.setattr(NHLWebClient, "_get", fake_pbp_get, raising=True)
    monkeypatch.setattr(web_app, "_v1_live_payload", fake_live_payload, raising=True)
    monkeypatch.setattr(web_app, "_v1_odds_payload", fake_v1_odds_payload, raising=True)
    monkeypatch.setattr(web_app, "_v1_props_odds_payload", fake_v1_props_odds_payload, raising=True)
    monkeypatch.setattr(web_app, "_read_all_players_projections", fake_read_all_players_projections, raising=True)
    monkeypatch.setattr(web_app, "_load_bundle_predictions_map", fake_bundle_predictions_map, raising=True)

    try:
        web_app._LIVE_LENS_CACHE.clear()
    except Exception:
        pass

    resp = client.get(f"/v1/live-lens/{date}?regions=us&best=1&include_non_live=1&inplay=1&include_pbp=1")
    assert resp.status_code == 200
    obj = resp.json()
    assert obj.get("ok") is True
    games = obj.get("games") or []
    assert len(games) == 1

    guidance = (games[0].get("guidance") or {})
    assert guidance.get("late_state_mode") == "one_goal_late_empty_net"
    assert guidance.get("pp_team") in {None, ""}
    assert guidance.get("mu_remaining") is not None
    assert guidance.get("mu_remaining_model") is not None
    assert guidance.get("mu_remaining_market") is not None
    assert float(guidance.get("market_blend_weight") or 0.0) > 0.0
    assert float(guidance.get("p_win_market_blend_weight") or 0.0) > 0.0
    assert float(guidance.get("mu_home_rem") or 0.0) > float(guidance.get("mu_away_rem") or 0.0)

    driver_tags = guidance.get("projection_driver_tags") or []
    assert "empty_net:away" in driver_tags
    assert "late:one_goal" in driver_tags
    assert "market:total_blend" in driver_tags

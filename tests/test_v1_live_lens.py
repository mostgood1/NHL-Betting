from fastapi.testclient import TestClient

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

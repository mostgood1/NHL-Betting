import pytest
from fastapi.testclient import TestClient

import nhl_betting.web.app as web_app
from nhl_betting.data.nhl_api import NHLClient as NHLStatsClient
from nhl_betting.data.nhl_api_web import NHLWebClient


client = TestClient(web_app.app)


def _intermission_live_feed() -> dict:
    return {
        "liveData": {
            "linescore": {
                "currentPeriod": 1,
                "currentPeriodTimeRemaining": "END",
                "intermissionInfo": {"inIntermission": True},
                "periods": [
                    {
                        "home": {"goals": 1},
                        "away": {"goals": 0},
                    }
                ],
            },
            "plays": {
                "currentPlay": {
                    "about": {
                        "period": 1,
                        "periodTimeRemaining": "01:26",
                    }
                }
            },
        }
    }


@pytest.mark.parametrize("game_state", ["LIVE", "INTERMISSION"])
def test_api_scoreboard_intermission_hides_stale_clock(monkeypatch, game_state):
    def _scoreboard_day(self, date: str):
        assert date == "2099-01-01"
        return [
            {
                "gamePk": 123,
                "gameDate": f"{date}T00:00:00Z",
                "home": "Detroit Red Wings",
                "away": "Florida Panthers",
                "home_goals": 1,
                "away_goals": 0,
                "gameState": game_state,
                "period": 1,
                "clock": "01:26",
            }
        ]

    def _linescore(self, gamePk: int):
        assert int(gamePk) == 123
        return {"period": 1, "clock": "01:26", "source": "linescore"}

    def _game_live_feed(self, gamePk: int):
        assert int(gamePk) == 123
        return _intermission_live_feed()

    monkeypatch.setattr(NHLWebClient, "scoreboard_day", _scoreboard_day, raising=True)
    monkeypatch.setattr(NHLWebClient, "linescore", _linescore, raising=True)
    monkeypatch.setattr(NHLStatsClient, "game_live_feed", _game_live_feed, raising=True)
    web_app._SCOREBOARD_STATS_CACHE.clear()

    resp = client.get("/api/scoreboard?date=2099-01-01")
    assert resp.status_code == 200
    rows = resp.json()
    assert isinstance(rows, list) and len(rows) == 1

    row = rows[0]
    assert row["gamePk"] == 123
    assert row["gameState"] == game_state
    assert row["period"] == 1
    assert row["intermission"] is True
    assert row["clock"] is None
    assert row["period_disp"] == "1st INT"
    assert row["source_clock"] == "stats-intermission"

import pandas as pd
from fastapi.testclient import TestClient

from nhl_betting.data.odds_api import OddsAPIClient
from nhl_betting.web.app import app


client = TestClient(app)


def test_v1_odds_smoke(monkeypatch):
    def _flat_snapshot(
        self,
        iso_date: str,
        regions: str = "us",
        markets: str = "h2h,totals,spreads",
        snapshot_iso=None,
        odds_format: str = "american",
        bookmaker=None,
        best: bool = False,
    ):
        rows = [
            {
                "date": iso_date,
                "home": "Boston Bruins",
                "away": "Montreal Canadiens",
                "home_ml": -140,
                "away_ml": 120,
                "over": -110,
                "under": -110,
                "total_line": 6.5,
                "home_pl_-1.5": 150,
                "away_pl_+1.5": -170,
                "home_ml_book": "draftkings",
                "away_ml_book": "draftkings",
                "over_book": "fanduel",
                "under_book": "fanduel",
                "home_pl_-1.5_book": "pinnacle",
                "away_pl_+1.5_book": "pinnacle",
            }
        ]
        return pd.DataFrame(rows)

    # Avoid ODDS_API_KEY requirement by bypassing __init__
    monkeypatch.setattr(OddsAPIClient, "__init__", lambda self: None, raising=True)
    monkeypatch.setattr(OddsAPIClient, "flat_snapshot", _flat_snapshot, raising=True)

    r = client.get("/v1/odds/2099-01-01")
    assert r.status_code == 200
    obj = r.json()
    assert obj.get("ok") is True
    assert obj.get("date") == "2099-01-01"
    assert "asof_utc" in obj

    games = obj.get("games")
    assert isinstance(games, list)
    assert len(games) == 1

    g0 = games[0]
    assert g0["home"] == "Boston Bruins"
    assert g0["away"] == "Montreal Canadiens"
    assert g0["ml"]["home"] == -140
    assert g0["total"]["line"] == 6.5

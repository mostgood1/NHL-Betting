import pandas as pd

import nhl_betting.web.app as web_app


def test_v1_odds_payload_inplay_requests_period_market_keys(monkeypatch):
    date = "2099-01-03"
    seen_markets: list[str] = []

    class FakeOddsAPIClient:
        def flat_snapshot(self, d, regions, markets, snapshot_iso, odds_format, bookmaker, best, inplay):
            seen_markets.append(str(markets))
            # Non-empty DF prevents in-play fallback to core markets.
            return pd.DataFrame(
                [
                    {
                        "date": d,
                        "home": "BOS",
                        "away": "MTL",
                    }
                ]
            )

    monkeypatch.setattr(web_app, "OddsAPIClient", FakeOddsAPIClient, raising=True)
    monkeypatch.setattr(web_app, "_live_odds_cache_get", lambda *_a, **_k: None, raising=True)
    monkeypatch.setattr(web_app, "_live_odds_cache_put", lambda *_a, **_k: None, raising=True)

    out = web_app._v1_odds_payload(date, regions="us", best=True, inplay=True)
    assert out.get("ok") is True
    assert seen_markets, "Expected OddsAPIClient.flat_snapshot to be called"

    m = seen_markets[0]
    for key in (
        "totals_p1",
        "totals_p2",
        "totals_p3",
        "h2h_p1",
        "h2h_p2",
        "h2h_p3",
        "spreads_p1",
        "spreads_p2",
        "spreads_p3",
        "h2h_3_way_p1",
        "h2h_3_way_p2",
        "h2h_3_way_p3",
    ):
        assert key in m

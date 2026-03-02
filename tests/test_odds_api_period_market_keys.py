from nhl_betting.data.odds_api import _extract_prices_from_markets


def test_extract_prices_supports_p_suffix_period_keys():
    markets = [
        {
            "key": "totals_p3",
            "outcomes": [
                {"name": "Over", "price": -160, "point": 0.5},
                {"name": "Under", "price": 120, "point": 0.5},
            ],
        },
        {
            "key": "h2h_p3",
            "outcomes": [
                {"name": "Home Team", "price": 100},
                {"name": "Away Team", "price": 1000},
            ],
        },
        {
            "key": "h2h_3_way_p3",
            "outcomes": [
                {"name": "Home Team", "price": 320},
                {"name": "Draw", "price": 270},
                {"name": "Away Team", "price": 2200},
            ],
        },
        {
            "key": "spreads_p3",
            "outcomes": [
                {"name": "Home Team", "price": -360, "point": -0.5},
                {"name": "Away Team", "price": 250, "point": 0.5},
            ],
        },
    ]

    prices = _extract_prices_from_markets(markets)

    assert prices.get("p3tot::Over") == -160
    assert prices.get("p3tot::Under") == 120
    assert prices.get("p3tot::point") == 0.5

    assert prices.get("p3ml::Home Team") == 100
    assert prices.get("p3ml::Away Team") == 1000

    assert prices.get("p33w::Home Team") == 320
    assert prices.get("p33w::Draw") == 270
    assert prices.get("p33w::Away Team") == 2200

    assert prices.get("p3spr::Home Team") == -360
    assert prices.get("p3spr::Home Team::point") == -0.5
    assert prices.get("p3spr::Away Team") == 250
    assert prices.get("p3spr::Away Team::point") == 0.5

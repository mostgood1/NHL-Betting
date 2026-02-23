import pandas as pd


def test_normalize_snapshot_to_rows_buckets_by_eastern_date():
    # 00:30Z on Feb 26 is still Feb 25 in US/Eastern.
    from nhl_betting.data.odds_api import normalize_snapshot_to_rows

    snapshot = [
        {
            "home_team": "New Jersey Devils",
            "away_team": "Buffalo Sabres",
            "commence_time": "2026-02-26T00:30:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [],
                }
            ],
        }
    ]

    df = normalize_snapshot_to_rows(snapshot, bookmaker=None, best_of_all=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["date"] == "2026-02-25"

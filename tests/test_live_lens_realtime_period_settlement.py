import pandas as pd


def test_realtime_period_settlement_from_snapshots():
    from check_live_lens_betting_performance import _settle_row, _update_period_scores_from_snapshot

    # Snapshot 1: live game in P2 (period not complete yet)
    snap_p2 = {
        "ok": True,
        "date": "2099-01-04",
        "games": [
            {
                "gamePk": 1,
                "gameState": "LIVE",
                "period": 2,
                "clock": "10:00",
                "score": {"home": 1, "away": 1},
                "lens": {
                    "periods": [
                        {"period": 1, "home": {"goals": 1}, "away": {"goals": 1}},
                        {"period": 2, "home": {"goals": 0}, "away": {"goals": 0}},
                    ]
                },
            }
        ],
    }

    # Snapshot 2: later, game is in P3 so P2 is complete
    snap_p3 = {
        "ok": True,
        "date": "2099-01-04",
        "games": [
            {
                "gamePk": 1,
                "gameState": "LIVE",
                "period": 3,
                "clock": "15:00",
                "score": {"home": 1, "away": 1},
                "lens": {
                    "periods": [
                        {"period": 1, "home": {"goals": 1}, "away": {"goals": 1}},
                        {"period": 2, "home": {"goals": 1}, "away": {"goals": 0}},
                    ]
                },
            }
        ],
    }

    period_scores = {}
    _update_period_scores_from_snapshot(snap_p2, period_scores)
    assert (1, 2) not in period_scores

    _update_period_scores_from_snapshot(snap_p3, period_scores)
    assert period_scores[(1, 2)] == (1, 0)

    # Now settle a PERIOD_TOTAL bet for P2 without any final game score.
    row_total = pd.Series(
        {
            "gamePk": 1,
            "market": "PERIOD_TOTAL",
            "side": "OVER",
            "line": 0.5,
            "sig_period": 2,
            "price_american": -110,
        }
    )
    out = _settle_row(row_total, final_scores={}, final_period_scores=period_scores)
    assert out["result"] == "WIN"
    assert out["profit_units"] is not None

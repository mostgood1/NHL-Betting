import pandas as pd

from scripts.live_lens_playoff_gate_delta_report import _playoff_gate_tag, build_gate_delta_table


def test_playoff_gate_tag_matches_runtime_gate_order():
    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "TOTAL",
            "side": "UNDER",
            "score_home": 1,
            "score_away": 2,
            "elapsed_min": 47.0,
            "driver_tags": ["score:away_leading", "goalie:strong"],
        })
    ) == "gate:playoff_under_away_leading_stale_block_35_60"

    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "ML",
            "side": "AWAY",
            "score_home": 1,
            "score_away": 2,
            "elapsed_min": 38.0,
            "edge": 0.14,
        })
    ) == "gate:playoff_away_ml_leading_block_35_50"

    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "ML",
            "side": "HOME",
            "score_home": 2,
            "score_away": 2,
            "edge": 0.075,
        })
    ) == "gate:playoff_home_ml_tied_edge>=0.08"

    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "TOTAL",
            "side": "OVER",
            "elapsed_min": 12.0,
        })
    ) == "gate:playoff_total_over_block_5_20"

    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "TOTAL",
            "side": "OVER",
            "elapsed_min": 26.0,
            "score_home": 2,
            "score_away": 2,
            "driver_meta": {"score_state_age_sec": 480},
        })
    ) == "gate:playoff_total_over_tied_stale_5m"

    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "PERIOD_TOTAL",
            "side": "OVER",
            "driver_meta": {"score_state_age_sec": 360},
            "elapsed_min": 28.0,
            "sig_period": 2,
            "score_home": 2,
            "score_away": 1,
        })
    ) == "gate:playoff_period_total_over_stale_score_state_2_10m"

    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "PERIOD_TOTAL",
            "side": "OVER",
            "elapsed_min": 8.0,
            "sig_period": 1,
            "score_home": 0,
            "score_away": 0,
        })
    ) == "gate:playoff_p1_over_block_5_15"

    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "PERIOD_TOTAL",
            "side": "OVER",
            "elapsed_min": 23.0,
            "sig_period": 2,
            "score_home": 1,
            "score_away": 1,
        })
    ) == "gate:playoff_period_total_over_tied"

    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "ML",
            "side": "HOME",
            "score_home": 2,
            "score_away": 2,
            "edge": 0.082,
        })
    ) is None

    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "ML",
            "side": "AWAY",
            "score_home": 1,
            "score_away": 2,
            "elapsed_min": 34.5,
            "edge": 0.14,
        })
    ) is None

    assert _playoff_gate_tag(
        pd.Series({
            "gamePk": 2025030131,
            "market": "TOTAL",
            "side": "UNDER",
            "score_home": 1,
            "score_away": 2,
            "elapsed_min": 47.0,
            "driver_tags": ["score:away_leading", "goal_away"],
        })
    ) is None


def test_build_gate_delta_table_includes_combined_row():
    df = pd.DataFrame(
        [
            {"season_type": "playoff", "profit_units": -1.0, "result": "LOSE", "gate_tag": "gate:playoff_total_over_block_5_20"},
            {"season_type": "playoff", "profit_units": -0.5, "result": "LOSE", "gate_tag": "gate:playoff_period_total_over_tied"},
            {"season_type": "playoff", "profit_units": 1.5, "result": "WIN", "gate_tag": None},
            {"season_type": "playoff", "profit_units": 0.5, "result": "WIN", "gate_tag": None},
        ]
    )

    out = build_gate_delta_table(df)

    combined = out[out["gate"] == "all_playoff_gates_combined"].iloc[0]
    assert int(combined["removed_bets"]) == 2
    assert int(combined["kept_bets"]) == 2
    assert float(combined["kept_units"]) == 2.0
    assert float(combined["delta_units"]) == 1.5
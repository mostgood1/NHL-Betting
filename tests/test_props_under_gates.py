import pandas as pd

from nhl_betting.cli import _apply_under_side_gates


def test_apply_under_side_gates_filters_only_unders():
    rows = pd.DataFrame(
        [
            {"player": "Over Favorite", "side": "Over", "ev": 0.02, "price": -260},
            {"player": "Under Balanced", "side": "Under", "ev": 0.19, "price": -150},
            {"player": "Under Too Juiced", "side": "Under", "ev": 0.25, "price": -185},
            {"player": "Under Low Ev", "side": "UNDER", "ev": 0.17, "price": -140},
            {"player": "Under Missing Price", "side": "Under", "ev": 0.21, "price": None},
        ]
    )

    filtered = _apply_under_side_gates(rows, under_min_ev=0.18, under_max_juice=170)

    assert filtered["player"].tolist() == [
        "Over Favorite",
        "Under Balanced",
        "Under Missing Price",
    ]


def test_apply_under_side_gates_is_noop_when_disabled():
    rows = pd.DataFrame(
        [
            {"player": "Over", "side": "Over", "ev": 0.01, "price": 120},
            {"player": "Under", "side": "Under", "ev": 0.05, "price": -250},
        ]
    )

    filtered = _apply_under_side_gates(rows, under_min_ev=0.0, under_max_juice=0.0)

    assert filtered["player"].tolist() == ["Over", "Under"]


def test_apply_under_side_gates_honors_market_specific_juice_override():
    rows = pd.DataFrame(
        [
            {"player": "Points Over", "market": "POINTS", "side": "Over", "ev": 0.03, "price": -210},
            {"player": "Points Under Keep", "market": "POINTS", "side": "Under", "ev": 0.24, "price": -120},
            {"player": "Points Under Drop", "market": "POINTS", "side": "Under", "ev": 0.24, "price": -130},
            {"player": "Goals Under Keep", "market": "GOALS", "side": "Under", "ev": 0.24, "price": -160},
            {"player": "Goals Under Drop", "market": "GOALS", "side": "Under", "ev": 0.24, "price": -175},
        ]
    )

    filtered = _apply_under_side_gates(
        rows,
        under_min_ev=0.18,
        under_max_juice=170,
        under_max_juice_per_market="POINTS=120",
    )

    assert filtered["player"].tolist() == [
        "Points Over",
        "Points Under Keep",
        "Goals Under Keep",
    ]
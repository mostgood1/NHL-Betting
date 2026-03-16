import json

import pytest

import nhl_betting.web.app as web_app
from scripts.fit_live_lens_driver_tag_priors import build_driver_tag_priors


def _ledger_row(*, market: str, profit_units: float, driver_tags=None, date: str = "2099-01-01", side: str | None = None) -> dict:
    if profit_units > 0:
        result = "WIN"
    elif profit_units < 0:
        result = "LOSE"
    else:
        result = "PUSH"
    return {
        "date": date,
        "market": market,
        "side": side,
        "profit_units": profit_units,
        "result": result,
        "driver_tags": list(driver_tags or []),
    }


def test_build_driver_tag_priors_loosen_good_total_tag():
    rows = []
    for _ in range(30):
        rows.append(_ledger_row(market="TOTAL", profit_units=0.18, driver_tags=["market:TOTAL", "pace:up"]))
    for _ in range(70):
        rows.append(_ledger_row(market="TOTAL", profit_units=0.0, driver_tags=["market:TOTAL"]))

    obj = build_driver_tag_priors(
        rows,
        start="2099-01-01",
        end="2099-01-31",
        min_bets=10,
        min_market_bets=20,
        shrink_bets=20.0,
        min_roi_gap=0.01,
        roi_to_edge=0.12,
        max_edge_adjustment=0.015,
    )

    markets = obj.get("markets") or {}
    assert "TOTAL" in markets
    assert "__all__" in markets

    total_pace_up = ((markets.get("TOTAL") or {}).get("pace:up") or {})
    assert total_pace_up
    assert float(total_pace_up.get("edge_delta") or 0.0) < 0.0
    assert int(total_pace_up.get("bets") or 0) == 30
    assert float(total_pace_up.get("baseline_roi") or 0.0) > 0.0


def test_live_lens_driver_tag_edge_adjustment_uses_market_family_fallback(tmp_path, monkeypatch: pytest.MonkeyPatch):
    priors_path = tmp_path / "live_lens_driver_tag_priors.json"
    priors_path.write_text(
        json.dumps(
            {
                "defaults": {"max_total_edge_adjustment": 0.015},
                "markets": {
                    "TOTAL": {
                        "pace:up": {"edge_delta": -0.02, "reliability": 1.0, "bets": 50},
                    },
                    "__all__": {
                        "goalie:weak": {"edge_delta": 0.004, "reliability": 1.0, "bets": 50},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("LIVE_LENS_DRIVER_TAG_PRIORS_JSON", str(priors_path))
    try:
        web_app._LIVE_LENS_DRIVER_TAG_PRIORS_CACHE.clear()
    except Exception:
        pass

    adj = web_app._live_lens_driver_tag_edge_adjustment(
        "PERIOD_TOTAL",
        ["market:PERIOD", "edge:>=0.04", "pace:up", "goalie:weak"],
        side="OVER",
    )

    assert adj
    assert pytest.approx(float(adj.get("edge_delta") or 0.0), abs=1e-6) == -0.015
    matched = adj.get("matched") or []
    matched_tags = {m.get("tag") for m in matched if isinstance(m, dict)}
    assert "pace:up" in matched_tags
    assert "goalie:weak" in matched_tags
    assert "market:PERIOD" not in matched_tags
    assert "edge:>=0.04" not in matched_tags


def test_build_driver_tag_priors_creates_side_specific_scopes():
    rows = []
    for _ in range(20):
        rows.append(_ledger_row(market="TOTAL", side="UNDER", profit_units=-1.0, driver_tags=["pressure:home"]))
    for _ in range(20):
        rows.append(_ledger_row(market="TOTAL", side="UNDER", profit_units=0.0, driver_tags=[]))
    for _ in range(20):
        rows.append(_ledger_row(market="TOTAL", side="OVER", profit_units=0.5, driver_tags=["pressure:home"]))
    for _ in range(20):
        rows.append(_ledger_row(market="TOTAL", side="OVER", profit_units=0.0, driver_tags=[]))

    obj = build_driver_tag_priors(
        rows,
        start="2099-01-01",
        end="2099-01-31",
        min_bets=10,
        min_market_bets=10,
        shrink_bets=10.0,
        min_roi_gap=0.01,
        roi_to_edge=0.12,
        max_edge_adjustment=0.015,
    )

    markets = obj.get("markets") or {}
    assert "TOTAL:UNDER" in markets
    assert "TOTAL:OVER" in markets

    under_pressure_home = ((markets.get("TOTAL:UNDER") or {}).get("pressure:home") or {})
    over_pressure_home = ((markets.get("TOTAL:OVER") or {}).get("pressure:home") or {})
    assert under_pressure_home
    assert over_pressure_home
    assert float(under_pressure_home.get("edge_delta") or 0.0) > 0.0
    assert float(over_pressure_home.get("edge_delta") or 0.0) < 0.0


def test_live_lens_driver_tag_edge_adjustment_prefers_side_specific_scope(tmp_path, monkeypatch: pytest.MonkeyPatch):
    priors_path = tmp_path / "live_lens_driver_tag_priors.json"
    priors_path.write_text(
        json.dumps(
            {
                "defaults": {"max_total_edge_adjustment": 0.015},
                "markets": {
                    "TOTAL": {
                        "pressure:home": {"edge_delta": -0.002, "reliability": 1.0, "bets": 50},
                    },
                    "TOTAL:UNDER": {
                        "pressure:home": {"edge_delta": 0.012, "reliability": 1.0, "bets": 50},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("LIVE_LENS_DRIVER_TAG_PRIORS_JSON", str(priors_path))
    try:
        web_app._LIVE_LENS_DRIVER_TAG_PRIORS_CACHE.clear()
    except Exception:
        pass

    adj = web_app._live_lens_driver_tag_edge_adjustment(
        "TOTAL",
        ["pressure:home"],
        side="UNDER",
    )

    assert adj
    assert pytest.approx(float(adj.get("edge_delta") or 0.0), abs=1e-6) == 0.012
    matched = adj.get("matched") or []
    assert matched
    assert matched[0].get("scope") == "TOTAL:UNDER"


def test_live_lens_total_under_toxic_gate_applies_after_20_minutes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_EM20_REQUIRED_EDGE", "0.10")
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_EM35_REQUIRED_EDGE", "0.12")

    gate = web_app._live_lens_total_under_toxic_gate(
        24.0,
        ["market:TOTAL", "pressure:home", "pace:down", "edge:>=0.06"],
    )

    assert gate
    assert pytest.approx(float(gate.get("min_required_edge") or 0.0), abs=1e-6) == 0.11
    assert gate.get("matched") == ["pressure:home", "pace:down"]


def test_live_lens_total_under_toxic_gate_uses_later_floor_at_35_minutes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_EM20_REQUIRED_EDGE", "0.10")
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_EM35_REQUIRED_EDGE", "0.12")

    gate = web_app._live_lens_total_under_toxic_gate(
        36.0,
        ["goalie:weak", "score:tied"],
    )

    assert gate
    assert pytest.approx(float(gate.get("min_required_edge") or 0.0), abs=1e-6) == 0.13
    assert gate.get("matched") == ["goalie:weak", "score:tied"]


def test_live_lens_total_under_toxic_gate_adds_extra_penalty_after_away_goal(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_EM20_REQUIRED_EDGE", "0.10")
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_EM35_REQUIRED_EDGE", "0.12")

    gate = web_app._live_lens_total_under_toxic_gate(
        24.0,
        ["score:away_leading", "goal_away"],
    )

    assert gate
    assert pytest.approx(float(gate.get("min_required_edge") or 0.0), abs=1e-6) == 0.13
    assert gate.get("matched") == ["score:away_leading", "goal_away"]


def test_live_lens_total_under_early_gate_blocks_through_first_period(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_MIN_ELAPSED_MIN", "20")

    gate = web_app._live_lens_total_under_early_gate(20.0)

    assert gate
    assert gate.get("blocked") is True
    assert pytest.approx(float(gate.get("min_elapsed") or 0.0), abs=1e-6) == 20.0


def test_live_lens_total_under_early_gate_allows_after_first_period(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_MIN_ELAPSED_MIN", "20")

    gate = web_app._live_lens_total_under_early_gate(20.01)

    assert gate
    assert gate.get("blocked") is False
    assert pytest.approx(float(gate.get("min_elapsed") or 0.0), abs=1e-6) == 20.0

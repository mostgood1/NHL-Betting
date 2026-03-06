import json

import pytest

import nhl_betting.web.app as web_app
from scripts.fit_live_lens_driver_tag_priors import build_driver_tag_priors


def _ledger_row(*, market: str, profit_units: float, driver_tags=None, date: str = "2099-01-01") -> dict:
    if profit_units > 0:
        result = "WIN"
    elif profit_units < 0:
        result = "LOSE"
    else:
        result = "PUSH"
    return {
        "date": date,
        "market": market,
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
    )

    assert adj
    assert pytest.approx(float(adj.get("edge_delta") or 0.0), abs=1e-6) == -0.015
    matched = adj.get("matched") or []
    matched_tags = {m.get("tag") for m in matched if isinstance(m, dict)}
    assert "pace:up" in matched_tags
    assert "goalie:weak" in matched_tags
    assert "market:PERIOD" not in matched_tags
    assert "edge:>=0.04" not in matched_tags

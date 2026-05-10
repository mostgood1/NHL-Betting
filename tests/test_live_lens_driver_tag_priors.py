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


def test_live_lens_required_edge_with_prior_is_informative_only_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LIVE_LENS_PRIOR_EDGE_IN_DECISION", raising=False)

    required = web_app._live_lens_required_edge_with_prior(
        0.06,
        {"edge_delta": 0.015, "matched": [{"tag": "score:tied"}]},
    )

    assert pytest.approx(float(required or 0.0), abs=1e-6) == 0.06


def test_live_lens_required_edge_with_prior_can_be_reenabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_PRIOR_EDGE_IN_DECISION", "1")

    required = web_app._live_lens_required_edge_with_prior(
        0.06,
        {"edge_delta": 0.015, "matched": [{"tag": "score:tied"}]},
    )

    assert pytest.approx(float(required or 0.0), abs=1e-6) == 0.075


def test_live_lens_flow_driver_meta_persists_trigger_and_pbp_context():
    meta = web_app._live_lens_flow_driver_meta(
        pbp_ctx={
            "last_goal_team": "away",
            "time_since_last_goal_sec": 92,
            "score_state_age_sec": 92,
            "pp_team": "away",
            "pp_sec_remaining_est": 48,
            "home_empty_net": False,
            "away_empty_net": True,
            "att_pace_60": 118.5,
            "home_att_l5": 3,
            "away_att_l5": 7,
            "home_fo_pct": 46.0,
            "away_fo_pct": 54.0,
            "home_fo_pct_l10": 40.0,
            "away_fo_pct_l10": 60.0,
            "home_xg_proxy": 1.1,
            "away_xg_proxy": 1.6,
            "home_xg_proxy_l10": 0.2,
            "away_xg_proxy_l10": 0.6,
        },
        trigger_tags=["score:away_leading", "goal_away", "pp_start_away", "pulled_goalie_away", "empty_net:away"],
        home_goals=1,
        away_goals=2,
        elapsed_min=47.5,
        late_state_mode="one_goal_late_empty_net",
    )

    assert meta["score_state"] == "away_leading"
    assert meta["score_diff"] == -1
    assert meta["recent_goal_state"] == "recent_goal"
    assert meta["recent_goal_team"] == "away"
    assert meta["last_goal_team"] == "away"
    assert meta["time_since_last_goal_sec"] == 92
    assert meta["score_state_age_sec"] == 92
    assert meta["pp_state_age_sec"] == 72
    assert meta["manpower_state"] == "even"
    assert meta["empty_net_state"] == "away"
    assert meta["pp_team"] == "away"
    assert pytest.approx(float(meta["pp_sec_remaining_est"]), abs=1e-6) == 48.0
    assert meta["att_diff_l5"] == -4
    assert pytest.approx(float(meta["fo_diff_l10"]), abs=1e-6) == -20.0
    assert pytest.approx(float(meta["xg_diff_l10"]), abs=1e-6) == -0.4
    assert meta["pp_start_away"] is True
    assert meta["pulled_goalie_away"] is True


def test_live_lens_total_under_toxic_gate_applies_after_20_minutes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_EM20_REQUIRED_EDGE", "0.10")
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_EM35_REQUIRED_EDGE", "0.12")
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_PRESSURE_HOME_EDGE_BUMP", "0.01")

    gate = web_app._live_lens_total_under_toxic_gate(
        24.0,
        ["market:TOTAL", "pressure:home", "pace:down", "edge:>=0.06"],
    )

    assert gate
    assert pytest.approx(float(gate.get("min_required_edge") or 0.0), abs=1e-6) == 0.12
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
    assert pytest.approx(float(gate.get("min_required_edge") or 0.0), abs=1e-6) == 0.14
    assert gate.get("matched") == ["score:away_leading", "goal_away"]


def test_live_lens_total_under_toxic_gate_adds_extra_pressure_home_bump(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_EM20_REQUIRED_EDGE", "0.10")
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_EM35_REQUIRED_EDGE", "0.12")
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_HIGH_RISK_EDGE_BUMP", "0.01")
    monkeypatch.setenv("LIVE_LENS_TOTAL_UNDER_TOXIC_PRESSURE_HOME_EDGE_BUMP", "0.02")

    gate = web_app._live_lens_total_under_toxic_gate(
        24.0,
        ["pressure:home"],
    )

    assert gate
    assert pytest.approx(float(gate.get("min_required_edge") or 0.0), abs=1e-6) == 0.13
    assert gate.get("matched") == ["pressure:home"]


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


def test_live_lens_is_playoff_game_detects_gamepk_type_code():
    assert web_app._live_lens_is_playoff_game(2025030131) is True
    assert web_app._live_lens_is_playoff_game(2025020947) is False


def test_live_lens_playoff_total_over_gate_blocks_between_five_and_twenty_minutes():
    gate = web_app._live_lens_playoff_over_gate(
        "TOTAL",
        "OVER",
        2025030131,
        elapsed_min=12.0,
    )

    assert gate
    assert gate.get("blocked") is True
    assert gate.get("tag") == "gate:playoff_total_over_block_5_20"


def test_live_lens_playoff_total_over_gate_skips_regular_season_games():
    gate = web_app._live_lens_playoff_over_gate(
        "TOTAL",
        "OVER",
        2025020947,
        elapsed_min=12.0,
    )

    assert gate
    assert gate.get("blocked") is False


def test_live_lens_playoff_total_over_gate_blocks_tied_stale_score_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_PLAYOFF_TOTAL_OVER_TIED_STALE_MIN_SEC", "300")

    gate = web_app._live_lens_playoff_over_gate(
        "TOTAL",
        "OVER",
        2025030131,
        elapsed_min=26.0,
        score_diff=0,
        pbp_ctx={"score_state_age_sec": 480},
    )

    assert gate
    assert gate.get("blocked") is True
    assert gate.get("tag") == "gate:playoff_total_over_tied_stale_5m"


def test_live_lens_playoff_total_over_gate_allows_tied_fresh_score_state():
    gate = web_app._live_lens_playoff_over_gate(
        "TOTAL",
        "OVER",
        2025030131,
        elapsed_min=26.0,
        score_diff=0,
        pbp_ctx={"score_state_age_sec": 120},
    )

    assert gate
    assert gate.get("blocked") is False
    assert gate.get("tag") is None


def test_live_lens_playoff_p1_period_total_over_gate_blocks_middle_of_first_period():
    gate = web_app._live_lens_playoff_over_gate(
        "PERIOD_TOTAL",
        "OVER",
        2025030131,
        period=1,
        period_elapsed_min=8.0,
    )

    assert gate
    assert gate.get("blocked") is True
    assert gate.get("tag") == "gate:playoff_p1_over_block_5_15"


def test_live_lens_playoff_p1_period_total_over_gate_allows_opening_five_minutes():
    gate = web_app._live_lens_playoff_over_gate(
        "PERIOD_TOTAL",
        "OVER",
        2025030131,
        period=1,
        period_elapsed_min=4.5,
    )

    assert gate
    assert gate.get("blocked") is False


def test_live_lens_playoff_period_total_over_gate_blocks_tied_game_states():
    gate = web_app._live_lens_playoff_over_gate(
        "PERIOD_TOTAL",
        "OVER",
        2025030131,
        period=2,
        period_elapsed_min=3.0,
        score_diff=0,
    )

    assert gate
    assert gate.get("blocked") is True
    assert gate.get("tag") == "gate:playoff_period_total_over_tied"


def test_live_lens_playoff_period_total_over_gate_allows_non_tied_game_states():
    gate = web_app._live_lens_playoff_over_gate(
        "PERIOD_TOTAL",
        "OVER",
        2025030131,
        period=2,
        period_elapsed_min=3.0,
        score_diff=1,
    )

    assert gate
    assert gate.get("blocked") is False


def test_live_lens_playoff_period_total_over_gate_blocks_stale_score_state_window(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_PLAYOFF_PERIOD_TOTAL_OVER_STALE_SCORE_STATE_MIN_SEC", "120")
    monkeypatch.setenv("LIVE_LENS_PLAYOFF_PERIOD_TOTAL_OVER_STALE_SCORE_STATE_MAX_SEC", "600")

    gate = web_app._live_lens_playoff_over_gate(
        "PERIOD_TOTAL",
        "OVER",
        2025030131,
        period=2,
        period_elapsed_min=6.0,
        score_diff=1,
        pbp_ctx={"score_state_age_sec": 360},
    )

    assert gate
    assert gate.get("blocked") is True
    assert gate.get("tag") == "gate:playoff_period_total_over_stale_score_state_2_10m"


def test_live_lens_playoff_period_total_over_gate_allows_fresh_score_state_window():
    gate = web_app._live_lens_playoff_over_gate(
        "PERIOD_TOTAL",
        "OVER",
        2025030131,
        period=2,
        period_elapsed_min=6.0,
        score_diff=1,
        pbp_ctx={"score_state_age_sec": 90},
    )

    assert gate
    assert gate.get("blocked") is False
    assert gate.get("tag") is None


def test_live_lens_playoff_ml_gate_raises_home_tied_floor(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_PLAYOFF_HOME_ML_TIED_REQUIRED_EDGE", "0.08")

    gate = web_app._live_lens_playoff_ml_gate(
        "HOME",
        2025030131,
        score_diff=0,
    )

    assert gate
    assert gate.get("blocked") is False
    assert pytest.approx(float(gate.get("min_required_edge") or 0.0), abs=1e-6) == 0.08
    assert gate.get("tag") == "gate:playoff_home_ml_tied_edge>=0.08"


def test_live_lens_playoff_ml_gate_skips_away_or_non_tied_cases():
    away_gate = web_app._live_lens_playoff_ml_gate(
        "AWAY",
        2025030131,
        score_diff=0,
    )
    assert away_gate.get("blocked") is False
    assert away_gate.get("min_required_edge") is None
    assert away_gate.get("tag") is None

    not_tied_gate = web_app._live_lens_playoff_ml_gate(
        "HOME",
        2025030131,
        score_diff=1,
    )
    assert not_tied_gate.get("min_required_edge") is None
    assert not_tied_gate.get("tag") is None

    regular_gate = web_app._live_lens_playoff_ml_gate(
        "HOME",
        2025020947,
        score_diff=0,
    )
    assert regular_gate.get("blocked") is False
    assert regular_gate.get("min_required_edge") is None
    assert regular_gate.get("tag") is None


def test_live_lens_playoff_ml_gate_blocks_away_ml_while_leading_midlate(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_PLAYOFF_AWAY_ML_LEADING_BLOCK_MIN_ELAPSED_MIN", "35")
    monkeypatch.setenv("LIVE_LENS_PLAYOFF_AWAY_ML_LEADING_BLOCK_MAX_ELAPSED_MIN", "50")

    gate = web_app._live_lens_playoff_ml_gate(
        "AWAY",
        2025030131,
        score_diff=-1,
        elapsed_min=38.0,
    )

    assert gate
    assert gate.get("blocked") is True
    assert gate.get("min_required_edge") is None
    assert gate.get("tag") == "gate:playoff_away_ml_leading_block_35_50"


def test_live_lens_playoff_ml_gate_allows_away_ml_leading_before_block_window():
    gate = web_app._live_lens_playoff_ml_gate(
        "AWAY",
        2025030131,
        score_diff=-1,
        elapsed_min=34.5,
    )

    assert gate.get("blocked") is False
    assert gate.get("min_required_edge") is None
    assert gate.get("tag") is None


def test_live_lens_playoff_total_under_flow_gate_blocks_stale_away_leading_late_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LIVE_LENS_PLAYOFF_UNDER_AWAY_LEADING_STALE_MIN_ELAPSED_MIN", "35")
    monkeypatch.setenv("LIVE_LENS_PLAYOFF_UNDER_AWAY_LEADING_STALE_MAX_ELAPSED_MIN", "60")

    gate = web_app._live_lens_playoff_total_under_flow_gate(
        "UNDER",
        2025030131,
        elapsed_min=47.0,
        score_diff=-1,
        driver_tags=["score:away_leading", "goalie:strong", "pressure:away"],
    )

    assert gate
    assert gate.get("blocked") is True
    assert gate.get("tag") == "gate:playoff_under_away_leading_stale_block_35_60"


def test_live_lens_playoff_total_under_flow_gate_allows_recent_goal_reset():
    gate = web_app._live_lens_playoff_total_under_flow_gate(
        "UNDER",
        2025030131,
        elapsed_min=47.0,
        score_diff=-1,
        driver_tags=["score:away_leading", "goal_away"],
    )

    assert gate.get("blocked") is False
    assert gate.get("tag") is None


def test_live_lens_playoff_total_under_flow_gate_skips_non_playoff_or_non_away_leading():
    regular_gate = web_app._live_lens_playoff_total_under_flow_gate(
        "UNDER",
        2025020947,
        elapsed_min=47.0,
        score_diff=-1,
        driver_tags=["score:away_leading"],
    )
    assert regular_gate.get("blocked") is False

    tied_gate = web_app._live_lens_playoff_total_under_flow_gate(
        "UNDER",
        2025030131,
        elapsed_min=47.0,
        score_diff=0,
        driver_tags=["score:tied"],
    )
    assert tied_gate.get("blocked") is False

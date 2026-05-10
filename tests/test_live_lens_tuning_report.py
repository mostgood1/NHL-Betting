from pathlib import Path

import pandas as pd
import pytest

from scripts.live_lens_tuning_report import _allowed_playoff_rows, _driver_tag_prior_sections, _filter_ledger, _flow_section, _guess_priors_path, _prepare_elapsed_columns, _prepare_flow_columns, _roi_table, _roi_table_multi, _top_n_edge_table


def test_guess_priors_path_from_perf_ledger():
    ledger = Path("data/processed/live_lens/perf/live_lens_bets_all.jsonl")
    got = _guess_priors_path(ledger)
    assert got.as_posix().endswith("data/processed/live_lens/live_lens_driver_tag_priors.json")


def test_driver_tag_prior_sections_render_loosen_and_tighten(tmp_path):
    priors = tmp_path / "live_lens_driver_tag_priors.json"
    priors.write_text(
        """
        {
          "defaults": {
            "max_total_edge_adjustment": 0.015
          },
          "markets": {
            "TOTAL": {
              "pace:up": {
                "edge_delta": -0.006,
                "reliability": 0.7,
                "bets": 24,
                "roi": 0.08,
                "baseline_roi": 0.02,
                "roi_gap": 0.06,
                "win_rate": 0.56
              }
            },
            "ML": {
              "goalie:weak": {
                "edge_delta": 0.004,
                "reliability": 0.65,
                "bets": 19,
                "roi": -0.03,
                "baseline_roi": 0.01,
                "roi_gap": -0.04,
                "win_rate": 0.47
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )

    sections = _driver_tag_prior_sections(priors)
    md = "\n".join(sections)

    assert "## Learned driver-tag priors" in md
    assert "### Tags that loosen gates" in md
    assert "### Tags that tighten gates" in md
    assert "pace:up" in md
    assert "goalie:weak" in md
    assert "-0.006" in md or "-0.600%" in md


def test_prepare_elapsed_columns_derives_precise_bins_and_sorts_chronologically():
    df = pd.DataFrame(
        [
            {"elapsed_min": 20.25, "profit_units": 1.0, "result": "WIN", "edge": 0.08},
            {"elapsed_min": 0.30, "profit_units": -1.0, "result": "LOSE", "edge": 0.05},
        ]
    )

    out = _prepare_elapsed_columns(df)

    assert list(out["elapsed_bucket"]) == ["20:00-21:00", "00:00-01:00"]
    assert list(out["elapsed_bucket_15s"]) == ["20:15-20:30", "00:15-00:30"]

    table = _roi_table(out, "elapsed_bucket", min_bets=1)
    assert list(table["elapsed_bucket"]) == ["00:00-01:00", "20:00-21:00"]


def test_filter_ledger_supports_playoff_gamepk_and_date_window():
    df = pd.DataFrame(
        [
            {"gamePk": 2025020947, "date": "2026-04-16", "profit_units": 1.0},
            {"gamePk": 2025030131, "date": "2026-04-18", "profit_units": -1.0},
            {"gamePk": 2025030132, "date": "2026-04-19", "profit_units": 0.5},
        ]
    )

    out = _filter_ledger(df, season_type="playoff", start_date="2026-04-19", end_date="2026-05-01")

    assert len(out) == 1
    assert int(out.iloc[0]["gamePk"]) == 2025030132


def test_prepare_flow_columns_derives_flow_shape_from_driver_tags():
    df = pd.DataFrame(
        [
            {
                "market": "TOTAL",
                "driver_tags": [
                    "score:away_leading",
                    "pressure:away",
                    "goal_away",
                    "manpower:pp_away",
                    "late:one_goal",
                    "goalie:strong",
                    "pace:down",
                ],
                  "meta_time_since_last_goal_sec": 92,
                  "meta_score_state_age_sec": 92,
                "meta_pp_team": "away",
                "meta_pp_state_age_sec": 72,
            }
        ]
    )

    out = _prepare_flow_columns(df)

    row = out.iloc[0]
    assert row["flow_score_state"] == "away_leading"
    assert row["flow_pressure_state"] == "away"
    assert row["flow_recent_goal_state"] == "recent_goal"
    assert row["flow_recent_goal_team"] == "away"
    assert row["flow_goal_age_bucket"] == "<=2m"
    assert row["flow_score_state_age_bucket"] == "<=2m"
    assert row["flow_pp_state_age_bucket"] == "60-90s"
    assert row["flow_manpower_state"] == "pp_away"
    assert row["flow_late_state"] == "one_goal"
    assert row["flow_goalie_state"] == "strong"
    assert row["flow_pace_state"] == "down"
    assert row["flow_shape_compact"] == "TOTAL | away_leading | recent_goal | one_goal | pp_away"


def test_prepare_flow_columns_supplements_missing_tags_from_meta_trigger_tags():
    df = pd.DataFrame(
        [
            {
                "market": "PERIOD_TOTAL",
                "driver_tags": ["score:tied"],
                "meta_trigger_tags": ["pressure:away", "pace:down", "late:multi_goal"],
                "meta_score_state_age_sec": 420,
            }
        ]
    )

    out = _prepare_flow_columns(df)
    row = out.iloc[0]

    assert row["flow_score_state"] == "tied"
    assert row["flow_pressure_state"] == "away"
    assert row["flow_pace_state"] == "down"
    assert row["flow_late_state"] == "multi_goal"
    assert row["flow_shape_compact"] == "PERIOD_TOTAL | tied | stale | multi_goal | even"


def test_prepare_flow_columns_derives_market_blend_bucket():
    df = pd.DataFrame([
        {"market": "TOTAL", "driver_tags": [], "meta_market_blend_weight": 0.18},
        {"market": "TOTAL", "driver_tags": [], "meta_market_blend_weight": 0.26},
        {"market": "TOTAL", "driver_tags": [], "meta_market_blend_weight": 0.41},
    ])

    out = _prepare_flow_columns(df)

    assert list(out["flow_market_blend_bucket"]) == ["<=0.20", "0.20-0.30", ">0.30"]


def test_roi_table_multi_groups_market_side_slices():
    df = pd.DataFrame(
        [
            {"side": "OVER", "flow_pressure_state": "even", "profit_units": -1.0, "result": "LOSE", "edge": 0.10},
            {"side": "OVER", "flow_pressure_state": "even", "profit_units": 1.2, "result": "WIN", "edge": 0.12},
            {"side": "UNDER", "flow_pressure_state": "home", "profit_units": 0.8, "result": "WIN", "edge": 0.08},
            {"side": "UNDER", "flow_pressure_state": "home", "profit_units": 0.9, "result": "WIN", "edge": 0.09},
        ]
    )

    out = _roi_table_multi(df, ["side", "flow_pressure_state"], min_bets=2)

    assert list(out[["side", "flow_pressure_state"]].itertuples(index=False, name=None)) == [
        ("OVER", "even"),
        ("UNDER", "home"),
    ]
    assert float(out.iloc[0]["roi"]) == pytest.approx(0.1)
    assert float(out.iloc[1]["roi"]) == pytest.approx(0.85)


def test_allowed_playoff_rows_filters_shipped_gate_slices():
    df = pd.DataFrame(
        [
            {"market": "TOTAL", "side": "OVER", "elapsed_min": 12.0, "score_home": 1, "score_away": 0, "profit_units": -1.0, "edge": 0.10},
            {"market": "TOTAL", "side": "OVER", "elapsed_min": 22.0, "score_home": 1, "score_away": 0, "profit_units": 1.0, "edge": 0.11},
            {"market": "ML", "side": "HOME", "elapsed_min": 25.0, "score_home": 2, "score_away": 2, "profit_units": -1.0, "edge": 0.07},
            {"market": "ML", "side": "HOME", "elapsed_min": 25.0, "score_home": 2, "score_away": 2, "profit_units": 1.0, "edge": 0.09},
        ]
    )

    out = _allowed_playoff_rows(df)

    assert len(out) == 2
    assert set(out["market"]) == {"TOTAL", "ML"}
    assert set(out["edge"].round(2)) == {0.11, 0.09}


def test_top_n_edge_table_summarizes_best_ranked_rows_by_market():
    df = pd.DataFrame(
        [
            {"market": "TOTAL", "edge": 0.20, "profit_units": 1.0},
            {"market": "TOTAL", "edge": 0.18, "profit_units": -1.0},
            {"market": "TOTAL", "edge": 0.15, "profit_units": 0.5},
            {"market": "ML", "edge": 0.25, "profit_units": 0.8},
            {"market": "ML", "edge": 0.12, "profit_units": 0.2},
        ]
    )

    out = _top_n_edge_table(df, top_ns=[2, 3])

    total_top2 = out[(out["market"] == "TOTAL") & (out["top_n"] == 2)].iloc[0]
    assert int(total_top2["bets"]) == 2
    assert float(total_top2["units"]) == pytest.approx(0.0)
    assert float(total_top2["avg_edge"]) == pytest.approx(0.19)


def test_flow_section_renders_pressure_by_side_tables_for_playoff_markets():
    df = pd.DataFrame(
        [
            {"market": "PERIOD_TOTAL", "side": "OVER", "elapsed_min": 22.0, "sig_period": 2.0, "period": 2, "score_home": 1, "score_away": 0, "edge_bucket": ">=0.06", "flow_pressure_state": "even", "flow_score_state": "home_leading", "flow_recent_goal_state": "stale", "flow_late_state": "normal", "flow_manpower_state": "even", "flow_market_blend_bucket": "0.20-0.30", "flow_shape_compact": "PERIOD_TOTAL | home_leading | stale | normal | even", "profit_units": -1.0, "result": "LOSE", "edge": 0.10},
            {"market": "PERIOD_TOTAL", "side": "OVER", "elapsed_min": 23.0, "sig_period": 2.0, "period": 2, "score_home": 1, "score_away": 0, "edge_bucket": ">=0.06", "flow_pressure_state": "even", "flow_score_state": "home_leading", "flow_recent_goal_state": "stale", "flow_late_state": "normal", "flow_manpower_state": "even", "flow_market_blend_bucket": "0.20-0.30", "flow_shape_compact": "PERIOD_TOTAL | home_leading | stale | normal | even", "profit_units": 0.2, "result": "WIN", "edge": 0.12},
            {"market": "PERIOD_TOTAL", "side": "UNDER", "edge_bucket": ">=0.06", "flow_pressure_state": "home", "flow_score_state": "home_leading", "flow_recent_goal_state": "stale", "flow_late_state": "one_goal", "flow_manpower_state": "even", "flow_market_blend_bucket": "0.20-0.30", "flow_shape_compact": "PERIOD_TOTAL | home_leading | stale | one_goal | even", "profit_units": 1.0, "result": "WIN", "edge": 0.09},
            {"market": "PERIOD_TOTAL", "side": "UNDER", "edge_bucket": ">=0.06", "flow_pressure_state": "home", "flow_score_state": "home_leading", "flow_recent_goal_state": "stale", "flow_late_state": "one_goal", "flow_manpower_state": "even", "flow_market_blend_bucket": "0.20-0.30", "flow_shape_compact": "PERIOD_TOTAL | home_leading | stale | one_goal | even", "profit_units": 0.8, "result": "WIN", "edge": 0.11},
            {"market": "TOTAL", "side": "OVER", "elapsed_min": 22.0, "score_home": 1, "score_away": 0, "edge_bucket": ">=0.06", "flow_pressure_state": "home", "flow_score_state": "home_leading", "flow_recent_goal_state": "stale", "flow_late_state": "normal", "flow_manpower_state": "even", "flow_market_blend_bucket": "0.20-0.30", "flow_shape_compact": "TOTAL | home_leading | stale | normal | even", "profit_units": -1.0, "result": "LOSE", "edge": 0.13},
            {"market": "TOTAL", "side": "OVER", "elapsed_min": 24.0, "score_home": 1, "score_away": 0, "edge_bucket": ">=0.06", "flow_pressure_state": "home", "flow_score_state": "home_leading", "flow_recent_goal_state": "stale", "flow_late_state": "normal", "flow_manpower_state": "even", "flow_market_blend_bucket": "0.20-0.30", "flow_shape_compact": "TOTAL | home_leading | stale | normal | even", "profit_units": -0.5, "result": "LOSE", "edge": 0.12},
            {"market": "ML", "side": "HOME", "elapsed_min": 24.0, "score_home": 2, "score_away": 2, "edge_bucket": ">=0.06", "flow_pressure_state": "even", "flow_score_state": "tied", "flow_recent_goal_state": "stale", "flow_late_state": "normal", "flow_manpower_state": "even", "flow_market_blend_bucket": "(missing)", "flow_shape_compact": "ML | tied | stale | normal | even", "profit_units": 1.0, "result": "WIN", "edge": 0.09},
            {"market": "ML", "side": "HOME", "elapsed_min": 26.0, "score_home": 1, "score_away": 0, "edge_bucket": ">=0.06", "flow_pressure_state": "home", "flow_score_state": "home_leading", "flow_recent_goal_state": "recent_goal", "flow_late_state": "normal", "flow_manpower_state": "even", "flow_market_blend_bucket": "(missing)", "flow_shape_compact": "ML | home_leading | recent_goal | normal | even", "profit_units": 0.6, "result": "WIN", "edge": 0.11},
        ]
    )

    md = "\n".join(_flow_section(df, min_bets=2))

    assert "### ROI by PERIOD_TOTAL side and pressure" in md
    assert "### ROI by TOTAL side and pressure" in md
    assert "### ROI by PERIOD_TOTAL side and late state" in md
    assert "### ROI by TOTAL side and late state" in md
    assert "### ROI by TOTAL side and market-blend bucket" in md
    assert "## Allowed-Row Ranking Quality" in md
    assert "### Allowed playoff rows by market" in md
    assert "### Allowed playoff rows by market and edge bucket" in md
    assert "### Allowed playoff rows top-N by edge" in md
    assert "| OVER | even | 2 |" in md
    assert "| UNDER | home | 2 |" in md
    assert "| OVER | 0.20-0.30 | 2 |" in md
    assert "| ML | 2 |" in md

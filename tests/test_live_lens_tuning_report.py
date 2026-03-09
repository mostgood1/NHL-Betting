from pathlib import Path

import pandas as pd

from scripts.live_lens_tuning_report import _driver_tag_prior_sections, _guess_priors_path, _prepare_elapsed_columns, _roi_table


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

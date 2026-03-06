from pathlib import Path

import nhl_betting.cli as cli
import nhl_betting.data.player_props as player_props_mod


def test_props_fast_passes_explicit_recommendation_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "PROC_DIR", tmp_path)
    monkeypatch.setenv("PROPS_SKIP_PROJECTIONS", "1")

    monkeypatch.setattr(player_props_mod, "PropsCollectionConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        player_props_mod,
        "collect_and_write",
        lambda date, roster_df=None, cfg=None: {"combined_count": 0},
    )

    captured = {}

    def _fake_props_recommendations(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "props_recommendations", _fake_props_recommendations)

    cli.props_fast(date="2026-03-06", min_ev=0.25, top=123, market="SOG")

    assert captured == {
        "date": "2026-03-06",
        "min_ev": 0.25,
        "top": 123,
        "market": "SOG",
        "min_ev_per_market": "",
        "min_prob": 0.0,
        "min_prob_per_market": "",
        "max_plus_odds": 0.0,
    }
    assert (tmp_path / "props_timing_2026-03-06.json").exists()
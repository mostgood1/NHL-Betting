import os
from pathlib import Path

import pandas as pd
import pytest

from nhl_betting.cli import props_simulate
from nhl_betting.data.nhl_api_web import NHLWebClient


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def test_strength_aware_scaling_affects_saves_and_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Use a synthetic future date to avoid interference
    date = "2099-01-01"
    repo_root = Path(os.getcwd())
    proc_dir = repo_root / "data" / "processed"
    lines_dir = repo_root / "data" / "props" / f"player_props_lines/date={date}"

    _ensure_dir(proc_dir)
    _ensure_dir(lines_dir)

    # Projections: goalie (SAVES), skater (BLOCKS), and skater (GOALS)
    proj_rows = [
        {"date": date, "player": "Test Goalie", "team": "BOS", "position": "G", "market": "SAVES", "proj_lambda": 28.0},
        {"date": date, "player": "Away D", "team": "MTL", "position": "D", "market": "BLOCKS", "proj_lambda": 2.0},
        {"date": date, "player": "Test Skater", "team": "BOS", "position": "F", "market": "GOALS", "proj_lambda": 0.25},
    ]
    proj_path = proc_dir / f"props_projections_all_{date}.csv"
    pd.DataFrame(proj_rows).to_csv(proj_path, index=False)

    # Lines: include team abbreviations so team_abbr is resolved
    lines_rows = [
        {"player_name": "Test Goalie", "market": "SAVES", "line": 25.5, "over_price": 100, "under_price": -120, "book": "DK", "team": "BOS"},
        {"player_name": "Away D", "market": "BLOCKS", "line": 2.5, "over_price": 100, "under_price": -120, "book": "DK", "team": "MTL"},
        {"player_name": "Test Skater", "market": "GOALS", "line": 0.5, "over_price": 100, "under_price": -120, "book": "DK", "team": "BOS"},
    ]
    (lines_dir / "oddsapi.csv").write_text(pd.DataFrame(lines_rows).to_csv(index=False), encoding="utf-8")

    # Penalty rates: set baseline values so penalty gamma has neutral effect
    pen_path = proc_dir / "team_penalty_rates.json"
    pen_path.write_text(
        "{"
        "\n  \"BOS\": { \"committed_per60\": 3.0 },"
        "\n  \"MTL\": { \"committed_per60\": 6.0 }"
        "\n}",
        encoding="utf-8",
    )

    # Possession events with asymmetric PP shot fractions: home 0.00, away 0.36 -> league ~0.18
    ev_rows = [{
        "home": "Boston Bruins",
        "away": "Montreal Canadiens",
        "shots_ev_home": 82,
        "shots_pp_home": 0,
        "shots_pk_home": 18,
        "shots_ev_away": 64,
        "shots_pp_away": 36,
        "shots_pk_away": 0,
    }]
    ev_path = proc_dir / f"sim_events_pos_{date}.csv"
    pd.DataFrame(ev_rows).to_csv(ev_path, index=False)

    # Patch NHL Web client used inside props_simulate to avoid external calls
    monkeypatch.setattr(NHLWebClient, "schedule_day", lambda self, d: [], raising=True)

    # Run props simulation (will write props_simulations_{date}.csv)
    props_simulate(date=date, markets="SAVES,BLOCKS,GOALS", n_sims=1000, sim_shared_k=1.0, props_strength_gamma=0.0)

    out_path = proc_dir / f"props_simulations_{date}.csv"
    assert out_path.exists(), "Simulation output CSV not written"
    out = pd.read_csv(out_path)
    assert not out.empty, "Simulation output is empty"

    # Check lam_scale_mean directionality via penalty gamma:
    # - BOS GOALS vs opponent with higher committed_per60 -> lam_scale_mean > 1
    bos_goals = out[(out["team"] == "BOS") & (out["market"] == "GOALS")]
    assert not bos_goals.empty
    assert float(bos_goals.iloc[0]["lam_scale_mean"]) > 1.0

    # - MTL BLOCKS vs opponent with lower committed_per60 -> lam_scale_mean < 1
    mtl_blocks = out[(out["team"] == "MTL") & (out["market"] == "BLOCKS")]
    assert not mtl_blocks.empty
    assert float(mtl_blocks.iloc[0]["lam_scale_mean"]) < 1.0

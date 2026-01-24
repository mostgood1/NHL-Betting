import pytest

from nhl_betting.sim.engine import SimConfig, GameSimulator
from nhl_betting.sim.models import RateModels


def _mk_player(pid: int, name: str, pos: str, toi: float):
    return {"player_id": pid, "full_name": name, "position": pos, "proj_toi": toi}


def _mk_line_slot(pid: int, slot: str, pp_unit=None, pk_unit=None):
    row = {"player_id": pid, "line_slot": slot}
    if pp_unit is not None:
        row["pp_unit"] = pp_unit
    if pk_unit is not None:
        row["pk_unit"] = pk_unit
    return row


def _build_minimal_lineups():
    roster_home = [
        _mk_player(1, "Home F1", "F", 18.0),
        _mk_player(2, "Home F2", "F", 16.0),
        _mk_player(3, "Home F3", "F", 15.0),
        _mk_player(4, "Home F4", "F", 14.0),
        _mk_player(5, "Home D1", "D", 20.0),
        _mk_player(6, "Home D2", "D", 19.0),
        _mk_player(7, "Home D3", "D", 18.0),
        _mk_player(8, "Home D4", "D", 17.0),
        _mk_player(9, "Home G1", "G", 60.0),
    ]
    roster_away = [
        _mk_player(101, "Away F1", "F", 18.0),
        _mk_player(102, "Away F2", "F", 16.0),
        _mk_player(103, "Away F3", "F", 15.0),
        _mk_player(104, "Away F4", "F", 14.0),
        _mk_player(105, "Away D1", "D", 20.0),
        _mk_player(106, "Away D2", "D", 19.0),
        _mk_player(107, "Away D3", "D", 18.0),
        _mk_player(108, "Away D4", "D", 17.0),
        _mk_player(109, "Away G1", "G", 60.0),
    ]
    lineup_home = [
        _mk_line_slot(1, "L1", pp_unit=1),
        _mk_line_slot(2, "L1", pp_unit=1),
        _mk_line_slot(3, "L2", pp_unit=2),
        _mk_line_slot(4, "L2", pp_unit=2),
        _mk_line_slot(5, "D1", pk_unit=1),
        _mk_line_slot(6, "D1", pk_unit=1),
        _mk_line_slot(7, "D2", pk_unit=2),
        _mk_line_slot(8, "D2", pk_unit=2),
    ]
    lineup_away = [
        _mk_line_slot(101, "L1", pp_unit=1),
        _mk_line_slot(102, "L1", pp_unit=1),
        _mk_line_slot(103, "L2", pp_unit=2),
        _mk_line_slot(104, "L2", pp_unit=2),
        _mk_line_slot(105, "D1", pk_unit=1),
        _mk_line_slot(106, "D1", pk_unit=1),
        _mk_line_slot(107, "D2", pk_unit=2),
        _mk_line_slot(108, "D2", pk_unit=2),
    ]
    st_home = {"pp_pct": 0.22, "pk_pct": 0.78, "drawn_per_game": 3.2, "committed_per_game": 3.0}
    st_away = {"pp_pct": 0.21, "pk_pct": 0.79, "drawn_per_game": 3.0, "committed_per_game": 3.1}
    return roster_home, roster_away, lineup_home, lineup_away, st_home, st_away


def test_block_calibration_rates_influence_counts():
    cfg = SimConfig(periods=3, seed=777)
    rates = RateModels.baseline(base_mu=3.2)
    sim = GameSimulator(cfg=cfg, rates=rates)

    roster_home, roster_away, lineup_home, lineup_away, st_home, st_away = _build_minimal_lineups()

    low_cal = {"blocks_ev_rate": 0.02, "blocks_pk_rate": 0.03, "blocks_pp_def_rate": 0.01}
    high_cal = {"blocks_ev_rate": 0.08, "blocks_pk_rate": 0.12, "blocks_pp_def_rate": 0.06}

    gs_low, events_low = sim.simulate_with_lineups(
        home_name="Home",
        away_name="Away",
        roster_home=roster_home,
        roster_away=roster_away,
        lineup_home=lineup_home,
        lineup_away=lineup_away,
        st_home=st_home,
        st_away=st_away,
        special_teams_cal=low_cal,
    )
    blocks_low = sum(1 for e in events_low if e.kind == "block")

    # Recreate simulator with same seed to keep RNG sequence comparable
    sim2 = GameSimulator(cfg=cfg, rates=rates)
    gs_high, events_high = sim2.simulate_with_lineups(
        home_name="Home",
        away_name="Away",
        roster_home=roster_home,
        roster_away=roster_away,
        lineup_home=lineup_home,
        lineup_away=lineup_away,
        st_home=st_home,
        st_away=st_away,
        special_teams_cal=high_cal,
    )
    blocks_high = sum(1 for e in events_high if e.kind == "block")

    assert blocks_high > blocks_low, "Higher calibrated block rates should produce more blocks"

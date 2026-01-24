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


def test_blocks_events_and_stats_present():
    cfg = SimConfig(periods=3, seed=123)
    rates = RateModels.baseline(base_mu=3.0)
    sim = GameSimulator(cfg=cfg, rates=rates)

    # Minimal rosters: 8 skaters + 1 goalie each
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

    # Lines: two forward lines (L1,L2) and two D pairs (D1,D2); basic PP/PK units
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

    # Use calibrated defaults-like block rates
    special_cal = {"blocks_ev_rate": 0.03, "blocks_pk_rate": 0.05, "blocks_pp_def_rate": 0.02}

    gs, events = sim.simulate_with_lineups(
        home_name="Home",
        away_name="Away",
        roster_home=roster_home,
        roster_away=roster_away,
        lineup_home=lineup_home,
        lineup_away=lineup_away,
        st_home=st_home,
        st_away=st_away,
        special_teams_cal=special_cal,
    )

    # Validate at least one block event occurred
    n_blocks = sum(1 for e in events if e.kind == "block")
    assert n_blocks >= 1, "Expected at least one block event in simulated game"

    # Validate player stats include blocks and sum matches events attribution team-wise
    home_blocks = sum(int(p.stats.get("blocks", 0)) for p in gs.home.players.values())
    away_blocks = sum(int(p.stats.get("blocks", 0)) for p in gs.away.players.values())
    # Since blocks are attributed per team events, total should equal counted events
    assert (home_blocks + away_blocks) == n_blocks

    # Ensure no goalie received blocks by allocation
    assert all(p.position != "G" or int(p.stats.get("blocks", 0)) == 0 for p in gs.home.players.values())
    assert all(p.position != "G" or int(p.stats.get("blocks", 0)) == 0 for p in gs.away.players.values())

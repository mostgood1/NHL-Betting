import pytest

from nhl_betting.cli import compute_props_lam_scale_mean


def test_multiplier_strength_and_penalty_effects():
    # Maps and baselines
    league_xg = 2.6
    xg_map = {"BOS": 2.8, "MTL": 2.4}
    league_pen = 4.5
    pen_comm = {"BOS": {"committed_per60": 3.0}, "MTL": {"committed_per60": 6.0}}
    league_sv = 0.905
    gf_map = {"BOS": 0.910, "MTL": 0.900}
    league_pp_frac = 0.18
    opp_pp_frac_map = {"BOS": 0.36, "MTL": 0.00}
    team_pp_frac_map = {"BOS": 0.00, "MTL": 0.36}
    gx = 0.02; gp = 0.06; gg = 0.02; gs = 0.04

    # SAVES for BOS vs MTL: strength above league should increase scale vs no-strength
    s_bos = compute_props_lam_scale_mean(
        market="SAVES", team_abbr="BOS", opp_abbr="MTL",
        league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
        league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
        opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
        props_xg_gamma=gx, props_penalty_gamma=gp, props_goalie_form_gamma=gg, props_strength_gamma=gs,
    )
    s_bos0 = compute_props_lam_scale_mean(
        market="SAVES", team_abbr="BOS", opp_abbr="MTL",
        league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
        league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
        opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
        props_xg_gamma=gx, props_penalty_gamma=gp, props_goalie_form_gamma=gg, props_strength_gamma=0.0,
    )
    assert s_bos > s_bos0

    # BLOCKS for MTL vs BOS: strength below league should reduce scale vs no-strength
    b_mtl = compute_props_lam_scale_mean(
        market="BLOCKS", team_abbr="MTL", opp_abbr="BOS",
        league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
        league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
        opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
        props_xg_gamma=gx, props_penalty_gamma=gp, props_goalie_form_gamma=gg, props_strength_gamma=gs,
    )
    b_mtl0 = compute_props_lam_scale_mean(
        market="BLOCKS", team_abbr="MTL", opp_abbr="BOS",
        league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
        league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
        opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
        props_xg_gamma=gx, props_penalty_gamma=gp, props_goalie_form_gamma=gg, props_strength_gamma=0.0,
    )
    assert b_mtl > b_mtl0

    # GOALS for BOS vs MTL: team_pp_frac below league should reduce scale vs no-strength
    g_bos = compute_props_lam_scale_mean(
        market="GOALS", team_abbr="BOS", opp_abbr="MTL",
        league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
        league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
        opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
        props_xg_gamma=gx, props_penalty_gamma=gp, props_goalie_form_gamma=gg, props_strength_gamma=gs,
    )
    g_bos0 = compute_props_lam_scale_mean(
        market="GOALS", team_abbr="BOS", opp_abbr="MTL",
        league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
        league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
        opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
        props_xg_gamma=gx, props_penalty_gamma=gp, props_goalie_form_gamma=gg, props_strength_gamma=0.0,
    )
    assert g_bos < g_bos0

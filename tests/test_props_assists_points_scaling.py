from nhl_betting.cli import compute_props_lam_scale_mean


def test_assists_and_points_strength_direction():
    league_xg = 2.6
    xg_map = {"BOS": 2.8, "MTL": 2.4}
    league_pen = 4.5
    pen_comm = {"BOS": {"committed_per60": 3.0}, "MTL": {"committed_per60": 6.0}}
    league_sv = 0.905
    gf_map = {"BOS": 0.910, "MTL": 0.900}
    league_pp_frac = 0.18
    # Team PP fractions: BOS below league, MTL above league
    opp_pp_frac_map = {"BOS": 0.36, "MTL": 0.00}
    team_pp_frac_map = {"BOS": 0.00, "MTL": 0.36}

    gx = 0.02
    gp = 0.00  # disable penalty effect to isolate strength
    gg = 0.00  # disable goalie form for simplicity
    gs = 0.05  # emphasize strength effect

    # GOALS for BOS vs MTL: team_pp_frac[BOS]=0.00 < league -> strength pulls down vs no-strength
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

    # ASSISTS for BOS vs MTL: same direction as GOALS (team_pp_frac below league)
    a_bos = compute_props_lam_scale_mean(
        market="ASSISTS", team_abbr="BOS", opp_abbr="MTL",
        league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
        league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
        opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
        props_xg_gamma=gx, props_penalty_gamma=gp, props_goalie_form_gamma=gg, props_strength_gamma=gs,
    )
    a_bos0 = compute_props_lam_scale_mean(
        market="ASSISTS", team_abbr="BOS", opp_abbr="MTL",
        league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
        league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
        opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
        props_xg_gamma=gx, props_penalty_gamma=gp, props_goalie_form_gamma=gg, props_strength_gamma=0.0,
    )
    assert a_bos < a_bos0

    # POINTS for MTL vs BOS: team_pp_frac[MTL]=0.36 > league -> strength lifts vs no-strength
    p_mtl = compute_props_lam_scale_mean(
        market="POINTS", team_abbr="MTL", opp_abbr="BOS",
        league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
        league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
        opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
        props_xg_gamma=gx, props_penalty_gamma=gp, props_goalie_form_gamma=gg, props_strength_gamma=gs,
    )
    p_mtl0 = compute_props_lam_scale_mean(
        market="POINTS", team_abbr="MTL", opp_abbr="BOS",
        league_xg=league_xg, xg_map=xg_map, league_pen=league_pen, pen_comm=pen_comm,
        league_sv=league_sv, gf_map=gf_map, league_pp_frac=league_pp_frac,
        opp_pp_frac_map=opp_pp_frac_map, team_pp_frac_map=team_pp_frac_map,
        props_xg_gamma=gx, props_penalty_gamma=gp, props_goalie_form_gamma=gg, props_strength_gamma=0.0,
    )
    assert p_mtl > p_mtl0

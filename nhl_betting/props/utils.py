from __future__ import annotations

def compute_props_lam_scale_mean(
    market: str,
    team_abbr: str | None,
    opp_abbr: str | None,
    *,
    league_xg: float,
    xg_map: dict,
    league_pen: float,
    pen_comm: dict,
    league_sv: float,
    gf_map: dict,
    league_pp_frac: float,
    opp_pp_frac_map: dict,
    team_pp_frac_map: dict,
    props_xg_gamma: float,
    props_penalty_gamma: float,
    props_goalie_form_gamma: float,
    props_strength_gamma: float,
) -> float:
    """Compute scalar lambda multiplier for a single props row based on team features.

    Applies strength-aware adjustments when PP fractions are available and uses xG, penalties,
    and opponent goalie form as configured. Returns a non-negative float.
    """
    mk = str(market).upper()
    team = (str(team_abbr).upper() if team_abbr else None)
    opp = (str(opp_abbr).upper() if opp_abbr else None)
    lam_scale = 1.0
    try:
        txg = xg_map.get(team) if team else None
        oxg = xg_map.get(opp) if opp else None
        if mk == "SAVES":
            if oxg is not None:
                lam_scale *= (1.0 + props_xg_gamma * ((float(oxg) / league_xg) - 1.0))
            pc = (pen_comm.get(team, {}) or {}).get("committed_per60") if team else None
            if pc is not None:
                lam_scale *= (1.0 + props_penalty_gamma * ((float(pc) / league_pen) - 1.0))
            pp_frac = opp_pp_frac_map.get(team) if team else None
            if pp_frac is not None:
                lam_scale *= (1.0 + props_strength_gamma * (float(pp_frac) - league_pp_frac))
        elif mk == "BLOCKS":
            pc = (pen_comm.get(opp, {}) or {}).get("committed_per60") if opp else None
            if pc is not None:
                lam_scale *= (1.0 + props_penalty_gamma * ((float(pc) / league_pen) - 1.0))
            opp_pp_frac = opp_pp_frac_map.get(opp) if opp else None
            if opp_pp_frac is not None:
                lam_scale *= (1.0 + props_strength_gamma * (float(opp_pp_frac) - league_pp_frac))
        else:
            if txg is not None:
                lam_scale *= (1.0 + props_xg_gamma * ((float(txg) / league_xg) - 1.0))
            pc = (pen_comm.get(opp, {}) or {}).get("committed_per60") if opp else None
            if pc is not None and mk in {"GOALS","ASSISTS","POINTS","SOG","BLOCKS"}:
                lam_scale *= (1.0 + props_penalty_gamma * ((float(pc) / league_pen) - 1.0))
            if mk in {"GOALS","ASSISTS","POINTS"}:
                tpp = team_pp_frac_map.get(team) if team else None
                if tpp is not None:
                    lam_scale *= (1.0 + props_strength_gamma * (float(tpp) - league_pp_frac))
                sv = gf_map.get(opp) if opp else None
                if sv is not None:
                    lam_scale *= (1.0 - props_goalie_form_gamma * (float(sv) - league_sv))
    except Exception:
        lam_scale = lam_scale
    return max(0.0, float(lam_scale))

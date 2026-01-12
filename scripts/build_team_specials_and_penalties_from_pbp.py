import json
from pathlib import Path
from typing import Dict

import duckdb
import pandas as pd


RAW_PBP_DEFAULT = "data/raw/nhl_pbp/pbp_2024.parquet"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _cap01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0


def _list_pbp_files() -> list[str]:
    """Return list of usable PBP parquet files across seasons, excluding tiny stubs."""
    root = Path(__file__).resolve().parents[1] / "data" / "raw" / "nhl_pbp"
    files: list[str] = []
    for p in sorted(root.glob("pbp_*.parquet")):
        try:
            if p.stat().st_size and p.stat().st_size > 100000:  # exclude tiny/empty files
                files.append(str(p))
        except Exception:
            continue
    return files or [RAW_PBP_DEFAULT]


def build_from_pbp(pbp_path: str = RAW_PBP_DEFAULT) -> Dict[str, Dict[str, Dict[str, float]]]:
    con = duckdb.connect()
    paths = _list_pbp_files()
    first = paths[0]
    con.execute(
        f"""
        create or replace table events as
        select
            game_id::INT as game_id,
            game_date::VARCHAR as game_date,
            season::INT as season,
            home_abbreviation::VARCHAR as home,
            away_abbreviation::VARCHAR as away,
            event_type::VARCHAR as event_type,
            coalesce(event_team_abbr, event_team)::VARCHAR as event_team,
            strength::VARCHAR as strength,
            lower(coalesce(penalty_severity, ''))::VARCHAR as penalty_severity,
            coalesce(penalty_minutes, 0)::INT as penalty_minutes,
            coalesce(empty_net, false)::BOOLEAN as empty_net
        from read_parquet('{first}')
        where game_id is not null
        """
    )
    for p in paths[1:]:
        try:
            con.execute(
                f"""
                insert into events
                select
                    game_id::INT as game_id,
                    game_date::VARCHAR as game_date,
                    season::INT as season,
                    home_abbreviation::VARCHAR as home,
                    away_abbreviation::VARCHAR as away,
                    event_type::VARCHAR as event_type,
                    coalesce(event_team_abbr, event_team)::VARCHAR as event_team,
                    strength::VARCHAR as strength,
                    lower(coalesce(penalty_severity, ''))::VARCHAR as penalty_severity,
                    coalesce(penalty_minutes, 0)::INT as penalty_minutes,
                    coalesce(empty_net, false)::BOOLEAN as empty_net
                from read_parquet('{p}')
                where game_id is not null
                """
            )
        except Exception:
            continue

    # Distinct games with teams
    games = con.execute(
        "select distinct game_id, game_date, season, home, away from events"
    ).fetchdf()

    # Penalties committed (per game, per team)
    penalties = con.execute(
        """
        select game_id, game_date, event_team as team, count(*) as penalties
        from events
        where event_type = 'PENALTY' and penalty_minutes >= 2
        group by 1,2,3
        """
    ).fetchdf()

    # Penalties drawn: assign to opponent
    drawn = con.execute(
        """
        select p.game_id, p.game_date,
               case when p.event_team = g.home then g.away else g.home end as team,
               count(*) as drawn
        from (
            select game_id, game_date, event_team from events
            where event_type = 'PENALTY' and penalty_minutes >= 2
        ) p
        join (select distinct game_id, home, away from events) g using (game_id)
        group by 1,2,3
        """
    ).fetchdf()

    # PP goals for (exclude empty net)
    ppg = con.execute(
        """
        select game_id, game_date, event_team as team, count(*) as ppg
        from events
        where event_type = 'GOAL' and lower(strength) like '%power%' and not empty_net
        group by 1,2,3
        """
    ).fetchdf()

    # PP opportunities approximated from opponent penalties (double minor counts as 2)
    ppo = con.execute(
        """
        select p.game_id, p.game_date,
               case when p.event_team = g.home then g.away else g.home end as team,
               sum(case when p.penalty_severity like '%double%' then 2 else 1 end) as ppo
        from (
            select game_id, game_date, event_team, penalty_severity
            from events
            where event_type = 'PENALTY' and penalty_minutes >= 2
        ) p
        join (select distinct game_id, home, away from events) g using (game_id)
        group by 1,2,3
        """
    ).fetchdf()

    # Aggregate to team level
    def agg_sum(df: pd.DataFrame, val_col: str, out_col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["team", out_col])
        a = df.groupby("team")[val_col].sum().reset_index().rename(columns={val_col: out_col})
        return a

    pen_comm = agg_sum(penalties, "penalties", "penalties_committed")
    pen_drawn = agg_sum(drawn, "drawn", "penalties_drawn")
    ppg_for = agg_sum(ppg, "ppg", "ppg")
    ppo_for = agg_sum(ppo, "ppo", "ppo")

    # Games played per team (from games table)
    gp_home = games.groupby("home").size().reset_index(name="gp").rename(columns={"home": "team"})
    gp_away = games.groupby("away").size().reset_index(name="gp").rename(columns={"away": "team"})
    gp = pd.concat([gp_home, gp_away], ignore_index=True).groupby("team")["gp"].sum().reset_index()

    # Merge all
    teams = sorted(set(list(gp.team)))
    df = pd.DataFrame({"team": teams}).merge(gp, on="team", how="left")
    for part in [pen_comm, pen_drawn, ppg_for, ppo_for]:
        df = df.merge(part, on="team", how="left")
    df = df.fillna(0)

    # Rates and percentages
    df["committed_per60"] = df.apply(lambda r: float(r["penalties_committed"]) / max(1, float(r["gp"])), axis=1)
    df["drawn_per60"] = df.apply(lambda r: float(r["penalties_drawn"]) / max(1, float(r["gp"])), axis=1)
    df["pp_pct"] = [
        _cap01((float(a) / float(b)) if b else 0.0) for a, b in zip(df["ppg"], df["ppo"])
    ]

    # Season derived from events
    seasons = [s[0] for s in con.execute("select distinct season from events order by 1").fetchall()]
    season = seasons[-1] if seasons else None

    # Build outputs
    team_specials = {str(row.team).upper(): {"pp_pct": float(row.pp_pct), "pk_pct": _cap01(1.0) } for _, row in df.iterrows()}
    # Estimate PK% from opponents' PP against
    # For a simple approximation using available aggregates: assume league average PPG per PPO,
    # or reuse (1 - opp_ppg/ppo_against). Compute ppo_against via penalties committed mapping.
    # ppo_against approximated as penalties_committed (double minors not expanded):
    # Keep simple: pk_pct = 1 - league_avg_pp_pct
    league_pp = float(df["ppg"].sum()) / float(df["ppo"].sum()) if df["ppo"].sum() > 0 else 0.0
    league_pk = _cap01(1.0 - league_pp)
    for k in list(team_specials.keys()):
        team_specials[k]["pk_pct"] = league_pk

    team_penalties = {str(row.team).upper(): {"committed_per60": float(row.committed_per60), "drawn_per60": float(row.drawn_per60)} for _, row in df.iterrows()}

    # Compute per-season outputs
    per_season: Dict[int, Dict[str, Dict[str, float]]] = {}
    for s in seasons:
        sub_games = games[games["season"] == s]
        if sub_games.empty:
            continue
        pen_s = con.execute(
            f"""
            select game_id, game_date, event_team as team, count(*) as penalties
            from events where season={int(s)} and event_type='PENALTY' and penalty_minutes>=2
            group by 1,2,3
            """
        ).fetchdf()
        drawn_s = con.execute(
            f"""
            select p.game_id, p.game_date,
                   case when p.event_team = g.home then g.away else g.home end as team,
                   count(*) as drawn
            from (
                select game_id, game_date, event_team from events
                where season={int(s)} and event_type='PENALTY' and penalty_minutes>=2
            ) p
            join (select distinct game_id, home, away from events where season={int(s)}) g using (game_id)
            group by 1,2,3
            """
        ).fetchdf()
        ppg_s = con.execute(
            f"""
            select game_id, game_date, event_team as team, count(*) as ppg
            from events
            where season={int(s)} and event_type='GOAL' and lower(strength) like '%power%' and not empty_net
            group by 1,2,3
            """
        ).fetchdf()
        ppo_s = con.execute(
            f"""
            select p.game_id, p.game_date,
                   case when p.event_team = g.home then g.away else g.home end as team,
                   sum(case when p.penalty_severity like '%double%' then 2 else 1 end) as ppo
            from (
                select game_id, game_date, event_team, penalty_severity
                from events
                where season={int(s)} and event_type='PENALTY' and penalty_minutes>=2
            ) p
            join (select distinct game_id, home, away from events where season={int(s)}) g using (game_id)
            group by 1,2,3
            """
        ).fetchdf()
        pen_comm_s = agg_sum(pen_s, "penalties", "penalties_committed")
        pen_drawn_s = agg_sum(drawn_s, "drawn", "penalties_drawn")
        ppg_for_s = agg_sum(ppg_s, "ppg", "ppg")
        ppo_for_s = agg_sum(ppo_s, "ppo", "ppo")
        gp_home_s = sub_games.groupby("home").size().reset_index(name="gp").rename(columns={"home": "team"})
        gp_away_s = sub_games.groupby("away").size().reset_index(name="gp").rename(columns={"away": "team"})
        gp_s = pd.concat([gp_home_s, gp_away_s], ignore_index=True).groupby("team")["gp"].sum().reset_index()
        teams_s = sorted(set(list(gp_s.team)))
        df_s = pd.DataFrame({"team": teams_s}).merge(gp_s, on="team", how="left")
        for part in [pen_comm_s, pen_drawn_s, ppg_for_s, ppo_for_s]:
            df_s = df_s.merge(part, on="team", how="left")
        df_s = df_s.fillna(0)
        df_s["committed_per60"] = df_s.apply(lambda r: float(r["penalties_committed"]) / max(1, float(r["gp"])), axis=1)
        df_s["drawn_per60"] = df_s.apply(lambda r: float(r["penalties_drawn"]) / max(1, float(r["gp"])), axis=1)
        df_s["pp_pct"] = [ _cap01((float(a) / float(b)) if b else 0.0) for a, b in zip(df_s["ppg"], df_s["ppo"]) ]
        league_pp_s = float(df_s["ppg"].sum()) / float(df_s["ppo"].sum()) if df_s["ppo"].sum() > 0 else 0.0
        league_pk_s = _cap01(1.0 - league_pp_s)
        team_specials_s = {str(row.team).upper(): {"pp_pct": float(row.pp_pct), "pk_pct": league_pk_s } for _, row in df_s.iterrows()}
        team_penalties_s = {str(row.team).upper(): {"committed_per60": float(row.committed_per60), "drawn_per60": float(row.drawn_per60)} for _, row in df_s.iterrows()}
        per_season[int(s)] = {"team_specials": team_specials_s, "team_penalties": team_penalties_s}

    return {
        "season": season,
        "team_specials": team_specials,
        "team_penalties": team_penalties,
        "per_season": per_season,
    }


def save_outputs(obj: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    season = obj.get("season")
    # Season-specific files
    if season:
        (PROC_DIR / f"team_special_teams_{season}.json").write_text(
            json.dumps({"season": season, "teams": obj["team_specials"]}, indent=2), encoding="utf-8"
        )
        (PROC_DIR / f"team_penalty_rates_{season}.json").write_text(
            json.dumps(obj["team_penalties"], indent=2), encoding="utf-8"
        )
    # Write per-season bundles
    per_season = obj.get("per_season") or {}
    for s, data in per_season.items():
        (PROC_DIR / f"team_special_teams_{s}.json").write_text(
            json.dumps({"season": s, "teams": data.get("team_specials", {})}, indent=2), encoding="utf-8"
        )
        (PROC_DIR / f"team_penalty_rates_{s}.json").write_text(
            json.dumps(data.get("team_penalties", {}), indent=2), encoding="utf-8"
        )
    # Generic fallbacks
    (PROC_DIR / "team_special_teams.json").write_text(
        json.dumps({"season": season, "teams": obj["team_specials"]}, indent=2), encoding="utf-8"
    )
    (PROC_DIR / "team_penalty_rates.json").write_text(
        json.dumps(obj["team_penalties"], indent=2), encoding="utf-8"
    )


def main():
    out = build_from_pbp()
    save_outputs(out)
    print("[done] wrote team_special_teams and team_penalty_rates from PBP")


if __name__ == "__main__":
    main()

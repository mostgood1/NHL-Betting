from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Optional

import duckdb
import pandas as pd


PBP_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "nhl_pbp"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def _pick_latest_pbp() -> Optional[str]:
    try:
        files = sorted(PBP_DIR.glob("pbp_*.parquet"))
        if not files:
            return None
        # Choose the file with the largest year suffix
        def _year(p: Path) -> int:
            try:
                return int(p.stem.split("_")[-1])
            except Exception:
                return 0
        latest = sorted(files, key=_year)[-1]
        return str(latest)
    except Exception:
        return None


def build_goalie_form_from_pbp(pbp_path: Optional[str] = None, lookback_games: int = 10) -> Dict[str, float]:
    con = duckdb.connect()
    src = pbp_path or _pick_latest_pbp()
    if not src:
        return {}
    con.execute(
        f"""
        create or replace table events as
        select
            game_id::INT as game_id,
            game_date::VARCHAR as game_date,
            home_abbreviation::VARCHAR as home,
            away_abbreviation::VARCHAR as away,
            event_type::VARCHAR as event_type,
            coalesce(event_team_abbr, event_team)::VARCHAR as shooting_team,
            coalesce(empty_net, false)::BOOLEAN as empty_net
        from read_parquet('{src}')
        where game_id is not null
        """
    )

    # Per-game shots and goals by shooting team
    shots = con.execute(
        """
        select game_id, game_date, shooting_team, count(*) as sog
        from events
        where event_type in ('SHOT','GOAL')
        group by 1,2,3
        """
    ).fetchdf()
    goals = con.execute(
        """
        select game_id, game_date, shooting_team, count(*) as goals
        from events
        where event_type = 'GOAL' and not empty_net
        group by 1,2,3
        """
    ).fetchdf()

    if shots.empty and goals.empty:
        return {}

    # Join to map defending team = opponent
    games = con.execute("select distinct game_id, home, away, game_date from events").fetchdf()
    shots = shots.merge(games, on=["game_id", "game_date"], how="left")
    goals = goals.merge(games, on=["game_id", "game_date"], how="left")

    def map_defending(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        df = df.copy()
        df["def_team"] = df.apply(lambda r: r["away"] if r["shooting_team"] == r["home"] else r["home"], axis=1)
        out = df[["game_id", "game_date", "def_team", value_col]].rename(columns={"def_team": "team"})
        return out

    sog_against = map_defending(shots, "sog").rename(columns={"sog": "sog_against"})
    ga = map_defending(goals, "goals").rename(columns={"goals": "ga"})

    per_game = pd.merge(sog_against, ga, on=["game_id", "game_date", "team"], how="left").fillna({"ga": 0})
    per_game["saves"] = per_game["sog_against"] - per_game["ga"]
    per_game = per_game.sort_values(["team", "game_date"])  # ensure order for rolling

    # Rolling sv% over last N games
    per_game["sv_pct"] = per_game.apply(lambda r: (float(r["saves"]) / float(r["sog_against"])) if r["sog_against"] else None, axis=1)
    forms: Dict[str, float] = {}
    for team, gdf in per_game.groupby("team"):
        gdf = gdf.sort_values("game_date")
        # Use rolling mean of sv_pct over last lookback_games, take last value available
        roll = gdf["sv_pct"].rolling(window=lookback_games, min_periods=max(3, min(lookback_games, len(gdf)))).mean()
        if not roll.empty and pd.notna(roll.iloc[-1]):
            forms[str(team).upper()] = float(max(0.0, min(1.0, roll.iloc[-1])))

    return forms


def main():
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    forms = build_goalie_form_from_pbp()
    if not forms:
        print("[warn] no forms computed from PBP")
        return
    today = date.today().strftime("%Y-%m-%d")
    out = pd.DataFrame({"team": list(forms.keys()), "sv_pct_l10": list(forms.values())})
    out_path = PROC_DIR / f"goalie_form_{today}.csv"
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()

"""Ingest NHL PBP parquet (from nhlfastR/fastRhockey) and aggregate period + first-10 goals.

Input: data/raw/nhl_pbp/pbp_*.parquet
Output: data/raw/games_with_periods_pbp.csv

Maps per gamePk the following integer counts:
- period1_home_goals, period1_away_goals
- period2_home_goals, period2_away_goals
- period3_home_goals, period3_away_goals
- goals_first_10min (total goals in first 10:00 of P1)
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nhl_betting.utils.io import RAW_DIR, save_df

PBP_DIR = RAW_DIR / "nhl_pbp"


def _load_pbp_frames() -> list[pd.DataFrame]:
    if not PBP_DIR.exists():
        raise FileNotFoundError(f"{PBP_DIR} not found. Run scripts/nhl_pbp_fetch.R first.")
    files = sorted(PBP_DIR.glob("pbp_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No pbp_*.parquet files in {PBP_DIR}")
    dfs: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    return dfs


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Different packages may name columns differently; try common variants
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    # Identify essential columns
    game_col = pick("game_id", "gamepk", "game_pk")
    period_col = pick("period", "period_number")
    team_col = pick("event_team", "team", "team_for")
    team_abbr_col = pick("event_team_abbr", "event_team_abbreviation", "team_abbreviation", "team_abbr")
    ev_type_col = pick("event_type", "event")
    event_col = pick("event",)
    # Prefer explicit seconds in period, else game_seconds (works for period 1), else period_time string
    clock_col = pick("period_seconds", "game_seconds", "clock", "clock_time", "period_time")
    home_abbrev_col = pick("home_abbreviation", "home_team", "home_team_abbr")
    away_abbrev_col = pick("away_abbreviation", "away_team", "away_team_abbr")
    home_name_col = pick("home_name", "home_team_name")
    away_name_col = pick("away_name", "away_team_name")
    season_col = pick("season",)
    game_date_col = pick("game_date", "date", "game_start")

    for required in [game_col, period_col, ev_type_col]:
        if required is None:
            raise KeyError("Missing required PBP columns. Got columns: " + ", ".join(df.columns))

    out = pd.DataFrame({
        "gamePk": df[game_col],
        "period": df[period_col],
        "event_type": df[ev_type_col].astype(str).str.upper(),
    })
    if event_col is not None and event_col in df.columns:
        out["event"] = df[event_col].astype(str).str.upper()

    # Team fields (may not always be present)
    if team_col is not None and team_col in df.columns:
        out["event_team"] = df[team_col]
    else:
        out["event_team"] = np.nan
    if team_abbr_col is not None and team_abbr_col in df.columns:
        out["event_team_abbr"] = df[team_abbr_col]

    if clock_col and clock_col in df.columns:
        out["clock"] = df[clock_col]
    else:
        out["clock"] = np.nan

    # Team abbreviations for mapping home/away when available
    if home_abbrev_col in df.columns and away_abbrev_col in df.columns:
        out["home"] = df[home_abbrev_col]
        out["away"] = df[away_abbrev_col]
    # Also keep full names if available for fallback
    if home_name_col is not None and home_name_col in df.columns:
        out["home_name"] = df[home_name_col]
    if away_name_col is not None and away_name_col in df.columns:
        out["away_name"] = df[away_name_col]
    # Season and game date (YYYY-MM-DD)
    if season_col is not None and season_col in df.columns:
        out["season"] = df[season_col]
    if game_date_col is not None and game_date_col in df.columns:
        out["game_date"] = df[game_date_col]

    return out


def _parse_seconds(clock_val) -> float:
    # Accept formats like MM:SS or numeric seconds
    if pd.isna(clock_val):
        return np.nan
    if isinstance(clock_val, (int, float)):
        return float(clock_val)
    s = str(clock_val)
    if ":" in s:
        try:
            mm, ss = s.split(":", 1)
            return int(mm) * 60 + int(ss)
        except Exception:
            return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def aggregate_periods() -> pd.DataFrame:
    dfs = _load_pbp_frames()
    std_frames: list[pd.DataFrame] = []
    for i, df in enumerate(dfs):
        try:
            std = _standardize_columns(df)
            std_frames.append(std)
        except KeyError as e:
            warnings.warn(f"[ingest] Skipping PBP file {i} due to missing columns: {e}")
            continue
    if not std_frames:
        raise RuntimeError("No usable PBP frames after standardization. Check Parquet files and schema.")
    pbp = pd.concat(std_frames, ignore_index=True)

    # Keep only goals (either event or event_type equals GOAL)
    ev_upper = pbp.get("event", pd.Series(index=pbp.index, dtype=object)).astype(str).str.upper()
    et_upper = pbp.get("event_type", pd.Series(index=pbp.index, dtype=object)).astype(str).str.upper()
    goals = pbp[ev_upper.eq("GOAL") | et_upper.eq("GOAL")].copy()
    goals["period"] = pd.to_numeric(goals["period"], errors="coerce").fillna(0).astype(int)
    # Compute seconds into period if clock available (assume mm:ss elapsed)
    goals["sec"] = goals["clock"].map(_parse_seconds)

    # Determine home vs away goal if possible
    # If home/away abbreviations present, tag by matching event_team
    if {"home", "away"}.issubset(goals.columns):
        # Prefer abbreviation match if available
        if "event_team_abbr" in goals.columns:
            goals["is_home_goal"] = (goals["event_team_abbr"].astype(str) == goals["home"].astype(str))
        else:
            # Fallback to name match when possible
            if "home_name" in goals.columns:
                goals["is_home_goal"] = goals["event_team"].astype(str).str.lower() == goals["home_name"].astype(str).str.lower()
            else:
                goals["is_home_goal"] = False
    else:
        goals["is_home_goal"] = False  # unknown; will be counted into totals only

    # Prefer robust keys: season, game_date, home, away
    if "game_date" not in goals.columns:
        goals["game_date"] = pd.NA
    if "season" not in goals.columns:
        goals["season"] = pd.NA
    key_cols = ["season", "game_date", "home", "away", "period"] if {"home","away"}.issubset(goals.columns) else ["gamePk", "period"]
    agg = goals.groupby(key_cols).agg(
        home_goals=("is_home_goal", lambda x: int(np.nansum(x.astype(int)))) ,
        total_goals=("event_type", "count"),
        first10=("sec", lambda s: int(np.sum(pd.to_numeric(s, errors="coerce").fillna(1e9) < 600)))
    ).reset_index()

    # Derive away goals as total - home
    agg["away_goals"] = agg["total_goals"] - agg["home_goals"]

    # Pivot to columns
    if {"season","game_date","home","away"}.issubset(agg.columns):
        out = agg.pivot(index=["season","game_date","home","away"], columns="period", values=["home_goals","away_goals","first10"]).fillna(0).astype(int)
    else:
        out = agg.pivot(index="gamePk", columns="period", values=["home_goals", "away_goals", "first10"]).fillna(0).astype(int)
    out.columns = [f"{a}{b}" for a, b in out.columns]
    out = out.reset_index()

    # Build expected columns
    def col_or_zero(name: str) -> pd.Series:
        return out[name] if name in out.columns else 0

    if "gamePk" in out.columns:
        result = pd.DataFrame({
            "gamePk": out["gamePk"],
            "period1_home_goals": col_or_zero("home_goals1"),
            "period1_away_goals": col_or_zero("away_goals1"),
            "period2_home_goals": col_or_zero("home_goals2"),
            "period2_away_goals": col_or_zero("away_goals2"),
            "period3_home_goals": col_or_zero("home_goals3"),
            "period3_away_goals": col_or_zero("away_goals3"),
            "goals_first_10min": col_or_zero("first101"),
        })
    else:
        result = pd.DataFrame({
            "season": out["season"],
            "game_date": out["game_date"],
            "home": out["home"],
            "away": out["away"],
            "period1_home_goals": col_or_zero("home_goals1"),
            "period1_away_goals": col_or_zero("away_goals1"),
            "period2_home_goals": col_or_zero("home_goals2"),
            "period2_away_goals": col_or_zero("away_goals2"),
            "period3_home_goals": col_or_zero("home_goals3"),
            "period3_away_goals": col_or_zero("away_goals3"),
            "goals_first_10min": col_or_zero("first101"),
        })
    result["period_source"] = "pbp"

    return result


def main():
    print("[ingest] Aggregating PBP...")
    periods = aggregate_periods()
    # Merge onto basic game identifiers if available
    games_path = RAW_DIR / "games_with_periods.csv"
    if games_path.exists():
        base = pd.read_csv(games_path)
        # Two merge strategies: prefer robust season/date/home/away when available; else fallback to gamePk
        if {"season","game_date","home","away"}.issubset(periods.columns):
            bj = base.copy()
            bj["home"] = bj["home"].astype(str).str.upper()
            bj["away"] = bj["away"].astype(str).str.upper()
            try:
                bj["date_only"] = pd.to_datetime(bj["date"]).dt.strftime("%Y-%m-%d")
            except Exception:
                bj["date_only"] = bj["date"].astype(str).str.slice(0,10)
            p = periods.copy()
            p["home"] = p["home"].astype(str).str.upper()
            p["away"] = p["away"].astype(str).str.upper()
            p["game_date"] = p["game_date"].astype(str).str.slice(0,10)
            merged = bj.merge(
                p,
                left_on=["season","date_only","home","away"],
                right_on=["season","game_date","home","away"],
                how="left",
                suffixes=("", "_pbp")
            )
        else:
            merged = base.merge(periods, on="gamePk", how="left", suffixes=("", "_pbp"))
        for c in [
            "period1_home_goals","period1_away_goals",
            "period2_home_goals","period2_away_goals",
            "period3_home_goals","period3_away_goals",
            "goals_first_10min",
        ]:
            pbp_col = c + "_pbp"
            if pbp_col in merged.columns:
                merged[c] = merged[pbp_col].fillna(merged.get(c))
        # Update period_source to pbp where we filled any of the pbp columns
        if "period_source" in merged.columns:
            pbp_cols = [c+"_pbp" for c in [
                "period1_home_goals","period1_away_goals",
                "period2_home_goals","period2_away_goals",
                "period3_home_goals","period3_away_goals",
                "goals_first_10min",
            ]]
            existing_pbp_cols = [c for c in pbp_cols if c in merged.columns]
            if existing_pbp_cols:
                filled_mask = merged[existing_pbp_cols].notna().any(axis=1)
                merged.loc[filled_mask, "period_source"] = "pbp"
        # Drop helper pbp columns and save
        drop_cols = [col for col in merged.columns if col.endswith("_pbp")] + ["date_only","game_date"]
        merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
        save_df(merged, games_path)
        print(f"[ingest] Updated {games_path} with PBP-derived counts")
    else:
        out = RAW_DIR / "games_with_periods_pbp.csv"
        save_df(periods, out)
        print(f"[ingest] Wrote {out}")


if __name__ == "__main__":
    main()

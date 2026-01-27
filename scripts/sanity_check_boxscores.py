import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from nhl_betting.utils.io import PROC_DIR
from nhl_betting.data.collect import collect_player_game_stats

METRICS = [
    ("shots", "shots", float),
    ("goals", "goals", float),
    ("assists", "assists", float),
    ("points", None, float),  # compute points from actuals
    ("blocks", "blocked", float),
    ("saves", "saves", float),
]


def _time_on_ice_to_minutes(val: str) -> float | None:
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip()
        if not s:
            return None
        parts = s.split(":")
        if len(parts) == 2:
            m = int(parts[0])
            sec = int(parts[1])
            return round(m + sec / 60.0, 2)
        # sometimes seconds total
        v = float(s)
        if v > 300:  # looks like seconds
            return round(v / 60.0, 2)
        return round(v, 2)
    except Exception:
        return None


def load_sim_boxscores(date: str) -> pd.DataFrame:
    p = PROC_DIR / f"props_boxscores_sim_{date}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Sim boxscores not found: {p}")
    df = pd.read_csv(p)
    # keep totals (period=0)
    if "period" in df.columns:
        df = df[df["period"].fillna(0).astype(int) == 0]
    # ensure numeric
    for c in ["shots","goals","assists","points","blocks","saves","toi_sec"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    return df


def load_actual_boxscores(date: str) -> pd.DataFrame:
    # Best-effort: collect from Stats API for the day
    df = collect_player_game_stats(date, date, source="stats")
    df = df[df["date"].astype(str).str.startswith(date)]
    # normalize points and TOI
    df["points"] = pd.to_numeric(df.get("goals"), errors="coerce") + pd.to_numeric(df.get("assists"), errors="coerce")
    df["toi_min"] = df["timeOnIce"].apply(_time_on_ice_to_minutes)
    return df


def compare(date: str) -> dict:
    sim = load_sim_boxscores(date)
    act = load_actual_boxscores(date)
    # join on team + player_id when available; fallback to name
    join_cols = ["team", "player_id"]
    if not set(join_cols).issubset(sim.columns):
        # try player name
        join_cols = ["team", "player"]
    left = sim.copy()
    right = act.copy()
    # rename actual columns to match join
    if join_cols[1] == "player":
        right = right.rename(columns={"player": "player"})
    else:
        right = right.rename(columns={"player_id": "player_id"})
    merged = pd.merge(left, right, on=join_cols, how="inner", suffixes=("_sim", "_act"))
    if merged.empty:
        return {"status": "no-join", "date": date, "sim_rows": len(sim), "act_rows": len(act)}
    # compute metrics
    report = {"date": date, "rows": len(merged)}
    for sim_key, act_key, cast in METRICS:
        sk = f"{sim_key}_sim"
        ak = f"{(act_key or sim_key)}_act"
        if sk not in merged.columns or ak not in merged.columns:
            continue
        x = pd.to_numeric(merged[sk], errors="coerce")
        y = pd.to_numeric(merged[ak], errors="coerce")
        diff = (x - y).dropna()
        report[f"mae_{sim_key}"] = float(np.mean(np.abs(diff))) if len(diff) > 0 else None
        report[f"bias_{sim_key}"] = float(np.mean(diff)) if len(diff) > 0 else None
    # TOI sanity: use sim toi_sec as minutes; actual from timeOnIce
    if "toi_sec_sim" in merged.columns and "toi_min_act" in merged.columns:
        x = pd.to_numeric(merged["toi_sec_sim"], errors="coerce") / 60.0
        y = pd.to_numeric(merged["toi_min_act"], errors="coerce")
        diff = (x - y).dropna()
        report["mae_toi_min"] = float(np.mean(np.abs(diff))) if len(diff) > 0 else None
        report["bias_toi_min"] = float(np.mean(diff)) if len(diff) > 0 else None
        # variance per team to catch identical TOI artifacts
        by_team = merged.groupby("team").apply(lambda d: float(np.std(pd.to_numeric(d["toi_sec_sim"], errors="coerce") / 60.0)))
        report["std_toi_min_by_team"] = by_team.to_dict()
    # sample worst players by shots diff
    if {"shots_sim", "shots_act"}.issubset(merged.columns):
        merged["shots_diff"] = pd.to_numeric(merged["shots_sim"], errors="coerce") - pd.to_numeric(merged["shots_act"], errors="coerce")
        worst = merged.sort_values("shots_diff", ascending=False).head(10)[[join_cols[1], "team", "shots_sim", "shots_act", "shots_diff"]]
        report["worst_shots"] = worst.to_dict(orient="records")
    return report


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    try:
        rep = compare(date)
        print("[sanity]", pd.Series(rep).to_json())
    except Exception as e:
        print(f"[sanity] ERROR: {e}")
        sys.exit(1)

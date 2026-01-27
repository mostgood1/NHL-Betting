import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from nhl_betting.utils.io import PROC_DIR
from nhl_betting.data.collect import collect_player_game_stats, _parse_boxscore_players
from nhl_betting.data.nhl_api import NHLClient as StatsClient
from nhl_betting.data.nhl_api_web import NHLWebClient as WebClient
from nhl_betting.web.teams import get_team_assets

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
    # normalize team to abbreviation
    def _sim_to_abbr(x: str) -> str | None:
        try:
            s = str(x or '').strip()
            if not s:
                return None
            if len(s) <= 3:
                return s.upper()
            a = get_team_assets(s).get('abbr')
            return str(a).upper() if a else None
        except Exception:
            return None
    if 'team' in df.columns:
        df['team'] = df['team'].map(lambda x: _sim_to_abbr(x) or (str(x).upper() if isinstance(x, str) else None))
    return df


def load_actual_boxscores(date: str) -> pd.DataFrame:
    """Load actual player boxscores for a date.

    Prefer Stats API; if empty, fallback to Web API. Normalize to include:
    columns: ['date','team','player','goals','assists','points','shots','blocked','saves','timeOnIce','toi_min'].
    Team must be abbreviation in 'team'.
    """
    # Try direct Stats API schedule+boxscores for the day
    def _direct_stats(day: str) -> pd.DataFrame:
        try:
            sc = StatsClient()
            games = sc.schedule(day, day)
            rows = []
            for g in games:
                # only final games
                if getattr(g, 'home_goals', None) is None or getattr(g, 'away_goals', None) is None:
                    continue
                box = sc.boxscore(getattr(g, 'gamePk'))
                rows.extend(_parse_boxscore_players(box, getattr(g, 'gamePk'), getattr(g, 'gameDate'), getattr(g, 'home'), getattr(g, 'away')))
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()
    def _direct_web(day: str) -> pd.DataFrame:
        try:
            wc = WebClient()
            games = wc.schedule_day(day)
            rows = []
            for g in games:
                if getattr(g, 'home_goals', None) is None or getattr(g, 'away_goals', None) is None:
                    continue
                box = wc.boxscore(getattr(g, 'gamePk'))
                rows.extend(_parse_boxscore_players(box, getattr(g, 'gamePk'), getattr(g, 'gameDate'), getattr(g, 'home'), getattr(g, 'away')))
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()
    df = _direct_stats(date)
    if df is None or df.empty:
        df = _direct_web(date)
    # If still empty, fall back to historical collector sources
    if df is None or df.empty:
        def _try_source(src: str) -> pd.DataFrame:
            try:
                dfx = collect_player_game_stats(date, date, source=src)
                if dfx is None:
                    return pd.DataFrame()
                dfx = dfx[dfx["date"].astype(str).str.startswith(date)]
                return dfx
            except Exception:
                return pd.DataFrame()
        df = _try_source("stats")
        if df is None or df.empty:
            df = _try_source("web")
    if df is None:
        df = pd.DataFrame()
    # ensure team abbreviation column
    if "team" not in df.columns:
        if "team_abbr" in df.columns:
            df["team"] = df["team_abbr"].astype(str).str.upper()
        else:
            # Best-effort: leave empty
            df["team"] = None
    else:
        df["team"] = df["team"].astype(str)
    # Map team names to abbreviations where needed
    def _to_abbr(s: str) -> str | None:
        try:
            x = str(s or '').strip()
            if not x:
                return None
            # If already short, return upper
            if len(x) <= 3:
                return x.upper()
            a = get_team_assets(x).get('abbr')
            return str(a).upper() if a else None
        except Exception:
            return None
    df["team"] = df["team"].map(lambda x: _to_abbr(x) or (str(x).upper() if isinstance(x, str) else None))
    # normalize points and TOI
    df["points"] = pd.to_numeric(df.get("goals"), errors="coerce") + pd.to_numeric(df.get("assists"), errors="coerce")
    df["toi_min"] = df.get("timeOnIce").apply(_time_on_ice_to_minutes) if "timeOnIce" in df.columns else None
    return df


def compare(date: str) -> dict:
    sim = load_sim_boxscores(date)
    act = load_actual_boxscores(date)
    # Decide join keys present in both sim and actual
    sim_cols = set(sim.columns)
    act_cols = set(act.columns)
    if {"team", "player_id"}.issubset(sim_cols) and {"team", "player_id"}.issubset(act_cols):
        join_cols = ["team", "player_id"]
    elif {"team", "player"}.issubset(sim_cols) and {"team", "player"}.issubset(act_cols):
        join_cols = ["team", "player"]
    else:
        # Fallback: attempt team + player name using available columns
        join_cols = ["team", "player"]
        if "player" not in sim_cols and "player_id" in sim_cols:
            # map sim player_id to name using actual when possible
            try:
                pid_to_name = act.dropna(subset=["player"]).set_index("player_id")["player"].to_dict() if "player_id" in act.columns else {}
                sim = sim.copy(); sim["player"] = sim["player_id"].map(lambda x: pid_to_name.get(int(x)) if pd.notna(x) else None)
            except Exception:
                sim = sim.copy(); sim["player"] = None
    left = sim.copy(); right = act.copy()
    merged = pd.merge(left, right, on=join_cols, how="inner", suffixes=("_sim", "_act"))
    if merged.empty:
        # Fallback: join on team + normalized player names if available
        def _norm(n):
            try:
                s = str(n or '').strip().lower()
                return ''.join(ch for ch in s if ch.isalnum())
            except Exception:
                return None
        if "player" in left.columns and "player" in right.columns:
            l = left.copy(); r = right.copy()
            l["player_norm"] = l["player"].map(_norm)
            r["player_norm"] = r["player"].map(_norm)
            merged = pd.merge(l, r, on=["team", "player_norm"], how="inner", suffixes=("_sim", "_act"))
        if merged is None or merged.empty:
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

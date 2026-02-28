from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ..utils.io import PROC_DIR, RAW_DIR
from ..web.teams import get_team_assets


def _norm_name(x: Any) -> str:
    try:
        s = str(x or "").strip()
    except Exception:
        return ""
    s = " ".join(s.split())
    return s


def _norm_name_key(x: Any) -> str:
    s = _norm_name(x).lower()
    # Keep it simple and stable: remove periods and collapse whitespace.
    s = s.replace(".", " ")
    s = " ".join(s.split())
    return s


def _safe_date(x: Any) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(x, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _ymd_to_date(d: str) -> datetime.date:
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def _sigmoid(x: pd.Series) -> pd.Series:
    # Numerically stable-ish sigmoid for moderate ranges.
    x = pd.to_numeric(x, errors="coerce")
    x = x.clip(lower=-10, upper=10)
    return 1.0 / (1.0 + np.exp(-x))


@lru_cache(maxsize=2)
def _load_player_game_history() -> pd.DataFrame:
    """Load player game stats with team/opponent abbreviations.

    Returns a DataFrame with columns:
      gamePk, game_date, team_abbr, opp_abbr, player_norm, shots, goals, assists, blocked, saves
    """
    p = RAW_DIR / "player_game_stats.csv"
    if not p.exists() or getattr(p.stat(), "st_size", 0) == 0:
        return pd.DataFrame()
    usecols = [
        "gamePk",
        "date",
        "team",
        "player",
        "shots",
        "goals",
        "assists",
        "blocked",
        "saves",
    ]
    try:
        df = pd.read_csv(p, usecols=usecols)
    except Exception:
        df = pd.read_csv(p)
        df = df[[c for c in usecols if c in df.columns]].copy()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["_ts"] = df["date"].map(_safe_date)
    df = df[df["_ts"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["game_date"] = df["_ts"].dt.date

    # Map full team name -> abbreviation.
    try:
        uniq = df["team"].dropna().astype(str).unique().tolist()
    except Exception:
        uniq = []
    team_map: dict[str, str] = {}
    for nm in uniq:
        try:
            ab = (get_team_assets(str(nm)).get("abbr") or "").upper()
            if ab:
                team_map[str(nm)] = ab
        except Exception:
            continue
    df["team_abbr"] = df["team"].astype(str).map(lambda x: team_map.get(str(x), ""))
    df["team_abbr"] = df["team_abbr"].where(df["team_abbr"].astype(str).str.len() > 0, None)

    df["player_norm"] = df["player"].map(_norm_name_key)

    # Opponent abbr per game/team
    teams = df[["gamePk", "team_abbr"]].dropna().drop_duplicates()
    if teams.empty:
        df["opp_abbr"] = None
    else:
        opp = teams.merge(teams, on="gamePk", suffixes=("", "_opp"))
        opp = opp[opp["team_abbr"] != opp["team_abbr_opp"]]
        opp = opp.drop_duplicates(subset=["gamePk", "team_abbr"], keep="first")
        opp = opp.rename(columns={"team_abbr_opp": "opp_abbr"})
        df = df.merge(opp[["gamePk", "team_abbr", "opp_abbr"]], on=["gamePk", "team_abbr"], how="left")

    # Coerce numeric stat columns
    for col in ["shots", "goals", "assists", "blocked", "saves"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = [
        "gamePk",
        "game_date",
        "team_abbr",
        "opp_abbr",
        "player_norm",
        "shots",
        "goals",
        "assists",
        "blocked",
        "saves",
    ]
    return df[keep].copy()


def build_team_opponent_map(date: str) -> dict[str, str]:
    """Map team abbreviation -> opponent abbreviation for a slate date."""
    d = str(date)
    # Prefer non-sim predictions when available, else sim.
    cand = [PROC_DIR / f"predictions_{d}.csv", PROC_DIR / f"predictions_sim_{d}.csv"]
    pred_path = next((p for p in cand if p.exists() and getattr(p.stat(), "st_size", 0) > 0), None)
    if pred_path is None:
        return {}
    try:
        preds = pd.read_csv(pred_path)
    except Exception:
        return {}
    if preds is None or preds.empty or not {"home", "away"}.issubset(set(preds.columns)):
        return {}
    out: dict[str, str] = {}
    for _, r in preds.iterrows():
        try:
            home = str(r.get("home") or "").strip()
            away = str(r.get("away") or "").strip()
            if not home or not away:
                continue
            h = (get_team_assets(home).get("abbr") or "").upper()
            a = (get_team_assets(away).get("abbr") or "").upper()
            if h and a:
                out[h] = a
                out[a] = h
        except Exception:
            continue
    return out


def attach_prop_edge_signals(
    *,
    date: str,
    props: pd.DataFrame,
    recent_games: int = 10,
    season_games: int = 30,
    h2h_games: int = 5,
    opp_games: int = 10,
) -> pd.DataFrame:
    """Attach non-EV edge signals and a composite edge score.

    Expected columns in `props`:
      team, player, market, line, p_over

    Adds columns:
      opp, player_recent_avg, player_season_avg, player_h2h_avg, opp_allow_avg,
      direction_score, side_suggested, chosen_prob, edge_score, edge_reasons
    """
    if props is None or props.empty:
        return props

    df = props.copy()
    for col in ["team", "player", "market"]:
        if col not in df.columns:
            df[col] = None
    df["team"] = df["team"].astype(str).str.upper()
    df["player"] = df["player"].map(_norm_name)
    df["player_norm"] = df["player"].map(_norm_name_key)
    df["market"] = df["market"].astype(str).str.upper()
    df["line"] = pd.to_numeric(df.get("line"), errors="coerce")
    df["p_over"] = pd.to_numeric(df.get("p_over"), errors="coerce")

    team_to_opp = build_team_opponent_map(date)
    df["opp"] = df["team"].map(lambda t: team_to_opp.get(str(t).upper(), None))

    hist = _load_player_game_history()
    if hist is None or hist.empty:
        # Still compute simple model-only direction/score.
        model_dir = (df["p_over"].fillna(0.5) - 0.5)
        df["direction_score"] = (2.0 * model_dir).astype(float)
        df["side_suggested"] = np.where(df["direction_score"] >= 0, "Over", "Under")
        df["chosen_prob"] = np.where(df["side_suggested"] == "Over", df["p_over"], (1.0 - df["p_over"]))
        df["edge_score"] = ((df["chosen_prob"].fillna(0.5) - 0.5) * 2.0).clip(lower=0.0, upper=1.0)
        df["edge_reasons"] = df.apply(
            lambda r: f"p {float(r.get('chosen_prob') or 0.0):.0%} (model-only)", axis=1
        )
        return df.drop(columns=["player_norm"], errors="ignore")

    cutoff = _ymd_to_date(date)
    markets = sorted(set([m for m in df["market"].dropna().astype(str).tolist() if m]))
    if not markets:
        return df.drop(columns=["player_norm"], errors="ignore")

    stat_cols = {
        "SOG": "shots",
        "GOALS": "goals",
        "ASSISTS": "assists",
        "BLOCKS": "blocked",
        "SAVES": "saves",
        "POINTS": "points",
    }
    need_points = "POINTS" in markets
    hsub = hist.copy()
    if need_points:
        hsub["points"] = pd.to_numeric(hsub.get("goals"), errors="coerce").fillna(0.0) + pd.to_numeric(
            hsub.get("assists"), errors="coerce"
        ).fillna(0.0)

    # Limit to cutoff and only players of interest.
    want_players = set(df["player_norm"].dropna().astype(str).tolist())
    hsub = hsub[(hsub["game_date"] < cutoff) & (hsub["player_norm"].isin(want_players))].copy()
    hsub = hsub.sort_values(["player_norm", "game_date"], ascending=True)

    # Player recent + season averages per market
    player_recent: dict[tuple[str, str], float] = {}
    player_season: dict[tuple[str, str], float] = {}
    player_h2h: dict[tuple[str, str, str], float] = {}

    for m in markets:
        col = stat_cols.get(m)
        if not col or col not in hsub.columns:
            continue
        v = pd.to_numeric(hsub[col], errors="coerce")
        # Recent (last N games)
        try:
            rmean = hsub.assign(_v=v).groupby("player_norm")["_v"].apply(lambda s: s.tail(int(recent_games)).mean())
            for k, vv in rmean.items():
                if pd.notna(vv):
                    player_recent[(k, m)] = float(vv)
        except Exception:
            pass
        # Season-ish (last M games)
        try:
            smean = hsub.assign(_v=v).groupby("player_norm")["_v"].apply(lambda s: s.tail(int(season_games)).mean())
            for k, vv in smean.items():
                if pd.notna(vv):
                    player_season[(k, m)] = float(vv)
        except Exception:
            pass
        # Head-to-head (vs opponent)
        try:
            hh = hsub.dropna(subset=["opp_abbr"]).assign(_v=v)
            hh = hh.groupby(["player_norm", "opp_abbr"])["_v"].apply(lambda s: s.tail(int(h2h_games)).mean())
            for (pn, opp), vv in hh.items():
                if pd.notna(vv):
                    player_h2h[(pn, str(opp), m)] = float(vv)
        except Exception:
            pass

    # Team defense: how much each team allows (opponent 'for') per game.
    # Build team_game frame from full history (not player-limited) for stable opponent defense signals.
    t = hist[hist["game_date"] < cutoff].copy()
    if need_points and "points" not in t.columns:
        t["points"] = pd.to_numeric(t.get("goals"), errors="coerce").fillna(0.0) + pd.to_numeric(
            t.get("assists"), errors="coerce"
        ).fillna(0.0)
    t = t.dropna(subset=["team_abbr"])
    if t.empty:
        opp_allow = {}
        league_mu = {}
        league_sd = {}
    else:
        # Sum team totals per game
        agg = {
            "shots": "sum",
            "goals": "sum",
            "assists": "sum",
            "blocked": "sum",
            "saves": "sum",
        }
        if need_points:
            agg["points"] = "sum"
        tg = t.groupby(["gamePk", "game_date", "team_abbr", "opp_abbr"], as_index=False).agg(agg)
        # Allowed stat for team = opponent's for-stat
        # self-join on (gamePk, opp_abbr)
        opp_for = tg.rename(columns={"team_abbr": "_opp_team"})
        merged = tg.merge(
            opp_for,
            left_on=["gamePk", "opp_abbr"],
            right_on=["gamePk", "_opp_team"],
            suffixes=("", "_oppfor"),
            how="left",
        )
        opp_allow: dict[tuple[str, str], float] = {}
        league_mu: dict[str, float] = {}
        league_sd: dict[str, float] = {}
        for m in markets:
            col = stat_cols.get(m)
            if not col:
                continue
            allow_col = f"{col}_oppfor"
            if allow_col not in merged.columns:
                continue
            try:
                s = pd.to_numeric(merged[allow_col], errors="coerce")
                # recent opponent allowed avg per team
                rr = merged.assign(_v=s).sort_values(["team_abbr", "game_date"]).groupby("team_abbr")["_v"].apply(
                    lambda x: x.tail(int(opp_games)).mean()
                )
                vals = [float(vv) for vv in rr.dropna().tolist()]
                league_mu[m] = float(np.mean(vals)) if vals else 0.0
                league_sd[m] = float(np.std(vals)) if vals else 1.0
                for k, vv in rr.items():
                    if pd.notna(vv):
                        opp_allow[(str(k), m)] = float(vv)
            except Exception:
                continue

    # Population-level stat SD used to scale player deltas
    stat_sd: dict[str, float] = {}
    for m in markets:
        col = stat_cols.get(m)
        if not col or col not in t.columns:
            stat_sd[m] = 1.0
            continue
        try:
            vals = pd.to_numeric(t[col], errors="coerce").dropna().astype(float)
            sd = float(vals.std()) if len(vals) else 1.0
            stat_sd[m] = sd if (sd and np.isfinite(sd) and sd > 1e-6) else 1.0
        except Exception:
            stat_sd[m] = 1.0

    # Attach signals per row
    def _get_player_val(dct: dict, pn: str, m: str) -> float | None:
        v = dct.get((pn, m))
        return float(v) if v is not None and np.isfinite(v) else None

    def _get_h2h(pn: str, opp: str | None, m: str) -> float | None:
        if not opp:
            return None
        v = player_h2h.get((pn, str(opp), m))
        return float(v) if v is not None and np.isfinite(v) else None

    df["player_recent_avg"] = df.apply(
        lambda r: _get_player_val(player_recent, str(r.get("player_norm") or ""), str(r.get("market") or "")), axis=1
    )
    df["player_season_avg"] = df.apply(
        lambda r: _get_player_val(player_season, str(r.get("player_norm") or ""), str(r.get("market") or "")), axis=1
    )
    df["player_h2h_avg"] = df.apply(
        lambda r: _get_h2h(str(r.get("player_norm") or ""), r.get("opp"), str(r.get("market") or "")), axis=1
    )
    df["opp_allow_avg"] = df.apply(
        lambda r: opp_allow.get((str(r.get("opp") or ""), str(r.get("market") or ""))), axis=1
    )

    # Direction score (non-EV): combine model direction + player recency + opponent defense + H2H.
    model_dir = (df["p_over"].fillna(0.5) - 0.5) * 2.0  # [-1, 1]
    mkt = df["market"].astype(str)
    sd = mkt.map(lambda mm: stat_sd.get(mm, 1.0)).astype(float)
    sd = sd.where(sd > 1e-6, 1.0)

    recent_z = (pd.to_numeric(df["player_recent_avg"], errors="coerce") - df["line"]) / sd
    h2h_z = (pd.to_numeric(df["player_h2h_avg"], errors="coerce") - df["line"]) / sd
    opp_mu = mkt.map(lambda mm: league_mu.get(mm, 0.0)).astype(float)
    opp_sd = mkt.map(lambda mm: league_sd.get(mm, 1.0)).astype(float)
    opp_sd = opp_sd.where(opp_sd > 1e-6, 1.0)
    opp_z = (pd.to_numeric(df["opp_allow_avg"], errors="coerce") - opp_mu) / opp_sd

    # Missing values contribute 0.
    recent_z = recent_z.fillna(0.0)
    h2h_z = h2h_z.fillna(0.0)
    opp_z = opp_z.fillna(0.0)

    direction = (0.60 * model_dir) + (0.20 * recent_z) + (0.15 * opp_z) + (0.05 * h2h_z)
    df["direction_score"] = pd.to_numeric(direction, errors="coerce").fillna(0.0)
    df["side_suggested"] = np.where(df["direction_score"] >= 0, "Over", "Under")
    df["chosen_prob"] = np.where(df["side_suggested"] == "Over", df["p_over"], (1.0 - df["p_over"]))

    # Support scores in [0,1] for ranking (higher = stronger).
    # Align deltas to the suggested side.
    sgn = np.where(df["side_suggested"] == "Over", 1.0, -1.0)
    recent_support = _sigmoid(sgn * recent_z)
    opp_support = _sigmoid(sgn * opp_z)
    h2h_support = _sigmoid(sgn * h2h_z)
    model_support = ((pd.to_numeric(df["chosen_prob"], errors="coerce").fillna(0.5) - 0.5) * 2.0).clip(0.0, 1.0)
    df["edge_score"] = (0.40 * model_support + 0.25 * recent_support + 0.20 * opp_support + 0.15 * h2h_support).clip(
        0.0, 1.0
    )

    def _reason_row(r: pd.Series) -> str:
        try:
            p = float(r.get("chosen_prob") or 0.0)
        except Exception:
            p = 0.0
        parts: list[str] = [f"p {p:.0%}"]
        try:
            ra = r.get("player_recent_avg")
            if ra is not None and pd.notna(ra):
                parts.append(f"L{int(recent_games)} {float(ra):.2f}")
        except Exception:
            pass
        try:
            hh = r.get("player_h2h_avg")
            if hh is not None and pd.notna(hh):
                parts.append(f"H2H {float(hh):.2f}")
        except Exception:
            pass
        try:
            oa = r.get("opp_allow_avg")
            if oa is not None and pd.notna(oa):
                parts.append(f"opp allow {float(oa):.2f}")
        except Exception:
            pass
        try:
            ln = r.get("line")
            if ln is not None and pd.notna(ln):
                parts.append(f"line {float(ln):.1f}")
        except Exception:
            pass
        return "; ".join(parts)

    df["edge_reasons"] = df.apply(_reason_row, axis=1)

    return df.drop(columns=["player_norm"], errors="ignore")

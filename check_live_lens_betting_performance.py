from __future__ import annotations

import argparse
import ast
import json
import math
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from nhl_betting.data.nhl_api_web import NHLWebClient
from nhl_betting.utils.odds import american_to_decimal


def _parse_ymd(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _daterange(start_ymd: str, end_ymd: str) -> list[str]:
    start = _parse_ymd(start_ymd)
    end = _parse_ymd(end_ymd)
    if end < start:
        raise ValueError("end before start")
    out: list[str] = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return out


def _norm_team(s: str) -> str:
    s = str(s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = " ".join(s.split())
    return s


def _norm_person_key(name: Any) -> str:
    """Normalize a person name into a robust key.

    Goal: allow matching between sources like "Nathan MacKinnon" and "N. MacKinnon".
    """
    s = str(name or "").strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("'", "")
    # Keep letters/spaces/dots/hyphens to interpret initials.
    cleaned = []
    for ch in s:
        if ch.isalnum() or ch in {" ", ".", "-"}:
            cleaned.append(ch)
    s = "".join(cleaned)
    s = s.lower().strip()
    s = " ".join(s.split())
    if not s:
        return ""
    parts = [p for p in s.replace("-", " ").split(" ") if p]
    if not parts:
        return ""
    # If we have at least 2 tokens, compress to "<first-initial> <last>".
    if len(parts) >= 2:
        first = parts[0]
        last = parts[-1]
        fi = first[0] if first else ""
        last = last.replace(".", "")
        return f"{fi} {last}".strip()
    # Single token: just return.
    return parts[0].replace(".", "")


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return v
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return int(x)
    except Exception:
        return None


def _as_dt(s: Any) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


@dataclass(frozen=True)
class BetKey:
    date: str
    gamePk: int
    market: str
    side: Optional[str]
    line: Optional[float]
    period: Optional[int]
    player_key: Optional[str]
    goalie_key: Optional[str]


def _iter_snapshots(jsonl_path: Path) -> Iterable[dict]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_mmss_clock(s: Any) -> Optional[int]:
    """Parse a MM:SS clock string into seconds."""
    if not s:
        return None
    try:
        parts = str(s).strip().split(":")
        if len(parts) != 2:
            return None
        mm = int(parts[0])
        ss = int(parts[1])
        if mm < 0 or ss < 0 or ss >= 60:
            return None
        return mm * 60 + ss
    except Exception:
        return None


def _is_live_state(game_state: Any) -> bool:
    st = str(game_state or "").upper().strip()
    if not st:
        return False
    # Be permissive: different sources use LIVE/CRIT/IN_PROGRESS.
    return ("LIVE" in st) or ("CRIT" in st) or ("IN_PROGRESS" in st)


def _is_final_state(game_state: Any) -> bool:
    st = str(game_state or "").upper().strip()
    return bool(st) and st.startswith("FINAL")


def _update_final_scores_from_snapshot(snapshot: dict, out: dict[int, tuple[int, int]]) -> None:
    """Update gamePk->(home_goals, away_goals) for games that are FINAL in a snapshot."""
    games = snapshot.get("games") or []
    if not isinstance(games, list):
        return
    for g in games:
        if not isinstance(g, dict):
            continue
        if not _is_final_state(g.get("gameState")):
            continue
        gamePk = _safe_int(g.get("gamePk"))
        if gamePk is None:
            continue
        score = g.get("score") if isinstance(g.get("score"), dict) else None
        if not isinstance(score, dict):
            continue
        hg = _safe_int(score.get("home"))
        ag = _safe_int(score.get("away"))
        if hg is None or ag is None:
            continue
        out[int(gamePk)] = (int(hg), int(ag))


def _update_period_scores_from_snapshot(snapshot: dict, out: dict[tuple[int, int], tuple[int, int]]) -> None:
    """Update (gamePk, period)->(home_goals, away_goals) when a period is complete.

    This enables settling PERIOD_* bets during a live game once we advance past that period
    (or the period clock hits 00:00).
    """
    games = snapshot.get("games") or []
    if not isinstance(games, list):
        return
    for g in games:
        if not isinstance(g, dict):
            continue
        gamePk = _safe_int(g.get("gamePk"))
        if gamePk is None:
            continue
        try:
            cur_period = int(g.get("period")) if g.get("period") is not None else None
        except Exception:
            cur_period = None
        clock_sec = _parse_mmss_clock(g.get("clock"))
        st_live = _is_live_state(g.get("gameState"))

        lens = g.get("lens") if isinstance(g.get("lens"), dict) else None
        per_list = lens.get("periods") if isinstance(lens, dict) else None
        if not isinstance(per_list, list):
            continue

        # If clock is 00:00, treat the current period as complete too.
        cur_period_complete = (cur_period is not None) and (clock_sec is not None) and (clock_sec <= 0)

        for p in per_list:
            if not isinstance(p, dict):
                continue
            pn = _safe_int(p.get("period"))
            if pn is None or pn < 1 or pn > 3:
                continue
            h = p.get("home") if isinstance(p.get("home"), dict) else {}
            a = p.get("away") if isinstance(p.get("away"), dict) else {}
            hg = _safe_int(h.get("goals"))
            ag = _safe_int(a.get("goals"))
            if hg is None or ag is None:
                continue

            complete = False
            if not st_live:
                complete = True
            elif cur_period is not None and cur_period > int(pn):
                complete = True
            elif cur_period_complete and cur_period == int(pn):
                complete = True

            if complete:
                out[(int(gamePk), int(pn))] = (int(hg), int(ag))


def _extract_bets_from_snapshot(snapshot: dict, source_path: Path) -> list[dict]:
    rows: list[dict] = []
    asof_utc = snapshot.get("asof_utc")
    odds_asof_utc = snapshot.get("odds_asof_utc")
    regions = snapshot.get("regions")
    best = snapshot.get("best")
    date = snapshot.get("date")
    for g in snapshot.get("games", []) or []:
        gamePk = _safe_int(g.get("gamePk"))
        if gamePk is None:
            continue
        score = g.get("score") or {}
        for sig in g.get("signals", []) or []:
            if str(sig.get("action") or "").upper() != "BET":
                continue
            price = _safe_int(sig.get("price"))
            if price is None:
                # Not settleable / not actually bettable
                continue
            market = str(sig.get("market") or "").strip().upper() or None
            if not market:
                continue

            line = _safe_float(sig.get("line"))
            side = sig.get("side")
            side = str(side).strip().upper() if side is not None else None
            period = _safe_int(sig.get("period"))
            player = sig.get("player")
            goalie = sig.get("goalie")
            team = sig.get("team")
            driver_tags = sig.get("driver_tags")
            driver_meta = sig.get("driver_meta")
            if not isinstance(driver_meta, dict):
                driver_meta = None
            if isinstance(driver_tags, str):
                driver_tags = [driver_tags]
            if not isinstance(driver_tags, list):
                driver_tags = None
            else:
                driver_tags = [str(x) for x in driver_tags if str(x).strip()]

            # Basic sanity: only support settling markets we know
            rows.append(
                {
                    "source": str(source_path.as_posix()),
                    "asof_utc": asof_utc,
                    "asof_dt": _as_dt(asof_utc),
                    "odds_asof_utc": odds_asof_utc,
                    "odds_asof_dt": _as_dt(odds_asof_utc),
                    "regions": regions,
                    "best": best,
                    "date": date,
                    "gamePk": int(gamePk),
                    "home": g.get("home"),
                    "away": g.get("away"),
                    "gameState": g.get("gameState"),
                    "period": _safe_int(g.get("period")),
                    "clock": g.get("clock"),
                    "score_home": _safe_int(score.get("home")),
                    "score_away": _safe_int(score.get("away")),
                    "scope": sig.get("scope"),
                    "market": market,
                    "label": sig.get("label"),
                    "side": side,
                    "line": line,
                    "sig_period": period,
                    "player": player,
                    "goalie": goalie,
                    "team": team,
                    "player_key": _norm_person_key(player),
                    "goalie_key": _norm_person_key(goalie),
                    "driver_tags": driver_tags,
                    "driver_meta": driver_meta,
                    "price_american": price,
                    "p_model": _safe_float(sig.get("p_model")),
                    "implied": _safe_float(sig.get("implied")),
                    "edge": _safe_float(sig.get("edge")),
                    "fair_price_american": _safe_int(sig.get("fair_price_american")),
                    "target_max_price_american": _safe_int(sig.get("target_max_price_american")),
                    "elapsed_min": _safe_float(sig.get("elapsed_min")),
                }
            )
    return rows


def _load_games_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    if "gamePk" not in df.columns:
        return None
    df["gamePk"] = pd.to_numeric(df["gamePk"], errors="coerce")
    df = df.dropna(subset=["gamePk"]).copy()
    df["gamePk"] = df["gamePk"].astype(int)
    return df


def _final_scores_from_player_stats(player_stats_csv: Path, game_pks: set[int], home_by_game: dict[int, str], away_by_game: dict[int, str]) -> dict[int, tuple[Optional[int], Optional[int]]]:
    if not player_stats_csv.exists() or player_stats_csv.stat().st_size == 0:
        return {}

    usecols = ["gamePk", "team", "goals"]
    df = pd.read_csv(player_stats_csv, usecols=usecols)
    df["gamePk"] = pd.to_numeric(df["gamePk"], errors="coerce")
    df = df.dropna(subset=["gamePk"]).copy()
    df["gamePk"] = df["gamePk"].astype(int)
    df = df[df["gamePk"].isin(list(game_pks))].copy()
    if df.empty:
        return {}

    df["goals"] = pd.to_numeric(df["goals"], errors="coerce").fillna(0.0)
    agg = df.groupby(["gamePk", "team"], as_index=False)["goals"].sum()

    out: dict[int, tuple[Optional[int], Optional[int]]] = {}
    for gamePk in game_pks:
        g = agg[agg["gamePk"] == gamePk]
        if g.empty:
            continue
        team_to_goals = {str(r["team"]): int(round(float(r["goals"]))) for _, r in g.iterrows()}
        home = home_by_game.get(gamePk)
        away = away_by_game.get(gamePk)
        home_norm = _norm_team(home)
        away_norm = _norm_team(away)
        mapped: dict[str, int] = {}
        for t, goals in team_to_goals.items():
            mapped[_norm_team(t)] = goals
        hg = mapped.get(home_norm)
        ag = mapped.get(away_norm)
        if hg is None or ag is None:
            # Last-resort: if exactly two teams, assign deterministically by string match proximity
            # (still better than failing outright)
            teams = sorted(mapped.items(), key=lambda kv: kv[0])
            if len(teams) == 2:
                # If one matches home, prefer it
                if teams[0][0] == home_norm:
                    hg, ag = teams[0][1], teams[1][1]
                elif teams[1][0] == home_norm:
                    hg, ag = teams[1][1], teams[0][1]
                else:
                    # Unknown mapping; assume teams[0]=away teams[1]=home is too risky.
                    # Leave missing rather than potentially flipping sides.
                    hg, ag = None, None
            else:
                hg, ag = None, None
        out[gamePk] = (hg, ag)

    return out


def _final_period_goals_web(game_pks: set[int], timeout_sec: float = 10.0) -> dict[tuple[int, int], tuple[int, int]]:
    """Fetch final per-period goal splits from NHL Web API.

    Returns mapping (gamePk, period) -> (home_goals, away_goals).

    Notes:
    - Best-effort: some games/periods may be missing.
    - Uses /gamecenter/{gamePk}/boxscore first; falls back to /landing summary.scoring.
    """

    def _extract_from_box(box: Any) -> dict[int, tuple[int, int]]:
        try:
            if not isinstance(box, dict):
                return {}
            periods = box.get("periods") or box.get("periodSummary") or []
            if not isinstance(periods, list):
                return {}
            out: dict[int, tuple[int, int]] = {}
            for p in periods:
                if not isinstance(p, dict):
                    continue
                pd = p.get("periodDescriptor") or {}
                per = None
                if isinstance(pd, dict):
                    per = pd.get("number") or pd.get("period")
                if per is None:
                    per = p.get("period") or p.get("currentPeriod")
                per_i = _safe_int(per)
                if per_i is None:
                    continue

                home_p = p.get("home") or p.get("homeTeam") or {}
                away_p = p.get("away") or p.get("awayTeam") or {}
                if not isinstance(home_p, dict):
                    home_p = {}
                if not isinstance(away_p, dict):
                    away_p = {}

                hg = _safe_int(home_p.get("goals"))
                if hg is None:
                    hg = _safe_int(home_p.get("score"))
                ag = _safe_int(away_p.get("goals"))
                if ag is None:
                    ag = _safe_int(away_p.get("score"))
                if hg is None or ag is None:
                    continue
                out[int(per_i)] = (int(hg), int(ag))
            return out
        except Exception:
            return {}

    def _extract_from_landing(landing: Any) -> dict[int, tuple[int, int]]:
        try:
            if not isinstance(landing, dict):
                return {}
            summary = landing.get("summary") if isinstance(landing, dict) else None
            scoring = (summary or {}).get("scoring") if isinstance(summary, dict) else None
            if not isinstance(scoring, list):
                return {}

            out: dict[int, tuple[int, int]] = {}
            for bucket in scoring:
                if not isinstance(bucket, dict):
                    continue
                pd = bucket.get("periodDescriptor") or {}
                per = None
                if isinstance(pd, dict):
                    per = pd.get("number") or pd.get("period")
                if per is None:
                    per = bucket.get("period")
                per_i = _safe_int(per)
                if per_i is None:
                    continue

                goals = bucket.get("goals")
                home_g = 0
                away_g = 0
                if isinstance(goals, list):
                    for goal in goals:
                        if not isinstance(goal, dict):
                            continue
                        ih = goal.get("isHome")
                        if ih is True:
                            home_g += 1
                        elif ih is False:
                            away_g += 1
                out[int(per_i)] = (int(home_g), int(away_g))
            return out
        except Exception:
            return {}

    if not game_pks:
        return {}

    client = NHLWebClient(timeout=float(timeout_sec))
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for gamePk in sorted(set(int(x) for x in game_pks)):
        per_map: dict[int, tuple[int, int]] = {}
        # Try boxscore first
        try:
            box = client.boxscore(int(gamePk))
            per_map = _extract_from_box(box)
        except Exception:
            per_map = {}

        # Fallback: landing summary scoring (counts goals list)
        if not per_map:
            try:
                landing = client._get(f"/gamecenter/{int(gamePk)}/landing", params=None, retries=2)
                per_map = _extract_from_landing(landing)
            except Exception:
                per_map = {}

        for per, (hg, ag) in per_map.items():
            if per is None:
                continue
            out[(int(gamePk), int(per))] = (int(hg), int(ag))

    return out


def _parse_player_field_to_name(player_field: Any) -> str:
    """player_game_stats.csv stores player as a stringified dict like {'default': 'N. Schmaltz'}"""
    s = str(player_field or "").strip()
    if not s:
        return ""
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return str(obj.get("default") or "").strip()
    except Exception:
        pass
    return s


def _final_player_stats(
    player_stats_csv: Path,
    game_pks: set[int],
    player_keys: set[str],
) -> dict[tuple[int, str], dict[str, float]]:
    """Return per-(gamePk, player_key) totals for stats needed to settle props."""
    if not player_stats_csv.exists() or player_stats_csv.stat().st_size == 0:
        return {}
    if not game_pks or not player_keys:
        return {}

    usecols = [
        "gamePk",
        "player",
        "role",
        "shots",
        "goals",
        "assists",
        "blocked",
        "saves",
        "shotsAgainst",
    ]
    df = pd.read_csv(player_stats_csv, usecols=usecols)
    df["gamePk"] = pd.to_numeric(df["gamePk"], errors="coerce")
    df = df.dropna(subset=["gamePk"]).copy()
    df["gamePk"] = df["gamePk"].astype(int)
    df = df[df["gamePk"].isin(list(game_pks))].copy()
    if df.empty:
        return {}

    df["player_name"] = df["player"].apply(_parse_player_field_to_name)
    df["player_key"] = df["player_name"].apply(_norm_person_key)
    df = df[df["player_key"].isin(list(player_keys))].copy()
    if df.empty:
        return {}

    for c in ["shots", "goals", "assists", "blocked", "saves", "shotsAgainst"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    agg = (
        df.groupby(["gamePk", "player_key"], as_index=False)[
            ["shots", "goals", "assists", "blocked", "saves", "shotsAgainst"]
        ]
        .sum()
    )

    out: dict[tuple[int, str], dict[str, float]] = {}
    for _, r in agg.iterrows():
        key = (int(r["gamePk"]), str(r["player_key"]))
        out[key] = {
            "shots": float(r.get("shots") or 0.0),
            "goals": float(r.get("goals") or 0.0),
            "assists": float(r.get("assists") or 0.0),
            "blocked": float(r.get("blocked") or 0.0),
            "saves": float(r.get("saves") or 0.0),
            "shotsAgainst": float(r.get("shotsAgainst") or 0.0),
        }
    return out


def _final_scores(
    bets: pd.DataFrame,
    games_csv_path: Path,
    player_stats_csv: Path,
    allow_web_fetch: bool,
) -> dict[int, tuple[Optional[int], Optional[int]]]:
    game_pks = set(pd.to_numeric(bets["gamePk"], errors="coerce").dropna().astype(int).tolist())
    if not game_pks:
        return {}

    home_by_game = {
        int(r["gamePk"]): str(r["home"]) for _, r in bets[["gamePk", "home"]].drop_duplicates().iterrows() if pd.notna(r["gamePk"])
    }
    away_by_game = {
        int(r["gamePk"]): str(r["away"]) for _, r in bets[["gamePk", "away"]].drop_duplicates().iterrows() if pd.notna(r["gamePk"])
    }

    out: dict[int, tuple[Optional[int], Optional[int]]] = {}

    # 1) data/raw/games.csv (fast)
    games_df = _load_games_csv(games_csv_path)
    if games_df is not None and not games_df.empty:
        gsub = games_df[games_df["gamePk"].isin(list(game_pks))].copy()
        for _, r in gsub.iterrows():
            hg = _safe_int(r.get("home_goals"))
            ag = _safe_int(r.get("away_goals"))
            if hg is not None and ag is not None:
                out[int(r["gamePk"])] = (hg, ag)

    # 2) NHL Web API schedule_day(date) (one call per date)
    if allow_web_fetch:
        missing = sorted([pk for pk in game_pks if pk not in out])
        if missing:
            client = NHLWebClient()
            for date in sorted(set(bets["date"].dropna().astype(str).tolist())):
                try:
                    games = client.schedule_day(date)
                except Exception:
                    continue
                for g in games:
                    if g.gamePk not in game_pks or g.gamePk in out:
                        continue
                    if g.home_goals is None or g.away_goals is None:
                        continue
                    out[g.gamePk] = (int(g.home_goals), int(g.away_goals))

    # 3) player_game_stats.csv aggregation (offline fallback)
    missing = sorted([pk for pk in game_pks if pk not in out])
    if missing:
        ps = _final_scores_from_player_stats(
            player_stats_csv=player_stats_csv,
            game_pks=set(missing),
            home_by_game=home_by_game,
            away_by_game=away_by_game,
        )
        out.update({k: v for k, v in ps.items() if v[0] is not None and v[1] is not None})

    return out


def _settle_row(
    row: pd.Series,
    final_scores: dict[int, tuple[Optional[int], Optional[int]]],
    final_period_scores: Optional[dict[tuple[int, int], tuple[int, int]]] = None,
) -> dict[str, Any]:
    gamePk = int(row["gamePk"])
    market = str(row["market"] or "").upper()
    side = str(row["side"] or "").upper() if row.get("side") is not None else None
    line = row.get("line")

    result: Optional[str] = None

    # NOTE: Period markets can settle as soon as we know the period-final goals,
    # even if the game is still live (final_scores may be unavailable).
    if market == "PERIOD_ML":
        per_i = _safe_int(row.get("sig_period"))
        if final_period_scores is None or per_i is None:
            result = None
        else:
            phg, pag = final_period_scores.get((gamePk, int(per_i)), (None, None))
            if phg is None or pag is None:
                result = None
            else:
                # Two-way period ML: treat a tied period as PUSH.
                if side == "HOME":
                    result = "WIN" if phg > pag else ("LOSE" if phg < pag else "PUSH")
                elif side == "AWAY":
                    result = "WIN" if pag > phg else ("LOSE" if pag < phg else "PUSH")
                else:
                    result = None

        # Profit only depends on odds.
        price = _safe_int(row.get("price_american"))
        profit: Optional[float] = None
        if result in {"WIN", "LOSE", "PUSH"} and price is not None:
            dec = float(american_to_decimal(int(price)))
            if result == "WIN":
                profit = dec - 1.0
            elif result == "LOSE":
                profit = -1.0
            else:
                profit = 0.0
        return {"home_goals_final": None, "away_goals_final": None, "result": result, "profit_units": profit}

    if market == "PERIOD_TOTAL":
        per_i = _safe_int(row.get("sig_period"))
        if final_period_scores is None or per_i is None:
            result = None
        elif line is None or not math.isfinite(float(line)):
            result = None
        else:
            phg, pag = final_period_scores.get((gamePk, int(per_i)), (None, None))
            if phg is None or pag is None:
                result = None
            else:
                total = int(phg) + int(pag)
                if side == "UNDER":
                    result = "WIN" if total < float(line) else ("LOSE" if total > float(line) else "PUSH")
                elif side == "OVER":
                    result = "WIN" if total > float(line) else ("LOSE" if total < float(line) else "PUSH")
                else:
                    result = None

        price = _safe_int(row.get("price_american"))
        profit = None
        if result in {"WIN", "LOSE", "PUSH"} and price is not None:
            dec = float(american_to_decimal(int(price)))
            if result == "WIN":
                profit = dec - 1.0
            elif result == "LOSE":
                profit = -1.0
            else:
                profit = 0.0
        return {"home_goals_final": None, "away_goals_final": None, "result": result, "profit_units": profit}

    # Non-period markets require final game score.
    hg, ag = final_scores.get(gamePk, (None, None))
    if hg is None or ag is None:
        return {"home_goals_final": None, "away_goals_final": None, "result": None, "profit_units": None}

    if market == "ML":
        if side == "HOME":
            result = "WIN" if hg > ag else ("LOSE" if hg < ag else "PUSH")
        elif side == "AWAY":
            result = "WIN" if ag > hg else ("LOSE" if ag < hg else "PUSH")
        else:
            result = None

    elif market == "TOTAL":
        if line is None or not math.isfinite(float(line)):
            result = None
        else:
            total = int(hg) + int(ag)
            if side == "UNDER":
                result = "WIN" if total < float(line) else ("LOSE" if total > float(line) else "PUSH")
            elif side == "OVER":
                result = "WIN" if total > float(line) else ("LOSE" if total < float(line) else "PUSH")
            else:
                result = None

    elif market == "PUCKLINE":
        # Expected sides: HOME_-1.5, AWAY_+1.5
        if side == "HOME_-1.5":
            result = "WIN" if (hg - ag) >= 2 else "LOSE"
        elif side == "AWAY_+1.5":
            result = "WIN" if (ag - hg) >= -1 else "LOSE"
        else:
            result = None

    price = _safe_int(row.get("price_american"))
    profit: Optional[float] = None
    if result in {"WIN", "LOSE", "PUSH"} and price is not None:
        dec = float(american_to_decimal(int(price)))
        if result == "WIN":
            profit = dec - 1.0
        elif result == "LOSE":
            profit = -1.0
        else:
            profit = 0.0

    return {"home_goals_final": int(hg), "away_goals_final": int(ag), "result": result, "profit_units": profit}


def _settle_prop_row(row: pd.Series, player_stats: dict[tuple[int, str], dict[str, float]]) -> dict[str, Any]:
    market = str(row.get("market") or "").upper()
    if not market.startswith("PROP_"):
        return {"prop_value_final": None}

    side = str(row.get("side") or "").upper()
    line = row.get("line")
    if line is None or not math.isfinite(float(line)):
        return {"prop_value_final": None, "result": None, "profit_units": None}

    gamePk = int(row.get("gamePk"))
    pk = str(row.get("player_key") or "").strip()
    if not pk:
        return {"prop_value_final": None, "result": None, "profit_units": None}

    stats = player_stats.get((gamePk, pk))
    if not stats:
        return {"prop_value_final": None, "result": None, "profit_units": None}

    mk = market.replace("PROP_", "", 1)
    if mk == "SOG":
        val = float(stats.get("shots") or 0.0)
    elif mk == "GOALS":
        val = float(stats.get("goals") or 0.0)
    elif mk == "ASSISTS":
        val = float(stats.get("assists") or 0.0)
    elif mk == "POINTS":
        val = float(stats.get("goals") or 0.0) + float(stats.get("assists") or 0.0)
    elif mk == "BLOCKS":
        val = float(stats.get("blocked") or 0.0)
    else:
        return {"prop_value_final": None, "result": None, "profit_units": None}

    # Settle over/under
    if side == "OVER":
        result = "WIN" if val > float(line) else ("LOSE" if val < float(line) else "PUSH")
    elif side == "UNDER":
        result = "WIN" if val < float(line) else ("LOSE" if val > float(line) else "PUSH")
    else:
        return {"prop_value_final": val, "result": None, "profit_units": None}

    price = _safe_int(row.get("price_american"))
    profit = None
    if result in {"WIN", "LOSE", "PUSH"} and price is not None:
        dec = float(american_to_decimal(int(price)))
        if result == "WIN":
            profit = dec - 1.0
        elif result == "LOSE":
            profit = -1.0
        else:
            profit = 0.0

    return {"prop_value_final": val, "result": result, "profit_units": profit}


def _bucket_edge(edge: Any) -> str:
    e = _safe_float(edge)
    if e is None:
        return "(missing)"
    if e < 0.02:
        return "<0.02"
    if e < 0.04:
        return "0.02-0.04"
    if e < 0.06:
        return "0.04-0.06"
    return ">=0.06"


_LIVE_LENS_MAX_ELAPSED_SECONDS = 65 * 60


def _format_elapsed_mmss(total_seconds: int) -> str:
    total_seconds = max(0, int(total_seconds))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


def _elapsed_seconds(elapsed_min: Any) -> Optional[float]:
    em = _safe_float(elapsed_min)
    if em is None:
        return None
    sec = max(0.0, float(em) * 60.0)
    if sec >= float(_LIVE_LENS_MAX_ELAPSED_SECONDS):
        sec = float(_LIVE_LENS_MAX_ELAPSED_SECONDS) - 1e-9
    return sec


def _bucket_elapsed(elapsed_min: Any, *, bin_seconds: int = 60) -> str:
    sec = _elapsed_seconds(elapsed_min)
    if sec is None:
        return "(missing)"
    bin_seconds = max(1, int(bin_seconds))
    start = int(math.floor(sec / float(bin_seconds))) * bin_seconds
    start = max(0, min(start, _LIVE_LENS_MAX_ELAPSED_SECONDS - bin_seconds))
    end = min(_LIVE_LENS_MAX_ELAPSED_SECONDS, start + bin_seconds)
    return f"{_format_elapsed_mmss(start)}-{_format_elapsed_mmss(end)}"


def _bucket_elapsed_coarse(elapsed_min: Any) -> str:
    em = _safe_float(elapsed_min)
    if em is None:
        return "(missing)"
    if em < 4:
        return "0-4"
    if em < 8:
        return "4-8"
    if em < 12:
        return "8-12"
    if em < 20:
        return "12-20"
    return ">=20"


def _bucket_odds_staleness_min(asof_dt: Any, odds_asof_dt: Any) -> str:
    try:
        if asof_dt is None or odds_asof_dt is None:
            return "(missing)"
        delta = (pd.Timestamp(asof_dt) - pd.Timestamp(odds_asof_dt)).total_seconds() / 60.0
        if not math.isfinite(float(delta)):
            return "(missing)"
        if delta <= 1.0:
            return "<=1m"
        if delta <= 3.0:
            return "1-3m"
        if delta <= 6.0:
            return "3-6m"
        return ">6m"
    except Exception:
        return "(missing)"


def _summarize(df: pd.DataFrame, title: str) -> None:
    if df.empty:
        print(f"{title}: no rows")
        return

    settled = df[df["profit_units"].notna()].copy()
    n = len(settled)
    if n == 0:
        print(f"{title}: no settled bets")
        return

    units = float(settled["profit_units"].sum())
    roi = units / float(n)
    wins = int((settled["result"] == "WIN").sum())
    pushes = int((settled["result"] == "PUSH").sum())
    losses = int((settled["result"] == "LOSE").sum())
    win_rate = wins / float(wins + losses) if (wins + losses) > 0 else float("nan")

    avg_edge = _safe_float(settled["edge"].mean())

    # model-implied EV per unit at bet time
    dec = settled["price_american"].apply(lambda x: float(american_to_decimal(int(x))) if pd.notna(x) else float("nan"))
    p = pd.to_numeric(settled["p_model"], errors="coerce")
    ev = p * (dec - 1.0) - (1.0 - p)
    ev = ev.replace([float("inf"), float("-inf")], float("nan"))
    avg_ev = _safe_float(ev.mean())

    print(
        f"{title}: bets={n} units={units:+.3f} roi={roi:+.3%} wins={wins} losses={losses} pushes={pushes} win%={win_rate:.3%} avg_edge={avg_edge if avg_edge is not None else float('nan'):+.3f} avg_ev={avg_ev if avg_ev is not None else float('nan'):+.3f}"
    )


def _summarize_grouped(df: pd.DataFrame, group_col: str, title: str, top_n: int = 12) -> None:
    if df.empty or group_col not in df.columns:
        return
    settled = df[df["profit_units"].notna()].copy()
    if settled.empty:
        return
    g = (
        settled.groupby(group_col, dropna=False)
        .agg(bets=("profit_units", "size"), units=("profit_units", "sum"), wins=("result", lambda x: int((x == "WIN").sum())), losses=("result", lambda x: int((x == "LOSE").sum())))
        .reset_index()
    )
    g["roi"] = g["units"] / g["bets"].replace(0, float("nan"))
    g = g.sort_values(["bets", "roi"], ascending=[False, False])
    print(f"{title} (top {top_n} by bets):")
    for _, r in g.head(top_n).iterrows():
        key = r.get(group_col)
        bets = int(r.get("bets") or 0)
        units = float(r.get("units") or 0.0)
        roi = float(r.get("roi") or 0.0)
        wins = int(r.get("wins") or 0)
        losses = int(r.get("losses") or 0)
        wr = wins / float(wins + losses) if (wins + losses) > 0 else float("nan")
        print(f"  - {group_col}={key}: bets={bets} units={units:+.3f} roi={roi:+.3%} win%={wr:.3%}")


def _json_sanitize(x: Any) -> Any:
    """Convert common pandas/python objects into JSON-friendly primitives."""
    try:
        if x is None:
            return None
        # pandas NA / NaN
        try:
            if isinstance(x, float) and math.isnan(x):
                return None
        except Exception:
            pass
        # pandas Timestamp
        try:
            if isinstance(x, pd.Timestamp):
                if pd.isna(x):
                    return None
                return x.to_pydatetime().isoformat()
        except Exception:
            pass
        # python datetime
        if isinstance(x, datetime):
            return x.isoformat()
        # numpy types
        try:
            import numpy as np

            if isinstance(x, (np.integer,)):
                return int(x)
            if isinstance(x, (np.floating,)):
                v = float(x)
                if math.isnan(v) or not math.isfinite(v):
                    return None
                return v
        except Exception:
            pass
        # Recursively sanitize
        if isinstance(x, dict):
            return {str(k): _json_sanitize(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_json_sanitize(v) for v in x]
        return x
    except Exception:
        return str(x)


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest realized P&L of Live Lens BET signals from saved JSONL snapshots")
    ap.add_argument("--date", help="Single date YYYY-MM-DD")
    ap.add_argument("--start", help="Start date YYYY-MM-DD (inclusive)")
    ap.add_argument("--end", help="End date YYYY-MM-DD (inclusive)")
    ap.add_argument(
        "--all",
        action="store_true",
        help="Scan all live_lens_signals_*.jsonl files under --signals-dir",
    )
    ap.add_argument(
        "--signals-dir",
        default=str(Path("data/processed/live_lens")),
        help="Directory containing live_lens_signals_YYYY-MM-DD.jsonl",
    )
    ap.add_argument(
        "--dedupe",
        default="first",
        choices=["first", "best_price", "none"],
        help="How to dedupe repeated snapshots of the same bet key",
    )
    ap.add_argument(
        "--allow-web-fetch",
        action="store_true",
        help="Allow NHLWebClient(schedule_day) calls to fill missing final scores",
    )
    ap.add_argument(
        "--out",
        help="Optional output CSV path for the settled bet ledger",
    )
    ap.add_argument(
        "--breakdowns",
        action="store_true",
        help="Print extra breakdowns (driver tags, edge/time buckets, odds staleness)",
    )
    args = ap.parse_args()

    if args.all:
        dates = []
    elif args.date:
        dates = [args.date]
    else:
        if not args.start or not args.end:
            ap.error("Provide --all, or --date, or --start/--end")
        dates = _daterange(args.start, args.end)

    signals_dir = Path(args.signals_dir)
    jsonl_paths: list[Path] = []
    if args.all:
        jsonl_paths = sorted([p for p in signals_dir.glob("live_lens_signals_*.jsonl") if p.is_file() and p.stat().st_size > 0])
    else:
        for d in dates:
            p = signals_dir / f"live_lens_signals_{d}.jsonl"
            if p.exists() and p.stat().st_size > 0:
                jsonl_paths.append(p)

    if not jsonl_paths:
        raise SystemExit(f"No JSONL files found for dates={dates} in {signals_dir}")

    bet_rows: list[dict] = []
    # Snapshot-derived settlement helpers (enables real-time settlement without waiting for game final web fetch)
    period_scores_from_snaps: dict[tuple[int, int], tuple[int, int]] = {}
    final_scores_from_snaps: dict[int, tuple[int, int]] = {}
    for p in jsonl_paths:
        for snap in _iter_snapshots(p):
            try:
                _update_period_scores_from_snapshot(snap, period_scores_from_snaps)
            except Exception:
                pass
            try:
                _update_final_scores_from_snapshot(snap, final_scores_from_snaps)
            except Exception:
                pass
            bet_rows.extend(_extract_bets_from_snapshot(snap, source_path=p))

    if not bet_rows:
        print("No BET signals with non-null price found.")
        return 0

    bets = pd.DataFrame(bet_rows)

    # Add analysis buckets
    bets["edge_bucket"] = bets["edge"].apply(_bucket_edge)
    bets["elapsed_bucket"] = bets["elapsed_min"].apply(_bucket_elapsed)
    bets["elapsed_bucket_1m"] = bets["elapsed_bucket"]
    bets["elapsed_bucket_15s"] = bets["elapsed_min"].apply(lambda x: _bucket_elapsed(x, bin_seconds=15))
    bets["elapsed_bucket_coarse"] = bets["elapsed_min"].apply(_bucket_elapsed_coarse)
    bets["odds_staleness_bucket"] = bets.apply(lambda r: _bucket_odds_staleness_min(r.get("asof_dt"), r.get("odds_asof_dt")), axis=1)

    # Dedupe repeated snapshots of same "bet".
    if args.dedupe != "none":
        bets = bets.sort_values(["date", "gamePk", "market", "side", "line", "asof_dt"], na_position="last")
        # Include period + player/goalie identity so props and period markets dedupe correctly.
        key_cols = ["date", "gamePk", "market", "side", "line", "sig_period", "player_key", "goalie_key"]
        if args.dedupe == "first":
            bets = bets.drop_duplicates(subset=key_cols, keep="first").copy()
        elif args.dedupe == "best_price":
            bets["decimal_odds"] = bets["price_american"].apply(lambda x: float(american_to_decimal(int(x))) if pd.notna(x) else float("nan"))
            bets = bets.sort_values(["date", "gamePk", "market", "side", "line", "decimal_odds", "asof_dt"], ascending=[True, True, True, True, True, False, True])
            bets = bets.drop_duplicates(subset=key_cols, keep="first").copy()

    # Final scores
    final_scores = _final_scores(
        bets=bets,
        games_csv_path=Path("data/raw/games.csv"),
        player_stats_csv=Path("data/raw/player_game_stats.csv"),
        allow_web_fetch=bool(args.allow_web_fetch),
    )

    # Overlay snapshot-observed FINAL scores (authoritative for that day; avoids needing web fetch).
    try:
        final_scores.update({int(k): (int(v[0]), int(v[1])) for k, v in final_scores_from_snaps.items()})
    except Exception:
        pass

    # Final per-period goal splits (PERIOD_*): prefer snapshot-derived period finals,
    # then optionally augment with web fetch.
    final_period_scores: dict[tuple[int, int], tuple[int, int]] = dict(period_scores_from_snaps)
    try:
        period_mask = bets["market"].astype(str).str.upper().isin(["PERIOD_ML", "PERIOD_TOTAL"])
        if bool(args.allow_web_fetch) and bool(period_mask.any()):
            pks = set(bets.loc[period_mask, "gamePk"].dropna().astype(int).tolist())
            # Keep timeout modest; this script is for backtesting and should finish.
            web_scores = _final_period_goals_web(pks, timeout_sec=10.0)
            # Web data is authoritative; let it overwrite snapshot-derived values.
            final_period_scores.update(web_scores)
    except Exception:
        pass

    # Preload player stats for priced props we can settle
    prop_mask = bets["market"].astype(str).str.upper().str.startswith("PROP_")
    prop_bets = bets[prop_mask].copy()
    player_stats = {}
    if not prop_bets.empty:
        game_pks = set(prop_bets["gamePk"].dropna().astype(int).tolist())
        pkeys = set([str(x) for x in prop_bets["player_key"].dropna().astype(str).tolist() if str(x).strip()])
        player_stats = _final_player_stats(Path("data/raw/player_game_stats.csv"), game_pks=game_pks, player_keys=pkeys)

    settled_rows = []
    for _, r in bets.iterrows():
        market = str(r.get("market") or "").upper()
        if market.startswith("PROP_"):
            settled_rows.append(_settle_prop_row(r, player_stats))
        else:
            settled_rows.append(_settle_row(r, final_scores, final_period_scores=final_period_scores))
    settled_df = pd.concat([bets.reset_index(drop=True), pd.DataFrame(settled_rows)], axis=1)

    # Summary
    _summarize(settled_df, title="ALL")
    markets = sorted(set(settled_df["market"].dropna().astype(str).tolist()))
    for market in markets:
        _summarize(settled_df[settled_df["market"] == market].copy(), title=f"market={market}")

    if args.breakdowns:
        _summarize_grouped(settled_df, group_col="edge_bucket", title="By edge bucket")
        _summarize_grouped(settled_df, group_col="elapsed_bucket", title="By elapsed bucket")
        _summarize_grouped(settled_df, group_col="odds_staleness_bucket", title="By odds staleness")
        # driver tags: explode list into individual rows
        try:
            tag_rows = settled_df.copy()
            tag_rows["driver_tag"] = tag_rows["driver_tags"]
            tag_rows = tag_rows.explode("driver_tag")
            tag_rows["driver_tag"] = tag_rows["driver_tag"].fillna("(none)")
            _summarize_grouped(tag_rows, group_col="driver_tag", title="By driver_tag", top_n=20)
        except Exception:
            pass

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suf = out_path.suffix.lower()
        if suf in {".jsonl", ".ndjson"}:
            # Preserve nested fields (driver_tags/driver_meta) without CSV stringification.
            with out_path.open("w", encoding="utf-8") as f:
                for rec in settled_df.to_dict(orient="records"):
                    f.write(json.dumps(_json_sanitize(rec), ensure_ascii=False) + "\n")
        else:
            settled_df.to_csv(out_path, index=False)
        print(f"wrote={out_path} rows={len(settled_df)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

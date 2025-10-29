from __future__ import annotations
"""Roster and usage inference utilities (draft).

This module provides functions to:
- Fetch current team rosters
- Gather recent game TOI splits
- Infer line slots (L1..L4, D1..D3) and special teams units (PP1/PP2, PK1/PK2)
- Produce a projected TOI snapshot DataFrame suitable for prop modeling feature inputs.

NOTE: Initial implementation uses NHL Stats API boxscore endpoints and
heuristics over recent games. Future iterations may integrate shift-level
data and external confirmed line sources.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import time

import pandas as pd
import requests

# Prefer primary Stats API host but keep alternates to survive DNS hiccups
STATS_BASES = [
    # Use the canonical Stats API host only; do not fall back to the deprecated statsapi.nhl.com
    "https://statsapi.web.nhl.com/api/v1",
]
RATE_LIMIT_SLEEP = 0.35  # ~3 req/sec safety
RECENT_GAMES = 10
MIN_GAMES_FOR_CONFIDENCE = 3


@dataclass
class RosterPlayer:
    player_id: int
    full_name: str
    position: str  # F, D, G (normalize later)
    team_id: int


def _get(path: str, params: Optional[Dict] = None, retries: int = 3) -> Dict:
    """GET helper with multi-base fallback and simple backoff.

    Tries multiple Stats API base URLs to avoid transient DNS/host issues.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        for base in STATS_BASES:
            try:
                time.sleep(RATE_LIMIT_SLEEP)
                r = requests.get(f"{base}{path}", params=params, timeout=12)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                continue
        # after cycling bases, back off and retry
        time.sleep(min(5.0, RATE_LIMIT_SLEEP * (2 ** attempt)))
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown request error")


def list_teams() -> List[Dict]:
    return _get("/teams")["teams"]


def fetch_current_roster(team_id: int) -> List[RosterPlayer]:
    data = _get(f"/teams/{team_id}", params={"expand": "team.roster"})
    roster_items = data["teams"][0]["roster"]["roster"]
    players: List[RosterPlayer] = []
    for p in roster_items:
        person = p["person"]
        pos = p["position"]["type"]  # e.g., Forward, Defenseman, Goalie
        if pos.startswith("Forw"):  # Forward
            norm = "F"
        elif pos.startswith("Def"):  # Defenseman
            norm = "D"
        elif pos.startswith("Goal"):  # Goalie
            norm = "G"
        else:
            norm = pos[:1]
        players.append(
            RosterPlayer(
                player_id=int(person["id"]),
                full_name=person["fullName"],
                position=norm,
                team_id=team_id,
            )
        )
    return players


def team_recent_game_pks(team_id: int, n: int = RECENT_GAMES) -> List[int]:
    # Fetch schedule backwards day by day until we have n final games
    # (Simpler initial approach; can be optimized using season ranges.)
    games: List[int] = []
    day_offset = 0
    while len(games) < n and day_offset < 120:  # safeguard
        # Look back one day
        from datetime import datetime, timedelta
        target_date = datetime.utcnow() - timedelta(days=day_offset)
        date_str = target_date.strftime("%Y-%m-%d")
        sched = _get("/schedule", params={"date": date_str, "teamId": team_id})
        for d in sched.get("dates", []):
            for g in d.get("games", []):
                status = g.get("status", {}).get("detailedState", "")
                if "Final" in status:
                    games.append(int(g["gamePk"]))
        day_offset += 1
    # maintain chronological order oldest->newest
    games = sorted(set(games))[-n:]
    return games


def fetch_boxscore(game_pk: int) -> Dict:
    return _get(f"/game/{game_pk}/boxscore")


def build_usage_frame(team_id: int, game_pks: List[int]) -> pd.DataFrame:
    rows: List[Dict] = []
    for gpk in game_pks:
        box = fetch_boxscore(gpk)
        teams = box.get("teams", {})
        # Determine if our team is home or away
        side = None
        for s in ("home", "away"):
            if int(teams.get(s, {}).get("team", {}).get("id", -1)) == team_id:
                side = s
                break
        if side is None:
            continue
        players = teams.get(side, {}).get("players", {})
        for pid_key, pdata in players.items():
            person = pdata.get("person", {})
            stats = pdata.get("stats", {})
            skater_stats = stats.get("skaterStats", {})
            goalie_stats = stats.get("goalieStats", {})
            # Parse TOI as mm:ss -> minutes float
            def toi_min(val: str | None) -> float:
                if not val or ":" not in val:
                    return 0.0
                mm, ss = val.split(":")
                try:
                    return int(mm) + int(ss) / 60.0
                except ValueError:
                    return 0.0
            position = pdata.get("position", {}).get("abbreviation", "")
            is_goalie = position == "G"
            row = {
                "player_id": int(person.get("id", -1)),
                "full_name": person.get("fullName"),
                "position": "G" if is_goalie else ("D" if position == "D" else "F"),
                "game_pk": gpk,
                "toi": toi_min(goalie_stats.get("timeOnIce")) if is_goalie else toi_min(skater_stats.get("timeOnIce")),
                "toi_pp": 0.0,
                "toi_sh": 0.0,
            }
            if not is_goalie:
                row["toi_pp"] = toi_min(skater_stats.get("powerPlayTimeOnIce"))
                row["toi_sh"] = toi_min(skater_stats.get("shortHandedTimeOnIce"))
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["player_id", "full_name", "position", "game_pk", "toi", "toi_pp", "toi_sh"])
    df = pd.DataFrame(rows)
    # Aggregate per player across games
    agg = df.groupby(["player_id", "full_name", "position"], as_index=False).agg(
        games_played=("game_pk", "nunique"),
        toi_total=("toi", "sum"),
        toi_pp_total=("toi_pp", "sum"),
        toi_sh_total=("toi_sh", "sum"),
    )
    agg["toi_avg"] = agg["toi_total"] / agg["games_played"].clip(lower=1)
    agg["toi_pp_avg"] = agg["toi_pp_total"] / agg["games_played"].clip(lower=1)
    agg["toi_sh_avg"] = agg["toi_sh_total"] / agg["games_played"].clip(lower=1)
    return agg


def infer_lines(usage_df: pd.DataFrame) -> pd.DataFrame:
    if usage_df.empty:
        usage_df["line_slot"] = None
        usage_df["pp_unit"] = None
        usage_df["pk_unit"] = None
        return usage_df
    df = usage_df.copy()
    # Forwards ranking by EV proxy = toi_avg (goalies excluded, defense separate)
    fw = df[df["position"] == "F"].sort_values("toi_avg", ascending=False).reset_index(drop=True)
    line_slots = []
    for idx, _ in fw.iterrows():
        if idx < 3:
            line_slots.append("L1")
        elif idx < 6:
            line_slots.append("L2")
        elif idx < 9:
            line_slots.append("L3")
        elif idx < 12:
            line_slots.append("L4")
        else:
            line_slots.append(None)
    fw["line_slot"] = line_slots
    dmen = df[df["position"] == "D"].sort_values("toi_avg", ascending=False).reset_index(drop=True)
    d_slots = []
    for idx, _ in dmen.iterrows():
        if idx < 2:
            d_slots.append("D1")
        elif idx < 4:
            d_slots.append("D2")
        elif idx < 6:
            d_slots.append("D3")
        else:
            d_slots.append(None)
    dmen["line_slot"] = d_slots
    goalies = df[df["position"] == "G"].copy()
    goalies["line_slot"] = None
    merged = pd.concat([fw, dmen, goalies], ignore_index=True)
    # PP Units from pp avg
    fw_pp = merged[merged["position"] != "G"].sort_values("toi_pp_avg", ascending=False)
    pp1_ids = fw_pp.head(5)["player_id"].tolist()
    pp2_ids = fw_pp.iloc[5:10]["player_id"].tolist()
    merged["pp_unit"] = merged["player_id"].apply(lambda pid: 1 if pid in pp1_ids else (2 if pid in pp2_ids else None))
    # PK Units from sh avg (top 4 + next 4)
    fw_pk = merged[merged["position"] != "G"].sort_values("toi_sh_avg", ascending=False)
    pk1_ids = fw_pk.head(4)["player_id"].tolist()
    pk2_ids = fw_pk.iloc[4:8]["player_id"].tolist()
    merged["pk_unit"] = merged["player_id"].apply(lambda pid: 1 if pid in pk1_ids else (2 if pid in pk2_ids else None))
    return merged


def project_toi(lines_df: pd.DataFrame) -> pd.DataFrame:
    if lines_df.empty:
        lines_df["proj_toi"] = []
        return lines_df
    df = lines_df.copy()
    # Simple projection = recent avg; future: blend + adjustments
    df["proj_toi"] = df["toi_avg"].fillna(0.0)
    df["proj_toi_pp"] = df["toi_pp_avg"].fillna(0.0)
    df["proj_toi_sh"] = df["toi_sh_avg"].fillna(0.0)
    # Goalie starter heuristic: highest toi_avg gets starter flag
    goalies = df[df["position"] == "G"].sort_values("toi_avg", ascending=False)
    starter_id = goalies.head(1)["player_id"].tolist()[0] if not goalies.empty else None
    df["starter_goalie"] = df["player_id"].eq(starter_id)
    df["projected"] = True
    return df


def build_roster_snapshot(team_id: int) -> pd.DataFrame:
    roster = fetch_current_roster(team_id)
    game_pks = team_recent_game_pks(team_id, n=RECENT_GAMES)
    usage = build_usage_frame(team_id, game_pks)
    lines = infer_lines(usage)
    proj = project_toi(lines)
    # Ensure all roster players included (even with 0 games)
    base = pd.DataFrame([r.__dict__ for r in roster])
    merged = base.merge(proj, on=["player_id", "full_name", "position"], how="left")
    cols = [
        "player_id","full_name","position","team_id","games_played","toi_avg","toi_pp_avg","toi_sh_avg","line_slot","pp_unit","pk_unit","proj_toi","proj_toi_pp","proj_toi_sh","starter_goalie","projected"
    ]
    for c in cols:
        if c not in merged.columns:
            merged[c] = None
    return merged[cols]


def build_all_team_roster_snapshots() -> pd.DataFrame:
    teams = list_teams()
    frames = []
    for t in teams:
        tid = int(t.get("id"))
        try:
            frames.append(build_roster_snapshot(tid))
        except Exception as e:
            print(f"[WARN] team {tid} roster snapshot failed: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "RosterPlayer",
    "list_teams",
    "fetch_current_roster",
    "team_recent_game_pks",
    "build_usage_frame",
    "infer_lines",
    "project_toi",
    "build_roster_snapshot",
    "build_all_team_roster_snapshots",
]

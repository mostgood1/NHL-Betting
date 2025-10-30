from __future__ import annotations
"""Roster and usage inference utilities.

Primary behavior for daily props workflows: build a lightweight, reliable roster
snapshot using the NHL Web API (api-web.nhle.com) only. This avoids any dependency
on the deprecated/unreliable Stats API hosts.

Functions retained for future usage inference (TOI, lines) remain, but the default
snapshot builder no longer calls Stats API endpoints.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import time

import pandas as pd
import requests

# Web API base (canonical)
WEB_BASE = "https://api-web.nhle.com/v1"
RATE_LIMIT_SLEEP = 0.35  # ~3 req/sec safety for any HTTP calls here
RECENT_GAMES = 10
MIN_GAMES_FOR_CONFIDENCE = 3

# Team abbreviations for the current league membership; used for roster fetch loop.
# Mirrors scripts/fetch_roster_snapshot.py to ensure consistency and zero external dependencies.
TEAM_ABBRS: list[str] = [
    "ANA","ARI","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET","EDM","FLA","LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SJS","SEA","STL","TBL","TOR","UTA","VAN","VGK","WPG","WSH"
]


def _current_season_code() -> int:
    """Return season code as integer, e.g., 20252026 for the 2025-26 season."""
    try:
        from datetime import datetime as _dt
        dt = _dt.utcnow()
        start_year = dt.year if dt.month >= 7 else (dt.year - 1)
        return int(f"{start_year}{start_year+1}")
    except Exception:
        # Fallback: conservative current year pairing
        from datetime import datetime as _dt
        y = _dt.utcnow().year
        try:
            return int(f"{y}{y+1}")
        except Exception:
            return 0


def _alias_team_abbr(abbr: str) -> str:
    """Handle relocations/renames for Web API lookups.

    - From 2025-26 season onward, ARI -> UTA (Arizona Coyotes relocated to Utah).
    """
    try:
        ab = str(abbr or "").upper()
        season_code = _current_season_code()
        if ab == "ARI" and season_code >= 20252026:
            return "UTA"
        return ab
    except Exception:
        return str(abbr or "").upper()


@dataclass
class RosterPlayer:
    player_id: int
    full_name: str
    position: str  # F, D, G (normalize later)
    team_id: int


def _get_web(path: str, params: Optional[Dict] = None, retries: int = 3) -> Dict:
    """GET helper against the NHL Web API with simple backoff."""
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            time.sleep(RATE_LIMIT_SLEEP)
            r = requests.get(f"{WEB_BASE}{path}", params=params, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(min(5.0, RATE_LIMIT_SLEEP * (2 ** attempt)))
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Unknown request error")


def list_teams() -> List[Dict]:
    """Return league teams using the NHL Web API.

    Attempts to call /v1/teams. If unavailable, falls back to TEAM_ABBRS with minimal info.
    """
    try:
        data = _get_web("/teams")
        teams = data.get("teams") if isinstance(data, dict) else None
        out: List[Dict] = []
        if isinstance(teams, list) and teams:
            for t in teams:
                try:
                    # Normalize common fields (best-effort)
                    tid = t.get("id") or t.get("teamId")
                    ab = t.get("abbrev") or t.get("abbreviation") or t.get("teamAbbrev")
                    name = t.get("name") or t.get("fullName") or t.get("commonName")
                    if isinstance(name, dict):
                        name = name.get("default") or next((v for v in name.values() if isinstance(v, str)), None)
                    out.append({"id": tid, "abbreviation": (ab or "").upper(), "name": name})
                except Exception:
                    continue
        if out:
            return out
    except Exception:
        pass
    # Fallback: minimal team dicts from abbreviations only
    return [{"id": None, "abbreviation": ab, "name": ab} for ab in TEAM_ABBRS]


def fetch_current_roster(team_abbr: str) -> List[RosterPlayer]:
    """Fetch current active roster using the NHL Web API for a team abbreviation.

    Endpoint: /v1/roster/{TEAM_ABBR}/current
    """
    ab = _alias_team_abbr(team_abbr)
    blob = _get_web(f"/roster/{ab}/current")
    # Collect any list-valued groups (forwards, defensemen, goalies, etc.)
    groups = [k for k, v in (blob or {}).items() if isinstance(v, list)]
    players: List[RosterPlayer] = []
    for g in groups:
        for p in (blob.get(g) or []):
            try:
                pid = p.get("playerId") or p.get("id")
                fn = p.get("firstName"); ln = p.get("lastName")
                first = fn.get("default") if isinstance(fn, dict) else fn
                last = ln.get("default") if isinstance(ln, dict) else ln
                full = (f"{first} {last}" if first and last else (first or last or "")).strip()
                # Normalize position from group name
                pos = None
                gl = g.lower()
                if 'forward' in gl:
                    pos = 'F'
                elif 'defense' in gl or 'defenc' in gl:
                    pos = 'D'
                elif 'goalie' in gl or 'goaltender' in gl:
                    pos = 'G'
                if pid and full:
                    players.append(RosterPlayer(player_id=int(pid), full_name=full, position=pos or '', team_id=None))
            except Exception:
                continue
    return players


def team_recent_game_pks(team_id: int, n: int = RECENT_GAMES) -> List[int]:
    """Deprecated: Stats API schedule lookup. Return empty list to avoid Stats API calls.

    Callers needing recent games should migrate to nhl_api_web.NHLWebClient().
    """
    return []


def fetch_boxscore(game_pk: int) -> Dict:
    # Deprecated Stats API path; use nhl_api_web.NHLWebClient().boxscore instead.
    return {}


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


def build_roster_snapshot(team_abbr: str) -> pd.DataFrame:
    """Build a minimal roster snapshot for a single team (Web API only).

    Returns columns: full_name, player_id, team (abbr), position.
    """
    players = fetch_current_roster(team_abbr)
    if not players:
        return pd.DataFrame(columns=["full_name","player_id","team","position","team_id"])
    rows = [{"full_name": p.full_name, "player_id": p.player_id, "team": str(team_abbr).upper(), "position": p.position, "team_id": None} for p in players]
    return pd.DataFrame(rows)


def build_all_team_roster_snapshots() -> pd.DataFrame:
    """Build roster snapshot for all teams using Web API only.

    This avoids any calls to statsapi.web.nhl.com and returns a simple, reliable
    frame for name->player_id/team enrichment in props workflows.
    """
    # Build the working team list with aliasing for current season to avoid defunct teams
    season_code = _current_season_code()
    abbrs = []
    seen = set()
    for ab in TEAM_ABBRS:
        try:
            a = _alias_team_abbr(ab)
            if a not in seen:
                seen.add(a)
                abbrs.append(a)
        except Exception:
            continue
    frames: List[pd.DataFrame] = []
    for ab in abbrs:
        try:
            frames.append(build_roster_snapshot(ab))
        except Exception as e:
            print(f"[WARN] team {ab} roster snapshot failed: {e}")
            continue
    if not frames:
        return pd.DataFrame(columns=["full_name","player_id","team","position","team_id"])
    # Ensure team_id column exists for downstream compatibility (may be None)
    out = pd.concat(frames, ignore_index=True)
    if 'team_id' not in out.columns:
        out['team_id'] = None
    return out


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

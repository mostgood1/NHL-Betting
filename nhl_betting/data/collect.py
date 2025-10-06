from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .nhl_api import NHLClient
from .nhl_api_web import NHLWebClient
from ..utils.io import RAW_DIR, save_df


def _parse_boxscore_players(box: Dict, gamePk: int, gameDate: str, home: str, away: str) -> List[Dict]:
    """Parse players from either Stats API or NHL Web API boxscore payloads.

    Stats API shape:
      box["teams"]["home"]["players"] -> {"ID": {"person": {..}, "stats": {skaterStats|goalieStats}}}

    NHL Web API shape:
      box["boxscore"]["teams"]["home"] has skaters and goalies lists with different field names.
    """
    rows: List[Dict] = []
    if "teams" in box and isinstance(box.get("teams"), dict):
        # Stats API format
        for side_key, team_name in [("home", home), ("away", away)]:
            players = box.get("teams", {}).get(side_key, {}).get("players", {})
            for pid, pdata in players.items():
                info = pdata.get("person", {})
                stats = pdata.get("stats", {})
                skater = stats.get("skaterStats")
                goalie = stats.get("goalieStats")
                base = {
                    "gamePk": gamePk,
                    "date": gameDate,
                    "team": team_name,
                    "player_id": info.get("id"),
                    "player": info.get("fullName"),
                    "primary_position": pdata.get("position", {}).get("abbreviation"),
                }
                if skater:
                    rows.append({
                        **base,
                        "role": "skater",
                        "shots": _safe_int(skater.get("shots")),
                        "goals": _safe_int(skater.get("goals")),
                        "assists": _safe_int(skater.get("assists")),
                        "blocked": _safe_int(skater.get("blocked")),
                        "timeOnIce": skater.get("timeOnIce"),
                    })
                if goalie:
                    rows.append({
                        **base,
                        "role": "goalie",
                        "saves": _safe_int(goalie.get("saves")),
                        "shotsAgainst": _safe_int(goalie.get("shots")),
                        "decision": goalie.get("decision"),
                        "timeOnIce": goalie.get("timeOnIce"),
                    })
        return rows

    # Try NHL Web API format
    try:
        # NHL Web API boxscore can be under top-level or nested 'boxscore'
        root = box.get("boxscore") or box
        # Preferred: playerByGameStats at root with 'homeTeam'/'awayTeam'
        pbg_root = root.get("playerByGameStats") if isinstance(root, dict) else None
        for side_key, team_name in [("home", home), ("away", away)]:
            skaters = []
            goalies = []
            if isinstance(pbg_root, dict):
                team_block = pbg_root.get("homeTeam" if side_key == "home" else "awayTeam") or {}
                # Skaters are forwards + defense
                fwd = team_block.get("forwards") or []
                dfd = team_block.get("defense") or []
                skaters = list(fwd) + list(dfd)
                goalies = team_block.get("goalies") or []
            else:
                # Fallbacks: Team containers: sometimes root["teams"]["home"/"away"], sometimes separate keys
                teams = root.get("teams") or {}
                if not teams:
                    teams = {
                        "home": root.get("homeTeam") or root.get("home") or {},
                        "away": root.get("awayTeam") or root.get("away") or {},
                    }
                t = teams.get(side_key) or {}
                # Players may be nested under these keys
                pbg = t.get("playerByGameStats") or {}
                if isinstance(pbg, dict):
                    skaters = list(pbg.get("skaters") or [])
                    goalies = list(pbg.get("goalies") or [])
                if not skaters:
                    skaters = t.get("skaters") or t.get("players") or []
                if not goalies:
                    goalies = t.get("goalies") or []
            # Skaters
            for p in skaters:
                # player object may be nested or flat depending on API version
                info = p.get("player") or p.get("playerId") or p
                # stats keys vary: skaterStats, stats, or flat
                stats = p.get("skaterStats") or p.get("stats") or p
                base = {
                    "gamePk": gamePk,
                    "date": gameDate,
                    "team": team_name,
                    "player_id": (info.get("id") if isinstance(info, dict) else info) or p.get("playerId"),
                    "player": (info.get("fullName") if isinstance(info, dict) else None) or 
                               (p.get("name", {}) if isinstance(p.get("name"), dict) else p.get("name")) or 
                               (p.get("name", {}) or {}).get("default") if isinstance(p.get("name"), dict) else p.get("firstInitialLastName"),
                    "primary_position": (p.get("position") or {}).get("abbreviation") if isinstance(p.get("position"), dict) else p.get("position"),
                }
                rows.append({
                    **base,
                    "role": "skater",
                    "shots": _safe_int(stats.get("shots") or stats.get("sog") or p.get("sog") or p.get("shots")),
                    "goals": _safe_int(stats.get("goals") or p.get("goals")),
                    "assists": _safe_int(stats.get("assists") or p.get("assists")),
                    "blocked": _safe_int(stats.get("blocked") or stats.get("blockedShots") or p.get("blocked") or p.get("blockedShots")),
                    "timeOnIce": stats.get("toi") or stats.get("timeOnIce") or p.get("toi"),
                })
            # Goalies
            for p in goalies:
                info = p.get("player") or p.get("playerId") or p
                stats = p.get("goalieStats") or p.get("stats") or p
                base = {
                    "gamePk": gamePk,
                    "date": gameDate,
                    "team": team_name,
                    "player_id": (info.get("id") if isinstance(info, dict) else info) or p.get("playerId"),
                    "player": (info.get("fullName") if isinstance(info, dict) else None) or 
                               (p.get("name", {}) if isinstance(p.get("name"), dict) else p.get("name")) or 
                               (p.get("name", {}) or {}).get("default") if isinstance(p.get("name"), dict) else p.get("firstInitialLastName"),
                    "primary_position": (p.get("position") or {}).get("abbreviation") if isinstance(p.get("position"), dict) else p.get("position"),
                }
                rows.append({
                    **base,
                    "role": "goalie",
                    "saves": _safe_int(stats.get("saves") or p.get("saves")),
                    "shotsAgainst": _safe_int(stats.get("shotsAgainst") or stats.get("shots") or p.get("shotsAgainst") or p.get("shots")),
                    "decision": stats.get("decision") or p.get("decision"),
                    "timeOnIce": stats.get("toi") or stats.get("timeOnIce") or p.get("toi"),
                })
    except Exception:
        # If structure unexpected, return empty and let caller decide
        return rows
    return rows


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _nhlpy_boxscore(game_id: int) -> Dict:
    try:
        from nhlpy import NHLClient as PyClient  # type: ignore
    except Exception as e:
        raise RuntimeError("nhl-api-py not installed. Run `pip install nhl-api-py`." ) from e
    c = PyClient()
    # nhl-api-py: game_center.boxscore expects nhl-style game_id string, the adapter should work with int
    return c.game_center.boxscore(game_id=str(game_id))


def collect_player_game_stats(start: str, end: str, source: str = "stats") -> pd.DataFrame:
    """
    Collect player skater shots/goals and goalie saves from boxscores for completed games.
    Saves to data/raw/player_game_stats.csv
    """
    source = (source or "stats").lower()
    if source == "web":
        client = NHLWebClient()
        games = client.schedule_range(start, end)
        boxscore_via = "web"  # use NHL Web API boxscore
    elif source == "stats":
        client = NHLClient()
        games = client.schedule(start, end)
        boxscore_via = "stats"
    elif source == "nhlpy":
        try:
            from nhlpy import NHLClient as PyClient  # type: ignore
        except Exception as e:
            raise RuntimeError("nhl-api-py not installed. Run `pip install nhl-api-py`." ) from e
        pyc = PyClient()
        # Build games list by day like our other clients
        from datetime import datetime, timedelta
        def to_dt(s: str):
            return datetime.fromisoformat(s)
        def fmt(dt):
            return dt.strftime("%Y-%m-%d")
        games = []
        cur = to_dt(start)
        end_dt = to_dt(end)
        while cur <= end_dt:
            day = pyc.schedule.daily_schedule(date=fmt(cur))
            for g in day.get("games", []):
                gid = int(g.get("id") or g.get("gameId") or g.get("gamePk") or 0)
                start_utc = g.get("startTimeUTC") or g.get("gameDate") or (fmt(cur) + "T00:00:00Z")
                season = str(g.get("season") or "")
                game_type = str(g.get("gameType") or g.get("gameTypeId") or "")
                home = g.get("homeTeam", {}).get("name") or g.get("homeTeam", {}).get("abbrev")
                away = g.get("awayTeam", {}).get("name") or g.get("awayTeam", {}).get("abbrev")
                hs = g.get("homeTeam", {}).get("score")
                as_ = g.get("awayTeam", {}).get("score")
                state = g.get("gameState") or g.get("status", {}).get("state")
                if state and str(state).upper() in {"FINAL", "OFF", "FINAL_SCORE"}:
                    hg = int(hs) if hs is not None else None
                    ag = int(as_) if as_ is not None else None
                else:
                    hg = None
                    ag = None
                games.append(type("G", (), {"gamePk": gid, "gameDate": start_utc, "season": season, "gameType": game_type, "home": home, "away": away, "home_goals": hg, "away_goals": ag}))
            cur += timedelta(days=1)
        boxscore_via = "nhlpy"
    else:
        raise ValueError("Unknown source. Use one of: web, stats, nhlpy")
    rows: List[Dict] = []
    for g in games:
        if g.home_goals is None or g.away_goals is None:
            continue  # skip unfinished
        if boxscore_via == "stats":
            box = client.boxscore(g.gamePk)
        elif boxscore_via == "web":
            box = client.boxscore(g.gamePk)
        else:
            box = _nhlpy_boxscore(g.gamePk)
        rows.extend(_parse_boxscore_players(box, g.gamePk, g.gameDate, g.home, g.away))
    # Build DataFrame and merge with existing history; avoid wiping file when no rows
    cols = [
        "gamePk","date","team","player_id","player","primary_position","role",
        "shots","goals","assists","blocked","timeOnIce","saves","shotsAgainst","decision",
    ]
    df = pd.DataFrame(rows)
    out = RAW_DIR / "player_game_stats.csv"
    if df.empty:
        # If file exists, ensure it has a readable header; if not, repair by writing columns
        if out.exists():
            try:
                _ = pd.read_csv(out)
                return df  # readable; keep existing
            except Exception:
                empty_df = pd.DataFrame(columns=cols)
                save_df(empty_df, out)
                return empty_df
        # Otherwise, initialize an empty file with stable columns
        empty_df = pd.DataFrame(columns=cols)
        save_df(empty_df, out)
        return empty_df
    # Ensure expected columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[cols]
    # Merge with existing (dedupe by gamePk + player_id + role)
    if out.exists():
        try:
            existing = pd.read_csv(out)
            # Ensure same columns
            for c in cols:
                if c not in existing.columns:
                    existing[c] = pd.NA
            existing = existing[cols]
        except Exception:
            existing = pd.DataFrame(columns=cols)
        merged = pd.concat([existing, df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["gamePk","player_id","role"], keep="last")
        save_df(merged, out)
        return merged
    else:
        save_df(df[cols], out)
        return df[cols]

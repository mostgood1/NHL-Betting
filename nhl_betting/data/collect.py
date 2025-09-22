from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .nhl_api import NHLClient
from .nhl_api_web import NHLWebClient
from ..utils.io import RAW_DIR, save_df


def _parse_boxscore_players(box: Dict, gamePk: int, gameDate: str, home: str, away: str) -> List[Dict]:
    rows: List[Dict] = []
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
        boxscore_via = "nhlpy"  # web client lacks boxscore; try nhl-api-py if available
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
        else:
            box = _nhlpy_boxscore(g.gamePk)
        rows.extend(_parse_boxscore_players(box, g.gamePk, g.gameDate, g.home, g.away))
    df = pd.DataFrame(rows)
    out = RAW_DIR / "player_game_stats.csv"
    save_df(df, out)
    return df

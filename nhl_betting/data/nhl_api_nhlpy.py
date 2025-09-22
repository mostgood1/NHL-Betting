from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class Game:
    gamePk: int
    gameDate: str  # ISO UTC
    season: str
    gameType: str
    home: str
    away: str
    home_goals: Optional[int]
    away_goals: Optional[int]


def _team_name(team: Dict) -> str:
    # nhl-api-py returns abbrev and names in nested dicts depending on endpoint
    # We try a few keys to get a reasonable display name
    if not team:
        return ""
    name = (
        team.get("name")
        or team.get("name", {}).get("default")
        or team.get("teamName", {}).get("default")
        or team.get("abbrev")
        or ""
    )
    return str(name)


class NHLNhlPyClient:
    """
    Thin adapter around nhl-api-py (NHLClient) to fetch schedules.

    This is optional; import happens at runtime to avoid hard dependency.
    """

    def __init__(self, rate_limit_per_sec: float = 3.0):
        try:
            from nhlpy import NHLClient  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "nhl-api-py not installed. Run `pip install nhl-api-py` to use source='nhlpy'."
            ) from e
        self._client = NHLClient()
        self.sleep = 1.0 / rate_limit_per_sec

    def schedule_day(self, date: str) -> List[Game]:
        # nhl-api-py schedule.daily_schedule returns dict with 'games'
        time.sleep(self.sleep)
        data = self._client.schedule.daily_schedule(date=date)
        games: List[Game] = []
        for g in data.get("games", []) if isinstance(data, dict) else data:
            # Fields vary; normalize
            gid = int(g.get("id") or g.get("gameId") or g.get("gamePk") or 0)
            start_utc = g.get("startTimeUTC") or g.get("gameDate") or (date + "T00:00:00Z")
            season = str(g.get("season") or "")
            game_type = str(g.get("gameType") or g.get("gameTypeId") or "")
            home_team = g.get("homeTeam") or {}
            away_team = g.get("awayTeam") or {}
            home_name = _team_name(home_team)
            away_name = _team_name(away_team)
            # Scores if final
            state = (g.get("gameState") or g.get("status", {}).get("state"))
            hs = home_team.get("score") if isinstance(home_team, dict) else None
            as_ = away_team.get("score") if isinstance(away_team, dict) else None
            if state and str(state).upper() in {"FINAL", "OFF", "FINAL_SCORE"}:
                hg = int(hs) if hs is not None else None
                ag = int(as_) if as_ is not None else None
            else:
                hg = None
                ag = None
            games.append(
                Game(
                    gamePk=gid,
                    gameDate=start_utc,
                    season=season,
                    gameType=game_type,
                    home=home_name,
                    away=away_name,
                    home_goals=hg,
                    away_goals=ag,
                )
            )
        return games

    def schedule_range(self, start_date: str, end_date: str) -> List[Game]:
        def to_dt(s: str) -> datetime:
            return datetime.fromisoformat(s)

        def fmt(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%d")

        start_dt = to_dt(start_date)
        end_dt = to_dt(end_date)
        all_games: List[Game] = []
        cur = start_dt
        while cur <= end_dt:
            all_games.extend(self.schedule_day(fmt(cur)))
            cur += timedelta(days=1)
        return all_games

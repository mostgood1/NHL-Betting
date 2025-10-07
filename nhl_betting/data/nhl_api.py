from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from datetime import datetime, timedelta

import requests

# Prefer primary Stats API host but keep alternates to survive DNS hiccups
BASES = [
    "https://statsapi.web.nhl.com/api/v1",
    # historical/alternate DNS entries sometimes used
    "https://statsapi.nhl.com/api/v1",
]


@dataclass
class Game:
    gamePk: int
    gameDate: str
    season: str
    gameType: str
    home: str
    away: str
    home_goals: Optional[int]
    away_goals: Optional[int]


class NHLClient:
    def __init__(self, rate_limit_per_sec: float = 3.0):
        self.sleep = 1.0 / rate_limit_per_sec

    def _get(self, path: str, params: Optional[Dict] = None, retries: int = 3) -> Dict:
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            for base in BASES:
                try:
                    time.sleep(self.sleep)
                    # Use shorter per-request timeout to avoid long stalls
                    r = requests.get(f"{base}{path}", params=params, timeout=10)
                    r.raise_for_status()
                    return r.json()
                except Exception as e:
                    last_exc = e
                    # try next base or backoff then retry
                    continue
            # simple backoff after cycling bases
            time.sleep(min(5.0, self.sleep * (2 ** attempt)))
        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown request error")

    def teams(self) -> List[Dict]:
        return self._get("/teams")["teams"]

    def schedule(self, start_date: str, end_date: str) -> List[Game]:
        """Fetch schedule from Stats API; on DNS failures, fallback to NHL Web API per-day.

        Note: For long ranges, prefer schedule_range to avoid server-side limits.
        """
        try:
            data = self._get("/schedule", {"startDate": start_date, "endDate": end_date})
            games: List[Game] = []
            for d in data.get("dates", []):
                for g in d.get("games", []):
                    teams = g["teams"]
                    home = teams["home"]["team"]["name"]
                    away = teams["away"]["team"]["name"]
                    linescore = g.get("linescore", {})
                    status = (g.get("status", {}).get("detailedState", "") or "").lower()
                    is_final = ("final" in status) or ("game over" in status) or (status == "off")
                    if is_final:
                        # Prefer linescore goals if present, else fallback to teams.*.score
                        home_goals = (
                            linescore.get("teams", {}).get("home", {}).get("goals")
                            if isinstance(linescore, dict) else None
                        )
                        away_goals = (
                            linescore.get("teams", {}).get("away", {}).get("goals")
                            if isinstance(linescore, dict) else None
                        )
                        if home_goals is None:
                            home_goals = teams.get("home", {}).get("score")
                        if away_goals is None:
                            away_goals = teams.get("away", {}).get("score")
                    else:
                        home_goals = None
                        away_goals = None
                    games.append(
                        Game(
                            gamePk=g["gamePk"],
                            gameDate=g["gameDate"],
                            season=g.get("season", ""),
                            gameType=g.get("gameType", ""),
                            home=home,
                            away=away,
                            home_goals=home_goals,
                            away_goals=away_goals,
                        )
                    )
            return games
        except Exception:
            # Fallback: use NHL Web API day-by-day
            from .nhl_api_web import NHLWebClient
            web = NHLWebClient()
            return web.schedule_range(start_date, end_date)

    def schedule_range(self, start_date: str, end_date: str, step_days: int = 30) -> List[Game]:
        def to_dt(s: str) -> datetime:
            return datetime.fromisoformat(s)

        def fmt(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%d")

        start_dt = to_dt(start_date)
        end_dt = to_dt(end_date)
        cursor = start_dt
        all_games: List[Game] = []
        while cursor <= end_dt:
            chunk_start = cursor
            chunk_end = min(cursor + timedelta(days=step_days), end_dt)
            try:
                chunk_games = self.schedule(fmt(chunk_start), fmt(chunk_end))
            except Exception:
                # As a last resort, use web client for this chunk
                from .nhl_api_web import NHLWebClient
                chunk_games = NHLWebClient().schedule_range(fmt(chunk_start), fmt(chunk_end))
            all_games.extend(chunk_games)
            cursor = chunk_end + timedelta(days=1)
        return all_games

    def boxscore(self, gamePk: int) -> Dict:
        """Fetch boxscore from Stats API; fallback to NHL Web API structure on failure.

        Callers that expect Stats API structure should be prepared to handle the Web format
        when fallback occurs.
        """
        try:
            return self._get(f"/game/{gamePk}/boxscore")
        except Exception:
            # Fallback: NHL Web API boxscore
            from .nhl_api_web import NHLWebClient
            return NHLWebClient().boxscore(gamePk)

    def game_live_feed(self, gamePk: int) -> Dict:
        return self._get(f"/game/{gamePk}/feed/live")

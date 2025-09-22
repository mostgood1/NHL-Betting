from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from datetime import datetime, timedelta

import requests

BASE = "https://statsapi.web.nhl.com/api/v1"


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
            try:
                time.sleep(self.sleep)
                r = requests.get(f"{BASE}{path}", params=params, timeout=30)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_exc = e
                # simple backoff
                time.sleep(self.sleep * (2 ** attempt))
        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown request error")

    def teams(self) -> List[Dict]:
        return self._get("/teams")["teams"]

    def schedule(self, start_date: str, end_date: str) -> List[Game]:
        data = self._get("/schedule", {"startDate": start_date, "endDate": end_date})
        games: List[Game] = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                teams = g["teams"]
                home = teams["home"]["team"]["name"]
                away = teams["away"]["team"]["name"]
                linescore = g.get("linescore", {})
                status = g.get("status", {}).get("detailedState", "")
                if "Final" in status:
                    home_goals = linescore.get("teams", {}).get("home", {}).get("goals")
                    away_goals = linescore.get("teams", {}).get("away", {}).get("goals")
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
            chunk_games = self.schedule(fmt(chunk_start), fmt(chunk_end))
            all_games.extend(chunk_games)
            cursor = chunk_end + timedelta(days=1)
        return all_games

    def boxscore(self, gamePk: int) -> Dict:
        return self._get(f"/game/{gamePk}/boxscore")

    def game_live_feed(self, gamePk: int) -> Dict:
        return self._get(f"/game/{gamePk}/feed/live")

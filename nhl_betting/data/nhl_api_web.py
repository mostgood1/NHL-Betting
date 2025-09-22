from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

BASE = "https://api-web.nhle.com/v1"


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
    venue: Optional[str] = None
    gameState: Optional[str] = None


def _team_name(team: Dict) -> str:
    place = team.get("placeName", {}).get("default") or team.get("placeName") or ""
    common = team.get("commonName", {}).get("default") or team.get("commonName") or ""
    name = f"{place} {common}".strip()
    return " ".join(name.split())


class NHLWebClient:
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
                time.sleep(self.sleep * (2 ** attempt))
        if last_exc:
            raise last_exc
        raise RuntimeError("Unknown request error")

    def schedule_day(self, date: str) -> List[Game]:
        data = self._get(f"/schedule/{date}")
        games: List[Game] = []
        for wk in data.get("gameWeek", []):
            if wk.get("date") != date:
                continue
            for g in wk.get("games", []):
                home = _team_name(g.get("homeTeam", {}))
                away = _team_name(g.get("awayTeam", {}))
                # Scores present for finished games
                home_score = g.get("homeTeam", {}).get("score")
                away_score = g.get("awayTeam", {}).get("score")
                state = g.get("gameState")
                if state and state.upper() in {"FINAL", "OFF"}:
                    hg = int(home_score) if home_score is not None else None
                    ag = int(away_score) if away_score is not None else None
                else:
                    hg = None
                    ag = None
                start_utc = g.get("startTimeUTC") or (date + "T00:00:00Z")
                season = str(g.get("season", ""))
                game_type = str(g.get("gameType", ""))
                venue = None
                try:
                    v = g.get("venue")
                    if isinstance(v, dict):
                        venue = v.get("default") or v.get("name")
                    elif isinstance(v, str):
                        venue = v
                except Exception:
                    venue = None
                games.append(
                    Game(
                        gamePk=int(g["id"]),
                        gameDate=start_utc,
                        season=season,
                        gameType=game_type,
                        home=home,
                        away=away,
                        home_goals=hg,
                        away_goals=ag,
                        venue=venue,
                        gameState=g.get("gameState"),
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
            d = fmt(cur)
            all_games.extend(self.schedule_day(d))
            cur += timedelta(days=1)
        return all_games

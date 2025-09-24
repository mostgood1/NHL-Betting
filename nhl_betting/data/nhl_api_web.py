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

    def scoreboard_day(self, date: str) -> List[dict]:
        """Return lightweight live scoreboard info for a given date.

        Includes current score even if game not final, plus state/period/clock when available.
        """
        data = self._get(f"/schedule/{date}")
        rows: List[dict] = []
        for wk in data.get("gameWeek", []):
            if wk.get("date") != date:
                continue
            for g in wk.get("games", []):
                home = _team_name(g.get("homeTeam", {}))
                away = _team_name(g.get("awayTeam", {}))
                home_score = g.get("homeTeam", {}).get("score")
                away_score = g.get("awayTeam", {}).get("score")
                state = g.get("gameState")
                # Attempt to extract period and clock in a robust way
                period = None
                try:
                    pd = g.get("periodDescriptor") or {}
                    period = pd.get("number") or pd.get("period") or g.get("period")
                except Exception:
                    period = None
                clock = None
                try:
                    cl = g.get("clock")
                    if isinstance(cl, dict):
                        clock = cl.get("timeRemaining") or cl.get("timeRemainingInPeriod") or cl.get("displayValue")
                    elif isinstance(cl, str):
                        clock = cl
                except Exception:
                    clock = None
                start_utc = g.get("startTimeUTC") or (date + "T00:00:00Z")
                try:
                    game_pk = int(g.get("id"))
                except Exception:
                    game_pk = None
                rows.append({
                    "gamePk": game_pk,
                    "gameDate": start_utc,
                    "home": home,
                    "away": away,
                    "home_goals": int(home_score) if home_score is not None else None,
                    "away_goals": int(away_score) if away_score is not None else None,
                    "gameState": state,
                    "period": period,
                    "clock": clock,
                })
        return rows

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

    def linescore(self, gamePk: int) -> dict:
        """Fetch live linescore for a game: period and clock when available.

        Uses /gamecenter/{gamePk}/linescore endpoint of the NHL Web API.
        Returns a dict like {"period": <int|str|None>, "clock": <str|None>}.
        """
        period = None
        clock = None
        source = None
        # Primary endpoint
        try:
            data = self._get(f"/gamecenter/{int(gamePk)}/linescore")
            try:
                period = data.get("currentPeriod") or data.get("period")
            except Exception:
                period = None
            try:
                cl = data.get("clock")
                if isinstance(cl, dict):
                    clock = cl.get("timeRemaining") or cl.get("displayValue") or cl.get("timeRemainingInPeriod")
                elif isinstance(cl, str):
                    clock = cl
                if not clock:
                    pd = data.get("periodDescriptor") or {}
                    clock = pd.get("timeRemaining") or pd.get("displayValue")
            except Exception:
                clock = None
            if clock or period is not None:
                source = "linescore"
        except Exception:
            # swallow; we'll try fallbacks below
            pass

        def _extract_from(data_obj, src_name: str):
            nonlocal period, clock, source
            try:
                if period is None:
                    pd = data_obj.get("periodDescriptor") or {}
                    per = pd.get("number") or pd.get("period") or data_obj.get("currentPeriod") or data_obj.get("period")
                    if per is not None:
                        period = per
                if not clock:
                    # Various nesting patterns for clock/time
                    cl = data_obj.get("clock")
                    if isinstance(cl, dict):
                        cands = [cl.get("timeRemaining"), cl.get("timeRemainingInPeriod"), cl.get("displayValue")]
                        for c in cands:
                            if isinstance(c, str) and c:
                                clock = c; break
                    elif isinstance(cl, str) and cl:
                        clock = cl
                if (not clock) and "plays" in data_obj:
                    # play-by-play style: derive remaining from latest play if available
                    try:
                        plays = data_obj.get("plays") or []
                        if isinstance(plays, list) and plays:
                            last = plays[-1]
                            tr = last.get("timeRemaining") or last.get("timeInPeriod")
                            if isinstance(tr, str) and tr:
                                clock = tr
                    except Exception:
                        pass
                if clock and not source:
                    source = src_name
            except Exception:
                pass

        # Fallback endpoints only if clock missing
        if not clock:
            # play-by-play
            try:
                pbp = self._get(f"/gamecenter/{int(gamePk)}/play-by-play")
                _extract_from(pbp, "play-by-play")
            except Exception:
                pass
        if not clock:
            # boxscore
            try:
                box = self._get(f"/gamecenter/{int(gamePk)}/boxscore")
                _extract_from(box, "boxscore")
            except Exception:
                pass
        if not clock:
            # landing
            try:
                land = self._get(f"/gamecenter/{int(gamePk)}/landing")
                _extract_from(land, "landing")
            except Exception:
                pass

        return {"period": period, "clock": clock, "source": source}

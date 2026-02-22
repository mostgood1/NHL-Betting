from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Iterable, Optional, Tuple


@dataclass
class RestTravelInfo:
    team_abbr: str
    rest_days: int
    b2b: int
    travel_km: float
    last_game_date: Optional[str] = None
    last_game_loc_abbr: Optional[str] = None
    cur_game_loc_abbr: Optional[str] = None


# Home city (arena metro) lat/lon approximations for travel distance estimates.
# This is intentionally lightweight and static (no external geocoding dependency).
_TEAM_HOME_LATLON: Dict[str, Tuple[float, float]] = {
    "ANA": (33.807, -117.876),  # Anaheim
    "ARI": (33.445, -112.071),  # Phoenix
    "BOS": (42.366, -71.062),
    "BUF": (42.875, -78.876),
    "CAR": (35.803, -78.721),  # Raleigh
    "CBJ": (39.969, -83.007),  # Columbus
    "CGY": (51.037, -114.051),
    "CHI": (41.880, -87.674),
    "COL": (39.748, -105.007),  # Denver
    "DAL": (32.790, -96.810),
    "DET": (42.341, -83.055),
    "EDM": (53.546, -113.497),
    "FLA": (26.158, -80.325),  # Sunrise
    "LAK": (34.043, -118.267),  # Los Angeles
    "MIN": (44.944, -93.101),  # St Paul
    "MTL": (45.496, -73.569),
    "NJD": (40.733, -74.172),  # Newark
    "NSH": (36.159, -86.778),
    "NYI": (40.721, -73.590),  # Elmont
    "NYR": (40.750, -73.994),  # Manhattan
    "OTT": (45.297, -75.927),
    "PHI": (39.901, -75.172),
    "PIT": (40.439, -79.990),
    "SEA": (47.622, -122.354),
    "SJS": (37.332, -121.901),  # San Jose
    "STL": (38.626, -90.203),
    "TBL": (27.943, -82.451),  # Tampa
    "TOR": (43.643, -79.379),
    "UTA": (40.768, -111.900),  # Salt Lake City
    "VAN": (49.277, -123.108),
    "VGK": (36.102, -115.178),  # Las Vegas
    "WPG": (49.892, -97.144),
    "WSH": (38.898, -77.021),
}


def _to_et_date(iso_utc: str) -> Optional[_date]:
    """Convert an ISO UTC timestamp to an America/New_York calendar date."""
    if not iso_utc:
        return None
    s = str(iso_utc).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except Exception:
        try:
            dt = datetime.fromisoformat(str(iso_utc)[:19]).replace(tzinfo=timezone.utc)
        except Exception:
            return None
    try:
        return dt.astimezone(ZoneInfo("America/New_York")).date()
    except Exception:
        return dt.date()


def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    import math

    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return float(2 * r * math.asin(math.sqrt(x)))


def _travel_km(loc_from_abbr: Optional[str], loc_to_abbr: Optional[str]) -> float:
    if not (loc_from_abbr and loc_to_abbr):
        return 0.0
    a = _TEAM_HOME_LATLON.get(str(loc_from_abbr).upper())
    b = _TEAM_HOME_LATLON.get(str(loc_to_abbr).upper())
    if not (a and b):
        return 0.0
    try:
        return float(_haversine_km(a, b))
    except Exception:
        return 0.0


def compute_rest_travel_by_abbr(
    *,
    date_ymd: str,
    slate_abbrs: Iterable[str],
    slate_home_away: Dict[str, str],
    lookback_days: int = 14,
    client=None,
    abbr_from_team_name=None,
) -> Dict[str, RestTravelInfo]:
    """Compute rest + B2B + travel distance for teams on the slate.

    Parameters
    - date_ymd: target slate date (ET) YYYY-MM-DD
    - slate_abbrs: set/list of team abbreviations for the slate
    - slate_home_away: mapping team_abbr -> current game location abbr (the HOME team abbr of today's game)
      Example: if BOS @ TOR then BOS->TOR, TOR->TOR

    Notes
    - Uses schedule data from NHL Web API to find each team's last game prior to date_ymd.
    - Travel distance is estimated as great-circle distance between arena metros.
    """
    # Lazy imports to keep module lightweight
    from .nhl_api_web import NHLWebClient

    try:
        target = _date.fromisoformat(str(date_ymd))
    except Exception:
        return {}

    slate_abbrs_u = [str(x).strip().upper() for x in slate_abbrs if str(x).strip()]
    if not slate_abbrs_u:
        return {}

    if client is None:
        client = NHLWebClient()

    # Look back a short window; for longer breaks we simply treat rest as large.
    end = target - timedelta(days=1)
    start = target - timedelta(days=int(max(1, lookback_days)))

    # Fetch games in range and identify last appearance per team.
    try:
        games = client.schedule_range(start.isoformat(), end.isoformat()) if start <= end else []
    except Exception:
        games = []

    last_by_team: Dict[str, Tuple[_date, str]] = {}
    for g in games or []:
        try:
            gd = _to_et_date(getattr(g, "gameDate", None))
        except Exception:
            gd = None
        if gd is None:
            continue
        if gd >= target:
            continue

        try:
            home_nm = getattr(g, "home", None)
            away_nm = getattr(g, "away", None)
        except Exception:
            home_nm = None
            away_nm = None

        if not (home_nm and away_nm):
            continue

        try:
            home_ab = abbr_from_team_name(str(home_nm)) if abbr_from_team_name else None
        except Exception:
            home_ab = None
        try:
            away_ab = abbr_from_team_name(str(away_nm)) if abbr_from_team_name else None
        except Exception:
            away_ab = None

        if not (home_ab and away_ab):
            continue

        home_ab = str(home_ab).upper()
        away_ab = str(away_ab).upper()

        # The relevant travel location for both teams is the home team's city.
        game_loc_ab = home_ab

        for team_ab in (home_ab, away_ab):
            if team_ab not in slate_abbrs_u:
                continue
            prev = last_by_team.get(team_ab)
            if prev is None or gd > prev[0]:
                last_by_team[team_ab] = (gd, game_loc_ab)

    out: Dict[str, RestTravelInfo] = {}
    for team_ab in slate_abbrs_u:
        cur_loc = str(slate_home_away.get(team_ab) or "").strip().upper() or None
        last = last_by_team.get(team_ab)

        if last is None:
            # Unknown last game (break / missing schedule data). Treat as well-rested.
            out[team_ab] = RestTravelInfo(
                team_abbr=team_ab,
                rest_days=7,
                b2b=0,
                travel_km=0.0,
                last_game_date=None,
                last_game_loc_abbr=None,
                cur_game_loc_abbr=cur_loc,
            )
            continue

        last_date, last_loc = last
        rest_days = int(max(0, (target - last_date).days))
        b2b = 1 if rest_days <= 1 else 0
        travel_km = _travel_km(last_loc, cur_loc)
        out[team_ab] = RestTravelInfo(
            team_abbr=team_ab,
            rest_days=rest_days,
            b2b=b2b,
            travel_km=float(travel_km),
            last_game_date=last_date.isoformat(),
            last_game_loc_abbr=last_loc,
            cur_game_loc_abbr=cur_loc,
        )

    return out

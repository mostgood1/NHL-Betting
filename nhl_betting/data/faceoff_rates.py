from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from nhl_betting.web.teams import get_team_assets

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _season_from_date(date: str) -> str:
    dt = pd.to_datetime(date)
    y = dt.year
    # NHL season spans fall-spring; seasonId like 20252026
    start = y if dt.month >= 8 else y - 1
    return f"{start}{start + 1}"


def fetch_team_faceoff_rates(season: str, timeout: float = 15.0) -> Dict[str, Dict[str, float]]:
    """Fetch team-level faceoff win% from NHL stats API.

    Returns a mapping: {ABBR: {faceoff_win_pct, ev_faceoff_win_pct, pp_faceoff_win_pct, sh_faceoff_win_pct}}.

    Notes:
    - The NHL stats endpoint is keyed by `franchiseName` (not team abbrev), so we map via `get_team_assets`.
    - Uses `cayenneExp` filtering because some reports ignore `seasonId` in query params.
    """

    return fetch_team_faceoff_rates_asof(season=season, as_of_date=None, timeout=timeout)


def _team_faceoff_cache_path(season: str, as_of_date: Optional[str]) -> Path:
    if as_of_date:
        safe = str(as_of_date).strip()
        return PROC_DIR / f"team_faceoff_rates_{season}_to_{safe}.json"
    return PROC_DIR / f"team_faceoff_rates_{season}.json"


def fetch_team_faceoff_rates_asof(
    season: str,
    as_of_date: Optional[str] = None,
    timeout: float = 15.0,
) -> Dict[str, Dict[str, float]]:
    """Fetch team-level faceoff win% from NHL stats API.

    If `as_of_date` is provided (YYYY-MM-DD), attempts to restrict aggregation to games
    with gameDate <= as_of_date (to support no-leak backtests).
    """

    cay = f"seasonId={season} and gameTypeId=2"
    if as_of_date:
        # Best-effort: the stats API generally supports filtering on gameDate.
        # If unsupported, callers will fall back to season-level cache.
        cay = f"{cay} and gameDate<='{str(as_of_date).strip()}'"

    url = (
        "https://api.nhle.com/stats/rest/en/team/faceoffpercentages"
        f"?isAggregate=true&isGame=true&cayenneExp={requests.utils.quote(cay)}"
    )

    out: Dict[str, Dict[str, float]] = {}
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json() or {}
    rows = data.get("data") or []

    for row in rows:
        try:
            name = str(row.get("franchiseName") or "").strip()
            if not name:
                continue
            abbr = (get_team_assets(name).get("abbr") or "").upper()
            if not abbr:
                continue

            def _pct(key: str) -> Optional[float]:
                try:
                    val = row.get(key)
                    if val is None:
                        return None
                    x = float(val)
                    # Some fields may be represented as 0..100 in other feeds; infer by magnitude.
                    if x > 1.5:
                        x = x / 100.0
                    return max(0.0, min(1.0, x))
                except Exception:
                    return None

            faceoff_win_pct = _pct("faceoffWinPct")
            ev = _pct("evFaceoffPct")
            pp = _pct("ppFaceoffPct")
            sh = _pct("shFaceoffPct")

            out[abbr] = {
                "faceoff_win_pct": float(faceoff_win_pct) if faceoff_win_pct is not None else 0.5,
                "ev_faceoff_win_pct": float(ev) if ev is not None else None,
                "pp_faceoff_win_pct": float(pp) if pp is not None else None,
                "sh_faceoff_win_pct": float(sh) if sh is not None else None,
            }
        except Exception:
            continue

    return out


def save_team_faceoff_rates(stats: Dict[str, Dict[str, float]], season: Optional[str] = None) -> None:
    path = PROC_DIR / (f"team_faceoff_rates_{season}.json" if season else "team_faceoff_rates.json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def save_team_faceoff_rates_asof(stats: Dict[str, Dict[str, float]], season: str, as_of_date: Optional[str]) -> None:
    path = _team_faceoff_cache_path(season=season, as_of_date=as_of_date)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_team_faceoff_rates(date: Optional[str] = None) -> Optional[Dict[str, Dict[str, float]]]:
    """Load cached team faceoff rates; if missing, attempt fetch and cache."""

    season = _season_from_date(date) if date else None

    # Prefer season-specific cache
    if season:
        p = PROC_DIR / f"team_faceoff_rates_{season}.json"
        if p.exists() and getattr(p.stat(), "st_size", 0) > 0:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and obj:
                    return obj
            except Exception:
                pass

    # Generic cache
    p2 = PROC_DIR / "team_faceoff_rates.json"
    if p2.exists() and getattr(p2.stat(), "st_size", 0) > 0:
        try:
            obj = json.loads(p2.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) and obj else None
        except Exception:
            pass

    # Try fetch then save
    try:
        if season:
            stats = fetch_team_faceoff_rates(season)
            if stats:
                save_team_faceoff_rates(stats, season=season)
                return stats
    except Exception:
        return None

    return None


def load_team_faceoff_rates_asof(date: str, as_of_date: Optional[str] = None) -> Optional[Dict[str, Dict[str, float]]]:
    """Load cached team faceoff rates for the season, optionally as-of a date.

    - If `as_of_date` is omitted, defaults to `date` (same day).
    - Useful for backtests where you want to avoid using future games.
    """
    season = _season_from_date(date)
    as_of = (str(as_of_date).strip() if as_of_date else str(date).strip())
    p = _team_faceoff_cache_path(season=season, as_of_date=as_of)
    if p.exists() and getattr(p.stat(), "st_size", 0) > 0:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) and obj else None
        except Exception:
            pass
    # Fall back to season cache if present
    p2 = _team_faceoff_cache_path(season=season, as_of_date=None)
    if p2.exists() and getattr(p2.stat(), "st_size", 0) > 0:
        try:
            obj = json.loads(p2.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and obj:
                return obj
        except Exception:
            pass
    # Fetch & cache best-effort
    try:
        stats = fetch_team_faceoff_rates_asof(season=season, as_of_date=as_of)
        if stats:
            save_team_faceoff_rates_asof(stats, season=season, as_of_date=as_of)
            return stats
    except Exception:
        pass
    return None


__all__ = [
    "fetch_team_faceoff_rates",
    "fetch_team_faceoff_rates_asof",
    "load_team_faceoff_rates",
    "load_team_faceoff_rates_asof",
    "save_team_faceoff_rates",
    "save_team_faceoff_rates_asof",
]

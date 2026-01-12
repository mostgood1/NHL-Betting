from __future__ import annotations

import json
from typing import Dict, Optional

import pandas as pd
import requests

from ..paths import PROC_DIR


def _season_from_date(date: str) -> str:
    dt = pd.to_datetime(date)
    y = dt.year
    # NHL season spans fall-spring; seasonId like 20252026
    start = y if dt.month >= 8 else y - 1
    return f"{start}{start+1}"


def fetch_team_penalty_rates(season: str, timeout: float = 15.0) -> Dict[str, Dict[str, float]]:
    """Fetch team-level penalty committed/drawn rates per 60 from NHL stats API.

    Attempts multiple summary endpoints; returns a mapping: {ABBR: {committed_per60, drawn_per60}}.
    Values are best-effort; missing fields default to None and will be ignored by callers.
    """
    urls = [
        # Team summary often includes penalties per game
        f"https://api.nhle.com/stats/rest/en/team/summary?isAggregate=true&isGame=true&reportType=season&seasonId={season}&gameType=R",
        # Team penalties endpoint (some environments)
        f"https://api.nhle.com/stats/rest/en/team/penalties?isAggregate=true&isGame=true&reportType=season&seasonId={season}&gameType=R",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            data = r.json() or {}
            rows = data.get("data") or []
            for row in rows:
                abbr = (row.get("teamAbbrev") or row.get("abbrev") or "").upper()
                if not abbr:
                    continue
                committed = None
                drawn = None
                # Common fields seen across endpoints
                # penaltiesPerGame, penaltiesDrawnPerGame, penaltyMinutesPerGame
                try:
                    val = row.get("penaltiesPerGame")
                    committed = float(val) if val is not None else None
                except Exception:
                    committed = committed
                try:
                    val = row.get("penaltiesDrawnPerGame")
                    drawn = float(val) if val is not None else None
                except Exception:
                    drawn = drawn
                # Fallback: use penaltyMinutesPerGame / 2 (approx two-minute minors)
                try:
                    if committed is None:
                        pm = row.get("penaltyMinutesPerGame")
                        committed = float(pm) / 2.0 if pm is not None else None
                except Exception:
                    pass
                prev = out.get(abbr) or {}
                out[abbr] = {
                    "committed_per60": committed if committed is not None else prev.get("committed_per60"),
                    "drawn_per60": drawn if drawn is not None else prev.get("drawn_per60"),
                }
        except Exception:
            continue
    return out


def save_team_penalty_rates(stats: Dict[str, Dict[str, float]], season: Optional[str] = None) -> None:
    path = PROC_DIR / (f"team_penalty_rates_{season}.json" if season else "team_penalty_rates.json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_team_penalty_rates(date: Optional[str] = None) -> Optional[Dict[str, Dict[str, float]]]:
    """Load cached team penalty rates; if missing, attempt fetch and cache.

    Returns None if unavailable.
    """
    season = _season_from_date(date) if date else None
    # Prefer season-specific cache
    if season:
        p = PROC_DIR / f"team_penalty_rates_{season}.json"
        if p.exists() and getattr(p.stat(), "st_size", 0) > 0:
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    # Generic cache
    p2 = PROC_DIR / "team_penalty_rates.json"
    if p2.exists() and getattr(p2.stat(), "st_size", 0) > 0:
        try:
            return json.loads(p2.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Try fetch then save
    try:
        if season:
            stats = fetch_team_penalty_rates(season)
            if stats:
                save_team_penalty_rates(stats, season=season)
                return stats
    except Exception:
        return None
    return None

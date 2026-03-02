from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

PROC_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def _season_from_date(date: str) -> str:
    dt = pd.to_datetime(date)
    y = dt.year
    start = y if dt.month >= 8 else y - 1
    return f"{start}{start + 1}"


def _player_faceoff_cache_path(season: str, as_of_date: Optional[str]) -> Path:
    if as_of_date:
        safe = str(as_of_date).strip()
        return PROC_DIR / f"player_faceoff_rates_{season}_to_{safe}.json"
    return PROC_DIR / f"player_faceoff_rates_{season}.json"


def fetch_player_faceoff_rates_asof(
    season: str,
    as_of_date: Optional[str] = None,
    timeout: float = 20.0,
) -> Dict[str, Dict[str, float]]:
    """Fetch skater faceoff rates from NHL stats API.

    Returns mapping keyed by `player_id` (string) with fields:
    - faceoff_win_pct (0..1)
    - faceoffs_taken
    - faceoffs_won
    - team_abbr (best-effort)

    If `as_of_date` is provided (YYYY-MM-DD), attempts to restrict aggregation to games
    with gameDate <= as_of_date (no-leak backtests).
    """

    cay = f"seasonId={season} and gameTypeId=2"
    if as_of_date:
        cay = f"{cay} and gameDate<='{str(as_of_date).strip()}'"

    # Endpoint name is best-effort; if it changes, callers will fall back gracefully.
    url = (
        "https://api.nhle.com/stats/rest/en/skater/faceoffpercentages"
        f"?isAggregate=true&isGame=true&start=0&limit=5000&cayenneExp={requests.utils.quote(cay)}"
    )

    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    obj = r.json() or {}
    rows = obj.get("data") or []

    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        try:
            pid = row.get("playerId")
            if pid is None:
                continue
            pid_s = str(int(pid))

            won = row.get("faceoffsWon")
            lost = row.get("faceoffsLost")
            pct = row.get("faceoffWinPct")

            faceoffs_won = int(won) if won is not None else None
            faceoffs_lost = int(lost) if lost is not None else None
            faceoffs_taken = None
            if faceoffs_won is not None and faceoffs_lost is not None:
                faceoffs_taken = int(faceoffs_won + faceoffs_lost)

            faceoff_win_pct = None
            if pct is not None:
                try:
                    x = float(pct)
                    if x > 1.5:
                        x = x / 100.0
                    faceoff_win_pct = max(0.0, min(1.0, x))
                except Exception:
                    faceoff_win_pct = None
            if faceoff_win_pct is None and faceoffs_taken and faceoffs_taken > 0 and faceoffs_won is not None:
                faceoff_win_pct = max(0.0, min(1.0, float(faceoffs_won) / float(faceoffs_taken)))

            team_abbr = (
                row.get("teamAbbrev")
                or row.get("teamAbbrevDefault")
                or row.get("teamAbbrevTricode")
                or row.get("teamAbbrevAbridged")
                or ""
            )
            team_abbr = str(team_abbr).strip().upper()

            if faceoff_win_pct is None:
                continue

            out[pid_s] = {
                "faceoff_win_pct": float(faceoff_win_pct),
                "faceoffs_taken": float(faceoffs_taken) if faceoffs_taken is not None else 0.0,
                "faceoffs_won": float(faceoffs_won) if faceoffs_won is not None else 0.0,
                "team_abbr": team_abbr,
            }
        except Exception:
            continue

    return out


def save_player_faceoff_rates_asof(stats: Dict[str, Dict[str, float]], season: str, as_of_date: Optional[str]) -> None:
    path = _player_faceoff_cache_path(season=season, as_of_date=as_of_date)
    with path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_player_faceoff_rates_asof(date: str, as_of_date: Optional[str] = None) -> Optional[Dict[str, Dict[str, float]]]:
    season = _season_from_date(date)
    as_of = (str(as_of_date).strip() if as_of_date else str(date).strip())

    p = _player_faceoff_cache_path(season=season, as_of_date=as_of)
    if p.exists() and getattr(p.stat(), "st_size", 0) > 0:
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) and obj else None
        except Exception:
            pass

    p2 = _player_faceoff_cache_path(season=season, as_of_date=None)
    if p2.exists() and getattr(p2.stat(), "st_size", 0) > 0:
        try:
            obj = json.loads(p2.read_text(encoding="utf-8"))
            if isinstance(obj, dict) and obj:
                return obj
        except Exception:
            pass

    try:
        stats = fetch_player_faceoff_rates_asof(season=season, as_of_date=as_of)
        if stats:
            save_player_faceoff_rates_asof(stats, season=season, as_of_date=as_of)
            return stats
    except Exception:
        pass

    return None


__all__ = [
    "fetch_player_faceoff_rates_asof",
    "load_player_faceoff_rates_asof",
    "save_player_faceoff_rates_asof",
]

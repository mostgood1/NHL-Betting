from __future__ import annotations

import json
import os
from typing import Dict, Optional

import requests

from .paths import PROC_DIR


def _season_for_date(date_str: str) -> str:
    import pandas as pd
    d = pd.to_datetime(date_str)
    y = d.year
    # NHL season starts in Sep
    start_year = y if d.month >= 9 else (y - 1)
    return f"{start_year}{start_year+1}"


def fetch_team_special_teams(season: str, timeout: float = 15.0) -> Dict[str, Dict[str, float]]:
    """Fetch team-level PP% and PK% for the given season from NHL Stats REST.

    Tries common endpoints and extracts Abbrev, powerPlayPct, penaltyKillPct.
    Returns mapping {ABBR: {pp_pct: float, pk_pct: float}} with values in [0,1].
    """
    # Known Stats API endpoint (subject to change): team summary
    urls = [
        f"https://api.nhle.com/stats/rest/en/team/summary?isGame=true&season={season}",
        f"https://api.nhle.com/stats/rest/en/team?isAggregate=false&isGame=true&reportType=summary&season={season}",
    ]
    last_exc: Optional[Exception] = None
    for url in urls:
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            obj = r.json()
            rows = obj.get("data") or obj
            out: Dict[str, Dict[str, float]] = {}
            for row in rows or []:
                abbr = (row.get("teamAbbrev") or row.get("teamAbbrevDefault") or row.get("teamAbbrevAbridged") or row.get("teamAbbrevTricode") or row.get("teamName") or "").strip().upper()
                if not abbr:
                    continue
                pp = row.get("powerPlayPct")
                pk = row.get("penaltyKillPct")
                try:
                    pp_f = float(pp) / (100.0 if (pp and pp > 1.0) else 1.0)
                    pk_f = float(pk) / (100.0 if (pk and pk > 1.0) else 1.0)
                except Exception:
                    pp_f = None; pk_f = None
                if (pp_f is None) and ("ppPct" in row):
                    try:
                        pp_f = float(row["ppPct"]) / (100.0 if row["ppPct"] and row["ppPct"] > 1.0 else 1.0)
                    except Exception:
                        pass
                if (pk_f is None) and ("pkPct" in row):
                    try:
                        pk_f = float(row["pkPct"]) / (100.0 if row["pkPct"] and row["pkPct"] > 1.0 else 1.0)
                    except Exception:
                        pass
                if (pp_f is None) or (pk_f is None):
                    continue
                out[abbr] = {"pp_pct": float(max(0.0, min(1.0, pp_f))), "pk_pct": float(max(0.0, min(1.0, pk_f)))}
            if out:
                return out
        except Exception as e:
            last_exc = e
            continue
    if last_exc:
        raise last_exc
    return {}


def save_team_special_teams(stats: Dict[str, Dict[str, float]], season: Optional[str] = None) -> None:
    path = PROC_DIR / (f"team_special_teams_{season}.json" if season else "team_special_teams.json")
    path.write_text(json.dumps({"season": season, "teams": stats}, indent=2), encoding="utf-8")


def load_team_special_teams(date: Optional[str] = None) -> Optional[Dict[str, Dict[str, float]]]:
    """Load team special teams mapping, trying season file then generic file."""
    # Prefer explicit season file based on date
    if date:
        season = _season_for_date(date)
        p = PROC_DIR / f"team_special_teams_{season}.json"
        if p.exists() and getattr(p.stat(), "st_size", 0) > 0:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                return obj.get("teams")
            except Exception:
                pass
    # Fallback generic
    p2 = PROC_DIR / "team_special_teams.json"
    if p2.exists() and getattr(p2.stat(), "st_size", 0) > 0:
        try:
            obj = json.loads(p2.read_text(encoding="utf-8"))
            return obj.get("teams")
        except Exception:
            return None
    return None

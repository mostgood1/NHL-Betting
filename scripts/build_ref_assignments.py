import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"


def daterange(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d"); e = datetime.strptime(end, "%Y-%m-%d")
    if e < s: s, e = e, s
    out = []
    d = s
    while d <= e:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def nhl_get(url: str, timeout: float = 15.0) -> Optional[dict]:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_schedule(date: str) -> List[dict]:
    data = nhl_get(f"https://statsapi.web.nhl.com/api/v1/schedule?date={date}&expand=schedule.teams,schedule.linescore")
    games = []
    try:
        for d in (data.get("dates") or []):
            for g in (d.get("games") or []):
                games.append(g)
    except Exception:
        pass
    # Fallback to NHL Web API schedule if Stats API yields nothing
    if not games:
        try:
            from nhl_betting.data.nhl_api_web import NHLWebClient
            web = NHLWebClient()
            gs = web.schedule_day(date)
            for g in gs:
                games.append({
                    "gamePk": g.gamePk,
                    "gameDate": g.gameDate,
                    "teams": {
                        "home": {"team": {"abbreviation": None, "name": g.home}},
                        "away": {"team": {"abbreviation": None, "name": g.away}},
                    },
                })
        except Exception:
            pass
    return games


def get_live(game_pk: int) -> Optional[dict]:
    return nhl_get(f"https://statsapi.web.nhl.com/api/v1/game/{game_pk}/feed/live")


def extract_officials(live: dict) -> Tuple[List[str], List[str]]:
    """Return (referees, linesmen) names from live feed if present."""
    refs: List[str] = []
    lines: List[str] = []
    try:
        # Try in liveData.boxscore.officials
        offs = (((live or {}).get("liveData") or {}).get("boxscore") or {}).get("officials") or []
        for o in offs:
            t = (o.get("officialType") or o.get("type") or "").lower()
            n = (o.get("official") or {}).get("fullName") or o.get("fullName") or o.get("name")
            if not n: continue
            if "ref" in t:
                refs.append(str(n).strip())
            elif "lines" in t:
                lines.append(str(n).strip())
    except Exception:
        pass
    # Fallback: sometimes in gameData.officials
    try:
        offs2 = (((live or {}).get("gameData") or {}).get("officials") or [])
        for o in offs2:
            t = (o.get("officialType") or o.get("type") or "").lower()
            n = (o.get("official") or {}).get("fullName") or o.get("fullName") or o.get("name")
            if not n: continue
            if "ref" in t and str(n).strip() not in refs:
                refs.append(str(n).strip())
            elif "lines" in t and str(n).strip() not in lines:
                lines.append(str(n).strip())
    except Exception:
        pass
    return refs, lines


def count_penalties(live: dict) -> Tuple[int, float]:
    """Return (penalties_count, minutes_basis). minutes_basis defaults to 60 if unknown."""
    cnt = 0
    try:
        plays = (((live or {}).get("liveData") or {}).get("plays") or {}).get("allPlays") or []
        for p in plays:
            if ((p.get("result") or {}).get("eventTypeId") or "") == "PENALTY":
                cnt += 1
    except Exception:
        pass
    minutes = 60.0
    try:
        # If linescore has current game periods final, estimate minutes
        lsc = (((live or {}).get("liveData") or {}).get("linescore") or {})
        pnum = int(lsc.get("currentPeriod", 3) or 3)
        minutes = 20.0 * max(3, pnum)
    except Exception:
        pass
    return cnt, minutes


def main():
    ap = argparse.ArgumentParser(description="Build per-date referee assignment rates")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    dates = daterange(args.start, args.end)
    # Historical ref rates up to date-1
    ref_hist: Dict[str, Dict[str, float]] = {}  # name -> {games, pens, mins}
    for day in dates:
        games = get_schedule(day)
        if not games:
            print(f"[ref] no schedule games found for {day}")
        rows = []
        # Compute league baseline from history
        try:
            total_p60 = []
            for rh in ref_hist.values():
                if rh.get("mins", 0) > 0:
                    total_p60.append(60.0 * rh.get("pens", 0) / rh.get("mins", 1e-6))
            base_rate = float(pd.Series(total_p60).mean()) if total_p60 else None
        except Exception:
            base_rate = None
        for g in games:
            try:
                game_pk = int(g.get("gamePk"))
                live = get_live(game_pk)
                # Teams
                home = (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("abbreviation") or (((g.get("teams") or {}).get("home") or {}).get("team") or {}).get("name")
                away = (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("abbreviation") or (((g.get("teams") or {}).get("away") or {}).get("team") or {}).get("name")
                home = str(home).upper(); away = str(away).upper()
                refs, _lines = extract_officials(live or {})
                if not refs:
                    # officials sometimes absent pregame; log and still write baseline row
                    print(f"[ref] no officials for game {game_pk} {away}@{home} {day}; using baseline only")
                # Predicted calls per60: mean of available ref historical p60; fallback to league base
                ref_p60 = []
                for r in refs:
                    rh = ref_hist.get(r)
                    if rh and rh.get("mins", 0) > 0:
                        ref_p60.append(60.0 * rh.get("pens", 0) / rh.get("mins", 1e-6))
                if ref_p60:
                    rate = float(pd.Series(ref_p60).mean())
                else:
                    rate = float(base_rate) if base_rate else None
                if rate is None:
                    # if still None, use 9 as a neutral league average guess
                    rate = 9.0
                rows.append({
                    "date": day,
                    "home": home,
                    "away": away,
                    "refs": "; ".join(refs) if refs else None,
                    "calls_per60": float(rate),
                    "baseline": float(base_rate) if base_rate else None,
                })
                # Update history with this game's actual penalties for future dates
                pens, mins = count_penalties(live or {})
                for r in refs:
                    st = ref_hist.setdefault(r, {"games": 0, "pens": 0.0, "mins": 0.0})
                    st["games"] += 1
                    st["pens"] += float(pens)
                    st["mins"] += float(mins)
            except Exception:
                continue
        # write per-date CSV
        if rows:
            df = pd.DataFrame(rows)
            out_path = PROC_DIR / f"ref_assignments_{day}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"[ref] wrote {out_path} ({len(df)} games)")
        else:
            print(f"[ref] no rows to write for {day}")


if __name__ == "__main__":
    main()

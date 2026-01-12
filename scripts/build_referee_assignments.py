import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict

import duckdb
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
PBP_DIR = RAW_DIR / "nhl_pbp"

STATS_BASE = "https://statsapi.web.nhl.com/api/v1"


def daterange(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d"); e = datetime.strptime(end, "%Y-%m-%d")
    if e < s: s, e = e, s
    out = []
    d = s
    while d <= e:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def fetch_schedule(date: str) -> List[Dict]:
    url = f"{STATS_BASE}/schedule?date={date}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return []
        js = r.json()
        dates = js.get("dates") or []
        if not dates:
            return []
        return dates[0].get("games") or []
    except Exception:
        return []


def fetch_officials(game_pk: int) -> List[Dict]:
    url = f"{STATS_BASE}/game/{game_pk}/feed/live"
    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return []
        js = r.json()
        # Officials may be under gameData -> officials
        od = js.get("gameData", {}).get("officials") or []
        return od
    except Exception:
        return []


def league_penalties_per_game(pbp_paths: List[Path]) -> Optional[float]:
    try:
        con = duckdb.connect()
        first = None
        for p in sorted(pbp_paths):
            if p.exists() and p.stat().st_size > 100_000:
                first = p; break
        if not first:
            return None
        con.execute(
            f"""
            create or replace table events as
            select game_id::INT as game_id, event_type::VARCHAR as event_type, coalesce(penalty_minutes,0)::INT as mins
            from read_parquet('{str(first)}')
            where game_id is not null
            """
        )
        for p in sorted(pbp_paths):
            if p == first:
                continue
            try:
                if p.exists() and p.stat().st_size > 100_000:
                    con.execute(
                        f"""
                        insert into events
                        select game_id::INT as game_id, event_type::VARCHAR as event_type, coalesce(penalty_minutes,0)::INT as mins
                        from read_parquet('{str(p)}')
                        where game_id is not null
                        """
                    )
            except Exception:
                continue
        per_game = con.execute(
            """
            select game_id, count(*) as n
            from events
            where event_type='PENALTY' and mins>=2
            group by 1
            """
        ).fetchdf()
        if per_game.empty:
            return None
        avg = float(per_game["n"].mean())
        return avg
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Build per-date referee assignments with baseline penalty rate")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    pbp_paths = list(PBP_DIR.glob("pbp_*.parquet"))
    base_rate = league_penalties_per_game(pbp_paths) or 0.0

    for day in daterange(args.start, args.end):
        games = fetch_schedule(day)
        rows = []
        for g in games:
            try:
                game_pk = int(g.get("gamePk"))
            except Exception:
                continue
            teams = g.get("teams") or {}
            home = teams.get("home", {}).get("team", {}).get("abbreviation") or teams.get("home", {}).get("team", {}).get("name")
            away = teams.get("away", {}).get("team", {}).get("abbreviation") or teams.get("away", {}).get("team", {}).get("name")
            officials = fetch_officials(game_pk)
            ref_names = ",".join(sorted([o.get("official", {}).get("fullName") for o in officials if str(o.get("officialType")).lower()=="referee" and o.get("official", {}).get("fullName")]))
            rows.append({"home": str(home).upper(), "away": str(away).upper(), "referees": ref_names, "rate": float(base_rate)})
        if not rows:
            print(f"[refs] No games on {day}")
            continue
        df = pd.DataFrame(rows)
        out = PROC_DIR / f"ref_assignments_{day}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[refs] wrote {out} ({len(df)} games), base_rate={base_rate:.2f}")


if __name__ == "__main__":
    main()

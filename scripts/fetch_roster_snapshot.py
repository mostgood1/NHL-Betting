"""Fetch daily NHL roster snapshot using api-web.nhle.com endpoints (replacement for deprecated statsapi.web.nhl.com).

The legacy statsapi.web.nhl.com roster endpoint is reportedly unavailable. This script
pulls each team's active roster from the newer api-web endpoints and writes a unified
DataFrame containing columns: full_name, player_id, team.

Endpoint pattern (observed 2024/2025 era):
  https://api-web.nhle.com/v1/roster/{TEAM_ABBR}/current

Example: https://api-web.nhle.com/v1/roster/TOR/current

Response structure (simplified):
{
    "forwards": [ { "playerId": 8479318, "firstName": {"default": "Auston"}, "lastName": {"default": "Matthews"}, ... }, ...],
    "defensemen": [...],
    "goalies": [...],
    ...
}

We normalize all position groups into a flat list.

If an endpoint returns non-200 or malformed JSON, we skip that team (best-effort).
Cached output path: data/models/roster_snapshot_<date>.parquet

Usage:
  python scripts/fetch_roster_snapshot.py --date 2025-10-14
  # date optional; defaults to today
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from typing import Iterable

import pandas as pd

# Allow running without PYTHONPATH exported
_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nhl_betting.utils.io import PROC_DIR  # noqa: E402

TEAM_ABBRS: list[str] = [
    "ANA","ARI","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET","EDM","FLA","LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SJS","SEA","STL","TBL","TOR","UTA","VAN","VGK","WPG","WSH"
]

import urllib.request

def _fetch(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": "props-roster-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
        data = r.read()
    try:
        return json.loads(data)
    except Exception:
        return None


def _extract_players(blob: dict, team: str) -> Iterable[dict]:
    if not isinstance(blob, dict):
        return []
    groups = [k for k in blob.keys() if isinstance(blob.get(k), list)]
    rows = []
    for g in groups:
        for p in (blob.get(g) or []):
            if not isinstance(p, dict):
                continue
            pid = p.get("playerId") or p.get("id")
            first = None
            last = None
            fn = p.get("firstName")
            ln = p.get("lastName")
            if isinstance(fn, dict):
                first = fn.get("default") or fn.get("en") or next((v for v in fn.values() if isinstance(v,str)), None)
            elif isinstance(fn, str):
                first = fn
            if isinstance(ln, dict):
                last = ln.get("default") or ln.get("en") or next((v for v in ln.values() if isinstance(v,str)), None)
            elif isinstance(ln, str):
                last = ln
            full = None
            if first and last:
                full = f"{first} {last}".strip()
            elif first:
                full = first
            elif last:
                full = last
            if not (pid and full):
                continue
            # Normalize position from group name g
            pos = None
            gu = g.lower()
            if 'forward' in gu:
                pos = 'F'
            elif 'defense' in gu or 'defenc' in gu:
                pos = 'D'
            elif 'goalie' in gu or 'goaltender' in gu:
                pos = 'G'
            rows.append({"full_name": full, "player_id": pid, "team": team, "position": pos})
    return rows


def build_snapshot() -> pd.DataFrame:
    all_rows = []
    for abbr in TEAM_ABBRS:
        url = f"https://api-web.nhle.com/v1/roster/{abbr}/current"
        try:
            blob = _fetch(url)
            players = list(_extract_players(blob, abbr))
            if players:
                all_rows.extend(players)
        except Exception as e:  # pragma: no cover
            print(f"[warn] roster fetch failed for {abbr}: {e}")
            continue
    if not all_rows:
        return pd.DataFrame(columns=["full_name","player_id","team","position"])
    df = pd.DataFrame(all_rows)
    # Deduplicate by player_id preferring rows with a non-null position
    # Add a helper flag to prioritize rows with a position value
    df['_pos_missing'] = df['position'].isna()
    df = df.sort_values(by=['player_id','_pos_missing']).drop_duplicates(['player_id'], keep='first').drop(columns=['_pos_missing'])
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="Date tag for snapshot (YYYY-MM-DD)")
    args = ap.parse_args()
    date_tag = args.date or dt.date.today().strftime("%Y-%m-%d")
    df = build_snapshot()
    out_dir = PROC_DIR.parent / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"roster_snapshot_{date_tag}.parquet"
    if df.empty:
        print("[warn] snapshot empty; nothing written")
    else:
        df.to_parquet(out_path, index=False)
        print(f"[ok] wrote {len(df)} players to {out_path}")
    # Also emit a minimal JSON for quick diffs
    try:
        jpath = out_dir / f"roster_snapshot_{date_tag}.json"
        df.to_json(jpath, orient="records")
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())

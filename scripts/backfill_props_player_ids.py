"""Backfill missing player_id and team for props lines.

Sources consulted (priority order):
 1. Roster enrichment snapshot via _build_roster_enrichment()
 2. Historical data/raw/player_game_stats.csv (last known player_id, team)

Writes updated Parquet/CSV in-place in data/props/player_props_lines/date=<date>/
and emits a summary.

Usage:
  python scripts/backfill_props_player_ids.py --date 2025-10-14

Safe to run multiple times (idempotent): only fills null player_id / team; does not overwrite existing non-null values.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import pandas as pd

# Ensure repo root on path
_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nhl_betting.data.player_props import _build_roster_enrichment  # type: ignore
from nhl_betting.utils.io import RAW_DIR, PROC_DIR
from nhl_betting.web.teams import get_team_assets


def _norm_name(s: str) -> str:
    return " ".join(str(s or "").strip().split()).lower()


def load_lines(date: str) -> list[tuple[pathlib.Path, pd.DataFrame]]:
    base = PROC_DIR.parent / "props" / f"player_props_lines/date={date}"
    out: list[tuple[pathlib.Path, pd.DataFrame]] = []
    if not base.exists():
        return out
    for name in ("oddsapi.parquet","oddsapi.csv"):
        p = base / name
        if not p.exists():
            continue
        try:
            if p.suffix == ".parquet":
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            if not isinstance(df, pd.DataFrame):
                continue
            out.append((p, df))
        except Exception as e:  # pragma: no cover
            print(f"[warn] failed reading {p}: {e}")
    return out


def build_lookup_from_stats() -> pd.DataFrame:
    sp = RAW_DIR / "player_game_stats.csv"
    if not sp.exists():
        return pd.DataFrame(columns=["full_name","player_id","team"])
    try:
        stats = pd.read_csv(sp)
    except Exception as e:
        print(f"[warn] read stats failed: {e}")
        return pd.DataFrame(columns=["full_name","player_id","team"])
    if stats.empty or not {"player","player_id"}.issubset(stats.columns):
        return pd.DataFrame(columns=["full_name","player_id","team"])
    stats = stats.dropna(subset=["player"])  # keep only with a player name
    # Keep last observed mapping per player name (date ascending)
    try:
        stats["_date"] = pd.to_datetime(stats["date"], errors="coerce")
        stats = stats.sort_values("_date")
    except Exception:
        pass
    last = stats.groupby("player").agg({"player_id":"last","team":"last"}).reset_index().rename(columns={"player":"full_name"})
    last["full_name"] = last["full_name"].astype(str)
    return last


def backfill(date: str) -> dict:
    lines = load_lines(date)
    if not lines:
        return {"date": date, "files": 0, "updated_rows": 0, "message": "no lines found"}
    roster = _build_roster_enrichment()
    stats_lookup = build_lookup_from_stats()
    # Merge roster + stats (roster priority)
    merged = pd.concat([roster, stats_lookup], ignore_index=True)
    # Deduplicate keeping first (roster rows appear first)
    merged = merged.dropna(subset=["full_name"]).drop_duplicates("full_name")
    # Build maps
    id_map = { _norm_name(r.full_name): r.player_id for r in merged.itertuples() if getattr(r, 'player_id', None) }
    team_map = { _norm_name(r.full_name): r.team for r in merged.itertuples() if getattr(r, 'team', None) }
    updated_rows = 0
    for path, df in lines:
        if df.empty:
            continue
        changed = False
        # Ensure player_name column exists (older files may have 'player')
        if 'player_name' not in df.columns and 'player' in df.columns:
            df['player_name'] = df['player']
        if 'player_id' not in df.columns:
            df['player_id'] = None
        if 'team' not in df.columns:
            df['team'] = None
        for idx, row in df.iterrows():
            pname = row.get('player_name') or row.get('player')
            if not isinstance(pname, str) or not pname.strip():
                continue
            key = _norm_name(pname)
            # Skip aggregate rows
            if key.startswith('total shots on goal') or key in ('1.5','2.5'):  # numeric artifacts
                continue
            cur_pid = row.get('player_id')
            if (cur_pid is None or (isinstance(cur_pid, float) and pd.isna(cur_pid))) and key in id_map:
                df.at[idx, 'player_id'] = id_map[key]
                changed = True
                updated_rows += 1
            cur_team = row.get('team')
            needs_team = (cur_team is None or (isinstance(cur_team, float) and pd.isna(cur_team)) or str(cur_team).strip()=='' )
            if needs_team and key in team_map:
                df.at[idx, 'team'] = team_map[key]
                changed = True
            # Normalize team to abbreviation
            try:
                tval = df.at[idx,'team']
                if tval and isinstance(tval, str):
                    assets = get_team_assets(tval)
                    ab = assets.get('abbr')
                    if ab:
                        df.at[idx,'team'] = ab
            except Exception:
                pass
        if changed:
            # Write back in same format
            try:
                if path.suffix == '.parquet':
                    df.to_parquet(path, index=False)
                else:
                    df.to_csv(path, index=False)
            except Exception as e:
                print(f"[warn] failed to write {path}: {e}")
    return {"date": date, "files": len(lines), "updated_rows": updated_rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=False, help='Date (YYYY-MM-DD); default today')
    args = ap.parse_args()
    import datetime as dt
    date = args.date or dt.date.today().strftime('%Y-%m-%d')
    summary = backfill(date)
    print(summary)
    if summary.get('updated_rows',0) == 0:
        print('[info] no rows updated (maybe already filled or no mapping available)')

if __name__ == '__main__':
    sys.exit(main())

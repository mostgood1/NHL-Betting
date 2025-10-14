"""Enrich props lines with player_id and team using a pre-fetched roster snapshot.

Reads data/models/roster_snapshot_<date>.parquet (created by fetch_roster_snapshot.py)
and updates files under data/props/player_props_lines/date=<date>/ (both parquet and csv)
filling missing player_id / team where possible via variant key matching.

Variant keys per roster full_name:
  - clean full name (lowercase, ascii, single space)
  - squashed (remove non-alphanumerics)
  - first-initial + last (with & without space)
  - multi-initials for hyphenated first names + last

Ambiguous variant collisions are ignored (first mapping wins; no overwrite of existing non-null ids).

Usage:
  python scripts/enrich_props_from_roster.py --date 2025-10-14
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, Set

import pandas as pd

# Ensure repo root on path
_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nhl_betting.utils.io import PROC_DIR  # type: ignore
from nhl_betting.web.teams import get_team_assets  # type: ignore


def _clean(s: str) -> str:
    import unicodedata, re
    s = unicodedata.normalize('NFKD', str(s or '')).encode('ascii','ignore').decode()
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s

def _squash(s: str) -> str:
    import re
    return re.sub(r'[^a-z0-9]', '', str(s or '').lower())

def _initials_parts(first: str) -> str:
    import re
    parts = re.split(r'[-\s]+', first.strip())
    return ''.join(p[0] for p in parts if p)

def variant_keys(full_name_clean: str) -> Set[str]:
    parts = full_name_clean.split()
    if not parts:
        return set()
    first = parts[0]
    last = parts[-1] if len(parts) > 1 else ''
    keys: Set[str] = {full_name_clean, _squash(full_name_clean)}
    if first and last:
        ini1 = first[0]
        iniall = _initials_parts(first)
        combos = {ini1 + ' ' + last, ini1 + last, _squash(ini1 + ' ' + last)}
        if iniall and iniall != ini1:
            combos |= {iniall + ' ' + last, iniall + last, _squash(iniall + ' ' + last)}
        keys |= combos
    return keys


def build_maps(roster_df: pd.DataFrame) -> tuple[Dict[str,int], Dict[str,str]]:
    id_map: Dict[str,int] = {}
    team_map: Dict[str,str] = {}
    for _, r in roster_df.iterrows():
        full = str(r.get('full_name') or '').strip()
        pid = r.get('player_id')
        team = r.get('team')
        if not full or pd.isna(pid):
            continue
        clean = _clean(full)
        for k in variant_keys(clean):
            id_map.setdefault(k, int(pid))
            if isinstance(team, str) and team.strip():
                team_map.setdefault(k, team.strip().upper())
    return id_map, team_map


def enrich(date: str) -> dict:
    snap_path = PROC_DIR.parent / 'models' / f'roster_snapshot_{date}.parquet'
    if not snap_path.exists():
        return {'date': date, 'updated_rows': 0, 'message': 'roster snapshot missing'}
    roster = pd.read_parquet(snap_path)
    if roster.empty:
        return {'date': date, 'updated_rows': 0, 'message': 'roster snapshot empty'}
    id_map, team_map = build_maps(roster)
    lines_dir = PROC_DIR.parent / 'props' / f'player_props_lines/date={date}'
    if not lines_dir.exists():
        return {'date': date, 'updated_rows': 0, 'message': 'lines dir missing'}
    updated_rows = 0
    files = 0
    for p in lines_dir.iterdir():
        if p.suffix not in ('.csv','.parquet'):
            continue
        try:
            if p.suffix == '.parquet':
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        files += 1
        # Ensure expected columns
        if 'player_name' not in df.columns and 'player' in df.columns:
            df['player_name'] = df['player']
        if 'player_id' not in df.columns:
            df['player_id'] = None
        if 'team' not in df.columns:
            df['team'] = None
        changed_rows = 0
        for idx, row in df.iterrows():
            pname = row.get('player_name')
            if not isinstance(pname, str) or not pname.strip():
                continue
            clean = _clean(pname)
            # Skip obvious aggregate artifacts
            if clean.startswith('total shots on goal'):
                continue
            if (pd.isna(row.get('player_id')) or row.get('player_id') in (None, '')):
                for k in variant_keys(clean):
                    pid = id_map.get(k)
                    if pid is not None:
                        df.at[idx,'player_id'] = pid
                        changed_rows += 1
                        break
            # Fill team if missing
            cur_team = row.get('team')
            if (cur_team is None or (isinstance(cur_team, float) and pd.isna(cur_team)) or str(cur_team).strip()=='' ):
                for k in variant_keys(clean):
                    tm = team_map.get(k)
                    if tm:
                        df.at[idx,'team'] = tm
                        break
            # Normalize team to abbreviation
            try:
                tv = df.at[idx,'team']
                if tv and isinstance(tv, str):
                    ab = get_team_assets(tv).get('abbr')
                    if ab:
                        df.at[idx,'team'] = ab
            except Exception:
                pass
        if changed_rows > 0:
            try:
                if p.suffix == '.parquet':
                    df.to_parquet(p, index=False)
                else:
                    df.to_csv(p, index=False)
                updated_rows += changed_rows
            except Exception as e:
                print(f"[warn] failed to write {p}: {e}")
    return {'date': date, 'updated_rows': updated_rows, 'files': files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=False)
    args = ap.parse_args()
    import datetime as dt
    date = args.date or dt.date.today().strftime('%Y-%m-%d')
    summary = enrich(date)
    print(summary)
    if summary.get('updated_rows',0) == 0:
        print('[info] no rows updated (maybe already enriched or no matches)')

if __name__ == '__main__':
    sys.exit(main())

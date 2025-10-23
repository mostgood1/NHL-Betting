"""Diagnose why player_photo mapping results in zero remote photos.

Loads canonical props lines for a given date (after enrichment) exactly as the
/props/recommendations route does, and prints:
 - total rows per file
 - non-null player_id counts
 - distinct players with ids
 - sample mapping entries built with the same logic
"""
from __future__ import annotations

import pathlib, sys, argparse
import pandas as pd

# Ensure repo root
_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nhl_betting.utils.io import PROC_DIR  # type: ignore

def _norm_name(x: str) -> str:
    try:
        return " "+" ".join(str(x or "").split())+" ".strip()
    except Exception:
        return str(x)

def build_mapping(date: str):
    base = PROC_DIR.parent / 'props' / f'player_props_lines/date={date}'
    parts = []
    meta = []
    for name in ("oddsapi.parquet","oddsapi.csv"):
        p = base / name
        if not p.exists():
            continue
        try:
            if p.suffix == '.parquet':
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
        except Exception as e:
            print(f"[warn] read fail {name}: {e}")
            continue
        meta.append((name, len(df), df['player_id'].notna().sum() if 'player_id' in df.columns else 0))
        parts.append(df)
    if not parts:
        print('[err] no parts loaded')
        return {}, meta
    lp = pd.concat(parts, ignore_index=True)
    mapping = {}
    if not lp.empty and {'player_name','player_id'}.issubset(lp.columns):
        grp = lp.groupby('player_name')['player_id'].agg(lambda s: s.dropna().astype(str).value_counts().head(1).index.tolist())
        for name_key, lst in grp.items():
            if not lst:
                continue
            pid = lst[0]
            if pid and pid.strip():
                mapping[_norm_name(name_key)] = f"https://cms.nhl.bamgrid.com/images/headshots/current/168x168/{pid}.jpg"
    return mapping, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=False)
    args = ap.parse_args()
    import datetime as dt
    date = args.date or dt.date.today().strftime('%Y-%m-%d')
    mapping, meta = build_mapping(date)
    print('FILES:')
    for name, rows, pid_nonnull in meta:
        print(f"  {name}: rows={rows} player_id_non_null={pid_nonnull}")
    print(f"mapping_size={len(mapping)}")
    if mapping:
        i = 0
        print('SAMPLE:')
        for k,v in mapping.items():
            print(' ', k, '->', v)
            i += 1
            if i>=10: break
    else:
        print('[warn] mapping empty')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

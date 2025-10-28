from pathlib import Path
import pandas as pd

def main():
    base = Path('data/raw/nhl_pbp')
    out = {}
    for yr in (2024, 2025):
        p = base / f'pbp_{yr}.parquet'
        if not p.exists():
            out[yr] = {'exists': False}
            continue
        try:
            df = pd.read_parquet(p, engine='pyarrow')
        except Exception as e:
            out[yr] = {'exists': True, 'read_error': str(e)}
            continue
        cols = list(df.columns)
        dt_col = next((c for c in ['gameDate','date','event_date','eventDate','game_date','dt','timestamp'] if c in cols), None)
        if dt_col:
            s = pd.to_datetime(df[dt_col], errors='coerce')
            out[yr] = {
                'exists': True,
                'rows': int(len(df)),
                'min_date': str(s.min().date()) if s.notna().any() else None,
                'max_date': str(s.max().date()) if s.notna().any() else None,
                'sample_cols': cols[:12],
            }
        else:
            out[yr] = {'exists': True, 'rows': int(len(df)), 'no_date_col': True, 'sample_cols': cols[:12]}
    print(out)

if __name__ == '__main__':
    main()

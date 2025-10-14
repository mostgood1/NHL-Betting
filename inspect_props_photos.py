import pandas as pd
from pathlib import Path
from datetime import date

today = date.today().strftime('%Y-%m-%d')
base = Path('data/props') / f'player_props_lines/date={today}'
rec_csv = Path('data/processed') / f'props_recommendations_{today}.csv'

print(f'=== Inspecting canonical props lines for {today} ===')
if not base.exists():
    print('No canonical lines directory:', base)
else:
    parts = []
    for name in ('bovada.parquet','oddsapi.parquet','bovada.csv','oddsapi.csv'):
        p = base / name
        if p.exists():
            try:
                if p.suffix == '.parquet':
                    dfp = pd.read_parquet(p)
                else:
                    dfp = pd.read_csv(p)
                dfp['__source_file'] = p.name
                parts.append(dfp)
                print(f'  Loaded {p.name}: rows={len(dfp)} cols={len(dfp.columns)}')
            except Exception as e:
                print(f'  Failed reading {p.name}: {e}')
    if parts:
        all_df = pd.concat(parts, ignore_index=True)
        has_cols = {'player_name','player_id'} <= set(all_df.columns)
        if has_cols:
            total_players = all_df['player_name'].nunique()
            non_null_ids = all_df.dropna(subset=['player_id'])['player_name'].nunique()
            print(f'  Unique players: {total_players}; with non-null player_id: {non_null_ids}')
            missing_sample = all_df[all_df['player_id'].isna()][['player_name','market','line']].head(15)
            if not missing_sample.empty:
                print('  Sample players missing player_id:')
                print(missing_sample.to_string())
        else:
            print('  player_name/player_id columns not both present; columns=', list(all_df.columns)[:20])
    else:
        print('  No parts loaded.')

print(f'\n=== Inspecting recommendations CSV {rec_csv} ===')
if rec_csv.exists():
    try:
        rec = pd.read_csv(rec_csv)
        print(f'  Rows: {len(rec)} Columns: {len(rec.columns)}')
        print('  Columns sample:', list(rec.columns)[:25])
        print('  First 5 rows:')
        print(rec.head().to_string())
        missing_team = rec[rec['team'].isna()]['player'].head(10) if 'team' in rec.columns else []
        if len(missing_team):
            print('  Players missing team (first 10):', list(missing_team))
    except Exception as e:
        print('  Failed to read recommendations CSV:', e)
else:
    print('  File does not exist.')

print('\nTip: Photos require player_id to be populated in canonical lines. If most player_id values are null, roster/stats enrichment failed. Consider running:')
print('  python -m nhl_betting.cli props-stats-backfill --start 2023-09-01 --end', today)
print('Then re-run props-collect and props-recommendations for the date.')

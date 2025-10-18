import pandas as pd
from pathlib import Path

base = Path('data/props/player_props_lines/date=2025-10-17')
print(f"Checking directory: {base}")
print(f"Directory exists: {base.exists()}")

if base.exists():
    files = list(base.glob("*"))
    print(f"\nFiles in directory: {[f.name for f in files]}")
    
    for file in files:
        if file.suffix in ['.parquet', '.csv']:
            print(f"\n=== {file.name} ===")
            try:
                if file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_csv(file)
                
                print(f"Total lines: {len(df)}")
                print(f"Columns: {df.columns.tolist()}")
                
                if 'team' in df.columns:
                    teams = sorted(df['team'].unique())
                    print(f"Teams with odds: {teams}")
                    print(f"\nLines by team:")
                    print(df.groupby('team').size().sort_values(ascending=False))
                
                if 'player_name' in df.columns:
                    print(f"\nUnique players: {df['player_name'].nunique()}")
            except Exception as e:
                print(f"Error reading {file.name}: {e}")
else:
    print(f"\nDirectory does not exist!")

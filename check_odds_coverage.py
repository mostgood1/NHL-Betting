import pandas as pd
from pathlib import Path

# Check canonical lines
proc_dir = Path("data/processed")
odds_file = proc_dir / "canonical_lines_oddsapi.parquet"

if odds_file.exists():
    df = pd.read_parquet(odds_file)
    print(f"Total lines in canonical_lines_oddsapi.parquet: {len(df)}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    if 'game' in df.columns:
        print(f"\nUnique games with odds:")
        for game in sorted(df['game'].unique()):
            count = len(df[df['game'] == game])
            print(f"  {game}: {count} lines")
    
    if 'market' in df.columns:
        print(f"\nMarkets available:")
        print(df['market'].value_counts())
    
    if 'player_name' in df.columns:
        print(f"\nUnique players: {df['player_name'].nunique()}")
        teams = df['team'].unique() if 'team' in df.columns else []
        print(f"Teams with player props odds: {sorted(teams)}")
else:
    print(f"Odds file not found: {odds_file}")

# Check today's games
pred_file = proc_dir / "predictions_2025-10-17.csv"
if pred_file.exists():
    games_df = pd.read_csv(pred_file)
    print(f"\n\nGames scheduled for 2025-10-17:")
    for _, row in games_df.iterrows():
        print(f"  {row['away']} @ {row['home']}")

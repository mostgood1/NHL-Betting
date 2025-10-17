import pandas as pd

df = pd.read_csv('data/raw/player_game_stats.csv')

print(f'Total rows: {len(df):,}')
print(f'Date range: {df["date"].min()} to {df["date"].max()}')
print(f'Unique players: {df["player_id"].nunique():,}')
print(f'\nColumns: {list(df.columns)}')
print(f'\nSample data:')
print(df.head())
print(f'\nData is ready for training!')

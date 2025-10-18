import pandas as pd

df = pd.read_csv('data/raw/games_with_features.csv')
print(f'Total games: {len(df)}')
print(f'Date range: {df["date"].min()} to {df["date"].max()}')
print(f'Seasons: {sorted(df["season"].unique())}')
print(f'\nFeatures ({len(df.columns)} columns):')
print(df.columns.tolist())
print('\nSample game:')
print(df.iloc[0].to_dict())

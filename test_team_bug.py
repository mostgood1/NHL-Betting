import pandas as pd

df = pd.read_csv('data/processed/props_recommendations_2025-10-17.csv')
grp_cols = [c for c in ['player','team'] if c in df.columns]
grouped = df.groupby(grp_cols, dropna=False) if grp_cols else []

teams_processed = []
for keys, g_all in grouped:
    if isinstance(keys, tuple):
        player = keys[0] if len(keys) > 0 else None
        team = keys[1] if len(keys) > 1 else None
    else:
        player = keys
        team = None
    teams_processed.append((player, team))

print(f"Total groups: {len(teams_processed)}")
print(f"Last 10 groups:")
for player, team in teams_processed[-10:]:
    print(f"  {player} - {team}")
print(f"\nFinal team value: {team}")

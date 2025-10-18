from nhl_betting.data.player_props import collect_oddsapi_props
import pandas as pd

print("Testing fixed collect_oddsapi_props for 2025-10-17...")
df = collect_oddsapi_props('2025-10-17')

print(f"\nTotal rows collected: {len(df)}")

if not df.empty:
    print(f"\nTeams with odds: {sorted(df['team'].unique()) if 'team' in df.columns else 'No team column'}")
    
    if 'book' in df.columns:
        print(f"\nBooks: {df['book'].unique()}")
    
    if 'player' in df.columns:
        print(f"\nUnique players: {df['player'].nunique()}")
        print(f"\nSample players by market:")
        for market in df['market'].unique():
            players = df[df['market'] == market]['player'].unique()[:3]
            print(f"  {market}: {', '.join(players)}")
else:
    print("No data collected!")

import pandas as pd
import numpy as np

# Load historical data
hist_df = pd.read_csv('data/raw/games_with_features.csv', parse_dates=['date'])

# Teams in today's games
teams = ['DET', 'TBL', 'WSH', 'MIN', 'CHI', 'VAN', 'UTA', 'SJS']
target_date = pd.to_datetime('2025-10-17 23:00:00+00:00')

for team in teams:
    # Get last 10 games for this team before target date
    team_games = hist_df[
        ((hist_df['home'] == team) | (hist_df['away'] == team)) &
        (hist_df['date'] < target_date)
    ].tail(10)
    
    if len(team_games) == 0:
        print(f"\n{team}: NO HISTORICAL GAMES FOUND")
        continue
    
    goals_for = []
    goals_against = []
    for _, g in team_games.iterrows():
        if g['home'] == team:
            goals_for.append(g['home_goals'])
            goals_against.append(g['away_goals'])
        else:
            goals_for.append(g['away_goals'])
            goals_against.append(g['home_goals'])
    
    print(f"\n{team}:")
    print(f"  Games found: {len(team_games)}")
    print(f"  Goals for (last 10): {goals_for}")
    print(f"  Mean: {np.mean(goals_for):.2f}")
    print(f"  Goals against: {goals_against}")
    print(f"  Mean: {np.mean(goals_against):.2f}")

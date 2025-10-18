"""Test props NN predictions to see variability."""
# CRITICAL: Import torch BEFORE pandas
try:
    import torch
    print(f"[info] PyTorch loaded: {torch.__version__}")
except Exception as e:
    print(f"[error] PyTorch failed: {e}")
    exit(1)

from pathlib import Path
import pandas as pd
from nhl_betting.models.nn_props import NNPropsModel

# Load some player stats
stats_path = Path('data/raw/player_game_stats.csv')
if not stats_path.exists():
    print("No stats found")
    exit()

df = pd.read_csv(stats_path, nrows=10000)

# Test SOG model
sog_model = NNPropsModel(market='SOG', use_npu=False)

if sog_model.model is None:
    print("SOG model not loaded")
    exit()

# Get some unique players
import json
def parse_player_name(p):
    if pd.isna(p):
        return ""
    p_str = str(p)
    if p_str.startswith("{"):
        try:
            p_dict = json.loads(p_str.replace("'", '"'))
            return p_dict.get("default", "")
        except:
            return p_str
    return p_str

df["player_name"] = df["player"].apply(parse_player_name)
df = df[df["role"] == "skater"]

# Get players from different teams
players_by_team = df.groupby('team')['player_name'].apply(lambda x: x.iloc[0] if len(x) > 0 else None).dropna()
test_players = players_by_team.head(10).tolist()

print(f"Testing {len(test_players)} players from different teams:")
print("="*80)

predictions = []
for player in test_players:
    try:
        pred = sog_model.predict_lambda(df, player)
        player_team = df[df['player_name'] == player]['team'].iloc[0] if len(df[df['player_name'] == player]) > 0 else 'Unknown'
        predictions.append(pred)
        pred_str = f"{pred:.4f}" if pred is not None else "None"
        print(f"{player:30} (Team: {player_team:20}): {pred_str}")
    except Exception as e:
        print(f"{player:30}: ERROR - {e}")

print("\n" + "="*80)
print(f"Statistics:")
print(f"  Mean: {sum(p for p in predictions if p) / len([p for p in predictions if p]):.4f}")
print(f"  Std:  {pd.Series([p for p in predictions if p]).std():.4f}")
print(f"  Min:  {min(p for p in predictions if p):.4f}")
print(f"  Max:  {max(p for p in predictions if p):.4f}")
print(f"  Range: {max(p for p in predictions if p) - min(p for p in predictions if p):.4f}")

if pd.Series([p for p in predictions if p]).std() < 0.1:
    print("\n❌ PROBLEM: Predictions are too similar! Variability is very low.")
    print("   This suggests team encodings are not being used properly.")
else:
    print("\n✅ Good variability in predictions")

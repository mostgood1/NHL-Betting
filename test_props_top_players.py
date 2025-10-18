"""Test props NN predictions with well-known players."""
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

# Load player stats
stats_path = Path('data/raw/player_game_stats.csv')
if not stats_path.exists():
    print("No stats found")
    exit()

df = pd.read_csv(stats_path)

# Test SOG model
sog_model = NNPropsModel(market='SOG', use_npu=False)

if sog_model.model is None:
    print("SOG model not loaded")
    exit()

# Parse player names
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

# Get players with most games (likely stars)
skaters = df[df["role"] == "skater"].copy()
player_game_counts = skaters.groupby('player_name').size().sort_values(ascending=False)
top_players = player_game_counts.head(30).index.tolist()

print(f"Testing top {len(top_players)} players by game count:")
print("="*100)

predictions = []
for player in top_players:
    try:
        pred = sog_model.predict_lambda(df, player)
        player_team = df[df['player_name'] == player]['team'].iloc[-1] if len(df[df['player_name'] == player]) > 0 else 'Unknown'
        player_games = len(df[df['player_name'] == player])
        
        if pred is not None:
            predictions.append({'player': player, 'team': player_team, 'pred': pred, 'games': player_games})
            pred_str = f"{pred:.4f}"
            print(f"{player:35} ({player_team:25}) Games: {player_games:4}  Pred: {pred_str}")
    except Exception as e:
        print(f"{player:35}: ERROR - {e}")

print("\n" + "="*100)
pred_values = [p['pred'] for p in predictions]
if pred_values:
    print(f"Statistics ({len(pred_values)} predictions):")
    print(f"  Mean:  {sum(pred_values) / len(pred_values):.4f}")
    print(f"  Std:   {pd.Series(pred_values).std():.4f}")
    print(f"  Min:   {min(pred_values):.4f}")
    print(f"  Max:   {max(pred_values):.4f}")
    print(f"  Range: {max(pred_values) - min(pred_values):.4f}")
    
    if pd.Series(pred_values).std() < 0.1:
        print("\n❌ PROBLEM: Predictions are too similar! Variability is very low.")
        print("   This suggests team encodings are not being used properly.")
    else:
        print("\n✅ Good variability in predictions")
    
    # Show top and bottom 5
    predictions_sorted = sorted(predictions, key=lambda x: x['pred'])
    print("\n📊 Lowest 5 predictions:")
    for p in predictions_sorted[:5]:
        print(f"  {p['player']:35} ({p['team']:25}): {p['pred']:.4f}")
    
    print("\n📊 Highest 5 predictions:")
    for p in predictions_sorted[-5:]:
        print(f"  {p['player']:35} ({p['team']:25}): {p['pred']:.4f}")
else:
    print("No predictions generated")

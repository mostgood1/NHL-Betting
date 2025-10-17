import pandas as pd
import json
from pathlib import Path

# Simulate what _prepare_features does
df = pd.read_csv('data/raw/player_game_stats.csv')

# Parse player names
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

# Test with top player
test_player = "C. Smith"
role = "skater"
metric_col = "shots"
window_games = 10

print(f"Testing feature prep for: {test_player}")

# Filter
df_copy = df.copy()
df_copy["player_name"] = df_copy["player"].apply(parse_player_name)
pdf = df_copy[df_copy["player_name"].astype(str).str.lower() == test_player.lower()].copy()
pdf = pdf[pdf["role"].astype(str).str.lower() == role.lower()].copy()

print(f"1. After filter: {len(pdf)} rows")

if pdf.empty:
    print("ERROR: Empty after filter!")
else:
    # Sort
    pdf = pdf.sort_values("date")
    
    # Check metric column
    if metric_col not in pdf.columns:
        print(f"ERROR: Column {metric_col} not found!")
    else:
        print(f"2. Column {metric_col} exists")
        
        # Convert and drop nulls
        pdf[metric_col] = pd.to_numeric(pdf[metric_col], errors="coerce")
        before_dropna = len(pdf)
        pdf = pdf.dropna(subset=[metric_col])
        print(f"3. After dropna: {len(pdf)} rows (dropped {before_dropna - len(pdf)})")
        
        if len(pdf) < 3:
            print("ERROR: Less than 3 rows!")
        else:
            # Create features
            features = pd.DataFrame()
            features[f"{metric_col}_mean_{window_games}"] = pdf[metric_col].rolling(
                window=window_games, min_periods=1
            ).mean()
            features[f"{metric_col}_std_{window_games}"] = pdf[metric_col].rolling(
                window=window_games, min_periods=1
            ).std().fillna(0)
            features[f"{metric_col}_last"] = pdf[metric_col].shift(1).fillna(pdf[metric_col].mean())
            
            # Target
            features["target"] = pdf[metric_col].values
            
            print(f"4. Features DataFrame: {len(features)} rows")
            print(f"   Columns: {list(features.columns)}")
            
            # Skip window
            result = features.iloc[window_games:]
            print(f"5. After skipping first {window_games} rows: {len(result)} rows")
            
            if len(result) == 0:
                print("ERROR: Empty after window skip!")
            else:
                print(f"SUCCESS: {len(result)} training samples for this player")

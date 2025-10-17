import pandas as pd
import json

# Load data
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

# Check a specific player
test_player = df["player_name"].value_counts().head(1).index[0]
print(f"Testing player: {test_player}")
print(f"Total games: {len(df[df['player_name'] == test_player])}")

# Filter for this player (skater)
pdf = df[df["player_name"] == test_player].copy()
pdf = pdf[pdf["role"] == "skater"].copy()

print(f"After role filter: {len(pdf)}")
print(f"\nColumns available: {list(pdf.columns)}")
print(f"\nSample data:")
print(pdf[["date", "player_name", "role", "shots"]].head())

# Check if shots column has data
print(f"\nShots stats:")
print(pdf["shots"].describe())
print(f"Non-null shots: {pdf['shots'].notna().sum()}")

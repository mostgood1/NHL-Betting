"""Validate player game stats data for neural network training."""
import pandas as pd
import json
from pathlib import Path
from collections import Counter

print("="*60)
print("DATA VALIDATION FOR NN TRAINING")
print("="*60)

# Load data
data_path = Path("data/raw/player_game_stats.csv")
if not data_path.exists():
    print(f"❌ ERROR: Data file not found: {data_path}")
    exit(1)

print(f"\n✓ Loading data from {data_path}...")
df = pd.read_csv(data_path)
print(f"✓ Loaded {len(df):,} rows")

# Check columns
print(f"\n📋 Columns ({len(df.columns)}):")
for col in df.columns:
    null_count = df[col].isna().sum()
    null_pct = (null_count / len(df)) * 100
    print(f"   - {col:20s} (null: {null_count:,} / {null_pct:5.1f}%)")

# Parse player names
print("\n🔍 Parsing player names...")
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

# Count parsed names
valid_names = df["player_name"][df["player_name"].str.strip() != ""]
print(f"✓ Parsed {len(valid_names):,} rows with valid player names")
print(f"✓ Unique players: {valid_names.nunique():,}")
print(f"✓ Empty/invalid names: {len(df) - len(valid_names):,}")

# Check roles
print(f"\n👥 Player roles:")
role_counts = df["role"].value_counts()
for role, count in role_counts.items():
    print(f"   - {role}: {count:,}")

# Check target metrics
print(f"\n🎯 Target metrics availability:")
metrics = {
    "SOG (shots)": "shots",
    "GOALS": "goals", 
    "ASSISTS": "assists",
    "POINTS": "points",
    "SAVES": "saves",
    "BLOCKS (blocked)": "blocked"
}

for market, col in metrics.items():
    if col in df.columns:
        non_null = df[col].notna().sum()
        non_null_pct = (non_null / len(df)) * 100
        print(f"   ✓ {market:20s}: {non_null:,} / {len(df):,} ({non_null_pct:.1f}%) non-null")
        if non_null > 0:
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            print(f"      Stats: min={numeric_vals.min():.1f}, max={numeric_vals.max():.1f}, mean={numeric_vals.mean():.2f}")
    else:
        print(f"   ❌ {market:20s}: Column '{col}' NOT FOUND")

# Sample player analysis
print(f"\n🔬 Sample player analysis:")
skaters = df[(df["role"] == "skater") & (df["player_name"].str.strip() != "")]
if len(skaters) > 0:
    # Find top players by game count
    top_players = skaters["player_name"].value_counts().head(5)
    
    for player_name, game_count in top_players.items():
        print(f"\n   Player: {player_name}")
        print(f"   Total games: {game_count}")
        
        player_df = skaters[skaters["player_name"] == player_name].copy()
        
        # Check each metric
        for market, col in [("SOG", "shots"), ("GOALS", "goals"), ("ASSISTS", "assists")]:
            if col in player_df.columns:
                player_df[col] = pd.to_numeric(player_df[col], errors='coerce')
                non_null = player_df[col].notna().sum()
                if non_null > 10:  # Need at least 10 for rolling window
                    print(f"      {market}: {non_null} games with data ✓")
                else:
                    print(f"      {market}: {non_null} games with data ⚠️ (need 10+)")

# Test feature preparation on one player
print(f"\n🧪 Testing feature preparation...")
test_role = "skater"
test_metric = "shots"
window = 10

# Find a player with good data
test_candidates = []
for player in skaters["player_name"].value_counts().head(20).index:
    player_df = skaters[skaters["player_name"] == player].copy()
    if test_metric in player_df.columns:
        player_df[test_metric] = pd.to_numeric(player_df[test_metric], errors='coerce')
        non_null = player_df[test_metric].notna().sum()
        if non_null > window + 10:  # Need more than window for training
            test_candidates.append((player, non_null))

if test_candidates:
    test_player, data_count = test_candidates[0]
    print(f"   Testing with: {test_player} ({data_count} valid games)")
    
    # Simulate feature prep
    pdf = df[df["player_name"] == test_player].copy()
    pdf = pdf[pdf["role"] == test_role].copy()
    pdf = pdf.sort_values("date")
    pdf[test_metric] = pd.to_numeric(pdf[test_metric], errors='coerce')
    pdf = pdf.dropna(subset=[test_metric])
    
    print(f"   After filtering: {len(pdf)} rows")
    
    if len(pdf) >= 3:
        # Create rolling features
        features = pd.DataFrame()
        features[f"{test_metric}_mean_{window}"] = pdf[test_metric].rolling(
            window=window, min_periods=1
        ).mean()
        features["target"] = pdf[test_metric].values
        
        # Skip initial window
        result = features.iloc[window:]
        print(f"   After rolling window ({window} games): {len(result)} training samples")
        
        if len(result) > 0:
            print(f"   ✅ SUCCESS: Feature preparation works!")
            print(f"   Sample features shape: {result.shape}")
        else:
            print(f"   ❌ ERROR: No samples after window skip")
    else:
        print(f"   ❌ ERROR: Not enough data after filtering")
else:
    print(f"   ❌ ERROR: No suitable test player found")

# Estimate total training samples
print(f"\n📊 Estimating training data for each market...")
for market, col in [("SOG", "shots"), ("GOALS", "goals"), ("ASSISTS", "assists"), 
                    ("POINTS", "points"), ("BLOCKS", "blocked")]:
    if col not in df.columns:
        print(f"   {market}: ❌ Column '{col}' not found")
        continue
    
    role = "skater"
    player_samples = []
    
    role_df = df[df["role"] == role].copy()
    players = role_df["player_name"].unique()
    players = [p for p in players if p and str(p).strip()]
    
    for player in players[:100]:  # Sample first 100 for speed
        pdf = role_df[role_df["player_name"] == player].copy()
        pdf = pdf.sort_values("date")
        pdf[col] = pd.to_numeric(pdf[col], errors='coerce')
        pdf = pdf.dropna(subset=[col])
        
        if len(pdf) > window:
            samples = len(pdf) - window
            player_samples.append(samples)
    
    if player_samples:
        total_estimate = sum(player_samples) * (len(players) / 100)
        avg_per_player = sum(player_samples) / len(player_samples)
        print(f"   {market:10s}: ~{total_estimate:,.0f} samples estimated ({avg_per_player:.1f} per player) ✓")
    else:
        print(f"   {market:10s}: ❌ No valid samples found")

print(f"\n{'='*60}")
print("VALIDATION COMPLETE")
print("="*60)

# Final recommendation
if len(test_candidates) > 0:
    print(f"\n✅ DATA IS READY FOR TRAINING")
    print(f"   - {len(players):,} players with valid data")
    print(f"   - Estimated {sum(player_samples) * (len(players) / 100):,.0f}+ training samples")
    print(f"\n   Next step: Debug why train() method returns 0 samples")
else:
    print(f"\n❌ DATA NOT READY")
    print(f"   - Need to collect more data or fix data format")

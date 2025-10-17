#!/usr/bin/env python3
"""Check status of all neural network models."""
from pathlib import Path
import os
import numpy as np

print("=" * 60)
print("NEURAL NETWORK MODELS STATUS")
print("=" * 60)

# Game Models
print("\n🎮 GAME MODELS (3,136 games trained):")
print("-" * 60)
games_dir = Path("data/models/nn_games")
for model_name in ["first_10min", "period_goals"]:
    pt = games_dir / f"{model_name}_model.pt"
    meta = games_dir / f"{model_name}_metadata.npz"
    
    if pt.exists():
        size_kb = os.path.getsize(pt) // 1024
        print(f"  ✓ {model_name.upper():15} trained  ({size_kb:,} KB)")
        
        if meta.exists():
            metadata = np.load(meta, allow_pickle=True)
            features = metadata['feature_columns']
            print(f"    └─ Features: {len(features)} (Elo, form, rest, season)")
    else:
        print(f"  ✗ {model_name.upper():15} NOT trained")

# Props Models
print("\n👤 PROPS MODELS (player-level, team-aware):")
print("-" * 60)
props_dir = Path("data/models/nn_props")
for model_name in ["goals", "assists", "points", "sog", "blocks", "saves"]:
    pt = props_dir / f"{model_name}_model.pt"
    meta = props_dir / f"{model_name}_metadata.npz"
    
    if pt.exists():
        size_kb = os.path.getsize(pt) // 1024
        print(f"  ✓ {model_name.upper():15} trained  ({size_kb:,} KB)")
        
        if meta.exists():
            metadata = np.load(meta, allow_pickle=True)
            features = metadata['feature_columns']
            team_features = [f for f in features if f.startswith('team_')]
            print(f"    └─ Features: {len(features)} total, {len(team_features)} team-specific")
    else:
        print(f"  ✗ {model_name.upper():15} NOT trained")

print("\n" + "=" * 60)
print("KEY INSIGHTS:")
print("=" * 60)
print("✓ Game models use LEAGUE-LEVEL features (Elo, form, rest)")
print("  - Trained on 3,136 games across all NHL teams")
print("  - Learns general NHL scoring patterns")
print("")
print("✓ Props models are TEAM-AWARE via one-hot encoding")
print("  - Each player's team is encoded as a feature")
print("  - Model learns team-specific tendencies")
print("  - E.g., 'team_Toronto Maple Leafs' feature = 1.0 for Leafs players")
print("=" * 60)

"""Validate trained NN game models produce reasonable predictions."""
import sys
from pathlib import Path

# Import order critical (torch before pandas)
try:
    import torch
    print(f"[info] PyTorch loaded: {torch.__version__}")
except Exception as e:
    print(f"[error] PyTorch failed: {e}")
    sys.exit(1)

import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from nhl_betting.models.nn_games import NNGameModel
from nhl_betting.utils.io import RAW_DIR, MODEL_DIR

def load_models():
    """Load the three trained models."""
    models = {}
    for model_type in ['TOTAL_GOALS', 'MONEYLINE', 'GOAL_DIFF']:
        try:
            # NNGameModel auto-loads from model_dir if models exist
            model = NNGameModel(model_type=model_type, model_dir=MODEL_DIR / 'nn_games')
            
            # Check if model was loaded
            if model.model is not None or model.onnx_session is not None:
                models[model_type] = model
                print(f"[load] ✓ {model_type} model loaded")
            else:
                print(f"[warn] Model not loaded for {model_type}")
        except Exception as e:
            print(f"[error] Failed to load {model_type}: {e}")
    
    return models

def prepare_test_features(home, away, home_elo=1500, away_elo=1500):
    """Prepare 95-feature vector matching the training data format."""
    # Base features matching nn_games.py _prepare_features()
    features = {
        # Elo features
        'home_elo': home_elo,
        'away_elo': away_elo,
        'elo_diff': home_elo - away_elo,  # CRITICAL: Must match training
        
        # Recent form (last 10 games)
        'home_goals_last10': 3.0,
        'home_goals_against_last10': 2.8,
        'home_wins_last10': 5,
        'away_goals_last10': 2.9,
        'away_goals_against_last10': 3.1,
        'away_wins_last10': 4,
        
        # Rest days
        'home_rest_days': 1,
        'away_rest_days': 2,
        
        # Season context
        'season_progress': 15 / 82.0,  # 15 games into season
        
        # Home ice advantage
        'is_home': 1.0,
    }
    
    # Team encodings must match training format: home_team_XXX and away_team_XXX
    # NOTE: predict() method will add these with different format, but we provide
    # the dictionary features which should not have team encodings
    # (The predict() method handles team encoding separately)
    
    return features

def test_predictions(models):
    """Test predictions on sample matchups."""
    test_matchups = [
        {'home': 'TOR', 'away': 'MTL', 'home_elo': 1550, 'away_elo': 1450},  # Strong vs weak
        {'home': 'EDM', 'away': 'COL', 'home_elo': 1520, 'away_elo': 1510},  # Even matchup
        {'home': 'SJS', 'away': 'VGK', 'home_elo': 1420, 'away_elo': 1540},  # Weak vs strong
        {'home': 'WSH', 'away': 'MIN', 'home_elo': 1480, 'away_elo': 1490},  # Even (recent game)
    ]
    
    print("\n" + "="*80)
    print("VALIDATION PREDICTIONS")
    print("="*80)
    
    for matchup in test_matchups:
        home = matchup['home']
        away = matchup['away']
        home_elo = matchup['home_elo']
        away_elo = matchup['away_elo']
        
        features = prepare_test_features(home, away, home_elo, away_elo)
        
        print(f"\n{home} (Elo: {home_elo}) vs {away} (Elo: {away_elo})")
        print("-" * 60)
        
        # Total goals prediction
        if 'TOTAL_GOALS' in models:
            try:
                total_pred = models['TOTAL_GOALS'].predict(home, away, features)
                print(f"  Total Goals:     {total_pred:.2f}")
            except Exception as e:
                print(f"  Total Goals:     ERROR - {e}")
        
        # Moneyline prediction
        if 'MONEYLINE' in models:
            try:
                ml_pred = models['MONEYLINE'].predict(home, away, features)
                print(f"  Home Win Prob:   {ml_pred*100:.1f}%")
                print(f"  Away Win Prob:   {(1-ml_pred)*100:.1f}%")
            except Exception as e:
                print(f"  Moneyline:       ERROR - {e}")
        
        # Goal differential prediction
        if 'GOAL_DIFF' in models:
            try:
                diff_pred = models['GOAL_DIFF'].predict(home, away, features)
                # Use predicted total if available
                total_used = total_pred if 'TOTAL_GOALS' in models else 6.0
                home_proj = (total_used / 2) + (diff_pred / 2)
                away_proj = (total_used / 2) - (diff_pred / 2)
                print(f"  Goal Diff:       {diff_pred:+.2f} (favors {'home' if diff_pred > 0 else 'away'})")
                print(f"  Score Proj:      {home} {home_proj:.2f} - {away} {away_proj:.2f}")
            except Exception as e:
                print(f"  Goal Diff:       ERROR - {e}")

def main():
    """Run validation."""
    print("[validate] Loading trained models...")
    models = load_models()
    
    if not models:
        print("[error] No models loaded. Train models first.")
        return
    
    print(f"[validate] Loaded {len(models)} model(s)")
    
    test_predictions(models)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check if predictions are reasonable (4-7 total goals)")
    print("  2. Verify team differentiation (strong teams favored)")
    print("  3. Compare vs current Elo/Poisson projections")
    print("  4. Backtest on Oct 1-17 games if predictions look good")

if __name__ == '__main__':
    main()

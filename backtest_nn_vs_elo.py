"""Backtest NN game models vs Elo/Poisson on recent games (Oct 1-17, 2025)."""
import sys
from pathlib import Path

# CRITICAL: Import torch before pandas
try:
    import torch
    print(f"[info] PyTorch loaded: {torch.__version__}")
except Exception as e:
    print(f"[warn] PyTorch unavailable: {e}")
    torch = None

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

from nhl_betting.models.nn_games import NNGameModel
from nhl_betting.models.elo import Elo
from nhl_betting.models.poisson import PoissonGoals
from nhl_betting.utils.io import RAW_DIR, MODEL_DIR

def load_nn_models():
    """Load trained NN models."""
    models = {}
    for model_type in ['TOTAL_GOALS', 'MONEYLINE', 'GOAL_DIFF']:
        try:
            model = NNGameModel(model_type=model_type, model_dir=MODEL_DIR / 'nn_games')
            if model.model is not None or model.onnx_session is not None:
                models[model_type] = model
                print(f"[load] ✓ NN {model_type} model loaded")
        except Exception as e:
            print(f"[error] Failed to load NN {model_type}: {e}")
    return models

def prepare_features_for_game(game_row):
    """Prepare 95-feature vector for a game from the dataset."""
    features = {
        # Elo features
        'home_elo': float(game_row.get('home_elo', 1500)),
        'away_elo': float(game_row.get('away_elo', 1500)),
        'elo_diff': float(game_row.get('home_elo', 1500)) - float(game_row.get('away_elo', 1500)),
        
        # Recent form
        'home_goals_last10': float(game_row.get('home_goals_last10', 3.0)),
        'home_goals_against_last10': float(game_row.get('home_goals_against_last10', 3.0)),
        'home_wins_last10': float(game_row.get('home_wins_last10', 5)),
        'away_goals_last10': float(game_row.get('away_goals_last10', 3.0)),
        'away_goals_against_last10': float(game_row.get('away_goals_against_last10', 3.0)),
        'away_wins_last10': float(game_row.get('away_wins_last10', 5)),
        
        # Rest days
        'home_rest_days': float(game_row.get('home_rest_days', 1)),
        'away_rest_days': float(game_row.get('away_rest_days', 1)),
        
        # Season context
        'season_progress': float(game_row.get('games_played_season', 0)) / 82.0,
        
        # Home ice
        'is_home': 1.0,
    }
    
    return features

def get_elo_poisson_predictions(game_row):
    """Get Elo/Poisson predictions for comparison."""
    home_elo = float(game_row.get('home_elo', 1500))
    away_elo = float(game_row.get('away_elo', 1500))
    
    # Elo win probability - use Elo.expected() method
    elo = Elo()
    elo.ratings[game_row['home']] = home_elo
    elo.ratings[game_row['away']] = away_elo
    p_home_elo = elo.expected(game_row['home'], game_row['away'], is_home=True)
    
    # Poisson total goals (using 70/30 split from total_line - the problematic approach)
    poisson = PoissonGoals(base_mu=3.05)
    # Use a default total of 6.0 (base_mu * 2 roughly)
    total_line = 6.0
    home_lambda, away_lambda = poisson.lambdas_from_total_split(total_line, p_home_elo)
    total_poisson = home_lambda + away_lambda
    diff_poisson = home_lambda - away_lambda
    
    return {
        'p_home_elo': p_home_elo,
        'total_poisson': total_poisson,
        'home_proj_poisson': home_lambda,
        'away_proj_poisson': away_lambda,
        'diff_poisson': diff_poisson,
    }

def calculate_metrics(predictions_df):
    """Calculate accuracy metrics for both systems."""
    metrics = {}
    
    # Moneyline accuracy
    nn_ml_correct = (predictions_df['home_won'] == (predictions_df['p_home_nn'] > 0.5)).sum()
    elo_ml_correct = (predictions_df['home_won'] == (predictions_df['p_home_elo'] > 0.5)).sum()
    total = len(predictions_df)
    
    metrics['nn_ml_accuracy'] = nn_ml_correct / total
    metrics['elo_ml_accuracy'] = elo_ml_correct / total
    
    # Total goals RMSE
    nn_total_errors = (predictions_df['actual_total'] - predictions_df['total_nn']) ** 2
    poisson_total_errors = (predictions_df['actual_total'] - predictions_df['total_poisson']) ** 2
    
    metrics['nn_total_rmse'] = np.sqrt(nn_total_errors.mean())
    metrics['poisson_total_rmse'] = np.sqrt(poisson_total_errors.mean())
    
    # Goal differential RMSE
    nn_diff_errors = (predictions_df['actual_diff'] - predictions_df['diff_nn']) ** 2
    poisson_diff_errors = (predictions_df['actual_diff'] - predictions_df['diff_poisson']) ** 2
    
    metrics['nn_diff_rmse'] = np.sqrt(nn_diff_errors.mean())
    metrics['poisson_diff_rmse'] = np.sqrt(poisson_diff_errors.mean())
    
    # Calibration (Brier score for ML predictions)
    nn_brier = ((predictions_df['p_home_nn'] - predictions_df['home_won'].astype(float)) ** 2).mean()
    elo_brier = ((predictions_df['p_home_elo'] - predictions_df['home_won'].astype(float)) ** 2).mean()
    
    metrics['nn_brier'] = nn_brier
    metrics['elo_brier'] = elo_brier
    
    # Check for extreme projections (< 1.5 or > 5.0 goals per team)
    nn_extremes = ((predictions_df['home_proj_nn'] < 1.5) | (predictions_df['home_proj_nn'] > 5.0) |
                   (predictions_df['away_proj_nn'] < 1.5) | (predictions_df['away_proj_nn'] > 5.0)).sum()
    poisson_extremes = ((predictions_df['home_proj_poisson'] < 1.5) | (predictions_df['home_proj_poisson'] > 5.0) |
                        (predictions_df['away_proj_poisson'] < 1.5) | (predictions_df['away_proj_poisson'] > 5.0)).sum()
    
    metrics['nn_extreme_projections'] = nn_extremes
    metrics['poisson_extreme_projections'] = poisson_extremes
    
    return metrics

def main():
    """Run backtest comparison."""
    print("="*80)
    print("BACKTEST: NN Models vs Elo/Poisson (Oct 1-17, 2025)")
    print("="*80)
    
    # Load NN models
    print("\n[1] Loading NN models...")
    nn_models = load_nn_models()
    
    if len(nn_models) < 3:
        print("[error] Not all NN models loaded. Need TOTAL_GOALS, MONEYLINE, GOAL_DIFF")
        return
    
    # Load test games (Oct 1-17, 2025)
    print("\n[2] Loading test games...")
    games_file = RAW_DIR / 'games_with_features.csv'
    games_df = pd.read_csv(games_file)
    games_df['date'] = pd.to_datetime(games_df['date'])
    
    # Filter to Oct 1-17, 2025
    test_games = games_df[
        (games_df['date'] >= '2025-10-01') & 
        (games_df['date'] <= '2025-10-17')
    ].copy()
    
    print(f"[load] Found {len(test_games)} games in test period")
    
    if len(test_games) == 0:
        print("[warn] No games found in test period. Using last 30 games instead.")
        test_games = games_df.tail(30).copy()
    
    # Generate predictions for each game
    print("\n[3] Generating predictions...")
    predictions = []
    
    for idx, game in test_games.iterrows():
        home = game['home']
        away = game['away']
        
        # Actual results
        actual_home_goals = float(game.get('home_goals', game.get('final_home_goals', 0)))
        actual_away_goals = float(game.get('away_goals', game.get('final_away_goals', 0)))
        actual_total = actual_home_goals + actual_away_goals
        actual_diff = actual_home_goals - actual_away_goals
        home_won = actual_home_goals > actual_away_goals
        
        # Prepare features
        features = prepare_features_for_game(game)
        
        # NN predictions
        try:
            p_home_nn = nn_models['MONEYLINE'].predict(home, away, features)
            total_nn = nn_models['TOTAL_GOALS'].predict(home, away, features)
            diff_nn = nn_models['GOAL_DIFF'].predict(home, away, features)
            
            home_proj_nn = (total_nn / 2) + (diff_nn / 2)
            away_proj_nn = (total_nn / 2) - (diff_nn / 2)
        except Exception as e:
            print(f"[error] NN prediction failed for {home} vs {away}: {e}")
            continue
        
        # Elo/Poisson predictions
        elo_poisson = get_elo_poisson_predictions(game)
        
        predictions.append({
            'date': game['date'],
            'home': home,
            'away': away,
            'actual_home_goals': actual_home_goals,
            'actual_away_goals': actual_away_goals,
            'actual_total': actual_total,
            'actual_diff': actual_diff,
            'home_won': home_won,
            # NN predictions
            'p_home_nn': p_home_nn,
            'total_nn': total_nn,
            'diff_nn': diff_nn,
            'home_proj_nn': home_proj_nn,
            'away_proj_nn': away_proj_nn,
            # Elo/Poisson predictions
            'p_home_elo': elo_poisson['p_home_elo'],
            'total_poisson': elo_poisson['total_poisson'],
            'diff_poisson': elo_poisson['diff_poisson'],
            'home_proj_poisson': elo_poisson['home_proj_poisson'],
            'away_proj_poisson': elo_poisson['away_proj_poisson'],
        })
    
    predictions_df = pd.DataFrame(predictions)
    print(f"[done] Generated {len(predictions_df)} predictions")
    
    # Calculate metrics
    print("\n[4] Calculating metrics...")
    metrics = calculate_metrics(predictions_df)
    
    # Display results
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    
    print("\n📊 MONEYLINE ACCURACY:")
    print(f"  NN Model:        {metrics['nn_ml_accuracy']*100:.1f}%")
    print(f"  Elo Model:       {metrics['elo_ml_accuracy']*100:.1f}%")
    winner = "NN" if metrics['nn_ml_accuracy'] > metrics['elo_ml_accuracy'] else "Elo"
    diff = abs(metrics['nn_ml_accuracy'] - metrics['elo_ml_accuracy']) * 100
    print(f"  Winner:          {winner} (+{diff:.1f}%)")
    
    print("\n📊 TOTAL GOALS RMSE (lower is better):")
    print(f"  NN Model:        {metrics['nn_total_rmse']:.3f} goals")
    print(f"  Poisson Model:   {metrics['poisson_total_rmse']:.3f} goals")
    winner = "NN" if metrics['nn_total_rmse'] < metrics['poisson_total_rmse'] else "Poisson"
    diff = abs(metrics['nn_total_rmse'] - metrics['poisson_total_rmse'])
    print(f"  Winner:          {winner} (-{diff:.3f} goals)")
    
    print("\n📊 GOAL DIFFERENTIAL RMSE (lower is better):")
    print(f"  NN Model:        {metrics['nn_diff_rmse']:.3f} goals")
    print(f"  Poisson Model:   {metrics['poisson_diff_rmse']:.3f} goals")
    winner = "NN" if metrics['nn_diff_rmse'] < metrics['poisson_diff_rmse'] else "Poisson"
    diff = abs(metrics['nn_diff_rmse'] - metrics['poisson_diff_rmse'])
    print(f"  Winner:          {winner} (-{diff:.3f} goals)")
    
    print("\n📊 CALIBRATION - Brier Score (lower is better):")
    print(f"  NN Model:        {metrics['nn_brier']:.4f}")
    print(f"  Elo Model:       {metrics['elo_brier']:.4f}")
    winner = "NN" if metrics['nn_brier'] < metrics['elo_brier'] else "Elo"
    diff = abs(metrics['nn_brier'] - metrics['elo_brier'])
    print(f"  Winner:          {winner} (-{diff:.4f})")
    
    print("\n📊 EXTREME PROJECTIONS (<1.5 or >5.0 goals per team):")
    print(f"  NN Model:        {metrics['nn_extreme_projections']} games")
    print(f"  Poisson Model:   {metrics['poisson_extreme_projections']} games")
    print(f"  Improvement:     {metrics['poisson_extreme_projections'] - metrics['nn_extreme_projections']} fewer extreme projections")
    
    # Show sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (First 5 games)")
    print("="*80)
    
    for idx, row in predictions_df.head(5).iterrows():
        print(f"\n{row['date'].strftime('%Y-%m-%d')}: {row['home']} vs {row['away']}")
        print(f"  Actual:          {row['home']} {row['actual_home_goals']:.0f} - {row['away']} {row['actual_away_goals']:.0f}")
        print(f"  NN Projection:   {row['home']} {row['home_proj_nn']:.2f} - {row['away']} {row['away_proj_nn']:.2f} (Win%: {row['p_home_nn']*100:.1f}%)")
        print(f"  Poisson Proj:    {row['home']} {row['home_proj_poisson']:.2f} - {row['away']} {row['away_proj_poisson']:.2f} (Win%: {row['p_home_elo']*100:.1f}%)")
    
    # Save results
    output_file = Path('backtest_results_nn_vs_elo.csv')
    predictions_df.to_csv(output_file, index=False)
    print(f"\n[save] Results saved to {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    nn_wins = sum([
        metrics['nn_ml_accuracy'] > metrics['elo_ml_accuracy'],
        metrics['nn_total_rmse'] < metrics['poisson_total_rmse'],
        metrics['nn_diff_rmse'] < metrics['poisson_diff_rmse'],
        metrics['nn_brier'] < metrics['elo_brier'],
        metrics['nn_extreme_projections'] < metrics['poisson_extreme_projections'],
    ])
    
    if nn_wins >= 4:
        print("\n✅ STRONG RECOMMENDATION: Deploy NN models")
        print("   NN models outperform Elo/Poisson on 4+ of 5 metrics")
        print("   Suggest: Full replacement of Elo/Poisson system")
    elif nn_wins >= 3:
        print("\n✅ RECOMMENDATION: Deploy NN models with ensemble")
        print("   NN models outperform on 3+ metrics")
        print("   Suggest: 70% NN + 30% Elo/Poisson ensemble")
    elif nn_wins >= 2:
        print("\n⚠️  NEUTRAL: Both systems have strengths")
        print("   Consider hybrid approach or further tuning")
    else:
        print("\n❌ NOT RECOMMENDED: Stick with Elo/Poisson for now")
        print("   NN models need more work or different architecture")
    
    print(f"\n   Score: NN wins {nn_wins}/5 metrics")

if __name__ == '__main__':
    main()

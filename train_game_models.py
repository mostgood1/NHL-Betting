"""Train NN game models with proper import order to avoid DLL conflicts."""
import sys
from pathlib import Path

# CRITICAL: Import torch/onnxruntime BEFORE pandas/numpy
try:
    import torch
    print(f"[info] PyTorch loaded: {torch.__version__}")
except (ImportError, OSError) as e:
    print(f"[warn] PyTorch unavailable: {e}")
    torch = None

try:
    import onnxruntime
    print(f"[info] ONNX Runtime loaded: {onnxruntime.__version__}")
except (ImportError, OSError) as e:
    print(f"[warn] ONNX Runtime unavailable: {e}")
    onnxruntime = None

# Now safe to import pandas/numpy
import pandas as pd
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from nhl_betting.models.nn_games import NNGameModel, NNGamesConfig
from nhl_betting.utils.io import RAW_DIR, MODEL_DIR


def train_model(model_type: str, epochs: int = 100, verbose: bool = True):
    """Train a single game prediction model."""
    
    if torch is None:
        print(f"[error] PyTorch not available - cannot train")
        return False
    
    print(f"\n{'='*80}")
    print(f"Training {model_type} Model")
    print(f"{'='*80}\n")
    
    # Load data
    games_csv = RAW_DIR / "games_with_features.csv"
    if not games_csv.exists():
        print(f"[error] Training data not found: {games_csv}")
        return False
    
    print(f"[load] Reading {games_csv}...")
    df = pd.read_csv(games_csv)
    print(f"[load] Loaded {len(df)} games")
    
    # Configure model
    task = "classification" if model_type == "MONEYLINE" else "regression"
    cfg = NNGamesConfig(
        hidden_dims=[128, 64, 32],
        epochs=epochs,
        batch_size=64,
        learning_rate=0.0005,
        dropout=0.3,
        task=task,
        include_team_encoding=True,
        include_recent_form=True,
        include_rest_days=True,
        include_time_of_season=True,
    )
    
    # Initialize model
    model_dir = MODEL_DIR / "nn_games"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    nn_model = NNGameModel(model_type=model_type, cfg=cfg, model_dir=model_dir)
    
    print(f"[train] Configuration:")
    print(f"  Hidden dims: {cfg.hidden_dims}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.learning_rate}")
    print(f"  Task: {task}")
    print(f"  Dropout: {cfg.dropout}")
    
    try:
        # Train model
        metrics = nn_model.train(df, validation_split=0.2, verbose=verbose)
        
        print(f"\n[done] Training complete!")
        print(f"  Best validation loss: {metrics['best_val_loss']:.4f}")
        print(f"  Training samples: {metrics['samples']}")
        print(f"  Features: {metrics['features']}")
        print(f"  Model saved: {nn_model._get_model_path()}")
        
        # Try ONNX export
        try:
            onnx_path = nn_model.export_onnx()
            print(f"  ONNX exported: {onnx_path}")
        except Exception as e:
            print(f"  [warn] ONNX export failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"[error] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NN game models")
    parser.add_argument("--model", type=str, default="all", 
                       help="Model to train: TOTAL_GOALS, MONEYLINE, GOAL_DIFF, or 'all'")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.model.lower() == "all":
        models = ["TOTAL_GOALS", "MONEYLINE", "GOAL_DIFF"]
    else:
        models = [args.model.upper()]
    
    print(f"[info] Training {len(models)} model(s)...\n")
    
    results = {}
    for model_type in models:
        success = train_model(model_type, epochs=args.epochs, verbose=args.verbose)
        results[model_type] = "✓" if success else "✗"
    
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    for model_type, status in results.items():
        print(f"  {status} {model_type}")
    
    all_success = all(status == "✓" for status in results.values())
    sys.exit(0 if all_success else 1)

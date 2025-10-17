"""Train neural network models for game outcome prediction and export to ONNX.

This script trains PyTorch models on historical game data and exports them
to ONNX format for NPU-accelerated inference via Qualcomm QNN.

Models:
- MONEYLINE: Home team win probability
- TOTAL_GOALS: Total goals in game
- GOAL_DIFF: Score differential (home - away)
- FIRST_10MIN: Goals in first 10 minutes
- PERIOD_GOALS: Goals per period for each team

Usage:
    python -m nhl_betting.scripts.train_nn_games train --model MONEYLINE --epochs 100
    python -m nhl_betting.scripts.train_nn_games train-all --epochs 100
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer
import pandas as pd
from rich import print

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nhl_betting.models.nn_games import NNGameModel, NNGamesConfig
from nhl_betting.utils.io import RAW_DIR, MODEL_DIR


app = typer.Typer(help="Train neural network models for game outcomes")


@app.command()
def train(
    model: str = typer.Option(
        ...,
        help="Model to train: MONEYLINE, TOTAL_GOALS, GOAL_DIFF, FIRST_10MIN, PERIOD_GOALS"
    ),
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(64, help="Training batch size"),
    learning_rate: float = typer.Option(0.0005, help="Learning rate"),
    hidden_dims: str = typer.Option("128,64,32", help="Comma-separated hidden layer sizes"),
    verbose: bool = typer.Option(True, help="Print training progress"),
    prepare_data: bool = typer.Option(False, help="Prepare training data first"),
):
    """Train a neural network model for a specific game prediction task."""
    
    model = model.upper()
    print(f"[info] Training {model} model...")
    
    # Prepare data if requested
    if prepare_data:
        print(f"[info] Preparing training data...")
        from nhl_betting.data.game_features import prepare_game_features
        prepare_game_features()
    
    # Load training data
    games_csv = RAW_DIR / "games_with_features.csv"
    if not games_csv.exists():
        print(f"[error] Training data not found: {games_csv}")
        print(f"[info] Run with --prepare-data flag to generate features")
        raise typer.Exit(1)
    
    print(f"[load] Reading {games_csv}...")
    df = pd.read_csv(games_csv)
    print(f"[load] Loaded {len(df)} games")
    
    # Parse hidden dims
    hidden_dims_list = [int(x.strip()) for x in hidden_dims.split(",")]
    
    # Determine task type
    if model in ["MONEYLINE"]:
        task = "classification"
    else:
        task = "regression"
    
    # Configure model
    cfg = NNGamesConfig(
        hidden_dims=hidden_dims_list,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        task=task,
    )
    
    # Initialize and train
    model_dir = MODEL_DIR / "nn_games"
    nn_model = NNGameModel(model_type=model, cfg=cfg, model_dir=model_dir)
    
    print(f"[train] Training {model} model...")
    print(f"  Hidden dims: {hidden_dims_list}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Task: {task}")
    
    try:
        metrics = nn_model.train(df, validation_split=0.2, verbose=verbose)
        
        print(f"[done] Training complete!")
        print(f"  Best validation loss: {metrics['best_val_loss']:.4f}")
        print(f"  Training samples: {metrics['samples']}")
        print(f"  Features: {metrics['features']}")
        print(f"  Model saved: {nn_model._get_model_path()}")
        print(f"  ONNX exported: {nn_model._get_onnx_path()}")
        
    except Exception as e:
        print(f"[error] Training failed: {e}")
        raise typer.Exit(1)


@app.command()
def train_all(
    epochs: int = typer.Option(100, help="Number of training epochs"),
    batch_size: int = typer.Option(64, help="Training batch size"),
    learning_rate: float = typer.Option(0.0005, help="Learning rate"),
    hidden_dims: str = typer.Option("128,64,32", help="Comma-separated hidden layer sizes"),
    prepare_data: bool = typer.Option(False, help="Prepare training data first"),
):
    """Train all game prediction models."""
    
    models = ["MONEYLINE", "TOTAL_GOALS", "GOAL_DIFF", "FIRST_10MIN", "PERIOD_GOALS"]
    
    print(f"[info] Training models for {len(models)} tasks...")
    
    # Prepare data once if requested
    if prepare_data:
        print(f"[info] Preparing training data...")
        from nhl_betting.data.game_features import prepare_game_features
        prepare_game_features()
    
    results = {}
    
    for model in models:
        print(f"\n{'='*60}")
        print(f"Training {model}...")
        print(f"{'='*60}")
        
        try:
            # Call train command
            train.callback(
                model=model,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_dims=hidden_dims,
                verbose=False,
                prepare_data=False,
            )
            results[model] = "✓ Success"
        except Exception as e:
            results[model] = f"✗ Failed: {e}"
            print(f"[error] {model} failed: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Training Summary:")
    print(f"{'='*60}")
    for model, status in results.items():
        print(f"  {model:15s} {status}")


@app.command()
def benchmark(
    model: str = typer.Option(..., help="Model to benchmark"),
    num_runs: int = typer.Option(1000, help="Number of inference runs"),
):
    """Benchmark NPU vs CPU inference speed."""
    
    model = model.upper()
    print(f"[benchmark] Testing {model} model inference speed...")
    
    # Load model
    model_dir = MODEL_DIR / "nn_games"
    nn_model = NNGameModel(model_type=model, model_dir=model_dir)
    
    if nn_model.model is None:
        print(f"[error] Model not trained yet")
        raise typer.Exit(1)
    
    # Generate dummy features
    dummy_features = {col: 0.0 for col in nn_model.feature_columns}
    dummy_features["home_elo"] = 1500
    dummy_features["away_elo"] = 1500
    
    # Benchmark PyTorch CPU
    import time
    
    print(f"[benchmark] Running {num_runs} inferences with PyTorch (CPU)...")
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = nn_model.predict("HOME", "AWAY", dummy_features)
    cpu_time = time.perf_counter() - start
    
    print(f"\n{'='*60}")
    print("Benchmark Results:")
    print(f"{'='*60}")
    print(f"  CPU (PyTorch):  {cpu_time:.4f}s total, {cpu_time/num_runs*1000:.4f}ms per inference")
    print(f"  Throughput:     {num_runs/cpu_time:.1f} inferences/sec")
    
    # TODO: Add ONNX Runtime NPU benchmarking once nn_inference supports game models
    print(f"\n[note] NPU benchmarking requires ONNX Runtime integration")
    print(f"[note] ONNX model available at: {nn_model._get_onnx_path()}")


if __name__ == "__main__":
    app()

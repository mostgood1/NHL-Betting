"""Train neural network models for player props prediction and export to ONNX.

This script trains PyTorch models on historical player game stats and exports them
to ONNX format for NPU-accelerated inference via Qualcomm QNN.

Usage:
    python -m nhl_betting.scripts.train_nn_props --market SOG --epochs 50
    python -m nhl_betting.scripts.train_nn_props --market GOALS --verbose
    python -m nhl_betting.scripts.train_nn_props --all  # train all markets
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer
import pandas as pd
from rich import print

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nhl_betting.models.nn_props import NNPropsModel, NNPropsConfig
from nhl_betting.utils.io import RAW_DIR, MODEL_DIR
from nhl_betting.data.collect import collect_player_game_stats


app = typer.Typer(help="Train neural network models for player props")


@app.command()
def train(
    market: str = typer.Option(
        ...,
        help="Market to train: SOG, GOALS, ASSISTS, POINTS, SAVES, BLOCKS"
    ),
    epochs: int = typer.Option(50, help="Number of training epochs"),
    batch_size: int = typer.Option(128, help="Training batch size"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    hidden_dims: str = typer.Option("64,32", help="Comma-separated hidden layer sizes"),
    window: int = typer.Option(10, help="Number of recent games for rolling features"),
    verbose: bool = typer.Option(True, help="Print training progress"),
    backfill: bool = typer.Option(False, help="Backfill player stats before training"),
):
    """Train a neural network model for a specific market."""
    
    if backfill:
        print("[backfill] Collecting player game stats...")
        try:
            # Backfill last 2 seasons
            from datetime import datetime, timedelta
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
            collect_player_game_stats(start, end, source="stats")
        except Exception as e:
            print(f"[warn] Backfill failed: {e}")
    
    # Load historical data
    stats_path = RAW_DIR / "player_game_stats.csv"
    if not stats_path.exists():
        print(f"[error] Player stats not found: {stats_path}")
        print("Run: python -m nhl_betting.cli collect_props --start 2023-01-01 --end 2025-10-16")
        raise typer.Exit(code=1)
    
    print(f"[load] Reading {stats_path}...")
    df = pd.read_csv(stats_path)
    print(f"[load] Loaded {len(df)} player-game records")
    
    # Configure model
    hidden_dims_list = [int(x) for x in hidden_dims.split(",")]
    cfg = NNPropsConfig(
        hidden_dims=hidden_dims_list,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        window_games=window,
    )
    
    # Initialize and train
    model_dir = MODEL_DIR / "nn_props"
    model = NNPropsModel(market=market, cfg=cfg, model_dir=model_dir)
    
    print(f"[train] Training {market} model...")
    print(f"  Hidden dims: {hidden_dims_list}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    try:
        metrics = model.train(df, validation_split=0.2, verbose=verbose)
        
        print(f"[done] Training complete!")
        print(f"  Best validation loss: {metrics['best_val_loss']:.4f}")
        print(f"  Training samples: {metrics['samples']}")
        print(f"  Features: {metrics['features']}")
        print(f"  Model saved: {model._get_model_path()}")
        print(f"  ONNX exported: {model._get_onnx_path()}")
        
    except Exception as e:
        print(f"[error] Training failed: {e}")
        raise typer.Exit(code=1)


@app.command()
def train_all(
    epochs: int = typer.Option(50, help="Number of training epochs"),
    verbose: bool = typer.Option(False, help="Print detailed training progress"),
):
    """Train neural network models for all markets."""
    markets = ["SOG", "GOALS", "ASSISTS", "POINTS", "SAVES", "BLOCKS"]
    
    print(f"[train_all] Training models for {len(markets)} markets...")
    
    results = {}
    for market in markets:
        print(f"\n{'='*60}")
        print(f"Training {market}...")
        print(f"{'='*60}")
        
        try:
            train(
                market=market,
                epochs=epochs,
                batch_size=128,
                learning_rate=0.001,
                hidden_dims="64,32",
                window=10,
                verbose=verbose,
                backfill=False,
            )
            results[market] = "✓ Success"
        except Exception as e:
            results[market] = f"✗ Failed: {e}"
    
    print(f"\n{'='*60}")
    print("Training Summary:")
    print(f"{'='*60}")
    for market, status in results.items():
        print(f"  {market:12s} {status}")


@app.command()
def benchmark(
    market: str = typer.Option("SOG", help="Market to benchmark"),
    num_runs: int = typer.Option(100, help="Number of inference runs"),
):
    """Benchmark CPU vs NPU inference performance."""
    from nhl_betting.models.nn_inference import benchmark_inference, check_npu_availability
    
    # Check NPU availability first
    npu_info = check_npu_availability()
    print("\nNPU Availability:")
    for k, v in npu_info.items():
        print(f"  {k}: {v}")
    
    model_dir = MODEL_DIR / "nn_props"
    onnx_path = model_dir / f"{market.lower()}_model.onnx"
    
    if not onnx_path.exists():
        print(f"\n[error] ONNX model not found: {onnx_path}")
        print(f"Train first: python -m nhl_betting.scripts.train_nn_props train --market {market}")
        raise typer.Exit(code=1)
    
    print(f"\nBenchmarking {market} model...")
    print(f"  ONNX path: {onnx_path}")
    print(f"  Runs: {num_runs}")
    
    # Get input shape from metadata
    import numpy as np
    meta_path = model_dir / f"{market.lower()}_metadata.npz"
    meta = np.load(meta_path, allow_pickle=True)
    feature_count = len(meta["feature_columns"])
    
    results = benchmark_inference(
        onnx_path=onnx_path,
        input_shape=(1, feature_count),
        num_runs=num_runs,
        warmup_runs=10,
    )
    
    print("\nBenchmark Results:")
    print(f"{'='*60}")
    
    for device, metrics in results.items():
        if device == "speedup":
            continue
        print(f"\n{device}:")
        if "error" in metrics:
            print(f"  Error: {metrics['error']}")
        else:
            print(f"  Avg time: {metrics['avg_time_ms']:.3f} ms")
            print(f"  Throughput: {metrics['throughput_per_sec']:.1f} inferences/sec")
            print(f"  Providers: {metrics['providers']}")
    
    if "speedup" in results:
        speedup = results["speedup"]
        print(f"\nSpeedup (NPU vs CPU): {speedup:.2f}x")
        if speedup > 1.5:
            print("✓ NPU is significantly faster!")
        elif speedup > 1.0:
            print("NPU is slightly faster.")
        else:
            print("Note: CPU is faster for this model size. NPU benefits larger models/batches.")


if __name__ == "__main__":
    app()

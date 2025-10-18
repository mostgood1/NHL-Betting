"""Neural network models for game outcome prediction using PyTorch.

Models for:
- Game outcomes (win/loss, totals, score differential)
- Time-based predictions (first 10 min goals, period goals)
- Team-specific period performance

Can be trained on historical data and exported to ONNX for NPU acceleration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# CRITICAL: Import onnxruntime and torch BEFORE numpy/pandas
# NumPy's MKL DLLs can interfere with ONNX Runtime's DLL loading
# This must be done at module level, not in try/except yet

# Try to import ONNX Runtime first (before numpy loads MKL DLLs)
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print(f"[info] ONNX Runtime available: {ort.__version__}")
except Exception as e:
    ONNX_AVAILABLE = False
    ort = None
    print(f"[warn] ONNX Runtime not available: {e}")

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    print(f"[info] PyTorch available: {torch.__version__}")
except Exception as e:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    print(f"[warn] PyTorch not available: {e}")

# NOW import numpy/pandas (after onnx/torch are loaded)
import numpy as np
import pandas as pd


@dataclass
class NNGamesConfig:
    """Configuration for neural network game models."""
    # Model architecture
    hidden_dims: List[int] = None  # [128, 64, 32] by default
    dropout: float = 0.3
    activation: str = "relu"  # relu, gelu, elu
    
    # Training
    learning_rate: float = 0.0005
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    
    # Features
    include_recent_form: bool = True  # last N games stats
    recent_games: int = 10
    include_h2h: bool = True  # head-to-head history
    include_roster_strength: bool = True  # aggregate player stats
    include_rest_days: bool = True
    include_time_of_season: bool = True
    include_team_encoding: bool = True  # one-hot encode teams for team-specific learning
    
    # Output type
    task: str = "classification"  # classification, regression, multi_output

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]


# Only define PyTorch model classes if torch is available
if TORCH_AVAILABLE:
    class GameOutcomeNN(nn.Module):
        """Neural network for game outcome prediction.
        
        Can be used for:
        - Binary classification (win/loss)
        - Regression (goal differential, total goals)
        - Multi-output (win prob + total goals)
        """
        
        def __init__(self, input_dim: int, output_dim: int, cfg: NNGamesConfig | None = None):
            super().__init__()
            self.cfg = cfg or NNGamesConfig()
            self.output_dim = output_dim
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in self.cfg.hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                
                if self.cfg.activation == "relu":
                    layers.append(nn.ReLU())
                elif self.cfg.activation == "gelu":
                    layers.append(nn.GELU())
                elif self.cfg.activation == "elu":
                    layers.append(nn.ELU())
                else:
                    layers.append(nn.ReLU())
                    
                layers.append(nn.Dropout(self.cfg.dropout))
                prev_dim = hidden_dim
            
            # Output layer
            layers.append(nn.Linear(prev_dim, output_dim))
            
            # Output activation depends on task
            if self.cfg.task == "classification":
                if output_dim == 1:
                    layers.append(nn.Sigmoid())  # binary
                else:
                    layers.append(nn.Softmax(dim=-1))  # multiclass
            elif self.cfg.task == "regression":
                # No activation for regression (can be negative)
                pass
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.network(x)
            if self.output_dim == 1:
                return out.squeeze(-1)
            return out


    class PeriodGoalsNN(nn.Module):
        """Neural network for predicting goals per period.
        
        Multi-output: [period1_home, period1_away, period2_home, period2_away, period3_home, period3_away]
        """
        
        def __init__(self, input_dim: int, cfg: NNGamesConfig | None = None):
            super().__init__()
            self.cfg = cfg or NNGamesConfig()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in self.cfg.hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.cfg.dropout))
                prev_dim = hidden_dim
            
            # Output: 6 values (3 periods × 2 teams), use softplus for positive output
            layers.append(nn.Linear(prev_dim, 6))
            layers.append(nn.Softplus())
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)
else:
    # Dummy classes when torch is not available (ONNX-only mode)
    class GameOutcomeNN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available - use ONNX models instead")
    
    class PeriodGoalsNN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available - use ONNX models instead")

class NNGameModel:
    """Wrapper for training and inference with game outcome neural networks.
    
    Supports multiple prediction tasks:
    - MONEYLINE: Home team win probability
    - TOTAL_GOALS: Total goals in game
    - GOAL_DIFF: Score differential (home - away)
    - FIRST_10MIN: Goals in first 10 minutes
    - PERIOD_GOALS: Goals per period for each team
    """
    
    def __init__(
        self,
        model_type: str,  # MONEYLINE, TOTAL_GOALS, GOAL_DIFF, FIRST_10MIN, PERIOD_GOALS
        cfg: NNGamesConfig | None = None,
        model_dir: Path | None = None,
    ):
        self.model_type = model_type.upper()
        self.cfg = cfg or NNGamesConfig()
        self.model_dir = model_dir or Path("data/models/nn_games")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.onnx_session = None  # ONNX Runtime session
        self.feature_columns = []
        self.scaler_mean = None
        self.scaler_std = None
        
        # Try to load existing model (ONNX first, then PyTorch)
        self._try_load()
    
    def _get_model_path(self) -> Path:
        return self.model_dir / f"{self.model_type.lower()}_model.pt"
    
    def _get_onnx_path(self) -> Path:
        return self.model_dir / f"{self.model_type.lower()}_model.onnx"
    
    def _get_metadata_path(self) -> Path:
        return self.model_dir / f"{self.model_type.lower()}_metadata.npz"
    
    def _try_load(self) -> bool:
        """Load model weights and metadata if available. Try ONNX first (works on Windows), then PyTorch."""
        onnx_path = self._get_onnx_path()
        model_path = self._get_model_path()
        meta_path = self._get_metadata_path()
        
        if not meta_path.exists():
            return False
        
        try:
            # Load metadata
            meta = np.load(meta_path, allow_pickle=True)
            self.feature_columns = meta["feature_columns"].tolist()
            self.scaler_mean = meta["scaler_mean"]
            self.scaler_std = meta["scaler_std"]
            
            # Try ONNX first (works on Windows without PyTorch DLL issues)
            if ONNX_AVAILABLE and onnx_path.exists():
                try:
                    self.onnx_session = ort.InferenceSession(
                        str(onnx_path),
                        providers=['CPUExecutionProvider']
                    )
                    print(f"[info] Loaded ONNX model for {self.model_type}")
                    return True
                except Exception as e:
                    print(f"[warn] Failed to load ONNX model for {self.model_type}: {e}")
            
            # Fall back to PyTorch if available
            if TORCH_AVAILABLE and model_path.exists():
                try:
                    input_dim = len(self.feature_columns)
                    
                    # Determine output dimension and architecture
                    if self.model_type == "PERIOD_GOALS":
                        self.model = PeriodGoalsNN(input_dim, self.cfg)
                    else:
                        output_dim = self._get_output_dim()
                        self.model = GameOutcomeNN(input_dim, output_dim, self.cfg)
                    
                    self.model.load_state_dict(torch.load(model_path, weights_only=True))
                    self.model.eval()
                    print(f"[info] Loaded PyTorch model for {self.model_type}")
                    return True
                except Exception as e:
                    print(f"[warn] Failed to load PyTorch model for {self.model_type}: {e}")
            
            return False
        except Exception as e:
            print(f"[warn] Failed to load NN model for {self.model_type}: {e}")
            return False
    
    def _get_output_dim(self) -> int:
        """Get output dimension based on model type."""
        if self.model_type in ["MONEYLINE", "TOTAL_GOALS", "GOAL_DIFF", "FIRST_10MIN"]:
            return 1
        elif self.model_type == "PERIOD_GOALS":
            return 6  # 3 periods × 2 teams
        else:
            return 1
    
    def _prepare_features(
        self,
        games_df: pd.DataFrame,
        player_stats_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Build feature matrix from game data.
        
        Features include:
        - Team Elo ratings
        - Recent form (last N games stats)
        - Head-to-head history
        - Roster strength aggregates
        - Rest days
        - Home ice advantage
        - Time of season
        
        Returns:
            features_df: DataFrame with all features
            targets: numpy array with target values
        """
        features = []
        targets = []
        
        # For each game, compute features
        for idx, game in games_df.iterrows():
            feat_dict = {}
            
            # Basic features
            home_team = game.get("home_team", game.get("home"))
            away_team = game.get("away_team", game.get("away"))
            game_date = pd.to_datetime(game.get("date"))
            
            # Elo ratings (if available in game data or compute from history)
            feat_dict["home_elo"] = game.get("home_elo", 1500)
            feat_dict["away_elo"] = game.get("away_elo", 1500)
            feat_dict["elo_diff"] = feat_dict["home_elo"] - feat_dict["away_elo"]
            
            # Recent form features (last N games)
            if self.cfg.include_recent_form:
                # Home team recent stats
                feat_dict["home_goals_last10"] = game.get("home_goals_last10", 0)
                feat_dict["home_goals_against_last10"] = game.get("home_goals_against_last10", 0)
                feat_dict["home_wins_last10"] = game.get("home_wins_last10", 0)
                
                # Away team recent stats
                feat_dict["away_goals_last10"] = game.get("away_goals_last10", 0)
                feat_dict["away_goals_against_last10"] = game.get("away_goals_against_last10", 0)
                feat_dict["away_wins_last10"] = game.get("away_wins_last10", 0)
            
            # Rest days
            if self.cfg.include_rest_days:
                feat_dict["home_rest_days"] = game.get("home_rest_days", 1)
                feat_dict["away_rest_days"] = game.get("away_rest_days", 1)
            
            # Time of season (normalized 0-1)
            if self.cfg.include_time_of_season:
                season_progress = game.get("games_played_season", 0) / 82.0
                feat_dict["season_progress"] = min(1.0, season_progress)
            
            # Home ice advantage indicator
            feat_dict["is_home"] = 1.0
            
            # Team-specific features (one-hot encoding for team awareness)
            # This allows the model to learn team-specific tendencies
            if self.cfg.include_team_encoding:
                # Add home team indicator
                feat_dict[f"home_team_{home_team}"] = 1.0
                # Add away team indicator
                feat_dict[f"away_team_{away_team}"] = 1.0
            
            features.append(feat_dict)
            
            # Target values based on model type
            if self.model_type == "MONEYLINE":
                home_goals = float(game.get("home_goals", game.get("final_home_goals", 0)))
                away_goals = float(game.get("away_goals", game.get("final_away_goals", 0)))
                target = 1.0 if home_goals > away_goals else 0.0
            elif self.model_type == "TOTAL_GOALS":
                home_goals = float(game.get("home_goals", game.get("final_home_goals", 0)))
                away_goals = float(game.get("away_goals", game.get("final_away_goals", 0)))
                target = home_goals + away_goals
            elif self.model_type == "GOAL_DIFF":
                home_goals = float(game.get("home_goals", game.get("final_home_goals", 0)))
                away_goals = float(game.get("away_goals", game.get("final_away_goals", 0)))
                target = home_goals - away_goals
            elif self.model_type == "FIRST_10MIN":
                target = float(game.get("goals_first_10min", 0))
            elif self.model_type == "PERIOD_GOALS":
                # 6 values: [p1_home, p1_away, p2_home, p2_away, p3_home, p3_away]
                target = [
                    float(game.get("period1_home_goals", 0)),
                    float(game.get("period1_away_goals", 0)),
                    float(game.get("period2_home_goals", 0)),
                    float(game.get("period2_away_goals", 0)),
                    float(game.get("period3_home_goals", 0)),
                    float(game.get("period3_away_goals", 0)),
                ]
            else:
                target = 0.0
            
            targets.append(target)
        
        features_df = pd.DataFrame(features)
        # Fill NaN values with 0 (for missing team encodings and other features)
        features_df = features_df.fillna(0.0)
        targets_array = np.array(targets)
        
        return features_df, targets_array
    
    def train(
        self,
        games_df: pd.DataFrame,
        player_stats_df: Optional[pd.DataFrame] = None,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train the neural network on historical game data.
        
        Args:
            games_df: DataFrame with game results and features
            player_stats_df: Optional player stats for roster aggregates
            validation_split: Fraction of data for validation
            verbose: Print training progress
        
        Returns:
            Dictionary with training metrics
        """
        if verbose:
            print(f"[train] Preparing features for {self.model_type}...")
        
        # Prepare features
        features_df, targets = self._prepare_features(games_df, player_stats_df)
        
        if verbose:
            print(f"[train] Feature matrix shape: {features_df.shape}")
            print(f"[train] Target shape: {targets.shape}")
        
        # Store feature columns
        self.feature_columns = features_df.columns.tolist()
        
        # Convert to numpy
        X = features_df.values.astype(np.float32)
        y = targets.astype(np.float32)
        
        # Drop rows with NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=0 if y.ndim == 1 else 1))
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError(f"No training data available for {self.model_type}")
        
        if verbose:
            print(f"[train] Training samples: {len(X)}")
        
        # Normalize features
        self.scaler_mean = np.mean(X, axis=0)
        self.scaler_std = np.std(X, axis=0) + 1e-8
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # Train/val split
        n = len(X_scaled)
        n_val = int(n * validation_split)
        indices = np.random.permutation(n)
        
        X_train = torch.FloatTensor(X_scaled[indices[n_val:]])
        y_train = torch.FloatTensor(y[indices[n_val:]])
        X_val = torch.FloatTensor(X_scaled[indices[:n_val]])
        y_val = torch.FloatTensor(y[indices[:n_val]])
        
        # Initialize model
        input_dim = len(self.feature_columns)
        if self.model_type == "PERIOD_GOALS":
            self.model = PeriodGoalsNN(input_dim, self.cfg)
        else:
            output_dim = self._get_output_dim()
            self.model = GameOutcomeNN(input_dim, output_dim, self.cfg)
        
        # Loss function
        if self.cfg.task == "classification":
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        
        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.cfg.epochs):
            self.model.train()
            
            # Mini-batch training
            perm = torch.randperm(len(X_train))
            total_loss = 0.0
            
            for i in range(0, len(X_train), self.cfg.batch_size):
                indices = perm[i : i + self.cfg.batch_size]
                batch_X = X_train[indices]
                batch_y = y_train[indices]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
            
            if verbose and (epoch % 10 == 0 or epoch == self.cfg.epochs - 1):
                print(f"[epoch {epoch+1}/{self.cfg.epochs}] train_loss: {total_loss/len(X_train):.4f}, val_loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self._get_model_path())
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.early_stopping_patience:
                    if verbose:
                        print(f"[early stop] No improvement for {self.cfg.early_stopping_patience} epochs")
                    break
        
        # Save metadata
        np.savez(
            self._get_metadata_path(),
            feature_columns=np.array(self.feature_columns, dtype=object),
            scaler_mean=self.scaler_mean,
            scaler_std=self.scaler_std,
        )
        
        # Export to ONNX
        try:
            self.export_onnx()
        except Exception as e:
            import warnings
            warnings.warn(f"ONNX export failed (model still saved as PyTorch): {e}")
            if verbose:
                print(f"[warn] ONNX export skipped: {e}")
        
        return {
            "best_val_loss": best_val_loss,
            "samples": len(X),
            "features": len(self.feature_columns),
        }
    
    def export_onnx(self, opset_version: int = 14) -> Path:
        """Export trained model to ONNX format for NPU inference."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.model.eval()
        dummy_input = torch.randn(1, len(self.feature_columns))
        onnx_path = self._get_onnx_path()
        
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["prediction"],
            dynamic_axes={"features": {0: "batch_size"}, "prediction": {0: "batch_size"}},
        )
        
        print(f"[export] ONNX model saved to {onnx_path}")
        return onnx_path
    
    def predict(
        self,
        home_team: str,
        away_team: str,
        game_features: Dict[str, float],
    ) -> float | np.ndarray:
        """Predict game outcome using the trained model (ONNX or PyTorch).
        
        Args:
            home_team: Home team name
            away_team: Away team name
            game_features: Dictionary of feature values
        
        Returns:
            Prediction (float for single output, array for multi-output)
        """
        if self.model is None and self.onnx_session is None:
            raise ValueError("Model not trained or loaded")
        
        # Add team encoding features if model was trained with them
        features_with_teams = game_features.copy()
        if self.cfg.include_team_encoding:
            # Add team indicators (will be 0.0 if not in feature_columns)
            features_with_teams[f"home_team_{home_team}"] = 1.0
            features_with_teams[f"away_team_{away_team}"] = 1.0
        
        # Build feature vector (uses 0.0 for missing columns)
        X = np.array([features_with_teams.get(col, 0) for col in self.feature_columns], dtype=np.float32)
        
        # Normalize
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # Use ONNX if available (preferred for Windows compatibility)
        if self.onnx_session is not None:
            try:
                # ONNX Runtime inference
                input_name = self.onnx_session.get_inputs()[0].name
                output_name = self.onnx_session.get_outputs()[0].name
                
                # Reshape to (1, n_features) for batch dimension
                X_input = X_scaled.reshape(1, -1)
                
                # Run inference
                pred = self.onnx_session.run([output_name], {input_name: X_input})[0]
                
                if self.model_type == "PERIOD_GOALS":
                    return pred.squeeze(0)  # Return 1D array of 6 values
                else:
                    return float(pred.squeeze())  # Return single value
            except Exception as e:
                print(f"[error] ONNX inference failed: {e}")
                raise
        
        # Fall back to PyTorch if ONNX not available
        if self.model is not None and TORCH_AVAILABLE:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
                pred = self.model(X_tensor)
                
                if self.model_type == "PERIOD_GOALS":
                    return pred.squeeze(0).numpy()
                else:
                    return pred.item()
        
        raise ValueError("No model available for inference")

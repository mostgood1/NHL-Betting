"""Neural network models for player props prediction using PyTorch.

These models can be trained on historical data and exported to ONNX for
NPU-accelerated inference via Qualcomm QNN execution provider.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort


@dataclass
class NNPropsConfig:
    """Configuration for neural network props models."""
    # Model architecture
    hidden_dims: List[int] = None  # [64, 32] by default
    dropout: float = 0.2
    activation: str = "relu"  # relu, gelu, elu
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 50
    early_stopping_patience: int = 5
    
    # Features
    window_games: int = 10  # recent games to include
    include_team_features: bool = True
    include_opponent_features: bool = True
    
    # Output
    predict_lambda: bool = True  # if True, output Poisson lambda; else raw count

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [64, 32]


class PlayerPropsNN(nn.Module):
    """Feedforward neural network for predicting player performance metrics.
    
    Inputs: recent game stats, team context, opponent strength, TOI, etc.
    Output: Poisson lambda parameter (or direct count prediction)
    """
    
    def __init__(self, input_dim: int, cfg: NNPropsConfig | None = None):
        super().__init__()
        self.cfg = cfg or NNPropsConfig()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.cfg.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
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
        
        # Output layer: single value (lambda for Poisson)
        layers.append(nn.Linear(prev_dim, 1))
        # Use softplus to ensure positive output (lambda > 0)
        layers.append(nn.Softplus())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class NNPropsModel:
    """Wrapper for training and inference with PyTorch NN models for props.
    
    Supports training on historical data and exporting to ONNX for NPU acceleration.
    """
    
    def __init__(
        self,
        market: str,  # SOG, GOALS, ASSISTS, POINTS, SAVES, BLOCKS
        cfg: NNPropsConfig | None = None,
        model_dir: Path | None = None,
        use_npu: bool = True,  # Use NPU (ONNX/QNN) vs CPU (PyTorch)
    ):
        self.market = market.upper()
        self.cfg = cfg or NNPropsConfig()
        self.model_dir = model_dir or Path("data/models/nn_props")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_npu = use_npu
        
        self.model: Optional[PlayerPropsNN] = None
        self.onnx_session: Optional[ort.InferenceSession] = None
        self.feature_columns: List[str] = []
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        
        # Try to load existing model if available
        self._try_load()
    
    def _get_model_path(self) -> Path:
        return self.model_dir / f"{self.market.lower()}_model.pt"
    
    def _get_onnx_path(self) -> Path:
        return self.model_dir / f"{self.market.lower()}_model.onnx"
    
    def _get_metadata_path(self) -> Path:
        return self.model_dir / f"{self.market.lower()}_metadata.npz"
    
    def _try_load(self) -> bool:
        """Load model weights and metadata if available."""
        meta_path = self._get_metadata_path()
        
        if not meta_path.exists():
            return False
        
        try:
            # Load metadata
            meta = np.load(meta_path, allow_pickle=True)
            self.feature_columns = meta["feature_columns"].tolist()
            self.scaler_mean = meta["scaler_mean"]
            self.scaler_std = meta["scaler_std"]
            input_dim = len(self.feature_columns)
            
            if self.use_npu:
                # Load ONNX model for NPU inference
                onnx_path = self._get_onnx_path()
                if not onnx_path.exists():
                    print(f"[warn] ONNX model not found for {self.market}, falling back to CPU")
                    self.use_npu = False
                else:
                    # QNN execution provider for Qualcomm NPU
                    qnn_sdk = os.environ.get("QNN_SDK_ROOT", r"C:\Qualcomm\QNN_SDK")
                    providers = [
                        (
                            "QNNExecutionProvider",
                            {
                                "backend_path": f"{qnn_sdk}\\lib\\arm64x-windows-msvc\\QnnHtp.dll",
                                "qnn_context_priority": "high",
                            },
                        ),
                        "CPUExecutionProvider",  # Fallback
                    ]
                    
                    self.onnx_session = ort.InferenceSession(
                        str(onnx_path),
                        providers=providers,
                    )
                    
                    # Verify NPU is being used
                    active_provider = self.onnx_session.get_providers()[0]
                    print(f"[npu] {self.market} model loaded with {active_provider}")
                    return True
            
            if not self.use_npu:
                # Load PyTorch model for CPU inference
                model_path = self._get_model_path()
                if not model_path.exists():
                    return False
                    
                self.model = PlayerPropsNN(input_dim, self.cfg)
                self.model.load_state_dict(torch.load(model_path, weights_only=True))
                self.model.eval()
                print(f"[cpu] {self.market} model loaded with PyTorch CPU")
                return True
                
        except Exception as e:
            print(f"[warn] Failed to load NN model for {self.market}: {e}")
            return False
        
        return False
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        player: str,
        role: str,
    ) -> Optional[pd.DataFrame]:
        """Build feature DataFrame for a player from historical game stats.
        
        Features include:
        - Recent performance (last N games average/std)
        - Team context (one-hot or embeddings)
        - Opponent strength (optional)
        - TOI, position, etc.
        """
        # Check if player_name column already exists (from train method)
        if "player_name" not in df.columns:
            # Parse player names if they're in dictionary format
            import json
            df = df.copy()
            
            # Handle player name format: {'default': 'Name'} or just 'Name'
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
        
        # Filter to player rows
        pdf = df[df["player_name"].astype(str).str.lower() == player.lower()].copy()
        pdf = pdf[pdf["role"].astype(str).str.lower() == role.lower()].copy()
        
        if pdf.empty:
            return None
        
        # Sort by date
        pdf = pdf.sort_values("date")
        
        # Rolling features for target metric
        metric_col = self._get_metric_column()
        
        # Special handling for POINTS - calculate from goals + assists
        if self.market == "POINTS" and metric_col == "points":
            if "goals" in pdf.columns and "assists" in pdf.columns:
                pdf["points"] = pd.to_numeric(pdf["goals"], errors="coerce").fillna(0) + \
                               pd.to_numeric(pdf["assists"], errors="coerce").fillna(0)
            else:
                return None
        
        if metric_col not in pdf.columns:
            return None
        
        pdf[metric_col] = pd.to_numeric(pdf[metric_col], errors="coerce")
        pdf = pdf.dropna(subset=[metric_col])
        
        if len(pdf) < 3:
            return None
        
        # Create rolling features
        features = pd.DataFrame()
        features[f"{metric_col}_mean_{self.cfg.window_games}"] = pdf[metric_col].rolling(
            window=self.cfg.window_games, min_periods=1
        ).mean()
        features[f"{metric_col}_std_{self.cfg.window_games}"] = pdf[metric_col].rolling(
            window=self.cfg.window_games, min_periods=1
        ).std().fillna(0)
        features[f"{metric_col}_last"] = pdf[metric_col].shift(1).fillna(pdf[metric_col].mean())
        
        # Team one-hot encoding (if available)
        if "team" in pdf.columns and self.cfg.include_team_features:
            team_dummies = pd.get_dummies(pdf["team"], prefix="team")
            features = pd.concat([features, team_dummies], axis=1)
        
        # Target
        features["target"] = pdf[metric_col].values
        
        return features.iloc[self.cfg.window_games:]  # skip initial window
    
    def _get_metric_column(self) -> str:
        """Map market to data column name."""
        mapping = {
            "SOG": "shots",
            "GOALS": "goals",
            "ASSISTS": "assists",
            "POINTS": "points",
            "SAVES": "saves",
            "BLOCKS": "blocked",  # Column is "blocked" not "blocked_shots"
        }
        return mapping.get(self.market, "shots")
    
    def train(
        self,
        df: pd.DataFrame,
        validation_split: float = 0.2,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train the neural network on historical player data.
        
        Args:
            df: DataFrame with columns: date, player, role, shots, goals, etc.
            validation_split: Fraction of data to use for validation
            verbose: Print training progress
        
        Returns:
            Dictionary with training metrics
        """
        # Collect features from all players
        all_features = []
        role = "goalie" if self.market == "SAVES" else "skater"
        
        # Parse player names from dictionary format if needed
        import json
        df_copy = df.copy()
        
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
        
        df_copy["player_name"] = df_copy["player"].apply(parse_player_name)
        players = df_copy["player_name"].unique()
        players = [p for p in players if p and str(p).strip()]  # Filter out empty names
        
        if verbose:
            print(f"[train] Preparing features for {len(players)} players...")
        
        success_count = 0
        for i, player in enumerate(players):
            # Pass the parsed df_copy instead of original df
            feats = self._prepare_features(df_copy, player, role)
            if feats is not None and not feats.empty:
                all_features.append(feats)
                success_count += 1
            
            # Progress logging
            if verbose and (i + 1) % 500 == 0:
                print(f"[train] Processed {i+1}/{len(players)} players, {success_count} with valid features...")
        
        if verbose:
            print(f"[train] Feature prep complete: {success_count}/{len(players)} players with valid data")
        
        if not all_features:
            raise ValueError(f"No training data available for {self.market}")
        
        # Combine all player features
        if verbose:
            print(f"[train] Combining {len(all_features)} player feature sets...")
            # Check sample sizes
            sample_sizes = [len(f) for f in all_features[:5]]
            print(f"[train] Sample feature set sizes: {sample_sizes}")
        
        data = pd.concat(all_features, ignore_index=True)
        
        if verbose:
            print(f"[train] Combined data shape: {data.shape}")
            print(f"[train] Columns: {list(data.columns)}")
            null_counts = data.isnull().sum()
            if null_counts.sum() > 0:
                print(f"[train] Columns with nulls: {null_counts[null_counts > 0].to_dict()}")
        
        # Fill NaN values in one-hot encoded columns with 0
        # (sparse team features should be 0, not NaN)
        team_cols = [c for c in data.columns if c.startswith('team_')]
        if team_cols:
            data[team_cols] = data[team_cols].fillna(0)
            if verbose:
                print(f"[train] Filled {len(team_cols)} team columns with 0s")
        
        # Now drop rows with NaN in critical columns only
        before_dropna = len(data)
        data = data.dropna()
        
        if verbose:
            print(f"[train] After dropna: {len(data)} samples (dropped {before_dropna - len(data)})")
        
        if len(data) < 100:
            raise ValueError(f"Insufficient training data: {len(data)} samples")
        
        # Split features and target
        target_col = "target"
        feature_cols = [c for c in data.columns if c != target_col]
        
        X = data[feature_cols].values
        y = data[target_col].values
        
        # Ensure X and y are proper numpy arrays
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        if verbose:
            print(f"[train] Feature matrix: {X.shape}, Target: {y.shape}")
        
        # Normalize features
        self.feature_columns = feature_cols
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
        input_dim = len(feature_cols)
        self.model = PlayerPropsNN(input_dim, self.cfg)
        
        # Loss and optimizer (Poisson NLL or MSE)
        if self.cfg.predict_lambda:
            criterion = nn.PoissonNLLLoss(log_input=False)
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
        
        # Export to ONNX (optional - will warn if dependencies missing)
        try:
            self.export_onnx()
        except Exception as e:
            import warnings
            warnings.warn(f"ONNX export failed (model still saved as PyTorch): {e}")
            if verbose:
                print(f"[warn] ONNX export skipped: {e}")
        
        return {
            "best_val_loss": best_val_loss,
            "samples": len(data),
            "features": len(feature_cols),
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
            output_names=["lambda"],
            dynamic_axes={"features": {0: "batch_size"}, "lambda": {0: "batch_size"}},
        )
        
        print(f"[export] ONNX model saved to {onnx_path}")
        return onnx_path
    
    def predict_lambda(
        self,
        df: pd.DataFrame,
        player: str,
        team: Optional[str] = None,
    ) -> Optional[float]:
        """Predict Poisson lambda for a player using the NN model.
        
        Returns None if model not available or player has insufficient history.
        """
        if self.model is None and self.onnx_session is None:
            return None
        
        role = "goalie" if self.market == "SAVES" else "skater"
        feats_df = self._prepare_features(df, player, role)
        
        if feats_df is None or feats_df.empty:
            return None
        
        # Use most recent row features
        latest = feats_df.iloc[-1]
        X = latest[[c for c in self.feature_columns if c in latest.index]].values
        
        # Handle missing team dummies
        if len(X) < len(self.feature_columns):
            X_full = np.zeros(len(self.feature_columns))
            X_full[: len(X)] = X
            X = X_full
        
        # Normalize
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        
        # Inference
        if self.use_npu and self.onnx_session is not None:
            # NPU inference via ONNX Runtime with QNN
            input_data = X_scaled.astype(np.float32).reshape(1, -1)
            input_name = self.onnx_session.get_inputs()[0].name
            output_name = self.onnx_session.get_outputs()[0].name
            
            result = self.onnx_session.run([output_name], {input_name: input_data})
            lam = float(result[0][0])
        else:
            # CPU inference via PyTorch
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0)
                lam = self.model(X_tensor).item()
        
        return float(lam)

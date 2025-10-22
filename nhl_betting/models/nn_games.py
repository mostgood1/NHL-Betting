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
    # Team/opponent embedding settings for PERIOD_GOALS model
    team_embed_dim: int = 16  # dimension of learned team embeddings derived from one-hot
    
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
        """Neural network for predicting goals per period with team/opponent embeddings.

        Inputs are a flat feature vector that may include one-hot blocks for
        home_team_* and away_team_* when cfg.include_team_encoding=True. This
        model learns dense embeddings from those one-hot blocks and models
        interactions between teams.

        Output: [p1_home, p1_away, p2_home, p2_away, p3_home, p3_away]
        """

        def __init__(self, input_dim: int, cfg: NNGamesConfig | None = None):
            super().__init__()
            self.cfg = cfg or NNGamesConfig()

            # Indices for slicing one-hot blocks will be set after feature prep
            self.register_buffer("home_idx_mask", torch.zeros(1, dtype=torch.bool), persistent=False)
            self.register_buffer("away_idx_mask", torch.zeros(1, dtype=torch.bool), persistent=False)
            self.register_buffer("other_idx_mask", torch.zeros(1, dtype=torch.bool), persistent=False)

            # These layers are initialized lazily once masks are set (see set_feature_indices)
            self.home_linear = None  # type: ignore
            self.away_linear = None  # type: ignore

            # Core MLP will be constructed after we know concat dim
            self.core = None  # type: ignore
            self.out = None  # type: ignore

            # Keep defaults in case masks are not provided (fallback to simple MLP)
            self._fallback_network = None
            layers = []
            prev_dim = input_dim
            for hidden_dim in self.cfg.hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.cfg.dropout))
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 6))
            layers.append(nn.Softplus())
            self._fallback_network = nn.Sequential(*layers)

        def set_feature_indices(self, home_idx: List[int], away_idx: List[int], other_idx: List[int]):
            """Provide index lists for home/away one-hot blocks and other features.

            This enables the embedding pathway. If not set, the model falls back to plain MLP.
            """
            # Create boolean masks for slicing in ONNX-friendly way
            max_idx = max(home_idx + away_idx + other_idx) + 1 if (home_idx or away_idx or other_idx) else 0
            mask_home = torch.zeros(max_idx, dtype=torch.bool)
            mask_away = torch.zeros(max_idx, dtype=torch.bool)
            mask_other = torch.zeros(max_idx, dtype=torch.bool)
            mask_home[home_idx] = True
            mask_away[away_idx] = True
            mask_other[other_idx] = True
            # Register (replace) buffers with correct size
            self.home_idx_mask = mask_home  # type: ignore
            self.away_idx_mask = mask_away  # type: ignore
            self.other_idx_mask = mask_other  # type: ignore

            # Define embedding layers as linear transforms from one-hot -> dense
            in_home = int(mask_home.sum().item())
            in_away = int(mask_away.sum().item())
            embed_dim = int(self.cfg.team_embed_dim)
            # Use bias=False to keep pure embedding semantics
            self.home_linear = nn.Linear(in_home, embed_dim, bias=False)
            self.away_linear = nn.Linear(in_away, embed_dim, bias=False)

            # Define core MLP over [other_features, home_embed, away_embed, home*away interaction]
            other_dim = int(mask_other.sum().item())
            concat_dim = other_dim + embed_dim + embed_dim + embed_dim  # interaction same dim

            core_layers = []
            prev = concat_dim
            for hidden_dim in self.cfg.hidden_dims:
                core_layers.append(nn.Linear(prev, hidden_dim))
                core_layers.append(nn.BatchNorm1d(hidden_dim))
                core_layers.append(nn.ReLU())
                core_layers.append(nn.Dropout(self.cfg.dropout))
                prev = hidden_dim
            self.core = nn.Sequential(*core_layers)
            self.out = nn.Sequential(nn.Linear(prev, 6), nn.Softplus())

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # If indices/masks not set, use fallback network
            if self.core is None or self.home_linear is None or self.away_linear is None:
                return self._fallback_network(x)

            # Slice using boolean masks
            # Ensure masks match x dimension; if not, fallback
            if self.home_idx_mask.numel() != x.shape[1]:
                return self._fallback_network(x)

            x_home = x[:, self.home_idx_mask]
            x_away = x[:, self.away_idx_mask]
            x_other = x[:, self.other_idx_mask]

            # Linear projection from one-hot to dense embedding
            e_home = self.home_linear(x_home)
            e_away = self.away_linear(x_away)
            # Simple interaction: elementwise product
            e_int = e_home * e_away

            z = torch.cat([x_other, e_home, e_away, e_int], dim=1)
            h = self.core(z)
            y = self.out(h)
            return y
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
        # Optional per-team, per-period calibration offsets (applied post-prediction)
        # Dict[str, List[float]] for home and away roles, each list length 3 for P1..P3
        self.period_calib_home = {}
        self.period_calib_away = {}
        # Global calibrations for other models
        self.moneyline_cal_x = None  # sorted breakpoints (0..1)
        self.moneyline_cal_y = None  # mapped probs (0..1)
        self.total_affine = None     # (a,b)
        self.goaldiff_affine = None  # (a,b)
        self.first10_alpha = None    # multiplicative factor on lambda
        
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
            # Optional calibration maps
            try:
                if "period_calib_home" in meta:
                    self.period_calib_home = dict(meta["period_calib_home"].item())
                if "period_calib_away" in meta:
                    self.period_calib_away = dict(meta["period_calib_away"].item())
                if "moneyline_cal_x" in meta and "moneyline_cal_y" in meta:
                    self.moneyline_cal_x = meta["moneyline_cal_x"]
                    self.moneyline_cal_y = meta["moneyline_cal_y"]
                if "total_affine" in meta:
                    self.total_affine = tuple(meta["total_affine"])  # (a,b)
                if "goaldiff_affine" in meta:
                    self.goaldiff_affine = tuple(meta["goaldiff_affine"])  # (a,b)
                if "first10_alpha" in meta:
                    self.first10_alpha = float(meta["first10_alpha"]) if meta["first10_alpha"] is not None else None
            except Exception:
                # If not present or incompatible, ignore (no calibration)
                self.period_calib_home = {}
                self.period_calib_away = {}
                self.moneyline_cal_x = None
                self.moneyline_cal_y = None
                self.total_affine = None
                self.goaldiff_affine = None
                self.first10_alpha = None
            
            # Try ONNX first (works on Windows without PyTorch DLL issues)
            if ONNX_AVAILABLE and onnx_path.exists():
                try:
                    import sys, platform
                    avail = set(ort.get_available_providers())
                    qnn_in_wheel = ("QNNExecutionProvider" in avail)
                    qnn_root = os.environ.get("QNN_SDK_ROOT", "")

                    # Try to discover a QNN backend DLL dynamically if SDK is available
                    backend_path = None
                    if qnn_root:
                        try:
                            # Search for QnnHtp*.dll under QNN_SDK_ROOT (depth-limited walk)
                            qnn_root_p = Path(qnn_root)
                            candidates = []
                            for sub in (
                                qnn_root_p / "lib",
                                qnn_root_p / "bin",
                                qnn_root_p,
                            ):
                                if sub.exists():
                                    for p in sub.rglob("QnnHtp*.dll"):
                                        candidates.append(p)
                            # Prefer paths that match platform architecture
                            arch = platform.machine().lower()
                            def _score(p: Path) -> int:
                                s = str(p).lower()
                                score = 0
                                if "arm64" in s or "aarch64" in s:
                                    score += 2
                                if "windows" in s or "msvc" in s:
                                    score += 1
                                return score
                            if candidates:
                                candidates = sorted(candidates, key=_score, reverse=True)
                                backend_path = str(candidates[0])
                        except Exception:
                            backend_path = None

                    providers: list = []
                    # Prefer QNN EP if available in this ORT build
                    if qnn_in_wheel:
                        if backend_path and Path(backend_path).exists():
                            providers.append((
                                "QNNExecutionProvider",
                                {"backend_path": backend_path, "qnn_context_priority": "high"},
                            ))
                        else:
                            # Try QNN without explicit backend_path (may resolve via PATH/QNN_SDK_ROOT)
                            providers.append("QNNExecutionProvider")
                    else:
                        print(
                            f"[info] QNN EP not in this onnxruntime build. avail={sorted(list(avail))}; "
                            f"platform={platform.system()} {platform.machine()}, qnn_root={'set' if qnn_root else 'unset'}"
                        )

                    # Prefer DirectML on Windows GPU if present
                    if "DmlExecutionProvider" in avail:
                        providers.append("DmlExecutionProvider")
                    # Always include CPU fallback
                    providers.append("CPUExecutionProvider")

                    self.onnx_session = ort.InferenceSession(
                        str(onnx_path),
                        providers=providers,
                    )
                    active_list = self.onnx_session.get_providers()
                    print(
                        f"[info] Loaded ONNX model for {self.model_type} with providers={active_list}; "
                        f"avail={sorted(list(avail))}; backend_path={backend_path or 'auto'}"
                    )
                    return True
                except Exception as e:
                    print(f"[warn] Failed ONNX load for {self.model_type} (QNN/CPU attempt): {e}")
            
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
            # If using team encoding, provide indices for embedding pathway
            if self.cfg.include_team_encoding:
                home_idx = [i for i, c in enumerate(self.feature_columns) if isinstance(c, str) and c.startswith("home_team_")]
                away_idx = [i for i, c in enumerate(self.feature_columns) if isinstance(c, str) and c.startswith("away_team_")]
                other_idx = [i for i in range(len(self.feature_columns)) if i not in set(home_idx) and i not in set(away_idx)]
                try:
                    # type: ignore[attr-defined]
                    self.model.set_feature_indices(home_idx, away_idx, other_idx)
                except Exception as e:
                    # If anything goes wrong, training still proceeds with fallback network
                    print(f"[warn] PeriodGoalsNN embedding setup skipped: {e}")
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
        
        # Compute simple per-team, per-period calibration offsets on validation set (for PERIOD_GOALS)
        calib_home: Dict[str, List[float]] = {}
        calib_away: Dict[str, List[float]] = {}
        if self.model_type == "PERIOD_GOALS" and self.cfg.include_team_encoding:
            try:
                # Recover validation indices
                val_idx = indices[:n_val]
                feats_val = features_df.iloc[val_idx].reset_index(drop=True)
                # Predict on validation features using the trained (best) model
                self.model.eval()
                with torch.no_grad():
                    Xv = torch.FloatTensor(X_scaled[indices[:n_val]])
                    yv = y_val  # shape (n_val, 6)
                    pv = self.model(Xv).detach().cpu().numpy()
                # Helper to extract team key from one-hot columns
                home_cols = [c for c in self.feature_columns if isinstance(c, str) and c.startswith("home_team_")]
                away_cols = [c for c in self.feature_columns if isinstance(c, str) and c.startswith("away_team_")]
                # Build maps of residuals per team
                from collections import defaultdict
                res_home = defaultdict(lambda: [[], [], []])
                res_away = defaultdict(lambda: [[], [], []])
                for i in range(len(feats_val)):
                    row = feats_val.iloc[i]
                    # Identify team names via one-hots
                    try:
                        h_team = next((str(c[len("home_team_"):]) for c in home_cols if row.get(c, 0.0) == 1.0), None)
                        a_team = next((str(c[len("away_team_"):]) for c in away_cols if row.get(c, 0.0) == 1.0), None)
                    except Exception:
                        h_team, a_team = None, None
                    if h_team:
                        # residuals = true - pred to add back mean underprediction
                        res_home[h_team][0].append(float(yv[i, 0] - pv[i, 0]))
                        res_home[h_team][1].append(float(yv[i, 2] - pv[i, 2]))
                        res_home[h_team][2].append(float(yv[i, 4] - pv[i, 4]))
                    if a_team:
                        res_away[a_team][0].append(float(yv[i, 1] - pv[i, 1]))
                        res_away[a_team][1].append(float(yv[i, 3] - pv[i, 3]))
                        res_away[a_team][2].append(float(yv[i, 5] - pv[i, 5]))
                # Aggregate means with small shrinkage to avoid overfitting
                def agg_mean(res_lists):
                    out = {}
                    for team, lists in res_lists.items():
                        vals = []
                        for lst in lists:
                            if len(lst) >= 10:
                                m = float(np.mean(lst))
                            elif len(lst) >= 3:
                                m = float(np.mean(lst) * (len(lst) / 10.0))  # shrink toward 0
                            else:
                                m = 0.0
                            vals.append(m)
                        out[team] = vals
                    return out
                calib_home = agg_mean(res_home)
                calib_away = agg_mean(res_away)
            except Exception:
                calib_home, calib_away = {}, {}

        # Save metadata (include calibration if available)
        if calib_home or calib_away:
            self.period_calib_home = calib_home
            self.period_calib_away = calib_away
            np.savez(
                self._get_metadata_path(),
                feature_columns=np.array(self.feature_columns, dtype=object),
                scaler_mean=self.scaler_mean,
                scaler_std=self.scaler_std,
                period_calib_home=np.array(self.period_calib_home, dtype=object),
                period_calib_away=np.array(self.period_calib_away, dtype=object),
            )
        else:
            np.savez(
                self._get_metadata_path(),
                feature_columns=np.array(self.feature_columns, dtype=object),
                scaler_mean=self.scaler_mean,
                scaler_std=self.scaler_std,
            )

        # Additional calibrations for other model types
        try:
            if self.model_type == "MONEYLINE":
                # Build decile calibration map from validation
                self.model.eval()
                with torch.no_grad():
                    pv = self.model(X_val).detach().cpu().numpy().reshape(-1)
                yv = y_val.detach().cpu().numpy().reshape(-1)
                order = np.argsort(pv)
                bins = np.array_split(order, 10)
                xs, ys = [], []
                for b in bins:
                    if len(b) == 0:
                        continue
                    p_bin = pv[b]
                    y_bin = yv[b]
                    if len(b) < 50:
                        continue
                    xs.append(float(np.mean(p_bin)))
                    ys.append(float(np.mean(y_bin)))
                if len(xs) >= 3:
                    cal_x = np.array([0.0] + xs + [1.0], dtype=np.float32)
                    cal_y = np.array([0.0] + ys + [1.0], dtype=np.float32)
                    # Ensure monotonic increasing cal_y
                    cal_y = np.maximum.accumulate(cal_y)
                    self.moneyline_cal_x = cal_x
                    self.moneyline_cal_y = cal_y
                    # Write back to metadata
                    m = dict(np.load(self._get_metadata_path(), allow_pickle=True))
                    m["moneyline_cal_x"] = cal_x
                    m["moneyline_cal_y"] = cal_y
                    np.savez(self._get_metadata_path(), **m)
            elif self.model_type == "TOTAL_GOALS":
                # Fit affine y ≈ a*pred + b on validation
                self.model.eval()
                with torch.no_grad():
                    pv = self.model(X_val).detach().cpu().numpy().reshape(-1)
                yv = y_val.detach().cpu().numpy().reshape(-1)
                n = len(pv)
                if n >= 100:
                    Spp = float(np.dot(pv, pv))
                    Sp1 = float(np.sum(pv))
                    S11 = float(n)
                    Spy = float(np.dot(pv, yv))
                    S1y = float(np.sum(yv))
                    den = (Spp * S11 - Sp1 * Sp1)
                    if abs(den) > 1e-8:
                        a = (Spy * S11 - Sp1 * S1y) / den
                        b = (Spp * S1y - Sp1 * Spy) / den
                        self.total_affine = (float(a), float(b))
                        m = dict(np.load(self._get_metadata_path(), allow_pickle=True))
                        m["total_affine"] = np.array(self.total_affine, dtype=np.float32)
                        np.savez(self._get_metadata_path(), **m)
            elif self.model_type == "GOAL_DIFF":
                # Affine calibration for goal differential
                self.model.eval()
                with torch.no_grad():
                    pv = self.model(X_val).detach().cpu().numpy().reshape(-1)
                yv = y_val.detach().cpu().numpy().reshape(-1)
                n = len(pv)
                if n >= 100:
                    Spp = float(np.dot(pv, pv))
                    Sp1 = float(np.sum(pv))
                    S11 = float(n)
                    Spy = float(np.dot(pv, yv))
                    S1y = float(np.sum(yv))
                    den = (Spp * S11 - Sp1 * Sp1)
                    if abs(den) > 1e-8:
                        a = (Spy * S11 - Sp1 * S1y) / den
                        b = (Spp * S1y - Sp1 * Spy) / den
                        self.goaldiff_affine = (float(a), float(b))
                        m = dict(np.load(self._get_metadata_path(), allow_pickle=True))
                        m["goaldiff_affine"] = np.array(self.goaldiff_affine, dtype=np.float32)
                        np.savez(self._get_metadata_path(), **m)
            elif self.model_type == "FIRST_10MIN":
                # Calibrate lambda via multiplicative factor to optimize Brier on P(goal >=1)
                self.model.eval()
                with torch.no_grad():
                    pv = self.model(X_val).detach().cpu().numpy().reshape(-1)
                # Observed binary: >0 first10 goals
                yv = (y_val.detach().cpu().numpy().reshape(-1) > 0.0).astype(np.float32)
                if len(pv) >= 200:
                    alphas = np.linspace(0.5, 1.5, 41, dtype=np.float32)
                    best_a, best_brier = 1.0, 1e9
                    for a in alphas:
                        p = 1.0 - np.exp(-np.maximum(0.0, a * pv))
                        brier = float(np.mean((p - yv) ** 2))
                        if brier < best_brier:
                            best_brier = brier
                            best_a = float(a)
                    self.first10_alpha = best_a
                    m = dict(np.load(self._get_metadata_path(), allow_pickle=True))
                    m["first10_alpha"] = np.array([self.first10_alpha], dtype=np.float32)
                    np.savez(self._get_metadata_path(), **m)
        except Exception as e:
            # Calibration is optional; proceed if anything fails
            print(f"[warn] Calibration step skipped: {e}")
        
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
    
    def export_onnx(self, opset_version: int = 18) -> Path:
        """Export trained model to ONNX format for NPU/CPU inference.

        Improvements:
        - Default opset_version=18 to align with modern PyTorch exporter behavior
        - Use dynamic_shapes (preferred) instead of dynamic_axes under dynamo exporter
        - Avoid external data files for smaller models (single .onnx artifact)
        - If only ONNX is loaded, auto-load PyTorch weights for export
        """
        # Ensure PyTorch model is available for export
        if self.model is None and TORCH_AVAILABLE:
            # Attempt to load PyTorch weights directly
            model_path = self._get_model_path()
            if model_path.exists():
                input_dim = len(self.feature_columns)
                if self.model_type == "PERIOD_GOALS":
                    self.model = PeriodGoalsNN(input_dim, self.cfg)
                else:
                    output_dim = self._get_output_dim()
                    self.model = GameOutcomeNN(input_dim, output_dim, self.cfg)
                try:
                    self.model.load_state_dict(torch.load(model_path, weights_only=True))
                except TypeError:
                    # Older torch versions may not support weights_only
                    self.model.load_state_dict(torch.load(model_path))
                self.model.eval()

        if self.model is None:
            raise ValueError("PyTorch model not available for ONNX export")

        dummy_input = torch.randn(1, len(self.feature_columns))
        onnx_path = self._get_onnx_path()

        # Use static export for compatibility across torch versions
        torch.onnx.export(
            self.model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["prediction"],
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
                    arr = pred.squeeze(0)
                    # Apply calibration offsets if available
                    try:
                        if isinstance(home_team, str) and isinstance(away_team, str):
                            h_off = self.period_calib_home.get(home_team, [0.0, 0.0, 0.0])
                            a_off = self.period_calib_away.get(away_team, [0.0, 0.0, 0.0])
                            arr[0] = max(0.0, float(arr[0] + h_off[0]))
                            arr[1] = max(0.0, float(arr[1] + a_off[0]))
                            arr[2] = max(0.0, float(arr[2] + h_off[1]))
                            arr[3] = max(0.0, float(arr[3] + a_off[1]))
                            arr[4] = max(0.0, float(arr[4] + h_off[2]))
                            arr[5] = max(0.0, float(arr[5] + a_off[2]))
                    except Exception:
                        pass
                    return arr  # Return calibrated array of 6 values
                else:
                    val = float(pred.squeeze())
                    # Apply global calibrations if available
                    try:
                        if self.model_type == "MONEYLINE" and (self.moneyline_cal_x is not None) and (self.moneyline_cal_y is not None):
                            p = max(1e-6, min(1 - 1e-6, val))
                            # numpy interp expects ascending x
                            val = float(np.interp(p, self.moneyline_cal_x, self.moneyline_cal_y))
                        elif self.model_type == "TOTAL_GOALS" and (self.total_affine is not None):
                            a, b = self.total_affine
                            val = float(a * val + b)
                        elif self.model_type == "GOAL_DIFF" and (self.goaldiff_affine is not None):
                            a, b = self.goaldiff_affine
                            val = float(a * val + b)
                        elif self.model_type == "FIRST_10MIN" and (self.first10_alpha is not None):
                            val = max(0.0, float(self.first10_alpha * val))
                    except Exception:
                        pass
                    return val  # Return single value
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
                    arr = pred.squeeze(0).numpy()
                    # Apply calibration offsets if available
                    try:
                        if isinstance(home_team, str) and isinstance(away_team, str):
                            h_off = self.period_calib_home.get(home_team, [0.0, 0.0, 0.0])
                            a_off = self.period_calib_away.get(away_team, [0.0, 0.0, 0.0])
                            arr[0] = max(0.0, float(arr[0] + h_off[0]))
                            arr[1] = max(0.0, float(arr[1] + a_off[0]))
                            arr[2] = max(0.0, float(arr[2] + h_off[1]))
                            arr[3] = max(0.0, float(arr[3] + a_off[1]))
                            arr[4] = max(0.0, float(arr[4] + h_off[2]))
                            arr[5] = max(0.0, float(arr[5] + a_off[2]))
                    except Exception:
                        pass
                    return arr
                else:
                    val = pred.item()
                    try:
                        if self.model_type == "MONEYLINE" and (self.moneyline_cal_x is not None) and (self.moneyline_cal_y is not None):
                            p = max(1e-6, min(1 - 1e-6, float(val)))
                            val = float(np.interp(p, self.moneyline_cal_x, self.moneyline_cal_y))
                        elif self.model_type == "TOTAL_GOALS" and (self.total_affine is not None):
                            a, b = self.total_affine
                            val = float(a * val + b)
                        elif self.model_type == "GOAL_DIFF" and (self.goaldiff_affine is not None):
                            a, b = self.goaldiff_affine
                            val = float(a * val + b)
                        elif self.model_type == "FIRST_10MIN" and (self.first10_alpha is not None):
                            val = max(0.0, float(self.first10_alpha * val))
                    except Exception:
                        pass
                    return val
        
        raise ValueError("No model available for inference")

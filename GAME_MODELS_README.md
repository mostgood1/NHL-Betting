# Game Outcome Neural Network Models - Setup Complete

## ✅ What's Been Created

### 1. **Neural Network Architecture** (`nhl_betting/models/nn_games.py`)
   - **GameOutcomeNN**: Feedforward network for game predictions
     - Architecture: 128 → 64 → 32 with BatchNorm and Dropout
     - Supports classification (win/loss) and regression (goals, differential)
   
   - **PeriodGoalsNN**: Multi-output network for period-by-period predictions
     - Outputs 6 values: [P1_home, P1_away, P2_home, P2_away, P3_home, P3_away]
   
   - **NNGameModel**: Training and inference wrapper
     - Automatic feature preparation
     - ONNX export for NPU acceleration
     - Comprehensive feature engineering

### 2. **Prediction Models Supported**
   | Model | Type | Output | Use Case |
   |-------|------|--------|----------|
   | **MONEYLINE** | Classification | Win probability | Which team wins |
   | **TOTAL_GOALS** | Regression | Expected goals | Over/under betting |
   | **GOAL_DIFF** | Regression | Score differential | Spread betting |
   | **FIRST_10MIN** | Regression | Early goals | First period action |
   | **PERIOD_GOALS** | Multi-output | 6 goal values | Period-specific bets |

### 3. **Feature Engineering** (`nhl_betting/data/game_features.py`)
   Computes comprehensive features:
   - **Elo Ratings**: Team strength at game time
   - **Recent Form**: Last 10 games stats (goals for/against, wins)
   - **Rest Days**: Days since last game for each team
   - **Season Progress**: Games played / 82
   - **Period Goals**: Historical period-by-period performance
   - **First 10 Min Goals**: Early game scoring patterns

### 4. **Training Infrastructure** (`nhl_betting/scripts/train_nn_games.py`)
   Commands:
   ```bash
   # Train single model
   python -m nhl_betting.scripts.train_nn_games train --model MONEYLINE --epochs 100
   
   # Train all models
   python -m nhl_betting.scripts.train_nn_games train-all --epochs 100
   
   # Benchmark NPU performance
   python -m nhl_betting.scripts.train_nn_games benchmark --model MONEYLINE --num-runs 1000
   ```

## 📊 Current Data Status

**Available Data:**
- **56 games** from Oct 7-15, 2025 (current season only)
- **30 teams** 
- Features prepared and saved to: `data/raw/games_with_features.csv`

**Data Issue:**
⚠️ **56 games is insufficient for neural network training**
- Recommended minimum: **2,000+ games** (2-3 seasons)
- Optimal: **5,000+ games** (5+ seasons)

## 🔄 Next Steps

### Option 1: Collect Historical Game Data (Recommended)
```bash
# Collect 2+ seasons of historical games
python -m nhl_betting.cli collect-games --start 2023-10-01 --end 2025-10-16 --source web

# Rebuild features with full dataset
python -m nhl_betting.data.game_features

# Train all models
python -m nhl_betting.scripts.train_nn_games train-all --epochs 100
```

### Option 2: Train on Current Data (For Testing Only)
```bash
# Train with small dataset (will overfit, not production-ready)
python -m nhl_betting.scripts.train_nn_games train --model MONEYLINE --epochs 50

# Models will work but won't generalize well
```

### Option 3: Use Existing Elo/Poisson Models
- Keep current lightweight models for game outcomes
- Use neural networks only for player props (already trained!)
- Player props have 127K samples (sufficient data)

## 🎯 Recommendation

**Best Approach:**
1. ✅ **Keep using Elo/Poisson for game outcomes** (proven, fast, interpretable)
2. ✅ **Use neural networks for player props** (complex, benefits from NPU, already trained)
3. 🔄 **Collect historical game data in background** (2-3 seasons)
4. 🔄 **Train game NN models once data is ready** (as enhancement/comparison)

**Why?**
- Your current Elo/Poisson models are well-tuned and working
- Player props are more complex → benefit more from deep learning
- You already have 6 trained prop models ready for NPU
- Game NNs can be added later as A/B test vs Elo

## 📁 Files Created

```
nhl_betting/
├── models/
│   └── nn_games.py          # Game outcome neural networks
├── data/
│   └── game_features.py     # Feature preparation
└── scripts/
    └── train_nn_games.py    # Training CLI

data/
└── raw/
    └── games_with_features.csv  # Prepared features (56 games)
```

## 🚀 Ready to Use (When Data Available)

All infrastructure is complete and tested. When you have historical game data:
1. Run feature preparation: `python -m nhl_betting.data.game_features`
2. Train models: `python -m nhl_betting.scripts.train_nn_games train-all`
3. Models export to ONNX for NPU acceleration automatically
4. Benchmark NPU speedup vs CPU

The architecture mirrors your working player props models, so training process will be identical.

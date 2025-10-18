# Neural Network Game Models - Training Plan

## Current Status ✅

**Data Available:**
- ✅ 3,136 games with full features (Oct 2023 - Oct 2025)
- ✅ 28 feature columns including:
  - Elo ratings
  - Recent form (goals_last10, wins_last10)
  - Rest days
  - Season progress
  - Period-by-period goals
  - First 10 minute goals
- ✅ Team abbreviations (DET, TBL, etc.)
- ✅ Training infrastructure ready

**Models to Train:**
1. **MONEYLINE** - Home team win probability (classification)
2. **TOTAL_GOALS** - Expected total goals (regression)
3. **GOAL_DIFF** - Score differential (regression)
4. **FIRST_10MIN** - ✅ Already trained! (0.0165 val loss)
5. **PERIOD_GOALS** - ✅ Already trained! (0.3411 val loss)

## Training Strategy

### Phase 1: Train Core Game Models (TODAY)

**Priority Order:**
1. **TOTAL_GOALS** (High priority)
   - Replace Poisson model's extreme projections
   - Input: 95 features (13 base + 82 team encodings)
   - Output: Single value (total goals in game)
   - Expected: Better than current 6.0 baseline

2. **MONEYLINE** (High priority)
   - Replace/supplement Elo predictions
   - Input: 95 features
   - Output: Home win probability
   - Expected: Better calibration than Elo alone

3. **GOAL_DIFF** (Medium priority)
   - For puck line predictions
   - Input: 95 features
   - Output: Home goals - Away goals
   - Use for spread betting

### Phase 2: Validate & Compare (Tomorrow)

**Backtest Protocol:**
1. Split data: Train on games before Oct 1, 2025
2. Test on Oct 1-17, 2025 (recent 2+ weeks)
3. Compare metrics:
   - NN vs Elo/Poisson accuracy
   - NN vs Elo/Poisson calibration
   - NN vs Elo/Poisson ROI

**Success Criteria:**
- Moneyline: >55% accuracy (vs Elo baseline)
- Totals: RMSE <2.0 goals (vs Poisson baseline)
- Calibration: Well-calibrated probabilities

### Phase 3: Deploy Hybrid System (This Weekend)

**Deployment Options:**

**Option A: Full Replacement** (If NN clearly superior)
```python
# Replace Elo/Poisson with NN predictions
p_home_ml = moneyline_model.predict(home, away, features)
total_goals = total_goals_model.predict(home, away, features)
```

**Option B: Ensemble** (If both have strengths)
```python
# Combine NN and Elo/Poisson
p_home_elo = elo.predict(home, away)
p_home_nn = moneyline_model.predict(home, away, features)
p_home_final = 0.6 * p_home_nn + 0.4 * p_home_elo

total_poisson = poisson.lambdas(...)
total_nn = total_goals_model.predict(home, away, features)
total_final = 0.7 * total_nn + 0.3 * total_poisson
```

**Option C: Hybrid by Context** (Best of both)
```python
# Use NN for complex situations, Elo/Poisson for simple
if extreme_situation(features):  # Back-to-back, injury, etc.
    use_nn_prediction()
else:
    use_elo_poisson()  # Faster, interpretable
```

## Training Commands

### Train Individual Models
```bash
# Total goals model (most important to fix projections)
python -m nhl_betting.scripts.train_nn_games train \
    --model TOTAL_GOALS \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --hidden-dims "128,64,32"

# Moneyline model (improve on Elo)
python -m nhl_betting.scripts.train_nn_games train \
    --model MONEYLINE \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --hidden-dims "128,64,32"

# Goal differential (puck line)
python -m nhl_betting.scripts.train_nn_games train \
    --model GOAL_DIFF \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --hidden-dims "128,64,32"
```

### Train All at Once
```bash
python -m nhl_betting.scripts.train_nn_games train-all \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --hidden-dims "128,64,32"
```

## Expected Outcomes

### TOTAL_GOALS Model
**Current Poisson Issues:**
- WSH vs MIN: 5.04 - 1.16 = 6.20 (unrealistic)
- UTA vs SJS: 4.08 - 0.90 = 4.98 (SJS <1 goal?)

**NN Should Learn:**
- Recent offensive performance (goals_last10)
- Recent defensive performance (goals_against_last10)
- Rest advantage patterns
- Team-specific scoring tendencies
- Historical head-to-head patterns

**Expected Improvement:**
- More realistic totals: 5.5-6.5 range instead of 4.5-7.5
- Better calibration: Predicted 6.0 games actually average 6.0
- Lower RMSE: <2.0 vs current unknown

### MONEYLINE Model
**Current Elo Baseline:**
- Uses only team strength (1500 ± adjustments)
- Home advantage: +50 Elo points
- No context beyond wins/losses

**NN Should Learn:**
- Elo + recent form + rest + matchup context
- Back-to-back disadvantage
- Playoff race urgency
- Goalie matchups (if we add goalie features later)

**Expected Improvement:**
- Better accuracy: 56-58% vs 55% Elo baseline
- Better calibration: 70% predictions win 70% of time
- Better edges: More confident on high-value bets

### GOAL_DIFF Model
**Current Approach:**
- Puck line derived from total goals split
- Very rough approximation

**NN Should Learn:**
- Which teams cover spreads vs just win
- Blowout vs close game patterns
- Comeback potential based on context

**Expected Improvement:**
- Better puck line accuracy: 54-56%
- Identify value on underdogs +1.5
- Identify value on favorites -1.5

## Integration Plan

### Step 1: Train Models (TODAY)
```bash
python -m nhl_betting.scripts.train_nn_games train-all --epochs 100
```

### Step 2: Update Prediction CLI (TOMORROW)
Modify `nhl_betting/cli.py` predict() function:
```python
# Load NN models (in addition to existing Elo/Poisson)
try:
    from .models.nn_games import NNGameModel
    moneyline_nn = NNGameModel(model_type="MONEYLINE", ...)
    total_goals_nn = NNGameModel(model_type="TOTAL_GOALS", ...)
    goal_diff_nn = NNGameModel(model_type="GOAL_DIFF", ...)
except Exception:
    # Fallback to Elo/Poisson if NN not available
    moneyline_nn = None
    ...

# For each game, get NN predictions
if moneyline_nn and total_goals_nn:
    # Build full 95-feature vector (already implemented!)
    game_features = build_features(g.home, g.away, ...)
    
    # Get NN predictions
    p_home_nn = moneyline_nn.predict(g.home, g.away, game_features)
    total_nn = total_goals_nn.predict(g.home, g.away, game_features)
    
    # Ensemble with Elo/Poisson
    p_home_final = 0.7 * p_home_nn + 0.3 * p_home_elo
    
    # Use NN total instead of Poisson allocation
    total_final = total_nn
```

### Step 3: Backtest (TOMORROW)
```bash
python backtest_nn_vs_elo.py --start 2025-10-01 --end 2025-10-17
```

### Step 4: Deploy (WEEKEND)
- Update daily_update.py to use NN models
- Keep Elo/Poisson as fallback
- Monitor performance for 1 week
- Adjust ensemble weights based on results

## Risk Mitigation

**If NN Models Underperform:**
1. Keep Elo/Poisson as primary
2. Use NN as secondary signal
3. Investigate what NN is learning
4. Collect more training data

**If NN Models Overfit:**
- Add L2 regularization
- Increase dropout (currently 0.3)
- Reduce model complexity (fewer layers)
- Add early stopping (already implemented)

**If Training Fails:**
- Check data quality
- Normalize features properly
- Adjust learning rate
- Try different architectures

## Timeline

**Today (Oct 17):**
- ✅ Assess current system
- ✅ Document plan
- 🔨 Train TOTAL_GOALS model (30 min)
- 🔨 Train MONEYLINE model (30 min)
- 🔨 Train GOAL_DIFF model (30 min)

**Tomorrow (Oct 18):**
- Backtest models vs Elo/Poisson
- Compare accuracy, calibration, ROI
- Decide on deployment strategy

**Weekend (Oct 19-20):**
- Integrate best-performing approach
- Update daily_update.py
- Deploy to production
- Monitor initial results

**Next Week:**
- Track performance daily
- Adjust ensemble weights if needed
- Consider adding goalie features
- Consider adding injury data

## Success Metrics

**We'll know NN models are better if:**
1. ✅ Moneyline accuracy >55% (need 52.4% to break even)
2. ✅ Totals RMSE <2.0 goals
3. ✅ Better calibration (Brier score <0.25)
4. ✅ Higher ROI on backtested bets
5. ✅ More realistic projections (no 5-1 or 4-0.9 games)

**We'll know to keep Elo/Poisson if:**
1. ❌ NN accuracy <53%
2. ❌ NN calibration worse than Elo
3. ❌ NN creates more extreme projections
4. ❌ Backtest ROI negative

Let's proceed! 🚀

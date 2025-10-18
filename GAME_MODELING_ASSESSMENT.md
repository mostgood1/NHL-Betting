# Game Modeling Assessment & Recommendations

## Current System Overview

### Models in Use (October 2025)
1. **Elo Rating System** (`nhl_betting/models/elo.py`)
   - Tracks team strength with dynamic ratings
   - K-factor: 20.0
   - Home advantage: 50 Elo points
   - Predicts moneyline probabilities

2. **Poisson Goals Model** (`nhl_betting/models/poisson.py`)
   - Base league average: 3.05 goals/game
   - Home attack: 1.05x multiplier
   - Away attack: 0.95x multiplier
   - Derives lambdas from total line + ML split
   - Predicts totals and puck line

3. **Probability Calibration** (`data/processed/model_calibration.json`)
   - Moneyline: t=1.5, b=-0.15
   - Totals: t=1.5, b=0.15
   - Trained on: 93 ML games, 82 totals games
   - Period: Sept 20 - Oct 14, 2025

4. **Neural Network Period Models** (NEW - Just Fixed!)
   - FIRST_10MIN: Predicts goals in first 10 minutes
   - PERIOD_GOALS: Predicts 6 values (home/away for P1/P2/P3)
   - Trained on 3,136 historical games with 95 features
   - Uses team encodings, recent form, rest days, etc.

## Performance Analysis

### Data Available
- **161 tracked predictions** from Sept 23 - Oct 17, 2025
- Markets: 66 moneyline, 40 puckline, 55 totals
- Reconciliation file exists but lacks win/loss tracking

### Current Predictions (Oct 17, 2025)
```
Detroit vs Tampa Bay:
  Model: DET 4.98 - 2.10 TBL (73.97% DET ML, 7.08 total)
  Edge: +0.70 EV on DET ML, +0.16 EV on Over

Washington vs Minnesota:
  Model: WSH 5.04 - 1.16 MIN (83.83% WSH ML, 6.19 total)
  Edge: +0.51 EV on WSH ML, +0.11 EV on Under

Chicago vs Vancouver:
  Model: CHI 2.29 - 3.60 VAN (33.79% CHI ML, 5.89 total)
  Edge: +0.15 EV on VAN ML, +0.20 EV on Under

Utah vs San Jose:
  Model: UTA 4.08 - 0.90 SJS (82.94% UTA ML, 4.98 total)
  Edge: +0.14 EV on UTA ML, +0.20 EV on Under
```

### Issues Identified

#### ❌ Missing Performance Tracking
- Reconciliation log tracks predictions but not actual outcomes
- No win/loss calculation
- No ROI tracking
- Can't assess model accuracy without backtest results

#### ⚠️ Extreme Projections
- WSH 5.04 - 1.16 MIN = 6.2 goals (MIN only 1.16?)
- DET 4.98 - 2.10 TBL = 7.08 goals (very high)
- UTA 4.08 - 0.90 SJS = 4.98 goals (SJS only 0.90?)

These projections seem unrealistic. Let's check why:

#### 🔍 Root Cause Analysis

Looking at the code in `cli.py` around line 750-850:
```python
# Poisson lambda derivation
lam_h, lam_a = pois.lambdas_from_total_split(per_game_total, p_home)

# With strong Elo advantage (83% ML), this allocates most goals to home
# Example: 6.0 total * 0.83 = ~5.0 home, 6.0 * 0.17 = ~1.0 away
```

The issue: **Poisson allocation is too aggressive when ML probability is extreme**.

## Recommendations

### 🎯 Priority 1: Add Performance Tracking (IMMEDIATE)

**Issue**: Can't evaluate model without tracking actual results.

**Solution**: Enhance reconciliation to calculate win/loss
```python
# In daily_update.py or reconciliation function
def reconcile_game_prediction(prediction, actual_result):
    """
    prediction: dict with home, away, bet, market, price
    actual_result: dict with home_goals, away_goals
    
    Returns: win/loss/push and payout
    """
    if market == 'moneyline':
        if bet == 'home_ml':
            result = 'win' if home_goals > away_goals else 'loss'
        elif bet == 'away_ml':
            result = 'win' if away_goals > home_goals else 'loss'
    elif market == 'totals':
        total_goals = home_goals + away_goals
        if bet == 'over':
            result = 'win' if total_goals > line else 'loss'
        elif bet == 'under':
            result = 'win' if total_goals < line else 'loss'
    # ... similar for puckline
    
    payout = stake * decimal_odds if result == 'win' else 0
    return {'result': result, 'payout': payout}
```

**Action Items**:
1. Create `reconcile_games()` function
2. Fetch actual scores from NHL API for past predictions
3. Calculate win/loss for each bet
4. Track cumulative ROI
5. Generate performance report

### 🎯 Priority 2: Calibrate Poisson Lambda Allocation (HIGH)

**Issue**: Extreme ML probabilities cause unrealistic goal projections.

**Current behavior**:
```python
# 70% by ML split, 30% by baseline
lam_h = 0.7 * (total_line * p_home) + 0.3 * lam_h0
lam_a = 0.7 * (total_line * p_away) + 0.3 * lam_a0
```

**Problem**: With p_home=0.83, we get:
- lam_h = 0.7 * (6.0 * 0.83) + 0.3 * 3.2 = 4.45
- lam_a = 0.7 * (6.0 * 0.17) + 0.3 * 2.9 = 1.58

**Solution Options**:

**A. Increase Baseline Weight** (Quick fix)
```python
# Change from 70/30 to 50/50 or 60/40
lam_h = 0.5 * (total_line * p_home) + 0.5 * lam_h0
lam_a = 0.5 * (total_line * p_away) + 0.5 * lam_a0
```

**B. Add Dampening for Extreme Probabilities**
```python
def dampen_extreme_probs(p, min_p=0.25, max_p=0.75):
    """Prevent extreme allocations"""
    return max(min_p, min(max_p, p))

p_home_dampened = dampen_extreme_probs(p_home)
p_away_dampened = 1.0 - p_home_dampened
```

**C. Use Recent Form Data** (Best - already available!)
```python
# We already compute this in the NN feature engineering!
home_form = compute_recent_form_features(home, date, historical_df)
away_form = compute_recent_form_features(away, date, historical_df)

# Use actual recent scoring instead of pure allocation
lam_h = 0.6 * home_form["goals_last10"] + 0.4 * (total_line * p_home)
lam_a = 0.6 * away_form["goals_last10"] + 0.4 * (total_line * p_away)
```

**Action Items**:
1. Test Option A (quick fix) - change weights to 50/50
2. Backtest on last 2 weeks of games
3. If improved, implement Option C for production
4. Compare projections before/after

### 🎯 Priority 3: Backtest Historical Performance (MEDIUM)

**Issue**: No baseline performance metrics.

**Solution**: Run backtest on completed games
```bash
# Collect actual scores for Sept 20 - Oct 17
python -m nhl_betting.cli collect-games --start 2025-09-20 --end 2025-10-17

# Regenerate predictions using saved odds
python backtest_game_predictions.py --start 2025-09-20 --end 2025-10-17

# Compare predictions vs actuals
# Calculate: accuracy, ROI, calibration curve
```

**Metrics to Track**:
- **Moneyline accuracy**: % of correct winner predictions
- **Totals accuracy**: % of correct over/under
- **Puckline accuracy**: % of correct spread predictions
- **Calibration**: Are 70% predictions correct 70% of the time?
- **ROI**: Profit/loss assuming flat stakes
- **Kelly Criterion ROI**: Profit/loss with optimal sizing

**Expected Baselines** (from similar systems):
- Moneyline: 55-58% accuracy (need 52.4% to break even at -110)
- Totals: 53-56% accuracy
- Puckline: 52-55% accuracy
- Overall ROI: 2-5% is excellent

### 🎯 Priority 4: Consider Hybrid Approach (LOW - Future)

**Option**: Keep Elo/Poisson for ML/Totals, use NN for special bets

**Rationale**:
- Elo/Poisson are fast, interpretable, proven
- NNs excel at complex patterns (period-specific, situational)
- Hybrid gets best of both worlds

**Recommended Architecture**:
```
Game Outcome Prediction:
├─ Moneyline: Elo model (current)
├─ Totals: Poisson model with recent form (enhanced)
├─ Puckline: Poisson model (current)
├─ Period bets: NN PERIOD_GOALS model ✓ (working now!)
└─ First 10min: NN FIRST_10MIN model ✓ (working now!)
```

**Action Items**:
1. Keep current system for core markets
2. Use NN models for period/situational bets
3. Add NN as secondary signal (ensemble)
4. A/B test: Elo vs NN for moneyline over 100 games

### 🎯 Priority 5: Retrain NN Game Models (OPTIONAL - Future)

**Current Status**:
- NN models exist but trained on only 56 games
- Not enough data for production use
- Period models working well with 3,136 games

**If You Want Full NN Game Models**:

**Step 1**: Collect 2-3 seasons of data
```bash
python -m nhl_betting.cli collect-games --start 2023-10-01 --end 2025-10-17 --source web
```

**Step 2**: Train models
```bash
python -m nhl_betting.scripts.train_nn_games train-all --epochs 100
```

**Step 3**: Backtest vs Elo/Poisson
```bash
python compare_models.py --start 2025-09-01 --end 2025-10-17
```

**Step 4**: Deploy if NN outperforms
- Replace Elo with NN for moneyline
- Replace Poisson with NN for totals
- Keep calibration layer

**But**: This is a lot of work for potentially marginal gains. The current Elo/Poisson system is likely sufficient if properly calibrated.

## Summary & Action Plan

### ✅ What's Working
1. Period predictions now show team differentiation (just fixed!)
2. Calibration system exists and is being updated
3. Feature engineering is comprehensive (95 features)
4. System makes predictions and tracks them

### ❌ What Needs Fixing
1. **No win/loss tracking** - Can't assess performance
2. **Extreme goal projections** - Poisson allocation too aggressive
3. **No backtest results** - Unknown if models are profitable

### 📋 Recommended Immediate Actions

**Week 1** (This week):
1. ✅ Fix NN prediction variability (DONE!)
2. 🔨 Add win/loss reconciliation (TODAY)
3. 🔨 Backtest last 2 weeks to get baseline metrics
4. 🔨 Adjust Poisson allocation to 50/50 if projections still extreme

**Week 2**:
5. Implement Option C (use recent form for lambda allocation)
6. Backtest again with new approach
7. Generate performance dashboard

**Month 1**:
8. Collect 2-3 seasons of historical game data
9. Set up automated nightly backtesting
10. Consider ensemble approach (Elo + NN)

### 🎯 Bottom Line

**Your game models DON'T need retraining right now**, but they DO need:
1. **Performance tracking** - Must measure before optimizing
2. **Calibration refinement** - Fix extreme projections
3. **Backtesting** - Validate on historical data

The underlying Elo/Poisson approach is sound. Focus on measurement and tuning before considering a full NN rebuild.

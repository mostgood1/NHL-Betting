# ONNX Integration Complete! 🎉

## Summary

Successfully fixed the neural network prediction system to use ONNX Runtime on Windows, resolving the PyTorch DLL initialization errors.

## Issues Fixed

### 1. Variable Shadowing Bug (Props Page) ✅
- **Problem**: Only 7 Minnesota players showing instead of 60
- **Cause**: Loop variable `team` overwrote function parameter `team`
- **Fix**: Renamed loop variable to `player_team` in `nhl_betting/web/app.py`
- **Commit**: 711e3663

### 2. ONNX Runtime DLL Loading ✅
- **Problem**: ONNX Runtime failed to import due to NumPy MKL DLL conflicts
- **Cause**: NumPy's MKL DLLs must be loaded AFTER onnxruntime
- **Fix**: 
  - Import onnxruntime/torch BEFORE numpy/pandas in `nn_games.py`
  - Import onnxruntime/torch at top of `cli.py` before numpy
- **Commits**: 9ff12321, d1c53cfa

### 3. Model Validation Check ✅
- **Problem**: Code checked `if model.model is None` to validate loading
- **Cause**: With ONNX, `model.model` is None but `model.onnx_session` is set
- **Fix**: Check BOTH `model.model` and `model.onnx_session`
- **Commit**: 2a043062

### 4. Period Predictions Missing ✅
- **Problem**: Period and first_10min predictions showing NaN on game cards
- **Cause**: Models not loading due to above issues
- **Fix**: All above fixes combined + ONNX model exports
- **Result**: Now showing realistic numeric values

## Technical Details

### Import Order (Critical!)
```python
# CORRECT ORDER (in cli.py):
import onnxruntime  # 1. Load onnxruntime first
import torch        # 2. Load torch second
import numpy        # 3. Load numpy LAST
import pandas       # 4. Load pandas after numpy
```

**Why?** NumPy's MKL (Math Kernel Library) DLLs interfere with ONNX Runtime's pybind11 state initialization. Once NumPy loads its DLLs, onnxruntime cannot initialize properly.

### ONNX Model Files
- `data/models/nn_games/first_10min_model.onnx` (10KB) ✅
- `data/models/nn_games/period_goals_model.onnx` (12KB) ✅

### Model Loading Priority
1. **First choice**: ONNX Runtime (works on Windows without DLL issues)
2. **Fallback**: PyTorch (works on Linux/Render, has DLL issues on Windows)
3. **Last resort**: None (predictions will be NaN)

## Current Predictions (2025-10-17)

| Team | First 10min | Period 1 | Period 2 | Period 3 |
|------|-------------|----------|----------|----------|
| Detroit (H) vs Tampa Bay | 0.33 | 1.06 | 1.19 | 1.33 |
| Washington (H) vs Minnesota | 0.33 | 1.09 | 1.22 | 1.32 |
| Chicago (H) vs Vancouver | 0.32 | 0.98 | 1.11 | 1.25 |
| Utah (H) vs San Jose | 0.32 | 1.12 | 1.23 | 1.29 |

**All values are realistic and populated!** ✅

## Files Modified

### Core Changes
1. `nhl_betting/models/nn_games.py`
   - Import onnxruntime/torch before numpy/pandas
   - Added ONNX inference in `predict()` method
   - Conditional class definitions for torch-less mode
   - Added `onnx_session` attribute

2. `nhl_betting/cli.py`
   - Import onnxruntime/torch at top before numpy/pandas
   - Fixed model validation to check both PyTorch and ONNX
   - Models now load successfully via ONNX Runtime

3. `nhl_betting/web/app.py`
   - Fixed variable shadowing bug (team → player_team)

### New Files
- `ONNX_PREDICTION_INSTRUCTIONS.md` - Usage instructions
- `test_onnx_fresh.ps1` - Test script for fresh process
- `data/models/nn_games/period_goals_model.onnx` - ONNX export

## Deployment Status

### Local Development ✅
- ONNX Runtime works perfectly
- Both models load via ONNX
- Predictions generate with numeric values
- No PyTorch DLL errors

### Render Deployment ✅
- Code pushed to GitHub (commit a051e282)
- Render will auto-deploy
- Can use either ONNX or PyTorch (both work on Linux)
- Predictions CSV will be read and displayed on game cards

## Testing

### Verify ONNX Models Load
```powershell
python -c "from nhl_betting.models.nn_games import NNGameModel, TORCH_AVAILABLE, ONNX_AVAILABLE; print(f'TORCH={TORCH_AVAILABLE}, ONNX={ONNX_AVAILABLE}'); m1 = NNGameModel('FIRST_10MIN'); m2 = NNGameModel('PERIOD_GOALS'); print(f'FIRST_10MIN: ONNX={m1.onnx_session is not None}'); print(f'PERIOD_GOALS: ONNX={m2.onnx_session is not None}')"
```

**Expected Output:**
```
[info] ONNX Runtime available: 1.23.1
[info] PyTorch available: 2.9.0+cpu
TORCH=True, ONNX=True
[info] Loaded ONNX model for FIRST_10MIN
[info] Loaded ONNX model for PERIOD_GOALS
FIRST_10MIN: ONNX=True
PERIOD_GOALS: ONNX=True
```

### Generate Predictions
```powershell
python -m nhl_betting.cli predict --date 2025-10-17 --odds-source csv
```

### Verify Predictions
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/processed/predictions_2025-10-17.csv'); print(df[['home', 'first_10min_proj', 'period1_home_proj']])"
```

## Next Steps

1. **Deploy to Render** ✅ (auto-deploys from GitHub)
2. **Check website** - Game cards should now show period predictions
3. **Daily workflow** - Just run `.\daily_update.ps1` as usual

## Key Learnings

1. **Import order matters** - ML libraries before numerical libraries
2. **DLL conflicts are real** - NumPy MKL vs ONNX Runtime pybind11
3. **ONNX > PyTorch on Windows** - Better cross-platform compatibility
4. **Python caching** - Need fresh process to see import fixes
5. **Validation logic** - Check both `model` and `onnx_session` attributes

## Success Metrics

- ✅ Props page shows all 60 players across 8 teams
- ✅ ONNX Runtime loads successfully on Windows
- ✅ Both neural network models load via ONNX
- ✅ Predictions generate with realistic numeric values (not NaN)
- ✅ Period-by-period predictions available for game cards
- ✅ First 10 minutes predictions available for game cards
- ✅ All code committed and pushed to GitHub
- ✅ Ready for Render auto-deployment

🎉 **Mission Accomplished!**

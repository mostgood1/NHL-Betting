# Daily Update DLL Fix - Import Order Critical ⚠️

## Problem

When running `daily_update.py` directly, NN models failed to load with DLL errors:

```
[warn] ONNX Runtime not available: DLL load failed while importing onnxruntime_pybind11_state: 
       A dynamic link library (DLL) initialization routine failed.
       
[warn] PyTorch not available: [WinError 1114] A dynamic link library (DLL) initialization 
       routine failed. Error loading "C:\...\torch\lib\c10.dll" or one of its dependencies.
```

**Result:** NN models not loaded → predictions fall back to ELO-only → less accurate predictions.

## Root Cause

**Import Order Violation:**

`daily_update.py` was importing packages in wrong order:

```python
# WRONG ORDER (before fix):
import pandas as pd  # ❌ pandas loads MKL DLLs first

from nhl_betting.cli import predict_core  # ❌ cli.py tries to load torch/onnx
# → DLL conflict → torch/onnx fail → no NN models
```

**Why This Fails:**
1. `pandas` imports NumPy which loads MKL (Math Kernel Library) DLLs
2. MKL DLLs initialize in a specific way
3. When `cli.py` later tries to import torch/onnx, their DLLs conflict with MKL
4. Import fails → `_torch = None`, `_ort = None` → NN models disabled

## Solution

**Fix Import Order:**

```python
# CORRECT ORDER (after fix):
from nhl_betting.cli import predict_core  # ✅ Load cli.py FIRST
from nhl_betting.cli import props_fetch_bovada
from nhl_betting.cli import props_predict
from nhl_betting.cli import props_build_dataset

import pandas as pd  # ✅ NOW safe to import pandas
```

**Why This Works:**
1. `cli.py` imports torch/onnx **before** pandas (see cli.py lines 3-13)
2. torch/onnx DLLs load first and initialize correctly
3. pandas/numpy can load after without conflict
4. NN models load successfully ✅

## Files Modified

**nhl_betting/scripts/daily_update.py:**
- Lines 12-39: Moved cli imports BEFORE pandas import
- Added comment explaining the critical import order requirement

```python
# CRITICAL: Import cli.py BEFORE pandas to avoid DLL conflicts with torch/onnx
# cli.py imports torch/onnx first, then pandas. If we import pandas here first,
# torch/onnx will fail to load later, disabling NN models.
from nhl_betting.cli import predict_core, featurize, train
from nhl_betting.cli import props_fetch_bovada as _props_fetch_bovada
from nhl_betting.cli import props_predict as _props_predict
from nhl_betting.cli import props_build_dataset as _props_build_dataset

# NOW safe to import pandas (after cli.py has loaded torch/onnx)
import pandas as pd
```

## Validation

### Before Fix
```powershell
PS> python nhl_betting/scripts/daily_update.py
[warn] ONNX Runtime not available: DLL load failed...
[warn] PyTorch not available: DLL initialization failed...
```
NN models: **NOT LOADED** ❌

### After Fix
```powershell
PS> python nhl_betting/scripts/daily_update.py
[info] ONNX Runtime available: 1.23.1
[info] PyTorch available: 2.9.0+cpu
[info] Loaded ONNX model for PERIOD_GOALS
[info] Loaded ONNX model for FIRST_10MIN
[info] Loaded ONNX model for TOTAL_GOALS
[info] Loaded ONNX model for MONEYLINE
[info] Loaded ONNX model for GOAL_DIFF
```
NN models: **LOADED** ✅

### Test NN Model Loading
```python
python -c "
from nhl_betting.scripts import daily_update
from nhl_betting.models.nn_games import NNGameModel
model = NNGameModel('first_10min')
print(f'Model loaded: {model.onnx_session is not None}')
"
# Output: Model loaded: True ✅
```

### Test Predictions Use NN
```powershell
PS> python -m nhl_betting.cli predict --date 2025-10-18 --odds-source csv
[info] Loaded ONNX model for FIRST_10MIN
[NN] Loaded models: PERIOD_GOALS, FIRST_10MIN, TOTAL_GOALS, MONEYLINE, GOAL_DIFF
```
Predictions file shows numeric `first_10min_proj` values (not NaN) ✅

## Critical Import Rules

**For ANY script that uses NN models:**

1. ✅ Import `nhl_betting.cli` (or other NN-using modules) **FIRST**
2. ✅ Import `pandas`, `numpy`, `sklearn` **AFTER** cli/NN modules
3. ❌ Never import pandas/numpy before torch/onnx modules

**Exception:** Web server (`nhl_betting/web/app.py`)
- Reads predictions from CSV (doesn't run NN models)
- Can import pandas first
- Catches OSError when importing cli (tolerates DLL conflict)

## Related Fixes

This is the **third** DLL conflict fix in the codebase:

1. **cli.py** (lines 3-13): Import torch/onnx before pandas
2. **web/app.py**: Catch OSError when importing cli (web server doesn't need NN)
3. **daily_update.py** (this fix): Import cli before pandas

All follow same principle: **torch/onnx DLLs must load before numpy/pandas MKL DLLs**

## Impact

- ✅ Daily update now uses NN models for predictions
- ✅ More accurate first_10min, total_goals, moneyline projections
- ✅ Props predictions use NN models
- ✅ No more DLL warnings in logs
- ✅ Consistent NN model availability across all scripts

## Testing

After this fix, verify daily update works:

```powershell
# Test import order fix
python -c "from nhl_betting.scripts import daily_update; print('✅ Import successful')"

# Test NN models load
python -c "from nhl_betting.scripts import daily_update; from nhl_betting.models.nn_games import NNGameModel; m = NNGameModel('first_10min'); print('✅ ONNX:', m.onnx_session is not None)"

# Test full predictions
python -m nhl_betting.cli predict --date 2025-10-18 --odds-source csv
# Should see: [info] Loaded ONNX model for FIRST_10MIN
```

All tests should pass without DLL warnings ✅

## Status

**COMPLETE AND VALIDATED** ✅

**Date:** October 18, 2025  
**Issue:** DLL errors preventing NN models from loading in daily_update.py  
**Root Cause:** pandas imported before cli.py  
**Solution:** Move cli imports before pandas import  
**Result:** NN models load successfully, predictions use ONNX models

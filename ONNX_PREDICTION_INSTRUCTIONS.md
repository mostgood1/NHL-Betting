# Instructions to Regenerate Predictions with ONNX Models

## ✅ What We Fixed

1. **Import Order Issue**: NumPy's MKL DLLs interfere with ONNX Runtime
   - Solution: Import onnxruntime/torch BEFORE numpy/pandas in nn_games.py
   
2. **ONNX Models Created**:
   - `first_10min_model.onnx` (10KB) ✓
   - `period_goals_model.onnx` (12KB) ✓

3. **ONNX Inference**: Both models now load via ONNX Runtime on Windows

## 🔄 To Regenerate Predictions (Fresh Terminal)

**IMPORTANT**: You need a fresh Python process because the old cached imports have the DLL errors.

### Option 1: Close VS Code and Reopen
1. Close VS Code completely
2. Reopen it
3. Open a new PowerShell terminal
4. Run:
   ```powershell
   cd c:\Users\mostg\OneDrive\Coding\NHL-Betting
   .\.venv\Scripts\Activate.ps1
   python -m nhl_betting.cli predict --date 2025-10-17 --odds-source csv
   ```

### Option 2: New PowerShell Window
1. Open a completely NEW PowerShell window (not a VS Code terminal)
2. Run:
   ```powershell
   cd c:\Users\mostg\OneDrive\Coding\NHL-Betting
   .\.venv\Scripts\Activate.ps1
   python -m nhl_betting.cli predict --date 2025-10-17 --odds-source csv
   ```

### Option 3: Quick Test (verifies ONNX works)
```powershell
# In a NEW terminal:
cd c:\Users\mostg\OneDrive\Coding\NHL-Betting
.\.venv\Scripts\Activate.ps1
python -c "from nhl_betting.models.nn_games import NNGameModel, TORCH_AVAILABLE, ONNX_AVAILABLE; print(f'TORCH={TORCH_AVAILABLE}, ONNX={ONNX_AVAILABLE}'); m1 = NNGameModel('FIRST_10MIN'); m2 = NNGameModel('PERIOD_GOALS'); print(f'FIRST_10MIN: ONNX={m1.onnx_session is not None}'); print(f'PERIOD_GOALS: ONNX={m2.onnx_session is not None}')"
```

**Expected output:**
```
[info] ONNX Runtime available: 1.23.1
[info] PyTorch available: 2.9.0+cpu
TORCH=True, ONNX=True
[info] Loaded ONNX model for FIRST_10MIN
[info] Loaded ONNX model for PERIOD_GOALS
FIRST_10MIN: ONNX=True
PERIOD_GOALS: ONNX=True
```

## 📊 Verify Predictions

After regenerating predictions, check the CSV:
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/processed/predictions_2025-10-17.csv'); print(df[['home', 'away', 'first_10min_proj', 'period1_home_proj', 'period2_home_proj']])"
```

**Should see numeric values** instead of NaN.

## 🚀 For Daily Use

Once verified working, use your normal workflow:
```powershell
# Daily update script will automatically use ONNX models
.\daily_update.ps1
```

The predictions CSV will be saved to `data/processed/predictions_YYYY-MM-DD.csv` and Render will read from there.

## 📝 Technical Notes

- **Local Development**: Uses ONNX Runtime (no PyTorch DLL issues on Windows)
- **Render Deployment**: Can use either ONNX or PyTorch (both work on Linux)
- **Model Priority**: ONNX is preferred over PyTorch (better cross-platform compatibility)
- **CSV Workflow**: Local generates predictions → saves to CSV → Render displays from CSV
- **No Training on Render**: All model training/export happens locally, Render just serves predictions

## 🎯 Summary

**The Fix**: Import order matters! ONNX Runtime must be imported before NumPy loads its MKL DLLs.

**Current Status**:
- ✅ ONNX Runtime works on Windows
- ✅ Both models have ONNX exports
- ✅ Code committed and pushed to GitHub
- ⏸️ Need fresh Python process to regenerate predictions (current terminal has cached imports with old errors)

**Next Step**: Open a new terminal and run the prediction command to get fresh imports without DLL errors.

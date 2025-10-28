# Test predictions generation in fresh Python process
cd c:\Users\mostg\OneDrive\Coding\NHL-Betting
if (Test-Path .\activate_npu.ps1) { . .\activate_npu.ps1 } else { if (Test-Path .\ensure_arm64_venv.ps1) { . .\ensure_arm64_venv.ps1; $null = Ensure-Arm64Venv -RepoRoot (Get-Location) -Activate } }

Write-Host "========================================" -ForegroundColor Green
Write-Host "Testing ONNX model loading..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

python -c "from nhl_betting.models.nn_games import NNGameModel, TORCH_AVAILABLE, ONNX_AVAILABLE; print(f'TORCH={TORCH_AVAILABLE}, ONNX={ONNX_AVAILABLE}'); m1 = NNGameModel('FIRST_10MIN'); m2 = NNGameModel('PERIOD_GOALS'); print(f'FIRST_10MIN ONNX={m1.onnx_session is not None}'); print(f'PERIOD_GOALS ONNX={m2.onnx_session is not None}')"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Regenerating predictions..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

python -m nhl_betting.cli predict --date 2025-10-17 --odds-source csv

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Checking predictions CSV..." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

python -c "import pandas as pd; df = pd.read_csv('data/processed/predictions_2025-10-17.csv'); print('Period and first_10min predictions:'); print(df[['home', 'first_10min_proj', 'period1_home_proj']].to_string()); has_values = not df['first_10min_proj'].isna().all(); print(f'\nHas numeric values: {has_values}'); exit(0 if has_values else 1)"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "SUCCESS! Predictions generated with ONNX" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "FAILED: Predictions still showing NaN" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}

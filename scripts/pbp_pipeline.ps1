param(
    [string]$Seasons = "2019 2020 2021 2022 2023 2024 2025",
    [string]$Out = "data/raw/nhl_pbp",
    [string]$Start = "2019-10-01",
    [string]$End = "2025-10-17"
)

Write-Host "[pbp] Starting NHL PBP pipeline" -ForegroundColor Cyan

# Resolve repo root and source NPU activation if present
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$NpuScript = Join-Path $RepoRoot "activate_npu.ps1"
if (Test-Path $NpuScript) { . $NpuScript }

# Ensure virtual env
if (Test-Path .\.venv\Scripts\Activate.ps1) {
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Warning "[pbp] Python venv not found at .\\.venv; continuing with system Python"
}

# Check for Rscript
$r = Get-Command Rscript -ErrorAction SilentlyContinue
if (-not $r) {
    Write-Warning "[pbp] Rscript not found. Install R to use nhlfastR/fastRhockey. Skipping fetch step."
} else {
    Write-Host "[pbp] Fetching PBP via R for seasons: $Seasons" -ForegroundColor Yellow
    Rscript scripts/nhl_pbp_fetch.R --seasons $Seasons --out $Out
}

# Ingest Parquet into periods and merge
Write-Host "[pbp] Ingesting PBP into games_with_periods.csv" -ForegroundColor Yellow
python scripts/ingest_pbp_to_periods.py

# Rebuild features
Write-Host "[pbp] Rebuilding games_with_features.csv" -ForegroundColor Yellow
python -m nhl_betting.data.game_features

# Backtests by source
Write-Host "[pbp] Backtest (source=pbp)" -ForegroundColor Yellow
python scripts/backtest_first10.py --start $Start --end $End --source pbp --out data/processed/first10_backtest_pbp.csv

Write-Host "[pbp] Backtest (source=api)" -ForegroundColor Yellow
python scripts/backtest_first10.py --start $Start --end $End --source api --out data/processed/first10_backtest_api.csv

Write-Host "[pbp] Pipeline complete" -ForegroundColor Green

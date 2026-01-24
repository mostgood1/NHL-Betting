param(
  [int]$Days = 30,
  [switch]$IncludeMultiplierSweep
)

$ErrorActionPreference = 'Stop'

# Activate environment
. .\activate_npu.ps1
. .\.venv\Scripts\Activate.ps1

# Date range (ET dates)
$end = (Get-Date).ToString('yyyy-MM-dd')
$start = (Get-Date).AddDays(-$Days).ToString('yyyy-MM-dd')

Write-Host "[weekly] Calibrating simulations for $start..$end" -ForegroundColor Cyan

# Global calibration (moneyline, totals, puckline)
python -m nhl_betting.cli game-calibrate-sim --start $start --end $end

# Per-total-line calibration for totals
python -m nhl_betting.cli game-calibrate-sim-per-line --start $start --end $end

# Special teams (PP/PK) calibration from possession sim events
try {
  Write-Host "[weekly] Calibrating special teams (PP/PK) …" -ForegroundColor Yellow
  python -m nhl_betting.cli game-calibrate-special-teams --start $start --end $end
} catch {
  Write-Warning "[weekly] Special teams calibration failed: $($_.Exception.Message)"
}

# Optional: sweep totals multipliers (can be slow)
if ($IncludeMultiplierSweep) {
  Write-Host "[weekly] Running totals multipliers sweep..." -ForegroundColor Yellow
  python scripts\calibrate_totals_multipliers.py
}

Write-Host "[weekly] Calibration complete" -ForegroundColor Green

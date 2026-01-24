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

# Precompute possession sim events over the window (ensures events exist)
try {
  $dates = @(for ($d = [DateTime]::Parse($start); $d -le [DateTime]::Parse($end); $d = $d.AddDays(1)) { $d.ToString('yyyy-MM-dd') })
  foreach ($dt in $dates) {
    # Skip dates with no scheduled games
    try {
      $hasGames = (python -c "from nhl_betting.data.nhl_api_web import NHLWebClient; import sys; print(1 if NHLWebClient().schedule_day('$dt') else 0)").Trim()
      if ($hasGames -ne '1') {
        Write-Host "[weekly] Skipping ${dt} (no games)" -ForegroundColor DarkGray
        continue
      }
    } catch {
      Write-Warning "[weekly] Failed schedule check for ${dt}: $($_.Exception.Message)"; 
    }
    $eventsPath = Join-Path 'data/processed' "sim_events_pos_${dt}.csv"
    if (-not (Test-Path $eventsPath)) {
      Write-Host "[weekly] Preparing lineup/shifts and simulating possession for ${dt} …" -ForegroundColor DarkYellow
      try { python -m nhl_betting.cli lineup-update --date $dt } catch { Write-Warning "[weekly] lineup-update failed for ${dt}: $($_.Exception.Message)" }
      try { python -m nhl_betting.cli shifts-update --date $dt } catch { Write-Warning "[weekly] shifts-update failed for ${dt}: $($_.Exception.Message)" }
      try { python -m nhl_betting.cli game-simulate-possession --date $dt } catch { Write-Warning "[weekly] game-simulate-possession failed for ${dt}: $($_.Exception.Message)" }
    }
  }
} catch {
  Write-Warning "[weekly] Failed to precompute possession events: $($_.Exception.Message)"
}

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

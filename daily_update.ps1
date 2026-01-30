Param(
  [int]$DaysAhead = 2,
  [int]$YearsBack = 2,
  [switch]$NoReconcile,
  [switch]$Quiet,
  [switch]$BootstrapModels,
  [double]$TrendsDecay = 0.98,
  [switch]$ResetTrends,
  [switch]$SkipProps,
  [switch]$SkipPropsProjections,
  [switch]$SkipPropsCalibration,
  [switch]$SkipGameCalibration,
  [switch]$InstallDeps,
  [switch]$RecomputeRecs,
  [switch]$RunBacktests,
  [int]$BacktestDays = 30,
  [string]$BacktestMinEvPerMarket = "",
  [switch]$RunSimBacktests,
  [int]$SimBacktestDays = 14,
  [string]$SimBacktestMinEvPerMarket = ""
)
$ErrorActionPreference = "Stop"

# Best-effort: set execution policy for this process so it runs without prompts
try { Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force | Out-Null } catch {}

# Resolve repo root (this file lives at repo root)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = $ScriptDir
# Fallback: if not run from root for some reason, try parent
if (-not (Test-Path (Join-Path $RepoRoot "requirements.txt"))) {
  $RepoRoot = Split-Path -Parent $ScriptDir
}

# Ensure we run with repo root as working directory (affects relative data paths)
Set-Location $RepoRoot

# Ensure ARM64 venv and activate
try {
  $Ensure = Join-Path $RepoRoot 'ensure_arm64_venv.ps1'
  if (Test-Path $Ensure) {
    . $Ensure
    $ok = Ensure-Arm64Venv -RepoRoot $RepoRoot -Activate
    if (-not $ok) { Write-Warning '[ARM64] Proceeding with existing venv (may be x64); QNN EP likely unavailable.'; . (Join-Path $RepoRoot '.venv/Scripts/Activate.ps1') }
  } else {
    # Fallback: legacy behavior
    $Venv = Join-Path $RepoRoot ".venv"
    $Activate = Join-Path $Venv "Scripts/Activate.ps1"
    if (-not (Test-Path $Activate)) { python -m venv $Venv }
    . $Activate
  }
} catch {
  Write-Warning "[ARM64] Failed to enforce ARM64 venv: $($_.Exception.Message)"; . (Join-Path $RepoRoot '.venv/Scripts/Activate.ps1')
}

# Enable NPU-accelerated NN model precomputation for props
$env:PROPS_PRECOMPUTE_ALL = "1"

# Ensure OddsAPI props lines are collected for today & tomorrow (event-level props)
try {
  $today = (Get-Date).ToString('yyyy-MM-dd')
  $tomorrow = (Get-Date).AddDays(1).ToString('yyyy-MM-dd')
  # Broaden coverage per provider docs
  if (-not $env:PROPS_ODDSAPI_REGIONS) { $env:PROPS_ODDSAPI_REGIONS = 'us,us2' }
  if (-not $env:PROPS_ODDSAPI_BOOKMAKERS) { $env:PROPS_ODDSAPI_BOOKMAKERS = '' }
  if (-not $Quiet) { Write-Host "[oddsapi] Collecting props lines for $today & $tomorrow" -ForegroundColor Cyan }
  python -m nhl_betting.cli props-collect --date $today --source oddsapi | Out-Null
  python -m nhl_betting.cli props-collect --date $tomorrow --source oddsapi | Out-Null
} catch {
  Write-Warning "[oddsapi] Props collection failed: $($_.Exception.Message)"
}

# Optional: allow callers to skip heavy props projections or calibration via switches
if ($SkipPropsProjections) { $env:PROPS_SKIP_PROJECTIONS = '1' }
if ($SkipPropsCalibration) { $env:SKIP_PROPS_CALIBRATION = '1' }

# Ensure dependencies (opt-in). Use -InstallDeps or DAILY_UPDATE_INSTALL_DEPS=1 to refresh deps.
try {
  $doInstall = $InstallDeps -or ("$env:DAILY_UPDATE_INSTALL_DEPS" -match '^(1|true|yes)$')
  if ($doInstall) {
    Write-Host "[deps] Installing requirements…" -ForegroundColor Cyan
    pip install -q -r (Join-Path $RepoRoot "requirements.txt")
  } else {
    Write-Host "[deps] Skipping requirements install (use -InstallDeps to enable)." -ForegroundColor DarkGray
  }
} catch {
  Write-Warning "[deps] Failed to install requirements: $($_.Exception.Message)"
}

# Build args and run
$argsList = @("-m", "nhl_betting.scripts.daily_update", "--days-ahead", "$DaysAhead", "--years-back", "$YearsBack")
if ($NoReconcile) { $argsList += "--no-reconcile" }
if (-not $Quiet) { $argsList += "--verbose" }
if ($BootstrapModels) { $argsList += "--bootstrap-models" }
$argsList += @("--trends-decay", "$TrendsDecay")
if ($ResetTrends) { $argsList += "--reset-trends" }
if ($SkipProps) { $argsList += "--skip-props" }
python @argsList

# Final status
if (-not $Quiet) { Write-Host "[run] Daily update complete." }

# Sim-based player props boxscores + recommendations (play-level rooted)
if (-not $SkipProps) {
  try {
    $today = (Get-Date).ToString('yyyy-MM-dd')
    $tomorrow = (Get-Date).AddDays(1).ToString('yyyy-MM-dd')
    if (-not $Quiet) { Write-Host "[props-sim] Boxscores $today & $tomorrow" -ForegroundColor Cyan }
    python -m nhl_betting.cli props-simulate-boxscores --date $today --n-sims 2000 --seed 42 | Out-Null
    python -m nhl_betting.cli props-simulate-boxscores --date $tomorrow --n-sims 2000 --seed 42 | Out-Null
    if (-not $Quiet) { Write-Host "[props-recs] From boxscores $today & $tomorrow (prob gating)" -ForegroundColor Cyan }
    # Apply per-market probability gates to improve daily accuracy (especially SOG)
    $probGates = 'SOG=0.68,GOALS=0.60,ASSISTS=0.60,POINTS=0.62,SAVES=0.60,BLOCKS=0.60'
    python -m nhl_betting.cli props-recommendations-boxscores --date $today --min-ev 0 --top 400 --min-prob-per-market $probGates | Out-Null
    python -m nhl_betting.cli props-recommendations-boxscores --date $tomorrow --min-ev 0 --top 400 --min-prob-per-market $probGates | Out-Null
  } catch {
    Write-Warning "[props-sim] Failed to generate boxscores/recs: $($_.Exception.Message)"
  }
}

# Automatically calibrate game model parameters (dc_rho, market anchor weights, totals temp)
if (-not $SkipGameCalibration) {
  try {
    # Determine current season start (ET): July boundary, start from Sep 1 for safety
    $now = Get-Date
    $seasonStartYear = if ($now.Month -ge 7) { $now.Year } else { $now.Year - 1 }
    $start = [datetime]::new($seasonStartYear, 9, 1).ToString('yyyy-MM-dd')
    $end = (Get-Date).ToString('yyyy-MM-dd')
    if (-not $Quiet) { Write-Host "[cal] Auto-calibrating games $start..$end" -ForegroundColor Cyan }
    python -m nhl_betting.cli game-auto-calibrate --start $start --end $end | Out-Null
    if (-not $Quiet) { Write-Host "[cal] Wrote calibration to data/processed/model_calibration.json" -ForegroundColor DarkGray }
  } catch {
    Write-Warning "[cal] Game auto-calibration failed: $($_.Exception.Message)"
  }
}

# Automatically learn per-market EV gates (ML & Totals) and persist
if (-not $SkipGameCalibration) {
  try {
    $now = Get-Date
    $seasonStartYear = if ($now.Month -ge 7) { $now.Year } else { $now.Year - 1 }
    $start = [datetime]::new($seasonStartYear, 9, 1).ToString('yyyy-MM-dd')
    $end = (Get-Date).ToString('yyyy-MM-dd')
    if (-not $Quiet) { Write-Host "[cal] Learning EV gates $start..$end" -ForegroundColor Cyan }
    python -m nhl_betting.cli game-learn-ev-gates --start $start --end $end | Out-Null
    if (-not $Quiet) { Write-Host "[cal] Updated min_ev_* thresholds in model_calibration.json" -ForegroundColor DarkGray }
  } catch {
    Write-Warning "[cal] EV gate learning failed: $($_.Exception.Message)"
  }
}

# Optional: recompute game recommendations for today and tomorrow using shared module
if ($RecomputeRecs) {
  try {
    $today = (Get-Date).ToString('yyyy-MM-dd')
    $tomorrow = (Get-Date).AddDays(1).ToString('yyyy-MM-dd')
    if (-not $Quiet) { Write-Host "[recs] Recomputing recommendations for $today and $tomorrow…" -ForegroundColor Cyan }
    # Enable First-10 blending using calibrated alpha from data/processed/model_calibration.json
    $env:FIRST10_BLEND = '1'
    python -c "from nhl_betting.core.recs_shared import recompute_edges_and_recommendations as R; R('$today', min_ev=0.0)" | Out-Null
    python -c "from nhl_betting.core.recs_shared import recompute_edges_and_recommendations as R; R('$tomorrow', min_ev=0.0)" | Out-Null
    if (-not $Quiet) { Write-Host "[recs] Done writing recommendations CSVs to data/processed." -ForegroundColor DarkGray }
  } catch {
    Write-Warning "[recs] Failed to recompute recommendations: $($_.Exception.Message)"
  }
}

# Optional: run a short projections backtest over recent days and print a summary
if ($RunBacktests) {
  try {
    $end = (Get-Date).ToString('yyyy-MM-dd')
    $start = (Get-Date).AddDays(-[int]$BacktestDays).ToString('yyyy-MM-dd')
    if (-not $Quiet) { Write-Host "[bt] Projections backtest $start..$end (EV≥2%)" -ForegroundColor Cyan }
    $btCmd = "python -m nhl_betting.cli props-backtest-from-projections --start $start --end $end --stake 100 --markets 'SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS' --min-ev 0.02 --out-prefix nn_daily"
    if ($BacktestMinEvPerMarket -and $BacktestMinEvPerMarket.Trim().Length -gt 0) { $btCmd += " --min-ev-per-market `"$BacktestMinEvPerMarket`"" }
    iex $btCmd | Out-Null
    $summ = Join-Path $RepoRoot "data/processed/nn_daily_props_backtest_summary_${start}_to_${end}.json"
    if (Test-Path $summ) {
      $obj = Get-Content $summ | ConvertFrom-Json
      $ov = $obj.overall
      $acc = if ($ov.accuracy) { [math]::Round([double]$ov.accuracy, 4) } else { $null }
      $brier = if ($ov.brier) { [math]::Round([double]$ov.brier, 4) } else { $null }
      Write-Host "[bt] Picks=$($ov.picks) Decided=$($ov.decided) Acc=$acc Brier=$brier" -ForegroundColor DarkGray
    } else {
      Write-Host "[bt] Summary not found: $summ" -ForegroundColor Yellow
    }
    $simSumm = $null
    $simSumm = Join-Path $RepoRoot "data/processed/sim_daily_props_backtest_sim_summary_${start}_to_${end}.json"
    $outCsv = Join-Path $RepoRoot "data/processed/backtest_daily_summary_${start}_to_${end}.csv"
    if (Test-Path $summ) {
      python -m nhl_betting.scripts.backtest_daily_summary --proj $summ --sim $simSumm --out $outCsv | Out-Null
      if (-not $Quiet) { Write-Host "[bt] Wrote dashboard: $outCsv" -ForegroundColor DarkGray }
    }
  } catch {
    Write-Warning "[bt] Backtest failed: $($_.Exception.Message)"
  }
}

# Optional: run a short sim-backed backtest over recent days and print a summary
if ($RunSimBacktests) {
  try {
    $end = (Get-Date).ToString('yyyy-MM-dd')
    $start = (Get-Date).AddDays(-[int]$SimBacktestDays).ToString('yyyy-MM-dd')
    if (-not $Quiet) { Write-Host "[bt] Sim-backed props backtest $start..$end" -ForegroundColor Cyan }
    $simCmd = "python -m nhl_betting.cli props-backtest-from-simulations --start $start --end $end --stake 100 --markets 'SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS' --min-ev -1 --out-prefix sim_daily"
    if ($SimBacktestMinEvPerMarket -and $SimBacktestMinEvPerMarket.Trim().Length -gt 0) { $simCmd += " --min-ev-per-market `"$SimBacktestMinEvPerMarket`"" }
    iex $simCmd | Out-Null
    $summ = Join-Path $RepoRoot "data/processed/sim_daily_props_backtest_sim_summary_${start}_to_${end}.json"
    if (Test-Path $summ) {
      $obj = Get-Content $summ | ConvertFrom-Json
      $ov = $obj.overall
      $acc = if ($ov.accuracy) { [math]::Round([double]$ov.accuracy, 4) } else { $null }
      $brier = if ($ov.brier) { [math]::Round([double]$ov.brier, 4) } else { $null }
      Write-Host "[bt-sim] Picks=$($ov.picks) Decided=$($ov.decided) Acc=$acc Brier=$brier" -ForegroundColor DarkGray
    } else {
      Write-Host "[bt-sim] Summary not found: $summ (ensure props simulations exist)" -ForegroundColor Yellow
    }
    $projSumm = $null
    $projSumm = Join-Path $RepoRoot "data/processed/nn_daily_props_backtest_summary_${start}_to_${end}.json"
    $outCsv = Join-Path $RepoRoot "data/processed/backtest_daily_summary_${start}_to_${end}.csv"
    if ((Test-Path $summ) -or (Test-Path $projSumm)) {
      python -m nhl_betting.scripts.backtest_daily_summary --proj $projSumm --sim $summ --out $outCsv | Out-Null
      if (-not $Quiet) { Write-Host "[bt-sim] Wrote dashboard: $outCsv" -ForegroundColor DarkGray }
    }
  } catch {
    Write-Warning "[bt-sim] Sim-backed backtest failed: $($_.Exception.Message)"
  }
}

# Optional: push changes to git (models/predictions/reconciliations)
try {
  $isGit = git rev-parse --is-inside-work-tree 2>$null
  if ($LASTEXITCODE -eq 0 -and $isGit -eq 'true') {
    # Check for changes
    $status = git --no-pager status -s
    if ($status) {
      if (-not $Quiet) { Write-Host "[git] Changes detected; staging and committing…" }
  # Stage common outputs from daily update
  git add data/models/*.json 2>$null | Out-Null
  git add data/processed/*.csv 2>$null | Out-Null
  git add data/processed/*.json 2>$null | Out-Null
  # Also stage canonical props lines so Render has odds inputs (OddsAPI preferred)
  git add data/props/player_props_lines/** 2>$null | Out-Null
      # Commit with timestamped message; ignore if nothing staged
  # Use PS5.1 compatible UTC timestamp (AsUTC not available on older shells)
  $date = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
  git commit -m "[auto] Daily update ${date}: models/predictions/reconciliations" 2>$null | Out-Null
      if ($LASTEXITCODE -eq 0) {
        if (-not $Quiet) { Write-Host "[git] Committed. Pushing…" }
        git push | Out-Null
        if (-not $Quiet) { Write-Host "[git] Push complete." }
      } else {
        if (-not $Quiet) { Write-Host "[git] Nothing to commit after staging or commit failed." }
      }
    } else {
      if (-not $Quiet) { Write-Host "[git] No changes to commit." }
    }
  } else {
    if (-not $Quiet) { Write-Host "[git] Not a git repository; skipping push." }
  }
} catch {
  if (-not $Quiet) { Write-Host "[git] Skipping git push due to error: $($_.Exception.Message)" }
}

exit $LASTEXITCODE

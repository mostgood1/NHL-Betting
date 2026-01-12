Param (
  [int]$DaysAhead = 2,
  [int]$YearsBack = 2,
  [switch]$NoReconcile,
  [switch]$Postgame,
  [string]$PostgameDate = "yesterday",
  [string]$PostgameStatsSource = "stats",
  [int]$PostgameWindow = 10,
  [double]$PostgameStake = 100,
  [switch]$PBPBackfill,
  [int]$PBPDaysBack = 7,
  [switch]$SimulateGames,
  [int]$SimSamples = 20000,
  [double]$SimOverK = 2.0,
  [double]$SimSharedK = 3.0,
  [double]$SimEmptyNetP = 0.18,
  [double]$SimEmptyNetTwoGoalScale = 0.30,
  [double]$TotalsPaceAlpha = 0.15,
  [double]$TotalsGoalieBeta = 0.10,
  [double]$TotalsFatigueBeta = 0.08,
  [double]$TotalsRollingPaceGamma = 0.10,
  [double]$TotalsPPGamma = 0.00,
  [double]$TotalsPKBeta = 0.00,
  [double]$TotalsPenaltyGamma = 0.08,
  [double]$TotalsXGGamma = 0.00,
  [double]$TotalsRefsGamma = 0.00,
  [double]$TotalsGoalieFormGamma = 0.00,
  [switch]$InstallDeps,
  [switch]$BacktestSimulations,
  [int]$BacktestWindowDays = 30,
  [switch]$SimRecommendations,
  [switch]$SimIncludeTotals,
  [double]$SimMLThr = 0.65,
  [double]$SimTotThr = 0.55,
  [double]$SimPLThr = 0.62
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$NpuScript = Join-Path $RepoRoot "activate_npu.ps1"

# Ensure QNN env (optional). Dot-source if available so QNN EP is found in this session.
if (Test-Path $NpuScript) {
  . $NpuScript
}

# Defaults for optional params if not bound
if (-not $PSBoundParameters.ContainsKey('SimIncludeTotals')) { $SimIncludeTotals = $false }
# Enable PBP backfill by default unless explicitly disabled
if (-not $PSBoundParameters.ContainsKey('PBPBackfill')) { $PBPBackfill = $true }

# Optional: load tuned totals multipliers from config if present and not overridden
try {
  $cfgPath = Join-Path $RepoRoot 'data/processed/totals_multipliers_config.json'
  if (Test-Path $cfgPath) {
    $cfg = Get-Content $cfgPath | ConvertFrom-Json
    if ($cfg.best) {
      if (-not $PSBoundParameters.ContainsKey('TotalsRefsGamma') -and $cfg.best.totals_refs_gamma) { $TotalsRefsGamma = [double]$cfg.best.totals_refs_gamma }
      if (-not $PSBoundParameters.ContainsKey('TotalsXGGamma') -and $cfg.best.totals_xg_gamma) { $TotalsXGGamma = [double]$cfg.best.totals_xg_gamma }
      if (-not $PSBoundParameters.ContainsKey('TotalsPenaltyGamma') -and $cfg.best.totals_penalty_gamma) { $TotalsPenaltyGamma = [double]$cfg.best.totals_penalty_gamma }
      if (-not $PSBoundParameters.ContainsKey('TotalsGoalieFormGamma') -and $cfg.best.totals_goalie_form_gamma) { $TotalsGoalieFormGamma = [double]$cfg.best.totals_goalie_form_gamma }
      if (-not $PSBoundParameters.ContainsKey('TotalsRollingPaceGamma') -and $cfg.best.totals_rolling_pace_gamma) { $TotalsRollingPaceGamma = [double]$cfg.best.totals_rolling_pace_gamma }
      if (-not $PSBoundParameters.ContainsKey('TotalsFatigueBeta') -and $cfg.best.totals_fatigue_beta) { $TotalsFatigueBeta = [double]$cfg.best.totals_fatigue_beta }
    }
  }
} catch {
  Write-Warning "[daily_update] Failed to load totals_multipliers_config: $($_.Exception.Message)"
}

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
if ($InstallDeps) {
  Write-Host "[deps] Installing Python dependencies from requirements.txt …" -ForegroundColor DarkGray
  pip install -q -r (Join-Path $RepoRoot "requirements.txt")
} else {
  Write-Host "[deps] Skipping pip install (use -InstallDeps to enable)" -ForegroundColor DarkGray
}
# Optional: lightweight PBP backfill via NHL Web API for recent days (true period splits)
if ($PBPBackfill) {
  try {
    $start = (Get-Date).AddDays(-1 * [int]$PBPDaysBack).ToString('yyyy-MM-dd')
    $end = (Get-Date).ToString('yyyy-MM-dd')
    Write-Host "[daily_update] PBP web backfill $start..$end" -ForegroundColor Yellow
    python scripts/backfill_pbp_webapi.py --start $start --end $end --sleep 0.0
  } catch {
    Write-Warning "[daily_update] PBP web backfill failed: $($_.Exception.Message)"
  }
}

# Always refresh PBP-derived feature caches (PP/PK, penalties) and today's goalie form
try {
  Write-Host "[daily_update] Refreshing PP/PK and penalty rates from PBP …" -ForegroundColor Yellow
  python scripts/build_team_specials_and_penalties_from_pbp.py
} catch {
  Write-Warning "[daily_update] Failed PP/PK & penalties refresh: $($_.Exception.Message)"
}
try {
  Write-Host "[daily_update] Refreshing goalie recent form for today …" -ForegroundColor Yellow
  python scripts/build_goalie_form_from_pbp.py
} catch {
  Write-Warning "[daily_update] Failed goalie form refresh: $($_.Exception.Message)"
}
try {
  Write-Host "[daily_update] Refreshing team xGF/60 (MoneyPuck) …" -ForegroundColor Yellow
  $today = (Get-Date).ToString('yyyy-MM-dd')
  python scripts/build_team_xg_moneypuck.py --date $today
} catch {
  Write-Warning "[daily_update] Failed team xG refresh: $($_.Exception.Message)"
}
# Run daily update workflow
$argsList = @("-m", "nhl_betting.scripts.daily_update", "--days-ahead", "$DaysAhead", "--years-back", "$YearsBack")
if ($NoReconcile) { $argsList += "--no-reconcile" }
python @argsList

# After core daily update, recompute edges for today and forward DaysAhead-1 days
try {
  $base = Get-Date
  for ($i = 0; $i -lt $DaysAhead; $i++) {
    $d = $base.AddDays($i).ToString('yyyy-MM-dd')
    Write-Host "[daily_update] Recomputing team market edges for $d …" -ForegroundColor Cyan
    python -m nhl_betting.cli game-recompute-edges --date $d
  }
} catch {
  Write-Warning "[daily_update] game-recompute-edges failed: $($_.Exception.Message)"
}

# Optional: run game simulations (ML/PL/Totals) driven by NN outputs
if ($SimulateGames) {
  try {
    $base = Get-Date
    for ($i = 0; $i -lt $DaysAhead; $i++) {
      $d = $base.AddDays($i).ToString('yyyy-MM-dd')
      try {
        Write-Host "[daily_update] Refreshing referee assignments for $d …" -ForegroundColor Yellow
        python scripts/build_referee_assignments.py --start $d --end $d
      } catch {
        Write-Warning "[daily_update] Failed referee assignments refresh for ${d}: $($_.Exception.Message)"
      }
      Write-Host "[daily_update] Simulating games for $d (n=$SimSamples) …" -ForegroundColor Yellow
      $pyArgs = @("-m", "nhl_betting.cli", "game-simulate", "--date", $d, "--n-sims", "$SimSamples")
      if ($SimOverK -gt 0) { $pyArgs += @("--sim-overdispersion-k", "$SimOverK") }
      if ($SimSharedK -gt 0) { $pyArgs += @("--sim-shared-k", "$SimSharedK") }
      if ($SimEmptyNetP -gt 0) { $pyArgs += @("--sim-empty-net-p", "$SimEmptyNetP") }
      if ($SimEmptyNetTwoGoalScale -gt 0) { $pyArgs += @("--sim-empty-net-two-goal-scale", "$SimEmptyNetTwoGoalScale") }
      if ($TotalsPaceAlpha -gt 0) { $pyArgs += @("--totals-pace-alpha", "$TotalsPaceAlpha") }
      if ($TotalsGoalieBeta -gt 0) { $pyArgs += @("--totals-goalie-beta", "$TotalsGoalieBeta") }
      if ($TotalsFatigueBeta -gt 0) { $pyArgs += @("--totals-fatigue-beta", "$TotalsFatigueBeta") }
      if ($TotalsPPGamma -gt 0) { $pyArgs += @("--totals-pp-gamma", "$TotalsPPGamma") }
      if ($TotalsPKBeta -gt 0) { $pyArgs += @("--totals-pk-beta", "$TotalsPKBeta") }
      if ($TotalsPenaltyGamma -gt 0) { $pyArgs += @("--totals-penalty-gamma", "$TotalsPenaltyGamma") }
      if ($TotalsXGGamma -gt 0) { $pyArgs += @("--totals-xg-gamma", "$TotalsXGGamma") }
      Write-Host "[daily_update] Totals adj: pace=$TotalsPaceAlpha xg=$TotalsXGGamma refs=$TotalsRefsGamma gform=$TotalsGoalieFormGamma goalie=$TotalsGoalieBeta fatigue=$TotalsFatigueBeta roll=$TotalsRollingPaceGamma pp=$TotalsPPGamma pk=$TotalsPKBeta pen=$TotalsPenaltyGamma en2=$SimEmptyNetTwoGoalScale" -ForegroundColor DarkCyan
      if ($TotalsRollingPaceGamma -gt 0) { $pyArgs += @("--totals-rolling-pace-gamma", "$TotalsRollingPaceGamma") }
      if ($TotalsRefsGamma -gt 0) { $pyArgs += @("--totals-refs-gamma", "$TotalsRefsGamma") }
      if ($TotalsGoalieFormGamma -gt 0) { $pyArgs += @("--totals-goalie-form-gamma", "$TotalsGoalieFormGamma") }
      python @pyArgs
    }
  } catch {
    Write-Warning "[daily_update] game-simulate failed: $($_.Exception.Message)"
  }
}

# Optional: generate threshold-based recommendations from simulations
if ($SimRecommendations) {
  try {
    $base = Get-Date
    for ($i = 0; $i -lt $DaysAhead; $i++) {
      $d = $base.AddDays($i).ToString('yyyy-MM-dd')
      $totFlag = if ($SimIncludeTotals) { "--include-totals" } else { "--no-include-totals" }
      Write-Host "[daily_update] Generating sim recommendations for $d (ML>=$SimMLThr, PL>=$SimPLThr, Tot>=$SimTotThr, includeTotals=$SimIncludeTotals) …" -ForegroundColor Cyan
      python -m nhl_betting.cli game-recommendations-sim --date $d --include-ml --include-puckline $totFlag --ml-thr $SimMLThr --tot-thr $SimTotThr --pl-thr $SimPLThr
    }
    Write-Host "[daily_update] Sim picks written to data/processed/sim_picks_YYYY-MM-DD.csv"
  } catch {
    Write-Warning "[daily_update] game-recommendations-sim failed: $($_.Exception.Message)"
  }
}

# Optional: run rolling backtest over last BacktestWindowDays (non-fatal)
if ($BacktestSimulations) {
  try {
    $end = (Get-Date).ToString('yyyy-MM-dd')
    $start = (Get-Date).AddDays(-1 * [int]$BacktestWindowDays).ToString('yyyy-MM-dd')
    Write-Host "[daily_update] Backtesting simulations $start..$end …" -ForegroundColor Yellow
    $pyBackArgs = @("-m", "nhl_betting.cli", "game-backtest-sim", "--start", $start, "--end", $end, "--n-sims", "6000", "--use-calibrated")
    if ($SimOverK -gt 0) { $pyBackArgs += @("--sim-overdispersion-k", "$SimOverK") }
    if ($SimSharedK -gt 0) { $pyBackArgs += @("--sim-shared-k", "$SimSharedK") }
    if ($SimEmptyNetP -gt 0) { $pyBackArgs += @("--sim-empty-net-p", "$SimEmptyNetP") }
    if ($SimEmptyNetTwoGoalScale -gt 0) { $pyBackArgs += @("--sim-empty-net-two-goal-scale", "$SimEmptyNetTwoGoalScale") }
    if ($TotalsPaceAlpha -gt 0) { $pyBackArgs += @("--totals-pace-alpha", "$TotalsPaceAlpha") }
    if ($TotalsGoalieBeta -gt 0) { $pyBackArgs += @("--totals-goalie-beta", "$TotalsGoalieBeta") }
    if ($TotalsFatigueBeta -gt 0) { $pyBackArgs += @("--totals-fatigue-beta", "$TotalsFatigueBeta") }
    Write-Host "[daily_update] Backtest totals adj: pace=$TotalsPaceAlpha xg=$TotalsXGGamma refs=$TotalsRefsGamma gform=$TotalsGoalieFormGamma goalie=$TotalsGoalieBeta fatigue=$TotalsFatigueBeta roll=$TotalsRollingPaceGamma pp=$TotalsPPGamma pk=$TotalsPKBeta" -ForegroundColor DarkCyan
    if ($TotalsRollingPaceGamma -gt 0) { $pyBackArgs += @("--totals-rolling-pace-gamma", "$TotalsRollingPaceGamma") }
    if ($TotalsPPGamma -gt 0) { $pyBackArgs += @("--totals-pp-gamma", "$TotalsPPGamma") }
    if ($TotalsPKBeta -gt 0) { $pyBackArgs += @("--totals-pk-beta", "$TotalsPKBeta") }
    if ($TotalsPenaltyGamma -gt 0) { $pyBackArgs += @("--totals-penalty-gamma", "$TotalsPenaltyGamma") }
    if ($TotalsXGGamma -gt 0) { $pyBackArgs += @("--totals-xg-gamma", "$TotalsXGGamma") }
    if ($TotalsRefsGamma -gt 0) { $pyBackArgs += @("--totals-refs-gamma", "$TotalsRefsGamma") }
    if ($TotalsGoalieFormGamma -gt 0) { $pyBackArgs += @("--totals-goalie-form-gamma", "$TotalsGoalieFormGamma") }
    python @pyBackArgs
    Write-Host "[daily_update] Backtest written to data/processed/sim_backtest_${start}_to_${end}.json"
  } catch {
    Write-Warning "[daily_update] game-backtest-sim failed: $($_.Exception.Message)"
  }
}

# Adaptive gate re-learn: if last learn older than 5 days OR 30-day overall ROI < -0.08
try {
  $calPath = Join-Path $RepoRoot 'data/processed/model_calibration.json'
  $monitorPath = Join-Path $RepoRoot 'data/processed/game_daily_monitor.json'
  $needLearn = $false
  $today = (Get-Date).ToString('yyyy-MM-dd')
  if (Test-Path $calPath) {
    $cal = Get-Content $calPath | ConvertFrom-Json
    if ($cal.ev_gates_last_learned_utc) {
      $last = [DateTime]::Parse($cal.ev_gates_last_learned_utc)
      if ((Get-Date) - $last -gt [TimeSpan]::FromDays(5)) { $needLearn = $true }
    } else { $needLearn = $true }
  } else { $needLearn = $true }
  if (Test-Path $monitorPath) {
    $mon = Get-Content $monitorPath | ConvertFrom-Json
    if ($mon.overall -and $mon.overall.roi -lt -0.08) { $needLearn = $true }
  }
  if ($needLearn) {
    $seasonStart = if ((Get-Date).Month -ge 9) { "$(Get-Date).Year-09-01" } else { "$(Get-Date).AddYears(-1).Year-09-01" }
    Write-Host "[daily_update] EV gate re-learn triggered (seasonStart=$seasonStart -> today)" -ForegroundColor Magenta
    python -m nhl_betting.cli game-learn-ev-gates --start $seasonStart --end $today
  } else {
    Write-Host "[daily_update] EV gate re-learn skipped (recent & stable)" -ForegroundColor DarkGreen
  }
} catch {
  Write-Warning "[daily_update] adaptive gate re-learn failed: $($_.Exception.Message)"
}

# Optionally run postgame pipeline after daily update
if ($Postgame) {
  Write-Host "[daily_update] Running postgame for $PostgameDate …"
  python -m nhl_betting.cli props-postgame --date $PostgameDate --stats-source $PostgameStatsSource --window $PostgameWindow --stake $PostgameStake
}

# Write rolling performance monitor JSON for dashboards (non-fatal)
try {
  $wd = 30
  Write-Host "[daily_update] Generating game_daily_monitor for last $wd days …"
  python -m nhl_betting.cli game-daily-monitor --window-days $wd
  Write-Host "[daily_update] Monitor written to data/processed/game_daily_monitor.json"
} catch {
  Write-Warning "[daily_update] game_daily_monitor failed: $($_.Exception.Message)"
}

# Generate anomaly alerts from latest monitor
try {
  Write-Host "[daily_update] Generating anomaly alerts …" -ForegroundColor Yellow
  python -m nhl_betting.cli game-monitor-anomalies
  Write-Host "[daily_update] Alerts written under data/processed/monitor_alerts_*.json"
} catch {
  Write-Warning "[daily_update] game-monitor-anomalies failed: $($_.Exception.Message)"
}

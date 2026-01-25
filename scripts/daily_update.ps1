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
  ,
  [switch]$PropsRecs,
  [switch]$PropsUseSim,
  [double]$PropsMinEv = 0,
  [int]$PropsTop = 400,
  [string]$PropsMinEvPerMarket = "SOG=0.00,GOALS=0.05,ASSISTS=0.00,POINTS=0.12,SAVES=0.02,BLOCKS=0.02",
  [double]$PropsMinProb = 0.0,
  [string]$PropsMinProbPerMarket = "",
  [switch]$PropsIncludeGoalies,
  # Props backtests (projections)
  [switch]$RunPropsBacktests,
  [int]$PropsBacktestDays = 30,
  [string]$PropsBacktestMinEvPerMarket = "",
  # Props backtests (sim-backed)
  [switch]$RunSimPropsBacktests,
  [int]$SimPropsBacktestDays = 13,
  [string]$SimPropsBacktestMinEvPerMarket = ""
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$ProcessedDir = Join-Path $RepoRoot 'data/processed'
$NpuScript = Join-Path $RepoRoot "activate_npu.ps1"

# Ensure QNN env (optional). Dot-source if available so QNN EP is found in this session.
if (Test-Path $NpuScript) {
  . $NpuScript
}

# Defaults for optional params if not bound
if (-not $PSBoundParameters.ContainsKey('SimIncludeTotals')) { $SimIncludeTotals = $false }
# Enable PBP backfill by default unless explicitly disabled
if (-not $PSBoundParameters.ContainsKey('PBPBackfill')) { $PBPBackfill = $true }
if (-not $PSBoundParameters.ContainsKey('PropsIncludeGoalies')) { $PropsIncludeGoalies = $true }

# Defaults for props backtests EV gates if not provided
if (-not $PropsBacktestMinEvPerMarket -or $PropsBacktestMinEvPerMarket.Trim() -eq '') {
  $PropsBacktestMinEvPerMarket = 'SOG=0.00,GOALS=0.05,ASSISTS=0.00,POINTS=0.12,SAVES=0.02,BLOCKS=0.02'
}
if (-not $SimPropsBacktestMinEvPerMarket -or $SimPropsBacktestMinEvPerMarket.Trim() -eq '') {
  # Mild gates to stabilize weaker markets for sim-backed
  $SimPropsBacktestMinEvPerMarket = 'SOG=0.04,GOALS=0.00,ASSISTS=0.00,POINTS=0.08,SAVES=0.02,BLOCKS=0.02'
}

# Optional: load tuned props sim multipliers if present
try {
  $propsCfgPath = Join-Path $ProcessedDir 'props_sim_multipliers_config.json'
  if (Test-Path $propsCfgPath) {
    $propsCfg = Get-Content $propsCfgPath | ConvertFrom-Json
    if ($propsCfg.best) {
      $best = $propsCfg.best
      $PropsXGGamma = [double]($best.props_xg_gamma)
      $PropsPenaltyGamma = [double]($best.props_penalty_gamma)
      $PropsGoalieFormGamma = [double]($best.props_goalie_form_gamma)
      $PropsStrengthGamma = [double]($best.props_strength_gamma)
      Write-Host "[daily_update] Using tuned props sim gammas: xg=$PropsXGGamma pen=$PropsPenaltyGamma gform=$PropsGoalieFormGamma str=$PropsStrengthGamma" -ForegroundColor DarkGreen
    }
  }
} catch { Write-Warning "[daily_update] Failed to load props_sim_multipliers_config: $($_.Exception.Message)" }

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

# Refresh roster, lineup (with co-TOI), and injuries for today & tomorrow
try {
  $today = (Get-Date).ToString('yyyy-MM-dd')
  $tomorrow = (Get-Date).AddDays(1).ToString('yyyy-MM-dd')
  Write-Host "[daily_update] Updating roster snapshot for $today …" -ForegroundColor Yellow
  python -m nhl_betting.cli roster-update --date $today
  Write-Host "[daily_update] Updating lineup + co-TOI for $today …" -ForegroundColor Yellow
  python -m nhl_betting.cli lineup-update --date $today
  Write-Host "[daily_update] Fetching shiftcharts + co-TOI for $today …" -ForegroundColor Yellow
  python -m nhl_betting.cli shifts-update --date $today
  Write-Host "[daily_update] Updating injury snapshot for $today …" -ForegroundColor Yellow
  python -m nhl_betting.cli injury-update --date $today
  Write-Host "[daily_update] Updating lineup + co-TOI for $tomorrow …" -ForegroundColor Yellow
  python -m nhl_betting.cli lineup-update --date $tomorrow
} catch {
  Write-Warning "[daily_update] roster/lineup/injuries update failed: $($_.Exception.Message)"
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
    # Calibrate special-teams multipliers before running possession sims (last 30 days)
    try {
      $calStart = (Get-Date).AddDays(-30).ToString('yyyy-MM-dd')
      $calEnd = (Get-Date).ToString('yyyy-MM-dd')
      Write-Host "[daily_update] Calibrating special teams $calStart..$calEnd …" -ForegroundColor DarkGreen
      python -m nhl_betting.cli game-calibrate-special-teams --start $calStart --end $calEnd
    } catch {
      Write-Warning "[daily_update] Special-teams calibration failed: $($_.Exception.Message)"
    }
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

      # Supplement: write baseline period-level sim outputs for quick props boxscore projections
      try {
        Write-Host "[daily_update] Baseline period-level sim for $d …" -ForegroundColor DarkYellow
        python -m nhl_betting.cli game-simulate-baseline --date $d
      } catch {
        Write-Warning "[daily_update] game-simulate-baseline failed for ${d}: $($_.Exception.Message)"
      }

      # Possession-aware sim using lineups and shift-based TOI
      try {
        Write-Host "[daily_update] Possession-aware sim for $d …" -ForegroundColor DarkYellow
        python -m nhl_betting.cli game-simulate-possession --date $d
      } catch {
        Write-Warning "[daily_update] game-simulate-possession failed for ${d}: $($_.Exception.Message)"
      }
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

# Optional: run rolling props backtests (projections) and write dashboard row
if ($RunPropsBacktests) {
  try {
    $end = (Get-Date).ToString('yyyy-MM-dd')
    $start = (Get-Date).AddDays(-1 * [int]$PropsBacktestDays).ToString('yyyy-MM-dd')
    Write-Host "[daily_update] Backtesting props (projections) $start..$end …" -ForegroundColor Yellow
    python -m nhl_betting.cli props-backtest-from-projections --start $start --end $end --stake 100 --markets "SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS" --min-ev-per-market $PropsBacktestMinEvPerMarket

    # Dashboard update with WoW deltas if previous window exists
    $projSummaryJson = Join-Path $ProcessedDir "nn_daily_props_backtest_summary_${start}_to_${end}.json"
    # previous window: [start-PropsBacktestDays .. (start-1)]
    $startDT = [datetime]::ParseExact($start, 'yyyy-MM-dd', $null)
    $prevEndDT = $startDT.AddDays(-1)
    $prevStartDT = $prevEndDT.AddDays(-$PropsBacktestDays + 1)
    $prevStart = $prevStartDT.ToString('yyyy-MM-dd')
    $prevEnd = $prevEndDT.ToString('yyyy-MM-dd')
    $projPrevSummaryJson = Join-Path $ProcessedDir "nn_daily_props_backtest_summary_${prevStart}_to_${prevEnd}.json"

    $dashboardCsv = Join-Path $ProcessedDir "backtest_daily_summary_${start}_to_${end}.csv"
    if (Test-Path $projPrevSummaryJson) {
      python .\\nhl_betting\\scripts\\backtest_daily_summary.py $projSummaryJson None $dashboardCsv $projPrevSummaryJson None
    } else {
      python .\\nhl_betting\\scripts\\backtest_daily_summary.py $projSummaryJson None $dashboardCsv
    }
    Write-Host "[daily_update] Props projections backtest summary written to $dashboardCsv" -ForegroundColor DarkGreen
  } catch {
    Write-Warning "[daily_update] props-backtest-from-projections failed: $($_.Exception.Message)"
  }
}

# Optional: run rolling props backtests (sim-backed) and update dashboard
if ($RunSimPropsBacktests) {
  try {
    $end = (Get-Date).ToString('yyyy-MM-dd')
    $start = (Get-Date).AddDays(-1 * [int]$SimPropsBacktestDays).ToString('yyyy-MM-dd')
    Write-Host "[daily_update] Backtesting props (sim-backed) $start..$end …" -ForegroundColor Yellow
    python -m nhl_betting.cli props-backtest-from-simulations --start $start --end $end --stake 100 --markets "SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS" --min-ev-per-market $SimPropsBacktestMinEvPerMarket

    $simSummaryJson = Join-Path $ProcessedDir "sim_daily_props_backtest_sim_summary_${start}_to_${end}.json"
    # previous window
    $startDT = [datetime]::ParseExact($start, 'yyyy-MM-dd', $null)
    $prevEndDT = $startDT.AddDays(-1)
    $prevStartDT = $prevEndDT.AddDays(-$SimPropsBacktestDays + 1)
    $prevStart = $prevStartDT.ToString('yyyy-MM-dd')
    $prevEnd = $prevEndDT.ToString('yyyy-MM-dd')
    $simPrevSummaryJson = Join-Path $ProcessedDir "sim_daily_props_backtest_sim_summary_${prevStart}_to_${prevEnd}.json"

    # Use existing projections summary if available
    $projSummaryJson = Join-Path $ProcessedDir "nn_daily_props_backtest_summary_${start}_to_${end}.json"
    $projPrevSummaryJson = Join-Path $ProcessedDir "nn_daily_props_backtest_summary_${prevStart}_to_${prevEnd}.json"

    $projArg = if (Test-Path $projSummaryJson) { $projSummaryJson } else { 'None' }
    $dashboardCsv = Join-Path $ProcessedDir "backtest_daily_summary_${start}_to_${end}.csv"
    if ((Test-Path $projPrevSummaryJson) -and (Test-Path $simPrevSummaryJson)) {
      python .\\nhl_betting\\scripts\\backtest_daily_summary.py $projArg $simSummaryJson $dashboardCsv $projPrevSummaryJson $simPrevSummaryJson
    } elseif (Test-Path $simPrevSummaryJson) {
      python .\\nhl_betting\\scripts\\backtest_daily_summary.py $projArg $simSummaryJson $dashboardCsv None $simPrevSummaryJson
    } elseif (Test-Path $projPrevSummaryJson) {
      python .\\nhl_betting\\scripts\\backtest_daily_summary.py $projArg $simSummaryJson $dashboardCsv $projPrevSummaryJson None
    } else {
      python .\\nhl_betting\\scripts\\backtest_daily_summary.py $projArg $simSummaryJson $dashboardCsv
    }
    Write-Host "[daily_update] Sim-backed props backtest summary written to $dashboardCsv" -ForegroundColor DarkGreen
  } catch {
    Write-Warning "[daily_update] props-backtest-from-simulations failed: $($_.Exception.Message)"
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

# Optional: precompute props projections and generate props recommendations
# Note: Web projections now compute strength-aware p_over using scaled lambda (proj_lambda_eff)
if ($PropsRecs) {
  try {
    $today = (Get-Date).ToString('yyyy-MM-dd')
    $tomorrow = (Get-Date).AddDays(1).ToString('yyyy-MM-dd')
    Write-Host "[daily_update] Precomputing props projections for $today & $tomorrow …" -ForegroundColor Yellow
    $projArgsBase = @("-m", "nhl_betting.cli", "props-project-all", "--ensure-history-days", "365")
    if ($PropsIncludeGoalies) { $projArgsBase += "--include-goalies" }
    python @($projArgsBase + @("--date", $today))
    python @($projArgsBase + @("--date", $tomorrow))

    if ($PropsUseSim) {
      Write-Host "[daily_update] Simulating props for $today & $tomorrow …" -ForegroundColor Yellow
      $simArgsBase = @("-m", "nhl_betting.cli", "props-simulate", "--markets", "SOG,GOALS,ASSISTS,POINTS,SAVES,BLOCKS", "--n-sims", "16000", "--sim-shared-k", "1.2")
      if ($PropsXGGamma) { $simArgsBase += @("--props-xg-gamma", "$PropsXGGamma") } else { $simArgsBase += @("--props-xg-gamma", "0.02") }
      if ($PropsPenaltyGamma) { $simArgsBase += @("--props-penalty-gamma", "$PropsPenaltyGamma") } else { $simArgsBase += @("--props-penalty-gamma", "0.06") }
      if ($PropsGoalieFormGamma) { $simArgsBase += @("--props-goalie-form-gamma", "$PropsGoalieFormGamma") } else { $simArgsBase += @("--props-goalie-form-gamma", "0.02") }
      if ($PropsStrengthGamma) { $simArgsBase += @("--props-strength-gamma", "$PropsStrengthGamma") } else { $simArgsBase += @("--props-strength-gamma", "0.04") }
      python @($simArgsBase + @("--date", $today))
      python @($simArgsBase + @("--date", $tomorrow))
      # Supplement: simulate SAVES/BLOCKS independent of provider lines
      Write-Host "[daily_update] Simulating nolines props (SAVES/BLOCKS) for $today & $tomorrow …" -ForegroundColor Yellow
      $nolArgsBase = @("-m", "nhl_betting.cli", "props-simulate-unlined", "--markets", "SAVES,BLOCKS", "--candidate-lines", "SAVES=24.5,26.5,28.5,30.5;BLOCKS=1.5,2.5,3.5", "--n-sims", "16000", "--sim-shared-k", "1.2")
      if ($PropsXGGamma) { $nolArgsBase += @("--props-xg-gamma", "$PropsXGGamma") } else { $nolArgsBase += @("--props-xg-gamma", "0.02") }
      if ($PropsPenaltyGamma) { $nolArgsBase += @("--props-penalty-gamma", "$PropsPenaltyGamma") } else { $nolArgsBase += @("--props-penalty-gamma", "0.06") }
      if ($PropsGoalieFormGamma) { $nolArgsBase += @("--props-goalie-form-gamma", "$PropsGoalieFormGamma") } else { $nolArgsBase += @("--props-goalie-form-gamma", "0.02") }
      python @($nolArgsBase + @("--date", $today))
      python @($nolArgsBase + @("--date", $tomorrow))
      Write-Host "[daily_update] Generating SIM-based props recommendations for $today & $tomorrow …" -ForegroundColor Cyan
      # Weekly auto-tune for SAVES nolines gate (range 0.65–0.68) based on 7-day monitor
      $SavesGate = 0.65
      try {
        if ((Get-Date).DayOfWeek -eq 'Monday') {
          Write-Host "[daily_update] Auto-tuning SAVES gate via 7-day monitor …" -ForegroundColor DarkGreen
          $monCmd = @("-m", "nhl_betting.cli", "props-nolines-monitor", "--window-days", "7", "--markets", "SAVES,BLOCKS", "--min-prob-per-market", "SAVES=$SavesGate,BLOCKS=0.92")
          python $monCmd
          $monPath = Join-Path "data\processed" "props_nolines_monitor.json"
          if (Test-Path $monPath) {
            $mon = Get-Content $monPath -Raw | ConvertFrom-Json
            $acc = [double]$mon.by_market.SAVES.accuracy
            $brier = [double]$mon.by_market.SAVES.brier
            if (($acc -lt 0.88) -or ($brier -gt 0.16)) { $SavesGate = 0.68 }
            elseif (($acc -lt 0.90) -or ($brier -gt 0.15)) { $SavesGate = 0.67 }
            elseif (($acc -lt 0.92) -or ($brier -gt 0.14)) { $SavesGate = 0.66 }
            else { $SavesGate = 0.65 }
            Write-Host "[daily_update] SAVES gate set to $SavesGate (acc=$acc brier=$brier)" -ForegroundColor DarkGreen
          }
        }
      } catch {
        Write-Warning "[daily_update] Auto-tune failed: $($_.Exception.Message)"
      }
      $recsSimBase = @("-m", "nhl_betting.cli", "props-recommendations-sim", "--min-ev", "$PropsMinEv", "--top", "$PropsTop", "--min-ev-per-market", $PropsMinEvPerMarket, "--min-prob", "$PropsMinProb", "--min-prob-per-market", $PropsMinProbPerMarket)
      python @($recsSimBase + @("--date", $today))
      python @($recsSimBase + @("--date", $tomorrow))
      # Also produce nolines-only recommendations (SAVES/BLOCKS) without odds
      Write-Host "[daily_update] Generating nolines props recommendations for $today & $tomorrow …" -ForegroundColor Cyan
      $recsNoBase = @("-m", "nhl_betting.cli", "props-recommendations-nolines", "--markets", "SAVES,BLOCKS", "--top", "$PropsTop", "--min-prob-per-market", "SAVES=$SavesGate,BLOCKS=0.92")
      python @($recsNoBase + @("--date", $today))
      python @($recsNoBase + @("--date", $tomorrow))
      # Combine EV-based and nolines into one output per day
      Write-Host "[daily_update] Combining EV-based and nolines recommendations …" -ForegroundColor DarkCyan
      python -m nhl_betting.cli props-recommendations-combined --date $today
      python -m nhl_betting.cli props-recommendations-combined --date $tomorrow
      # Write 7-day nolines monitor
      Write-Host "[daily_update] Generating nolines monitor (7-day) …" -ForegroundColor DarkGreen
      python -m nhl_betting.cli props-nolines-monitor --window-days 7 --markets "SAVES,BLOCKS" --min-prob-per-market "SAVES=$SavesGate,BLOCKS=0.92"
    } else {
      Write-Host "[daily_update] Generating props recommendations (model-only) for $today & $tomorrow …" -ForegroundColor Cyan
      $recsArgsBase = @("-m", "nhl_betting.cli", "props-recommendations", "--min-ev", "$PropsMinEv", "--top", "$PropsTop", "--min-ev-per-market", $PropsMinEvPerMarket)
      python @($recsArgsBase + @("--date", $today))
      python @($recsArgsBase + @("--date", $tomorrow))
    }
  } catch {
    Write-Warning "[daily_update] Props projections/recommendations failed: $($_.Exception.Message)"
  }
}

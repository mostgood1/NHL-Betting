Param (
  # Optional: override the "today" anchor date for all date computations in this script.
  # Useful during schedule breaks (e.g., Olympics) to test against a known slate.
  [string]$BaseDate = "",
  [int]$DaysAhead = 2,
  [int]$YearsBack = 2,
  [int]$CoreTimeoutSec = 600,
  [int]$PropsBoxscoreNSims = 6000,
  [int]$PropsBoxscoreTimeoutSec = 3600,
  [int]$GameSimTimeoutSec = 600,
  [switch]$StrictOutputs,
  # Legacy compatibility switches retained so older tasks/helpers can call the canonical script.
  [switch]$Quiet,
  [switch]$BootstrapModels,
  [double]$TrendsDecay = 0.98,
  [switch]$ResetTrends,
  [switch]$SkipProps,
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
  [int]$SimCalibrationTrainDays = 52,
  [int]$SimCalibrationRefreshDays = 7,
  [int]$SimCalibrationNSims = 6000,
  [bool]$SimCalibrationTotalsOnly = $true,
  [bool]$MoneylineAnchorOnly = $true,
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
  [string]$PropsMinProbPerMarket = "SOG=0.75,GOALS=0.60,ASSISTS=0.60,POINTS=0.60,SAVES=0.60,BLOCKS=0.60",
  [string]$PropsNolinesMarkets = "SAVES,BLOCKS,SOG",
  [string]$PropsNolinesMinProbPerMarket = "SAVES=0.65,BLOCKS=0.92,SOG=0.72",
  # When omitted, inherit the CLI defaults for props under-side gating.
  [double]$PropsUnderMinEv = 0,
  [int]$PropsUnderMaxJuice = 0,
  [string]$PropsUnderMaxJuicePerMarket = "",
  [int]$PropsMaxPlusOdds = 300,
  [switch]$PropsIncludeGoalies,
  # Props historical backfill (OddsAPI historical event props)
  [switch]$PropsBackfillRange,
  [int]$PropsBackfillDays = 30,
  # Props backtests (projections)
  [Alias('RunBacktests')]
  [switch]$RunPropsBacktests,
  [Alias('BacktestDays')]
  [int]$PropsBacktestDays = 30,
  [Alias('BacktestMinEvPerMarket')]
  [string]$PropsBacktestMinEvPerMarket = "",
  # Props backtests (sim-backed)
  [Alias('RunSimBacktests')]
  [switch]$RunSimPropsBacktests,
  [Alias('SimBacktestDays')]
  [int]$SimPropsBacktestDays = 30,
  [Alias('SimBacktestMinEvPerMarket')]
  [string]$SimPropsBacktestMinEvPerMarket = "",

  [switch]$SkipPropsProjections,
  [switch]$SkipPropsCalibration,
  [switch]$SkipGameCalibration,
  [switch]$RecomputeRecs,

  # Play-level props boxscore sim tuning (impacts web-facing props lambdas)
  [string]$PropsToiMode = "auto",
  [string]$PropsUsageModel = "deterministic",
  [double]$PropsUsageNoisySigma = 0.18,
  [string]$PropsStarterSource = "auto",
  [double]$PropsSavesCal = 0.85,
  [int]$PropsSeed = 42,

  # Optional: sync disk-backed artifacts from Render down to local before reconciliation.
  # This keeps local analysis/accuracy in lockstep with production-collected Live Lens + lines.
  [switch]$NoRenderSync,
  [string]$RenderBaseUrl = "",
  [string]$RenderToken = "",
  [int]$RenderSyncDaysBack = 14,
  [int]$RenderSyncDaysAhead = 0,

  # Git auto-push controls (enabled by default to keep generated artifacts published)
  [switch]$NoGitPush,
  [string]$GitRemote = "origin",
  [string]$GitBranch = ""
)
$ErrorActionPreference = "Stop"
$PropsUnderMinEvBound = $PSBoundParameters.ContainsKey('PropsUnderMinEv')
$PropsUnderMaxJuiceBound = $PSBoundParameters.ContainsKey('PropsUnderMaxJuice')
$PropsUnderMaxJuicePerMarketBound = $PSBoundParameters.ContainsKey('PropsUnderMaxJuicePerMarket')
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$ProcessedDir = Join-Path $RepoRoot 'data/processed'
$NpuScript = Join-Path $RepoRoot "activate_npu.ps1"

# Resolve anchor date for this run
$AnchorNow = $null
try {
  if ($BaseDate -and $BaseDate.Trim() -ne "") {
    $AnchorNow = Get-Date $BaseDate
  } else {
    $AnchorNow = Get-Date
  }
} catch {
  $AnchorNow = Get-Date
}

function Assert-AnyPath {
  param(
    [Parameter(Mandatory=$true)][string]$Label,
    [Parameter(Mandatory=$true)][string[]]$Paths,
    [switch]$NonEmpty,
    [switch]$Strict
  )
  $found = $false
  foreach ($p in $Paths) {
    if (Test-Path $p) {
      if ($NonEmpty) {
        try {
          $len = (Get-Item $p).Length
          if ($len -gt 0) { $found = $true; break }
        } catch {
          # If we can't stat it, treat existence as sufficient
          $found = $true; break
        }
      } else {
        $found = $true; break
      }
    }
  }
  if (-not $found) {
    $msg = "[daily_update] Missing expected output for $Label. Checked: $($Paths -join '; ')"
    if ($Strict) { throw $msg } else { Write-Warning $msg }
  }
}

function _PathInProcessed {
  Param([Parameter(Mandatory=$true)][string]$FileName)
  return (Join-Path $ProcessedDir $FileName)
}

function Has-NonEmptyCsv {
  Param([Parameter(Mandatory=$true)][string]$Path)
  try {
    if (-not (Test-Path $Path)) { return $false }
    # Read first two lines: header + at least one data row
    $lines = Get-Content -LiteralPath $Path -TotalCount 2 -ErrorAction Stop
    return ($lines.Count -ge 2)
  } catch {
    return $false
  }
}

function Set-MarketThreshold {
  param(
    [Parameter(Mandatory=$true)][string]$Thresholds,
    [Parameter(Mandatory=$true)][string]$Market,
    [Parameter(Mandatory=$true)][double]$Value
  )

  $entries = [ordered]@{}
  foreach ($part in ($Thresholds -split ',')) {
    $trimmed = $part.Trim()
    if (-not $trimmed -or $trimmed.IndexOf('=') -lt 0) { continue }
    $pieces = $trimmed -split '=', 2
    $key = $pieces[0].Trim().ToUpper()
    $rawVal = $pieces[1].Trim()
    if (-not $key) { continue }
    $entries[$key] = $rawVal
  }
  $entries[$Market.Trim().ToUpper()] = ('{0:0.##}' -f $Value)
  return (($entries.GetEnumerator() | ForEach-Object { "{0}={1}" -f $_.Key, $_.Value }) -join ',')
}

function Set-MoneylineAnchorOnlyCalibration {
  param(
    [Parameter(Mandatory=$true)][string]$CalibrationPath,
    [switch]$Strict
  )

  if (-not (Test-Path $CalibrationPath)) {
    $msg = "[daily_update] model calibration not found at $CalibrationPath; cannot enforce moneyline anchor-only policy."
    if ($Strict) { throw $msg }
    Write-Host $msg -ForegroundColor DarkGray
    return
  }

  try {
    $raw = Get-Content -LiteralPath $CalibrationPath -Raw -ErrorAction Stop
    if (-not $raw -or $raw.Trim() -eq '') {
      throw "calibration file is empty"
    }
    $cal = $raw | ConvertFrom-Json -ErrorAction Stop
    $needsWrite = $false

    if ($null -eq $cal.moneyline) {
      $cal | Add-Member -NotePropertyName moneyline -NotePropertyValue ([pscustomobject]@{ t = 1.0; b = 0.0 })
      $needsWrite = $true
    } else {
      if ([double]$cal.moneyline.t -ne 1.0) { $cal.moneyline.t = 1.0; $needsWrite = $true }
      if ([double]$cal.moneyline.b -ne 0.0) { $cal.moneyline.b = 0.0; $needsWrite = $true }
    }
    if ([double]$cal.ml_temp -ne 1.0) { $cal.ml_temp = 1.0; $needsWrite = $true }
    if ([double]$cal.ml_bias -ne 0.0) { $cal.ml_bias = 0.0; $needsWrite = $true }
    if ($cal.ml_post_calibration_policy -ne 'anchor_only') {
      if ($cal.PSObject.Properties.Name -contains 'ml_post_calibration_policy') {
        $cal.ml_post_calibration_policy = 'anchor_only'
      } else {
        $cal | Add-Member -NotePropertyName ml_post_calibration_policy -NotePropertyValue 'anchor_only'
      }
      $needsWrite = $true
    }
    if ($needsWrite) {
      $cal | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $CalibrationPath -Encoding UTF8
      Write-Host "[daily_update] Enforced moneyline anchor-only calibration policy at $CalibrationPath" -ForegroundColor DarkGreen
    } else {
      Write-Host "[daily_update] Moneyline anchor-only calibration already active at $CalibrationPath" -ForegroundColor DarkGreen
    }
  } catch {
    $msg = "[daily_update] Failed to enforce moneyline anchor-only calibration: $($_.Exception.Message)"
    if ($Strict) { throw $msg }
    Write-Warning $msg
  }
}

function Sync-RenderArtifacts {
  param(
    [Parameter(Mandatory=$true)][string]$BaseUrl,
    [Parameter(Mandatory=$true)][string]$Token,
    [Parameter(Mandatory=$true)][string]$StartDate,
    [Parameter(Mandatory=$true)][string]$EndDate,
    [Parameter(Mandatory=$true)][string]$RepoRoot
  )
  try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  } catch { }

  $base = ($BaseUrl.TrimEnd('/'))
  $Headers = @{ Authorization = "Bearer $Token" }
  $url = "$base/api/cron/export-artifacts?start=$StartDate&end=$EndDate&include_live_lens=true&include_perf=true&include_props=true&include_odds=true&include_processed=false"
  $tmpZip = Join-Path $env:TEMP ("nhl_artifacts_{0}.zip" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

  Write-Host "[daily_update] Syncing Render artifacts ($StartDate..$EndDate) …" -ForegroundColor Yellow
  Invoke-WebRequest -Uri $url -Method GET -Headers $Headers -OutFile $tmpZip | Out-Null

  Write-Host "[daily_update] Extracting artifacts into repo …" -ForegroundColor Yellow
  Expand-Archive -LiteralPath $tmpZip -DestinationPath $RepoRoot -Force
  Remove-Item -LiteralPath $tmpZip -Force -ErrorAction SilentlyContinue
}

function Invoke-GitCommand {
  param(
    [Parameter(Mandatory=$true)][string[]]$Args,
    [string]$FailureMessage = "git command failed"
  )

  # Disable auto-gc for automation runs to avoid interactive prompts (e.g. failed object cleanup on OneDrive).
  $cmdArgs = @('-c', 'gc.auto=0') + $Args
  $previousErrorActionPreference = $ErrorActionPreference
  try {
    # Native git warnings may write to stderr even when the command succeeds.
    # Keep those as captured output and rely on the actual exit code for failure.
    $ErrorActionPreference = 'Continue'
    $result = & git @cmdArgs 2>&1
  } finally {
    $ErrorActionPreference = $previousErrorActionPreference
  }
  $exitCode = $LASTEXITCODE
  $output = @($result | ForEach-Object {
    if ($_ -is [System.Management.Automation.ErrorRecord]) {
      $_.ToString()
    } else {
      [string]$_
    }
  })

  if ($exitCode -ne 0) {
    $detail = ($output -join [Environment]::NewLine).Trim()
    if (-not $detail) { $detail = "git exited with code $exitCode" }
    throw "$FailureMessage`n$detail"
  }

  return $output
}

function Test-GitIdentityLooksConfigured {
  param(
    [string]$UserName,
    [string]$UserEmail
  )

  if (-not $UserName -or -not $UserEmail) { return $false }

  $trimmedName = $UserName.Trim()
  $trimmedEmail = $UserEmail.Trim()
  if (-not $trimmedName -or -not $trimmedEmail) { return $false }
  if ($trimmedName -eq 'Your Name') { return $false }
  if ($trimmedEmail -eq 'your.email@example.com') { return $false }

  return $true
}

# Ensure QNN env (optional). Dot-source if available so QNN EP is found in this session.
if (Test-Path $NpuScript) {
  . $NpuScript
}

# Defaults for optional params if not bound
if (-not $PSBoundParameters.ContainsKey('SimIncludeTotals')) { $SimIncludeTotals = $false }
# Enable PBP backfill by default unless explicitly disabled
if (-not $PSBoundParameters.ContainsKey('PBPBackfill')) { $PBPBackfill = $true }
if (-not $PSBoundParameters.ContainsKey('PropsIncludeGoalies')) { $PropsIncludeGoalies = $true }
if (-not $PSBoundParameters.ContainsKey('PropsRecs')) { $PropsRecs = $true }
if (-not $PSBoundParameters.ContainsKey('PropsUseSim')) { $PropsUseSim = $true }
if (-not $PSBoundParameters.ContainsKey('PropsMinProbPerMarket') -or [string]::IsNullOrWhiteSpace($PropsMinProbPerMarket)) {
  $PropsMinProbPerMarket = 'SOG=0.75,GOALS=0.60,ASSISTS=0.60,POINTS=0.60,SAVES=0.60,BLOCKS=0.60'
}
if ($SkipPropsProjections) { $env:PROPS_SKIP_PROJECTIONS = '1' }
if ($SkipPropsCalibration) { $env:SKIP_PROPS_CALIBRATION = '1' }
if ($SkipProps) {
  $PropsRecs = $false
  $PropsBackfillRange = $false
}

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
# Use venv Python for timed subprocesses
$PyExe = Join-Path $RepoRoot '.venv/Scripts/python.exe'

if ($MoneylineAnchorOnly) {
  Set-MoneylineAnchorOnlyCalibration -CalibrationPath (Join-Path $ProcessedDir 'model_calibration.json') -Strict:$StrictOutputs
} else {
  Write-Host "[daily_update] Moneyline anchor-only override disabled." -ForegroundColor DarkGray
}

# Optional: pull production disk artifacts down to local for reconciliation.
try {
  if (-not $NoRenderSync) {
    $baseUrl = $RenderBaseUrl
    if (-not $baseUrl -or $baseUrl.Trim() -eq "") { $baseUrl = $env:NHL_RENDER_BASE_URL }
    if (-not $baseUrl -or $baseUrl.Trim() -eq "") { $baseUrl = $env:RENDER_BASE_URL }
    if (-not $baseUrl -or $baseUrl.Trim() -eq "") { $baseUrl = "https://nhl-betting.onrender.com" }

    $tok = $RenderToken
    if (-not $tok -or $tok.Trim() -eq "") { $tok = $env:NHL_RENDER_CRON_TOKEN }
    if (-not $tok -or $tok.Trim() -eq "") { $tok = $env:RENDER_CRON_TOKEN }
    if (-not $tok -or $tok.Trim() -eq "") { $tok = $env:REFRESH_CRON_TOKEN }

    if ($tok -and $tok.Trim() -ne "") {
      $syncStart = $AnchorNow.AddDays(-1 * [int]$RenderSyncDaysBack).ToString('yyyy-MM-dd')
      $syncEnd = $AnchorNow.AddDays([int]$RenderSyncDaysAhead).ToString('yyyy-MM-dd')
      Sync-RenderArtifacts -BaseUrl $baseUrl -Token $tok -StartDate $syncStart -EndDate $syncEnd -RepoRoot $RepoRoot
    } else {
      Write-Host "[daily_update] Render sync skipped (missing token). Set NHL_RENDER_CRON_TOKEN or REFRESH_CRON_TOKEN." -ForegroundColor DarkGreen
    }
  } else {
    Write-Host "[daily_update] Render sync skipped (-NoRenderSync)" -ForegroundColor DarkGreen
  }
} catch {
  Write-Warning "[daily_update] Render sync failed (continuing): $($_.Exception.Message)"
}
# Optional: lightweight PBP backfill via NHL Web API for recent days (true period splits)
if ($PBPBackfill) {
  try {
    $start = $AnchorNow.AddDays(-1 * [int]$PBPDaysBack).ToString('yyyy-MM-dd')
    $end = $AnchorNow.ToString('yyyy-MM-dd')
    Write-Host "[daily_update] PBP web backfill $start..$end" -ForegroundColor Yellow
    python scripts/backfill_pbp_webapi.py --start $start --end $end --sleep 0.0
  } catch {
    Write-Warning "[daily_update] PBP web backfill failed: $($_.Exception.Message)"
  }
}

# Refresh roster, lineup (with co-TOI), and injuries for today & tomorrow
try {
  $today = $AnchorNow.ToString('yyyy-MM-dd')
  $tomorrow = $AnchorNow.AddDays(1).ToString('yyyy-MM-dd')
  Write-Host "[daily_update] Updating roster snapshot for $today …" -ForegroundColor Yellow
  python -m nhl_betting.cli roster-update --date $today
  Write-Host "[daily_update] Updating roster snapshot for $tomorrow …" -ForegroundColor Yellow
  python -m nhl_betting.cli roster-update --date $tomorrow
  Write-Host "[daily_update] Updating lineup + co-TOI for $today …" -ForegroundColor Yellow
  python -m nhl_betting.cli lineup-update --date $today
  Write-Host "[daily_update] Fetching shiftcharts + co-TOI for $today …" -ForegroundColor Yellow
  python -m nhl_betting.cli shifts-update --date $today
  Write-Host "[daily_update] Updating injury snapshot for $today …" -ForegroundColor Yellow
  python -m nhl_betting.cli injury-update --date $today
  Write-Host "[daily_update] Updating lineup + co-TOI for $tomorrow …" -ForegroundColor Yellow
  python -m nhl_betting.cli lineup-update --date $tomorrow

  # Sanity checks (lineups are foundational; shifts can be missing and will fall back to TOI history)
  Assert-AnyPath -Label "lineups $today" -Paths @(_PathInProcessed "lineups_${today}.csv") -NonEmpty -Strict:$StrictOutputs
  Assert-AnyPath -Label "lineups $tomorrow" -Paths @(_PathInProcessed "lineups_${tomorrow}.csv") -NonEmpty -Strict:$StrictOutputs
  Assert-AnyPath -Label "shifts $today (optional)" -Paths @(_PathInProcessed "shifts_${today}.csv") -NonEmpty -Strict:$false
} catch {
  Write-Warning "[daily_update] roster/lineup/injuries update failed: $($_.Exception.Message)"
}

# Helper date array (today/tomorrow) used by multiple sections
$dates = @($AnchorNow.ToString('yyyy-MM-dd'), $AnchorNow.AddDays(1).ToString('yyyy-MM-dd'))

# Generate simulated player boxscores for today & tomorrow (play-level aggregation for web cards)
try {
  foreach ($d in $dates) {
    $lineupsPath = _PathInProcessed "lineups_${d}.csv"
    $hasSlate = Has-NonEmptyCsv $lineupsPath
    if (-not $hasSlate) {
      Write-Host "[daily_update] No slate detected for ${d} (lineups empty/missing); skipping sims." -ForegroundColor DarkGray
      continue
    }
    Write-Host "[daily_update] Generating simulated player boxscores for $d …" -ForegroundColor DarkYellow
    try {
      # Run with timeout to prevent stalls on future slates
      $job1 = Start-Job -ScriptBlock {
        Set-Location $using:RepoRoot
        & $using:PyExe -m nhl_betting.cli props-simulate-boxscores --date $using:d --n-sims $using:PropsBoxscoreNSims --seed $using:PropsSeed --toi-mode $using:PropsToiMode --usage-model $using:PropsUsageModel --usage-noisy-sigma $using:PropsUsageNoisySigma --starter-source $using:PropsStarterSource --saves-cal $using:PropsSavesCal 2>&1
      }
      $done1 = Wait-Job $job1 -Timeout $PropsBoxscoreTimeoutSec
      if (-not $done1) {
        try { Stop-Job $job1 -Force } catch {}
        Remove-Job $job1 -Force
        Write-Warning "[daily_update] props-simulate-boxscores timed out for ${d}; skipping."
        if ($StrictOutputs) { throw "[daily_update] props-simulate-boxscores timed out for ${d} (timeout=${PropsBoxscoreTimeoutSec}s)" }
      } else {
        Receive-Job $job1 | Write-Host
        Remove-Job $job1 -Force

        # Verify expected sim artifacts were produced (only if slate exists)
        if ($StrictOutputs) {
          Assert-AnyPath -Label "props_boxscores_sim $d" -Paths @(
            (_PathInProcessed "props_boxscores_sim_${d}.csv")
          ) -NonEmpty -Strict:$StrictOutputs
          Assert-AnyPath -Label "props_boxscores_sim_hist $d" -Paths @(
            (_PathInProcessed "props_boxscores_sim_hist_${d}.csv")
          ) -NonEmpty -Strict:$StrictOutputs
          Assert-AnyPath -Label "props_boxscores_sim_samples $d" -Paths @(
            (_PathInProcessed "props_boxscores_sim_samples_${d}.parquet"),
            (_PathInProcessed "props_boxscores_sim_samples_${d}.csv")
          ) -NonEmpty -Strict:$StrictOutputs
        }
      }
    } catch {
      Write-Warning "[daily_update] props-simulate-boxscores failed for ${d}: $($_.Exception.Message)"
      if ($StrictOutputs) { throw }
    }

    # Ensure team odds exist before sim-native EV computations
    try {
      $teamDir = Join-Path $RepoRoot "data/odds/team/date=$d"
      if (-not (Test-Path $teamDir)) { New-Item -ItemType Directory -Path $teamDir | Out-Null }
      $teamCsv = Join-Path $teamDir 'oddsapi.csv'
      $needTeam = $true
      if (Test-Path $teamCsv) {
        try {
          $cnt = (Import-Csv $teamCsv).Count
          if ($cnt -gt 0) { $needTeam = $false }
        } catch { }
      }
      if ($needTeam) {
        Write-Host "[daily_update] Collecting team odds for $d (pre-sim EV) …" -ForegroundColor Yellow
        python -m nhl_betting.cli team-odds-collect --date $d
      }
    } catch {
      Write-Warning "[daily_update] team-odds-collect (pre-sim) failed for ${d}: $($_.Exception.Message)"
      if ($StrictOutputs) { throw }
    }

    # Produce sim-native game predictions and edges (ML/Totals) directly from per-sim samples
    try {
      $totFlag = if ($SimIncludeTotals) { "--include-totals" } else { "--no-include-totals" }
      Write-Host "[daily_update] Computing sim-native game predictions for $d (ML>=$SimMLThr, PL>=$SimPLThr, Tot>=$SimTotThr, includeTotals=$SimIncludeTotals) …" -ForegroundColor DarkYellow
      # Run with timeout to prevent stalls if samples are unavailable
      $job2 = Start-Job -ScriptBlock { Set-Location $using:RepoRoot; & $using:PyExe -m nhl_betting.cli game-recommendations-sim --date $using:d --top 200 $using:totFlag --ml-prob-thr $using:SimMLThr --tot-prob-thr $using:SimTotThr --pl-prob-thr $using:SimPLThr 2>&1 }
      $done2 = Wait-Job $job2 -Timeout $GameSimTimeoutSec
      if (-not $done2) {
        try { Stop-Job $job2 -Force } catch {}
        Remove-Job $job2 -Force
        Write-Warning "[daily_update] game-recommendations-sim timed out for ${d}; skipping."
        if ($StrictOutputs) { throw "[daily_update] game-recommendations-sim timed out for ${d} (timeout=${GameSimTimeoutSec}s)" }
      } else {
        Receive-Job $job2 | Write-Host
        Remove-Job $job2 -Force

        # Verify expected sim-native game outputs were produced (only if slate exists)
        if ($StrictOutputs) {
          Assert-AnyPath -Label "predictions_sim $d" -Paths @((_PathInProcessed "predictions_sim_${d}.csv")) -NonEmpty -Strict:$StrictOutputs
          Assert-AnyPath -Label "edges_sim $d" -Paths @((_PathInProcessed "edges_sim_${d}.csv")) -NonEmpty -Strict:$StrictOutputs
        }
      }
    } catch {
      Write-Warning "[daily_update] game-recommendations-sim failed for ${d}: $($_.Exception.Message)"
      if ($StrictOutputs) { throw }
    }
  }
} catch {
  Write-Warning "[daily_update] props-simulate-boxscores block failed: $($_.Exception.Message)"
  if ($StrictOutputs) { throw }
}

# Ensure props lines are saved once per slate (CSV+Parquet) without refetching repeatedly
if (-not $SkipProps) {
  try {
    foreach ($d in $dates) {
      $dir = Join-Path $RepoRoot "data/props/player_props_lines/date=$d"
      if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
      $haveRows = $false
      foreach ($fn in @('oddsapi.csv','bovada.csv')) {
        $p = Join-Path $dir $fn
        if (Test-Path $p) {
          try {
            $cnt = (Import-Csv $p).Count
            if ($cnt -gt 0) { $haveRows = $true }
          } catch { }
        }
      }
      if (-not $haveRows) {
        Write-Host "[daily_update] Collecting props lines for $d (oddsapi+bovada) …" -ForegroundColor Yellow
        try { python -m nhl_betting.cli props-collect --date $d --source oddsapi } catch { Write-Warning "[daily_update] props-collect oddsapi failed: $($_.Exception.Message)" }
        try { python -m nhl_betting.cli props-collect --date $d --source bovada } catch { Write-Warning "[daily_update] props-collect bovada failed: $($_.Exception.Message)" }
      } else {
        Write-Host "[daily_update] Props lines already present for $d; skipping collection." -ForegroundColor DarkGreen
      }
    }
  } catch { Write-Warning "[daily_update] props lines ensure failed: $($_.Exception.Message)" }
} else {
  Write-Host "[daily_update] SkipProps set; skipping props line collection." -ForegroundColor DarkGreen
}

# Optional: backfill historical player props via OddsAPI (per-day partitions)
try {
  if ($PropsBackfillRange) {
    $start = $AnchorNow.AddDays(-1 * [int]$PropsBackfillDays).ToString('yyyy-MM-dd')
    $end = $AnchorNow.ToString('yyyy-MM-dd')
    Write-Host "[daily_update] Backfilling OddsAPI player props $start..$end …" -ForegroundColor Yellow
    # Broaden coverage to US/US2/EU and allow any bookmaker (collector has fallbacks)
    $env:PROPS_ODDSAPI_REGIONS = 'us,us2,eu'
    $env:PROPS_ODDSAPI_BOOKMAKERS = ''
    try {
      python -m nhl_betting.cli props-collect-range --start $start --end $end --source oddsapi
    } catch {
      Write-Warning "[daily_update] props-collect-range failed: $($_.Exception.Message)"
    }
  }
} catch { Write-Warning "[daily_update] props historical backfill block failed: $($_.Exception.Message)" }

# Ensure team odds (ML/PL/Totals) and scoreboard snapshots are archived daily
try {
  foreach ($d in $dates) {
    # Team odds via OddsAPI
    $teamDir = Join-Path $RepoRoot "data/odds/team/date=$d"
    if (-not (Test-Path $teamDir)) { New-Item -ItemType Directory -Path $teamDir | Out-Null }
    $teamCsv = Join-Path $teamDir 'oddsapi.csv'
    $needTeam = $true
    if (Test-Path $teamCsv) {
      try { $cnt = (Import-Csv $teamCsv).Count; if ($cnt -gt 0) { $needTeam = $false } } catch { }
    }
    if ($needTeam) {
      Write-Host "[daily_update] Archiving team odds for $d …" -ForegroundColor Yellow
      try { python -m nhl_betting.cli team-odds-collect --date $d } catch { Write-Warning "[daily_update] team-odds-collect failed for ${d}: $($_.Exception.Message)" }
    } else {
      Write-Host "[daily_update] Team odds already present for $d; skipping collection." -ForegroundColor DarkGreen
    }
    # Games scoreboard archive
    $gamesDir = Join-Path $RepoRoot "data/odds/games/date=$d"
    if (-not (Test-Path $gamesDir)) { New-Item -ItemType Directory -Path $gamesDir | Out-Null }
    $sbCsv = Join-Path $gamesDir 'scoreboard.csv'
    if (-not (Test-Path $sbCsv) -or ((Get-Item $sbCsv).Length -lt 64)) {
      Write-Host "[daily_update] Archiving scoreboard for $d …" -ForegroundColor Yellow
      try { python -m nhl_betting.cli games-archive --date $d } catch { Write-Warning "[daily_update] games-archive failed for ${d}: $($_.Exception.Message)" }
    } else {
      Write-Host "[daily_update] Scoreboard already present for $d; skipping." -ForegroundColor DarkGreen
    }
  }
} catch { Write-Warning "[daily_update] team odds / games archive failed: $($_.Exception.Message)" }

# Compute accuracy JSON for yesterday (post-settlement)
try {
  $yesterday = $AnchorNow.AddDays(-1).ToString('yyyy-MM-dd')
  Write-Host "[daily_update] Writing per-day accuracy for $yesterday …" -ForegroundColor Yellow
  python -m nhl_betting.cli game-accuracy-day --date $yesterday
} catch { Write-Warning "[daily_update] game-accuracy-day failed: $($_.Exception.Message)" }

# Always refresh PBP-derived feature caches (PP/PK, penalties) and today's goalie form
try {
  Write-Host "[daily_update] Refreshing PP/PK and penalty rates from PBP …" -ForegroundColor Yellow
  python scripts/build_team_specials_and_penalties_from_pbp.py
} catch {
  Write-Warning "[daily_update] Failed PP/PK & penalties refresh: $($_.Exception.Message)"
}
try {
  Write-Host "[daily_update] Refreshing goalie recent form for today …" -ForegroundColor Yellow
  $today = $AnchorNow.ToString('yyyy-MM-dd')
  python scripts/build_goalie_form_from_pbp.py --date $today
} catch {
  Write-Warning "[daily_update] Failed goalie form refresh: $($_.Exception.Message)"
}
try {
  Write-Host "[daily_update] Refreshing team xGF/60 (MoneyPuck) …" -ForegroundColor Yellow
  $today = $AnchorNow.ToString('yyyy-MM-dd')
  python scripts/build_team_xg_moneypuck.py --date $today
} catch {
  Write-Warning "[daily_update] Failed team xG refresh: $($_.Exception.Message)"
}
# Run daily update workflow
$argsList = @("-m", "nhl_betting.scripts.daily_update", "--days-ahead", "$DaysAhead", "--years-back", "$YearsBack")
if ($NoReconcile) { $argsList += "--no-reconcile" }
if ($BootstrapModels) { $argsList += "--bootstrap-models" }
$argsList += @("--trends-decay", "$TrendsDecay")
if ($ResetTrends) { $argsList += "--reset-trends" }
if ($SkipProps) { $argsList += "--skip-props" }
Write-Host "[daily_update] Running core daily_update module with timeout=${CoreTimeoutSec}s …" -ForegroundColor Yellow
try {
  $coreJob = Start-Job -ScriptBlock { Set-Location $using:RepoRoot; & $using:PyExe @using:argsList }
  $coreDone = Wait-Job $coreJob -Timeout $CoreTimeoutSec
  if (-not $coreDone) {
    try { Stop-Job $coreJob -Force } catch {}
    Remove-Job $coreJob -Force
    Write-Warning "[daily_update] Core daily_update timed out after ${CoreTimeoutSec}s; continuing with downstream steps."
  } else {
    Receive-Job $coreJob | Write-Host
    Remove-Job $coreJob -Force
  }
} catch {
  Write-Warning "[daily_update] Core daily_update failed: $($_.Exception.Message)"
}

# After core daily update, recompute edges for today and forward DaysAhead-1 days
try {
  $base = $AnchorNow
  for ($i = 0; $i -lt $DaysAhead; $i++) {
    $d = $base.AddDays($i).ToString('yyyy-MM-dd')
    Write-Host "[daily_update] Recomputing team market edges for $d …" -ForegroundColor Cyan
    python -m nhl_betting.cli game-recompute-edges --date $d
  }
} catch {
  Write-Warning "[daily_update] game-recompute-edges failed: $($_.Exception.Message)"
}

if ($RecomputeRecs) {
  try {
    $base = $AnchorNow
    $recomputeDays = [Math]::Max($DaysAhead, 2)
    $env:FIRST10_BLEND = '1'
    for ($i = 0; $i -lt $recomputeDays; $i++) {
      $d = $base.AddDays($i).ToString('yyyy-MM-dd')
      Write-Host "[daily_update] Recomputing recommendations for $d …" -ForegroundColor Cyan
      python -c "from nhl_betting.core.recs_shared import recompute_edges_and_recommendations as R; R('$d', min_ev=0.0)"
    }
  } catch {
    Write-Warning "[daily_update] recommendation recompute failed: $($_.Exception.Message)"
  }
}

# Refresh simulation probability calibration on a leakage-safe historical window.
if ((-not $SkipGameCalibration) -and ($SimulateGames -or $BacktestSimulations -or $SimRecommendations)) {
  try {
    $simCalPath = Join-Path $ProcessedDir 'sim_calibration.json'
    $simCalLinePath = Join-Path $ProcessedDir 'sim_calibration_per_line.json'
    $trainEndDt = $AnchorNow.AddDays(-1 * ([int]$BacktestWindowDays + 1))
    $trainStartDt = $trainEndDt.AddDays(-1 * ([int]$SimCalibrationTrainDays - 1))
    $trainStart = $trainStartDt.ToString('yyyy-MM-dd')
    $trainEnd = $trainEndDt.ToString('yyyy-MM-dd')
    $refreshCutoff = $AnchorNow.AddDays(-1 * [int]$SimCalibrationRefreshDays)
    $needSimCalRefresh = $false

    if (([int]$SimCalibrationTrainDays -gt 0) -and ($trainEndDt -ge $trainStartDt)) {
      if (-not ((Test-Path $simCalPath) -and (Test-Path $simCalLinePath))) {
        $needSimCalRefresh = $true
      } else {
        try {
          $simCalObj = Get-Content $simCalPath | ConvertFrom-Json
          $simCalEnd = $null
          if ($simCalObj.meta -and $simCalObj.meta.end) { $simCalEnd = [datetime]::Parse($simCalObj.meta.end) }
          $simCalAge = (Get-Item $simCalPath).LastWriteTime
          $simCalLineAge = (Get-Item $simCalLinePath).LastWriteTime
          if (($simCalAge -lt $refreshCutoff) -or ($simCalLineAge -lt $refreshCutoff)) {
            $needSimCalRefresh = $true
          } elseif (($null -eq $simCalEnd) -or ($simCalEnd -lt $trainEndDt)) {
            $needSimCalRefresh = $true
          }
        } catch {
          $needSimCalRefresh = $true
        }
      }
    }

    if ($needSimCalRefresh) {
      Write-Host "[daily_update] Refreshing sim calibration on $trainStart..$trainEnd …" -ForegroundColor Magenta
      $builtSimCount = 0
      $availSimCount = 0
      for ($d = $trainStartDt; $d -le $trainEndDt; $d = $d.AddDays(1)) {
        $ds = $d.ToString('yyyy-MM-dd')
        $predPath = Join-Path $ProcessedDir ("predictions_{0}.csv" -f $ds)
        $simPath = Join-Path $ProcessedDir ("simulations_{0}.csv" -f $ds)
        if (-not (Test-Path $predPath)) { continue }
        if ((-not (Test-Path $simPath)) -or ((Get-Item $simPath).Length -le 0)) {
          Write-Host "[daily_update] Building historical sim for $ds (n=$SimCalibrationNSims) …" -ForegroundColor DarkYellow
          $pySimCalArgs = @("-m", "nhl_betting.cli", "game-simulate", "--date", $ds, "--n-sims", "$SimCalibrationNSims")
          if ($SimOverK -gt 0) { $pySimCalArgs += @("--sim-overdispersion-k", "$SimOverK") }
          if ($SimSharedK -gt 0) { $pySimCalArgs += @("--sim-shared-k", "$SimSharedK") }
          if ($SimEmptyNetP -gt 0) { $pySimCalArgs += @("--sim-empty-net-p", "$SimEmptyNetP") }
          if ($SimEmptyNetTwoGoalScale -gt 0) { $pySimCalArgs += @("--sim-empty-net-two-goal-scale", "$SimEmptyNetTwoGoalScale") }
          if ($TotalsPaceAlpha -gt 0) { $pySimCalArgs += @("--totals-pace-alpha", "$TotalsPaceAlpha") }
          if ($TotalsGoalieBeta -gt 0) { $pySimCalArgs += @("--totals-goalie-beta", "$TotalsGoalieBeta") }
          if ($TotalsFatigueBeta -gt 0) { $pySimCalArgs += @("--totals-fatigue-beta", "$TotalsFatigueBeta") }
          if ($TotalsRollingPaceGamma -gt 0) { $pySimCalArgs += @("--totals-rolling-pace-gamma", "$TotalsRollingPaceGamma") }
          if ($TotalsPPGamma -gt 0) { $pySimCalArgs += @("--totals-pp-gamma", "$TotalsPPGamma") }
          if ($TotalsPKBeta -gt 0) { $pySimCalArgs += @("--totals-pk-beta", "$TotalsPKBeta") }
          if ($TotalsPenaltyGamma -gt 0) { $pySimCalArgs += @("--totals-penalty-gamma", "$TotalsPenaltyGamma") }
          if ($TotalsXGGamma -gt 0) { $pySimCalArgs += @("--totals-xg-gamma", "$TotalsXGGamma") }
          if ($TotalsRefsGamma -gt 0) { $pySimCalArgs += @("--totals-refs-gamma", "$TotalsRefsGamma") }
          if ($TotalsGoalieFormGamma -gt 0) { $pySimCalArgs += @("--totals-goalie-form-gamma", "$TotalsGoalieFormGamma") }
          python @pySimCalArgs
          $builtSimCount += 1
        }
        if ((Test-Path $simPath) -and ((Get-Item $simPath).Length -gt 0)) { $availSimCount += 1 }
      }

      if ($availSimCount -gt 0) {
        Write-Host "[daily_update] Fitting sim calibration from $availSimCount training days …" -ForegroundColor Magenta
        python -m nhl_betting.cli game-calibrate-sim --start $trainStart --end $trainEnd
        python -m nhl_betting.cli game-calibrate-sim-per-line --start $trainStart --end $trainEnd
        if ($SimCalibrationTotalsOnly -and (Test-Path $simCalPath)) {
          try {
            $simCalObj = Get-Content $simCalPath | ConvertFrom-Json
            if (-not $simCalObj.moneyline) { $simCalObj | Add-Member -NotePropertyName moneyline -NotePropertyValue (@{}) }
            if (-not $simCalObj.puckline) { $simCalObj | Add-Member -NotePropertyName puckline -NotePropertyValue (@{}) }
            $simCalObj.moneyline.t = 1.0
            $simCalObj.moneyline.b = 0.0
            $simCalObj.puckline.t = 1.0
            $simCalObj.puckline.b = 0.0
            $simCalObj | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 $simCalPath
            Write-Host "[daily_update] Applied totals-only sim calibration (ML/PL reset to identity)." -ForegroundColor DarkGreen
          } catch {
            Write-Warning "[daily_update] Failed to enforce totals-only sim calibration: $($_.Exception.Message)"
          }
        }
        Write-Host "[daily_update] Sim calibration refresh complete (built $builtSimCount missing simulation files)." -ForegroundColor DarkGreen
      } else {
        Write-Warning "[daily_update] No historical simulation inputs available for sim calibration refresh."
      }
    } else {
      Write-Host "[daily_update] Sim calibration refresh skipped (recent coverage already available)." -ForegroundColor DarkGreen
    }
  } catch {
    Write-Warning "[daily_update] sim calibration refresh failed: $($_.Exception.Message)"
  }
}

# Optional: run game simulations (ML/PL/Totals) driven by NN outputs
if ($SimulateGames) {
  try {
    # Calibrate special-teams multipliers before running possession sims (last 30 days)
    try {
      $calStart = $AnchorNow.AddDays(-30).ToString('yyyy-MM-dd')
      $calEnd = $AnchorNow.ToString('yyyy-MM-dd')
      Write-Host "[daily_update] Calibrating special teams $calStart..$calEnd …" -ForegroundColor DarkGreen
      python -m nhl_betting.cli game-calibrate-special-teams --start $calStart --end $calEnd
    } catch {
      Write-Warning "[daily_update] Special-teams calibration failed: $($_.Exception.Message)"
    }
    $base = $AnchorNow
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
    $base = $AnchorNow
    for ($i = 0; $i -lt $DaysAhead; $i++) {
      $d = $base.AddDays($i).ToString('yyyy-MM-dd')
      $totFlag = if ($SimIncludeTotals) { "--include-totals" } else { "--no-include-totals" }
      Write-Host "[daily_update] Generating sim recommendations for $d (ML>=$SimMLThr, PL>=$SimPLThr, Tot>=$SimTotThr, includeTotals=$SimIncludeTotals) …" -ForegroundColor Cyan
      python -m nhl_betting.cli game-predictions-sim --date $d --include-ml --include-puckline $totFlag --ml-thr $SimMLThr --tot-thr $SimTotThr --pl-thr $SimPLThr
    }
    Write-Host "[daily_update] Sim picks written to data/processed/sim_picks_YYYY-MM-DD.csv"
  } catch {
    Write-Warning "[daily_update] game-predictions-sim failed: $($_.Exception.Message)"
  }
}

# Optional: run rolling props backtests (projections) and write dashboard row
if ($RunPropsBacktests) {
  try {
    $end = $AnchorNow.ToString('yyyy-MM-dd')
    $start = $AnchorNow.AddDays(-1 * [int]$PropsBacktestDays).ToString('yyyy-MM-dd')
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
    $end = $AnchorNow.ToString('yyyy-MM-dd')
    $start = $AnchorNow.AddDays(-1 * [int]$SimPropsBacktestDays).ToString('yyyy-MM-dd')
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
    $end = $AnchorNow.ToString('yyyy-MM-dd')
    $start = $AnchorNow.AddDays(-1 * [int]$BacktestWindowDays).ToString('yyyy-MM-dd')
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
if (-not $SkipGameCalibration) {
  try {
    $calPath = Join-Path $RepoRoot 'data/processed/model_calibration.json'
    $monitorPath = Join-Path $RepoRoot 'data/processed/game_daily_monitor.json'
    $needLearn = $false
    $today = $AnchorNow.ToString('yyyy-MM-dd')
    if (Test-Path $calPath) {
      $cal = Get-Content $calPath | ConvertFrom-Json
      if ($cal.ev_gates_last_learned_utc) {
        $last = [DateTime]::Parse($cal.ev_gates_last_learned_utc)
        if ($AnchorNow - $last -gt [TimeSpan]::FromDays(5)) { $needLearn = $true }
      } else { $needLearn = $true }
    } else { $needLearn = $true }
    if (Test-Path $monitorPath) {
      $mon = Get-Content $monitorPath | ConvertFrom-Json
      if ($mon.overall -and $mon.overall.roi -lt -0.08) { $needLearn = $true }
    }
    if ($needLearn) {
      $seasonStart = if ($AnchorNow.Month -ge 9) { "$($AnchorNow.Year)-09-01" } else { "$($AnchorNow.AddYears(-1).Year)-09-01" }
      Write-Host "[daily_update] EV gate re-learn triggered (seasonStart=$seasonStart -> today)" -ForegroundColor Magenta
      python -m nhl_betting.cli game-learn-ev-gates --start $seasonStart --end $today
    } else {
      Write-Host "[daily_update] EV gate re-learn skipped (recent & stable)" -ForegroundColor DarkGreen
    }
  } catch {
    Write-Warning "[daily_update] adaptive gate re-learn failed: $($_.Exception.Message)"
  }
} else {
  Write-Host "[daily_update] SkipGameCalibration set; skipping adaptive EV gate re-learn." -ForegroundColor DarkGreen
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

# Fit/update Live Lens in-play win-prob calibration (non-fatal)
try {
  # Backfill final outcomes into Live Lens state snapshots (best-effort)
  try {
    $llEnd = $AnchorNow.ToString('yyyy-MM-dd')
    $llLabelStart = $AnchorNow.AddDays(-14).ToString('yyyy-MM-dd')
    Write-Host "[daily_update] Backfilling Live Lens state outcomes ($llLabelStart..$llEnd) …" -ForegroundColor DarkCyan
    $oldPythonPath = $env:PYTHONPATH
    try {
      $env:PYTHONPATH = $RepoRoot
      Push-Location $RepoRoot
      & $PyExe scripts/backfill_live_lens_states_outcomes.py --start $llLabelStart --end $llEnd
    } finally {
      Pop-Location
      $env:PYTHONPATH = $oldPythonPath
    }
  } catch {
    Write-Warning "[daily_update] Live Lens outcome backfill failed: $($_.Exception.Message)"
  }

  # Build/refresh settled Live Lens bet ledger from saved signals (best-effort)
  try {
    $signalsDir = Join-Path $ProcessedDir 'live_lens'
    $perfDir = Join-Path $signalsDir 'perf'
    if (-not (Test-Path $perfDir)) { New-Item -ItemType Directory -Force -Path $perfDir | Out-Null }
    $outLedger = Join-Path $perfDir 'live_lens_bets_all.jsonl'
    Write-Host "[daily_update] Building Live Lens settled ledger from snapshots …" -ForegroundColor DarkCyan
    $oldPythonPath = $env:PYTHONPATH
    try {
      $env:PYTHONPATH = $RepoRoot
      Push-Location $RepoRoot
      & $PyExe .\check_live_lens_betting_performance.py --all --signals-dir $signalsDir --dedupe first --allow-web-fetch --out $outLedger
    } finally {
      Pop-Location
      $env:PYTHONPATH = $oldPythonPath
    }
    Assert-AnyPath -Label "live_lens_bets_all" -Paths @($outLedger) -NonEmpty -Strict:$StrictOutputs
  } catch {
    Write-Warning "[daily_update] Live Lens ledger build failed: $($_.Exception.Message)"
  }

  Write-Host "[daily_update] Fitting Live Lens driver-tag priors …" -ForegroundColor DarkCyan
  $llPriorsStart = $AnchorNow.AddDays(-60).ToString('yyyy-MM-dd')
  $llPriorsEnd = $AnchorNow.ToString('yyyy-MM-dd')
  $oldPythonPath = $env:PYTHONPATH
  try {
    $env:PYTHONPATH = $RepoRoot
    Push-Location $RepoRoot
    & $PyExe scripts/fit_live_lens_driver_tag_priors.py --start $llPriorsStart --end $llPriorsEnd
  } finally {
    Pop-Location
    $env:PYTHONPATH = $oldPythonPath
  }
  Assert-AnyPath -Label "live_lens_driver_tag_priors" -Paths @(
    (Join-Path $ProcessedDir 'live_lens\live_lens_driver_tag_priors.json'),
    (Join-Path $ProcessedDir 'live_lens/live_lens_driver_tag_priors.json'),
    (_PathInProcessed 'live_lens_driver_tag_priors.json')
  ) -NonEmpty -Strict:$StrictOutputs

  # Generate tuning report with learned priors context (non-fatal)
  try {
    $llPerfDir = Join-Path (Join-Path $ProcessedDir 'live_lens') 'perf'
    $llLedger = Join-Path $llPerfDir 'live_lens_bets_all.jsonl'
    $llPriors = Join-Path (Join-Path $ProcessedDir 'live_lens') 'live_lens_driver_tag_priors.json'
    $llTuningReport = Join-Path $llPerfDir 'live_lens_tuning_report.md'
    if (Test-Path $llLedger) {
      Write-Host "[daily_update] Building Live Lens tuning report …" -ForegroundColor DarkCyan
      $oldPythonPath = $env:PYTHONPATH
      try {
        $env:PYTHONPATH = $RepoRoot
        Push-Location $RepoRoot
        & $PyExe scripts/live_lens_tuning_report.py --ledger $llLedger --priors-json $llPriors --out $llTuningReport
      } finally {
        Pop-Location
        $env:PYTHONPATH = $oldPythonPath
      }
      Assert-AnyPath -Label "live_lens_tuning_report" -Paths @($llTuningReport) -NonEmpty -Strict:$StrictOutputs
    }
  } catch {
    Write-Warning "[daily_update] Live Lens tuning report failed: $($_.Exception.Message)"
  }

  Write-Host "[daily_update] Fitting Live Lens win-prob calibration …" -ForegroundColor DarkCyan
  $llFitStart = $AnchorNow.AddDays(-30).ToString('yyyy-MM-dd')
  $llFitEnd = $AnchorNow.ToString('yyyy-MM-dd')
  $oldPythonPath = $env:PYTHONPATH
  try {
    $env:PYTHONPATH = $RepoRoot
    Push-Location $RepoRoot
    & $PyExe scripts/fit_live_lens_winprob_calibration.py --start $llFitStart --end $llFitEnd
  } finally {
    Pop-Location
    $env:PYTHONPATH = $oldPythonPath
  }
  Assert-AnyPath -Label "live_lens_winprob_calibration" -Paths @(_PathInProcessed "live_lens_winprob_calibration.json") -NonEmpty -Strict:$StrictOutputs

  # Win-prob monitor + drift alert (non-fatal)
  try {
    Write-Host "[daily_update] Reporting Live Lens win-prob monitor/drift …" -ForegroundColor DarkCyan
    $oldPythonPath = $env:PYTHONPATH
    try {
      $env:PYTHONPATH = $RepoRoot
      Push-Location $RepoRoot
      & $PyExe scripts/report_live_lens_winprob_monitor.py --start $llFitStart --end $llFitEnd
    } finally {
      Pop-Location
      $env:PYTHONPATH = $oldPythonPath
    }
    Assert-AnyPath -Label "live_lens_winprob_monitor" -Paths @(
      (Join-Path $ProcessedDir 'live_lens\live_lens_winprob_monitor.json'),
      (Join-Path $ProcessedDir 'live_lens/live_lens_winprob_monitor.json')
    ) -NonEmpty -Strict:$StrictOutputs
    Assert-AnyPath -Label "live_lens_winprob_drift" -Paths @(
      (Join-Path $ProcessedDir 'live_lens\live_lens_winprob_drift_alert.json'),
      (Join-Path $ProcessedDir 'live_lens/live_lens_winprob_drift_alert.json')
    ) -NonEmpty -Strict:$StrictOutputs

    try {
      $driftPath = Join-Path $ProcessedDir 'live_lens\live_lens_winprob_drift_alert.json'
      if (-not (Test-Path $driftPath)) { $driftPath = Join-Path $ProcessedDir 'live_lens/live_lens_winprob_drift_alert.json' }
      if (Test-Path $driftPath) {
        $drift = Get-Content $driftPath | ConvertFrom-Json
        if ($drift -and $drift.alert -eq $true) {
          Write-Warning "[daily_update] Live Lens win-prob drift alert: $($drift.reasons -join ', ')"
        }
      }
    } catch {
      Write-Warning "[daily_update] Failed to parse Live Lens drift alert JSON: $($_.Exception.Message)"
    }
  } catch {
    Write-Warning "[daily_update] Live Lens win-prob monitor/drift failed: $($_.Exception.Message)"
  }
} catch {
  Write-Warning "[daily_update] Live Lens win-prob calibration fit failed: $($_.Exception.Message)"
}

# Optional: precompute props projections and generate props recommendations
# Note: Web projections now compute strength-aware p_over using scaled lambda (proj_lambda_eff)
if ($PropsRecs) {
  try {
    $today = $AnchorNow.ToString('yyyy-MM-dd')
    $tomorrow = $AnchorNow.AddDays(1).ToString('yyyy-MM-dd')
    # Step 0: Collect OddsAPI props lines and verify presence (non-fatal)
    try {
      Write-Host "[daily_update] Collecting canonical props lines (OddsAPI) for $today & $tomorrow …" -ForegroundColor Yellow
      python -m nhl_betting.cli props-collect --date $today
      python -m nhl_betting.cli props-collect --date $tomorrow
      Write-Host "[daily_update] Verifying props lines files …" -ForegroundColor Yellow
      python -m nhl_betting.cli props-verify --date $today
      python -m nhl_betting.cli props-verify --date $tomorrow
    } catch {
      Write-Warning "[daily_update] props-collect/verify failed: $($_.Exception.Message)"
    }
    # Step 0b: Ensure projections_all exist (needed for nolines simulation fallback)
    try {
      $pfToday = Join-Path "data\processed" "props_projections_all_${today}.csv"
      $pfTomorrow = Join-Path "data\processed" "props_projections_all_${tomorrow}.csv"
      $needProjToday = (-not (Test-Path $pfToday)) -or ((Get-Item $pfToday).Length -lt 64)
      $needProjTomorrow = (-not (Test-Path $pfTomorrow)) -or ((Get-Item $pfTomorrow).Length -lt 64)
      if ($needProjToday -or $needProjTomorrow) {
        Write-Host "[daily_update] Ensuring props projections_all for missing/empty days …" -ForegroundColor Yellow
        $projForceBase = @("-m", "nhl_betting.cli", "props-project-all", "--ensure-history-days", "365", "--include-goalies")
        if ($needProjToday) { python @($projForceBase + @("--date", $today)) }
        if ($needProjTomorrow) { python @($projForceBase + @("--date", $tomorrow)) }
      }
    } catch {
      Write-Warning "[daily_update] projections_all ensure failed: $($_.Exception.Message)"
    }
    Write-Host "[daily_update] Precomputing props projections for $today & $tomorrow …" -ForegroundColor Yellow
    $projArgsBase = @("-m", "nhl_betting.cli", "props-project-all", "--ensure-history-days", "365")
    if ($PropsIncludeGoalies) { $projArgsBase += "--include-goalies" }
    python @($projArgsBase + @("--date", $today))
    python @($projArgsBase + @("--date", $tomorrow))

    if ($PropsUseSim) {
      Write-Host "[daily_update] Simulating props for $today & $tomorrow …" -ForegroundColor Yellow
      $simArgsBase = @("-m", "nhl_betting.cli", "props-simulate", "--markets", "SOG,GOALS,ASSISTS,POINTS,SAVES,BLOCKS", "--n-sims", "16000", "--sim-shared-k", "1.2", "--props-sog-dispersion", "0.20", "--props-sog-pp-gamma", "0.40", "--props-sog-pk-gamma", "0.30")
      if ($PropsXGGamma) { $simArgsBase += @("--props-xg-gamma", "$PropsXGGamma") } else { $simArgsBase += @("--props-xg-gamma", "0.02") }
      if ($PropsPenaltyGamma) { $simArgsBase += @("--props-penalty-gamma", "$PropsPenaltyGamma") } else { $simArgsBase += @("--props-penalty-gamma", "0.06") }
      if ($PropsGoalieFormGamma) { $simArgsBase += @("--props-goalie-form-gamma", "$PropsGoalieFormGamma") } else { $simArgsBase += @("--props-goalie-form-gamma", "0.02") }
      if ($PropsStrengthGamma) { $simArgsBase += @("--props-strength-gamma", "$PropsStrengthGamma") } else { $simArgsBase += @("--props-strength-gamma", "0.04") }
      python @($simArgsBase + @("--date", $today))
      python @($simArgsBase + @("--date", $tomorrow))
      # Supplement: simulate SAVES/BLOCKS independent of provider lines
      Write-Host "[daily_update] Simulating nolines props (SAVES/BLOCKS/SOG) for $today & $tomorrow …" -ForegroundColor Yellow
      $nolArgsBase = @("-m", "nhl_betting.cli", "props-simulate-unlined", "--markets", "SAVES,BLOCKS,SOG", "--candidate-lines", "SAVES=24.5,26.5,28.5,30.5;BLOCKS=1.5,2.5,3.5;SOG=1.5,2.5,3.5,4.5", "--n-sims", "16000", "--sim-shared-k", "1.2")
      if ($PropsXGGamma) { $nolArgsBase += @("--props-xg-gamma", "$PropsXGGamma") } else { $nolArgsBase += @("--props-xg-gamma", "0.02") }
      if ($PropsPenaltyGamma) { $nolArgsBase += @("--props-penalty-gamma", "$PropsPenaltyGamma") } else { $nolArgsBase += @("--props-penalty-gamma", "0.06") }
      if ($PropsGoalieFormGamma) { $nolArgsBase += @("--props-goalie-form-gamma", "$PropsGoalieFormGamma") } else { $nolArgsBase += @("--props-goalie-form-gamma", "0.02") }
      python @($nolArgsBase + @("--date", $today))
      python @($nolArgsBase + @("--date", $tomorrow))
      Write-Host "[daily_update] Generating SIM-based props recommendations for $today & $tomorrow …" -ForegroundColor Cyan
      # Weekly auto-tune for SAVES nolines gate (range 0.65–0.68) based on 7-day monitor
      $SavesGate = 0.65
      $nolinesMarkets = $PropsNolinesMarkets
      $nolinesMinProbPerMarket = $PropsNolinesMinProbPerMarket
      try {
        if ($AnchorNow.DayOfWeek -eq 'Monday') {
          Write-Host "[daily_update] Auto-tuning SAVES gate via 7-day monitor …" -ForegroundColor DarkGreen
          $tunedNolinesMinProbPerMarket = Set-MarketThreshold -Thresholds $nolinesMinProbPerMarket -Market 'SAVES' -Value $SavesGate
          $monCmd = @("-m", "nhl_betting.cli", "props-nolines-monitor", "--window-days", "7", "--markets", $nolinesMarkets, "--min-prob-per-market", $tunedNolinesMinProbPerMarket)
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
      $nolinesMinProbPerMarket = Set-MarketThreshold -Thresholds $nolinesMinProbPerMarket -Market 'SAVES' -Value $SavesGate
      $recsSimBase = @("-m", "nhl_betting.cli", "props-recommendations-sim", "--min-ev", "$PropsMinEv", "--top", "$PropsTop", "--min-ev-per-market", $PropsMinEvPerMarket, "--min-prob", "$PropsMinProb", "--min-prob-per-market", $PropsMinProbPerMarket)
      if ($PropsUnderMinEvBound) {
        if ($PropsUnderMinEv -gt 0) {
          Write-Host "[daily_update] Props UNDER EV gate override: min EV = $PropsUnderMinEv" -ForegroundColor DarkGray
        } else {
          Write-Host "[daily_update] Props UNDER EV gate override: disabled" -ForegroundColor DarkGray
        }
        $recsSimBase += @("--under-min-ev", "$PropsUnderMinEv")
      }
      if ($PropsUnderMaxJuiceBound) {
        if ($PropsUnderMaxJuice -gt 0) {
          Write-Host "[daily_update] Props UNDER juice cap override: max juice = -$PropsUnderMaxJuice" -ForegroundColor DarkGray
        } else {
          Write-Host "[daily_update] Props UNDER juice cap override: disabled" -ForegroundColor DarkGray
        }
        $recsSimBase += @("--under-max-juice", "$PropsUnderMaxJuice")
      }
      if ($PropsUnderMaxJuicePerMarketBound) {
        if (-not [string]::IsNullOrWhiteSpace($PropsUnderMaxJuicePerMarket)) {
          Write-Host "[daily_update] Props UNDER per-market juice cap override: $PropsUnderMaxJuicePerMarket" -ForegroundColor DarkGray
        } else {
          Write-Host "[daily_update] Props UNDER per-market juice cap override: disabled" -ForegroundColor DarkGray
        }
        $recsSimBase += @("--under-max-juice-per-market", $PropsUnderMaxJuicePerMarket)
      }
      if ($PropsMaxPlusOdds -and $PropsMaxPlusOdds -gt 0) {
        Write-Host "[daily_update] Props odds clamp enabled: max plus odds = +$PropsMaxPlusOdds" -ForegroundColor DarkGray
        $recsSimBase += @("--max-plus-odds", "$PropsMaxPlusOdds")
      }
      python @($recsSimBase + @("--date", $today))
      python @($recsSimBase + @("--date", $tomorrow))
      # Also produce nolines-only recommendations without odds
      Write-Host "[daily_update] Generating nolines props recommendations for $today & $tomorrow …" -ForegroundColor Cyan
      $recsNoBase = @("-m", "nhl_betting.cli", "props-recommendations-nolines", "--markets", $nolinesMarkets, "--top", "$PropsTop", "--min-prob-per-market", $nolinesMinProbPerMarket)
      python @($recsNoBase + @("--date", $today))
      python @($recsNoBase + @("--date", $tomorrow))
      # Combine EV-based and nolines into one output per day
      Write-Host "[daily_update] Combining EV-based and nolines recommendations …" -ForegroundColor DarkCyan
      python -m nhl_betting.cli props-recommendations-combined --date $today
      python -m nhl_betting.cli props-recommendations-combined --date $tomorrow
      # Write 7-day nolines monitor
      Write-Host "[daily_update] Generating nolines monitor (7-day) …" -ForegroundColor DarkGreen
      python -m nhl_betting.cli props-nolines-monitor --window-days 7 --markets $nolinesMarkets --min-prob-per-market $nolinesMinProbPerMarket
    } else {
      Write-Host "[daily_update] Generating props recommendations (model-only) for $today & $tomorrow …" -ForegroundColor Cyan
      $recsArgsBase = @(
        "-m", "nhl_betting.cli", "props-recommendations",
        "--min-ev", "$PropsMinEv",
        "--top", "$PropsTop",
        "--min-ev-per-market", $PropsMinEvPerMarket,
        "--min-prob", "$PropsMinProb",
        "--min-prob-per-market", $PropsMinProbPerMarket
      )
      if ($PropsUnderMinEvBound) {
        if ($PropsUnderMinEv -gt 0) {
          Write-Host "[daily_update] Props UNDER EV gate override: min EV = $PropsUnderMinEv" -ForegroundColor DarkGray
        } else {
          Write-Host "[daily_update] Props UNDER EV gate override: disabled" -ForegroundColor DarkGray
        }
        $recsArgsBase += @("--under-min-ev", "$PropsUnderMinEv")
      }
      if ($PropsUnderMaxJuiceBound) {
        if ($PropsUnderMaxJuice -gt 0) {
          Write-Host "[daily_update] Props UNDER juice cap override: max juice = -$PropsUnderMaxJuice" -ForegroundColor DarkGray
        } else {
          Write-Host "[daily_update] Props UNDER juice cap override: disabled" -ForegroundColor DarkGray
        }
        $recsArgsBase += @("--under-max-juice", "$PropsUnderMaxJuice")
      }
      if ($PropsUnderMaxJuicePerMarketBound) {
        if (-not [string]::IsNullOrWhiteSpace($PropsUnderMaxJuicePerMarket)) {
          Write-Host "[daily_update] Props UNDER per-market juice cap override: $PropsUnderMaxJuicePerMarket" -ForegroundColor DarkGray
        } else {
          Write-Host "[daily_update] Props UNDER per-market juice cap override: disabled" -ForegroundColor DarkGray
        }
        $recsArgsBase += @("--under-max-juice-per-market", $PropsUnderMaxJuicePerMarket)
      }
      if ($PropsMaxPlusOdds -and $PropsMaxPlusOdds -gt 0) {
        Write-Host "[daily_update] Props odds clamp enabled: max plus odds = +$PropsMaxPlusOdds" -ForegroundColor DarkGray
        $recsArgsBase += @("--max-plus-odds", "$PropsMaxPlusOdds")
      }
      python @($recsArgsBase + @("--date", $today))
      python @($recsArgsBase + @("--date", $tomorrow))
    }
  } catch {
    Write-Warning "[daily_update] Props projections/recommendations failed: $($_.Exception.Message)"
  }
}

  # Post-settlement props reconciliation and summary (yesterday)
  try {
    if (-not $NoReconcile) {
      $yesterday = $AnchorNow.AddDays(-1).ToString('yyyy-MM-dd')
      Write-Host "[daily_update] Backfilling actuals for $yesterday …" -ForegroundColor Yellow
      $oldPythonPath = $env:PYTHONPATH
      try {
        $env:PYTHONPATH = $RepoRoot
        Push-Location $RepoRoot
        & $PyExe .\scripts\backfill_actuals_day.py $yesterday
      } catch {
        Write-Warning "[daily_update] backfill_actuals_day failed: $($_.Exception.Message)"
      } finally {
        Pop-Location
        $env:PYTHONPATH = $oldPythonPath
      }
      Write-Host "[daily_update] Reconciling props for $yesterday …" -ForegroundColor Cyan
      $oldPythonPath = $env:PYTHONPATH
      try {
        $env:PYTHONPATH = $RepoRoot
        Push-Location $RepoRoot
        & $PyExe .\scripts\props_reconcile_day.py $yesterday
      } catch {
        Write-Warning "[daily_update] props_reconcile_day failed: $($_.Exception.Message)"
      } finally {
        Pop-Location
        $env:PYTHONPATH = $oldPythonPath
      }
      Write-Host "[daily_update] Aggregating reconciliation summaries …" -ForegroundColor DarkGreen
      $oldPythonPath = $env:PYTHONPATH
      try {
        $env:PYTHONPATH = $RepoRoot
        Push-Location $RepoRoot
        & $PyExe .\scripts\props_reconciliations_aggregate.py $yesterday
      } catch {
        Write-Warning "[daily_update] props_reconciliations_aggregate failed: $($_.Exception.Message)"
      } finally {
        Pop-Location
        $env:PYTHONPATH = $oldPythonPath
      }
    } else {
      Write-Host "[daily_update] Props reconciliation skipped (-NoReconcile)" -ForegroundColor DarkGreen
    }
  } catch {
    Write-Warning "[daily_update] props reconciliation pipeline failed: $($_.Exception.Message)"
  }

# Publish stable web/UI artifacts: daily bundles + manifest
try {
  $y = $AnchorNow.AddDays(-1).ToString('yyyy-MM-dd')
  $t = $AnchorNow.ToString('yyyy-MM-dd')
  $tm = $AnchorNow.AddDays(1).ToString('yyyy-MM-dd')
  Write-Host "[daily_update] Publishing bundles for $y, $t, $tm …" -ForegroundColor DarkCyan
  python -m nhl_betting.cli bundle-build --date $y
  python -m nhl_betting.cli bundle-build --date $t
  python -m nhl_betting.cli bundle-build --date $tm
  python -m nhl_betting.cli bundle-manifest
} catch {
  Write-Warning "[daily_update] Bundle publish failed: $($_.Exception.Message)"
  if ($StrictOutputs) { throw "[daily_update] Bundle publish failed under -StrictOutputs: $($_.Exception.Message)" }
}

$DailyUpdateExitCode = $LASTEXITCODE

# Push generated artifacts to git by default (matches documented daily-update workflow).
try {
  if ($NoGitPush) {
    Write-Host "[daily_update] Git push skipped (-NoGitPush)" -ForegroundColor DarkGreen
  } else {
    Push-Location $RepoRoot
    try {
      $isGit = (Invoke-GitCommand -Args @('rev-parse', '--is-inside-work-tree') -FailureMessage '[daily_update] Failed to detect git repository' | Select-Object -First 1)
      if ("$isGit".Trim() -eq 'true') {
        $hadDisableAutoGc = Test-Path Env:GIT_DISABLE_AUTO_GC
        $oldDisableAutoGc = $env:GIT_DISABLE_AUTO_GC
        $env:GIT_DISABLE_AUTO_GC = '1'
        try {
        $gitPaths = @(
          'data/models',
          'data/processed',
          'data/props/player_props_lines',
          'data/raw/player_game_stats.csv'
        )

        # Bundles are intentionally ignored by default to avoid staging a backlog.
        # Force-add only the dates needed for web deploy parity (yesterday/today/tomorrow) + manifest.
        $bundleDates = @(
          $AnchorNow.AddDays(-1).ToString('yyyy-MM-dd'),
          $AnchorNow.ToString('yyyy-MM-dd'),
          $AnchorNow.AddDays(1).ToString('yyyy-MM-dd')
        )
        $forceAddFiles = @('data/processed/bundles/manifest.json')
        foreach ($d in $bundleDates) {
          $forceAddFiles += "data/processed/bundles/date=$d/bundle.json"
        }

        $branch = $GitBranch
        if (-not $branch -or $branch.Trim() -eq '') {
          $branch = (Invoke-GitCommand -Args @('rev-parse', '--abbrev-ref', 'HEAD') -FailureMessage '[daily_update] Failed to resolve current branch' | Select-Object -First 1)
        }
        if (-not $branch -or $branch.Trim() -eq '') { $branch = 'master' }
        $branch = $branch.Trim()

        $userName = (& git config --get user.name 2>$null | Select-Object -First 1)
        $userEmail = (& git config --get user.email 2>$null | Select-Object -First 1)
        if (-not (Test-GitIdentityLooksConfigured -UserName $userName -UserEmail $userEmail)) {
          Write-Warning '[daily_update] Git user.name/user.email are unset or placeholder values; auto-generated commits may use bad author metadata.'
        }

        $commitFailed = $false

        Write-Host "[daily_update] Staging generated artifacts …" -ForegroundColor Yellow
        foreach ($p in $gitPaths) {
          if (Test-Path (Join-Path $RepoRoot $p)) {
            Invoke-GitCommand -Args @('add', '-A', '--', $p) -FailureMessage "[daily_update] Failed to stage generated artifacts under $p" | Out-Null
          }
        }

        foreach ($f in $forceAddFiles) {
          if (Test-Path (Join-Path $RepoRoot $f)) {
            Invoke-GitCommand -Args @('add', '-f', '--', $f) -FailureMessage "[daily_update] Failed to force-add ignored artifact $f" | Out-Null
          }
        }

        $cachedArgs = @('diff', '--cached', '--name-only', '--') + $gitPaths
        $cached = Invoke-GitCommand -Args $cachedArgs -FailureMessage '[daily_update] Failed to inspect staged generated artifacts'
        if ($cached) {
          $targetDates = @()
          try {
            $targetDates = @($dates | Where-Object { $_ -and $_.Trim() -ne '' })
          } catch {
            $targetDates = @()
          }
          if (-not $targetDates -or $targetDates.Count -eq 0) {
            $targetDates = @($AnchorNow.ToString('yyyy-MM-dd'))
          }
          $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm zzz'
          $msg = "[auto] daily update: $($targetDates -join ', ') @ $stamp"

          try {
            Invoke-GitCommand -Args @('commit', '--no-gpg-sign', '-m', $msg) -FailureMessage '[daily_update] Git commit failed for generated artifacts' | Out-Null
            Write-Host "[daily_update] Git commit created." -ForegroundColor Yellow
          } catch {
            $commitFailed = $true
            Write-Warning $_.Exception.Message
            if ($StrictOutputs) { throw }
          }
        } else {
          Write-Host "[daily_update] No staged artifact changes after git add." -ForegroundColor DarkGreen
        }

        if (-not $commitFailed) {
          Write-Host "[daily_update] Pushing to $GitRemote/$branch …" -ForegroundColor Yellow
          $pushOutput = Invoke-GitCommand -Args @('push', $GitRemote, $branch) -FailureMessage "[daily_update] Git push failed (remote=$GitRemote branch=$branch)"
          $pushSummary = ($pushOutput -join ' ').Trim()
          if (-not $pushSummary) { $pushSummary = 'push completed' }
          Write-Host "[daily_update] Git push complete: $pushSummary" -ForegroundColor DarkGreen
        } else {
          Write-Warning '[daily_update] Git push skipped because the artifact commit failed.'
        }
        } finally {
          if ($hadDisableAutoGc) {
            $env:GIT_DISABLE_AUTO_GC = $oldDisableAutoGc
          } else {
            Remove-Item Env:GIT_DISABLE_AUTO_GC -ErrorAction SilentlyContinue
          }
        }
      } else {
        Write-Host "[daily_update] Not a git repository; skipping push." -ForegroundColor DarkGreen
      }
    } finally {
      Pop-Location
    }
  }
} catch {
  Write-Warning "[daily_update] Git push skipped due to error: $($_.Exception.Message)"
  if ($StrictOutputs) { throw }
}

if ($DailyUpdateExitCode -ne 0) {
  exit $DailyUpdateExitCode
}

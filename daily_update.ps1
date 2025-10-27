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
  [switch]$SkipPropsCalibration
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

# Optional: allow callers to skip heavy props projections or calibration via switches
if ($SkipPropsProjections) { $env:PROPS_SKIP_PROJECTIONS = '1' }
if ($SkipPropsCalibration) { $env:SKIP_PROPS_CALIBRATION = '1' }

# Ensure dependencies
pip install -q -r (Join-Path $RepoRoot "requirements.txt")

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

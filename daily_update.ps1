Param(
  [int]$DaysAhead = 2,
  [int]$YearsBack = 2,
  [switch]$NoReconcile,
  [switch]$Quiet,
  [switch]$BootstrapModels,
  [double]$TrendsDecay = 0.98,
  [switch]$ResetTrends,
  [switch]$SkipProps
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

# Ensure venv exists and activate
$Venv = Join-Path $RepoRoot ".venv"
$Activate = Join-Path $Venv "Scripts/Activate.ps1"
if (-not (Test-Path $Activate)) { python -m venv $Venv }
. $Activate

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

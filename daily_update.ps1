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
exit $LASTEXITCODE

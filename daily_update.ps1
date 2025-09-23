Param(
  [int]$DaysAhead = 2,
  [int]$YearsBack = 2,
  [switch]$NoReconcile,
  [switch]$__BypassOk
)
$ErrorActionPreference = "Stop"

# If not already relaunched with ExecutionPolicy Bypass, relaunch self so it runs without prompts
if (-not $__BypassOk) {
  $argsList = @("-NoProfile","-ExecutionPolicy","Bypass","-File", $PSCommandPath, "-DaysAhead", "$DaysAhead", "-YearsBack", "$YearsBack")
  if ($NoReconcile) { $argsList += "-NoReconcile" }
  $argsList += "-__BypassOk"
  Start-Process -FilePath "powershell.exe" -ArgumentList $argsList -Wait
  exit $LASTEXITCODE
}

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
python @argsList

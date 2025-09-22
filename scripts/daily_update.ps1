Param(
  [int]$DaysAhead = 2
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$Venv = Join-Path $RepoRoot ".venv"
$Activate = Join-Path $Venv "Scripts/Activate.ps1"
if (-not (Test-Path $Activate)) { python -m venv $Venv }
. $Activate
pip install -q -r (Join-Path $RepoRoot "requirements.txt")
# Run daily update via CLI
python -m nhl_betting.cli daily-update --days-ahead $DaysAhead

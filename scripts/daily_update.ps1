Param(
  [int]$DaysAhead = 2,
  [int]$YearsBack = 2,
  [switch]$NoReconcile,
  [switch]$Postgame,              # If set, also run postgame (stats backfill -> props reconciliation -> backtest)
  [string]$PostgameDate = "yesterday",  # Date for postgame step ("yesterday" | "today" | YYYY-MM-DD)
  [string]$PostgameStatsSource = "stats",
  [int]$PostgameWindow = 10,
  [double]$PostgameStake = 100
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$Venv = Join-Path $RepoRoot ".venv"
$Activate = Join-Path $Venv "Scripts/Activate.ps1"
if (-not (Test-Path $Activate)) { python -m venv $Venv }
. $Activate
pip install -q -r (Join-Path $RepoRoot "requirements.txt")
# Run daily update workflow
$argsList = @("-m", "nhl_betting.scripts.daily_update", "--days-ahead", "$DaysAhead", "--years-back", "$YearsBack")
if ($NoReconcile) { $argsList += "--no-reconcile" }
python @argsList

# Optionally run postgame pipeline after daily update
if ($Postgame) {
  Write-Host "[daily_update] Running postgame for $PostgameDate …"
  python -m nhl_betting.cli props-postgame --date $PostgameDate --stats-source $PostgameStatsSource --window $PostgameWindow --stake $PostgameStake
}

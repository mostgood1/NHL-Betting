Param(
  [string]$Date = "yesterday",
  [string]$StatsSource = "stats",
  [int]$Window = 10,
  [double]$Stake = 100
)

# Activate venv and run the postgame pipeline (stats backfill -> props reconciliation -> backtest)
$ErrorActionPreference = "Stop"

$venv = Join-Path $PSScriptRoot "..\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
  . $venv
}

# Ensure QNN env (optional) for any ONNX usage
$RepoRoot = Split-Path -Parent $PSScriptRoot
$NpuScript = Join-Path $RepoRoot "activate_npu.ps1"
if (Test-Path $NpuScript) { . $NpuScript }

python -m nhl_betting.cli props-postgame --date $Date --stats-source $StatsSource --window $Window --stake $Stake

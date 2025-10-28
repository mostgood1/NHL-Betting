Param(
  [string]$Date = "yesterday",
  [string]$StatsSource = "stats",
  [int]$Window = 10,
  [double]$Stake = 100
)

# Activate venv and run the postgame pipeline (stats backfill -> props reconciliation -> backtest)
$ErrorActionPreference = "Stop"

# Ensure ARM64 venv and QNN env
$RepoRoot = Split-Path -Parent $PSScriptRoot
$NpuScript = Join-Path $RepoRoot "activate_npu.ps1"
if (Test-Path $NpuScript) { . $NpuScript } else {
  $Ensure = Join-Path $RepoRoot 'ensure_arm64_venv.ps1'
  if (Test-Path $Ensure) { . $Ensure; $null = Ensure-Arm64Venv -RepoRoot $RepoRoot -Activate }
}

python -m nhl_betting.cli props-postgame --date $Date --stats-source $StatsSource --window $Window --stake $Stake

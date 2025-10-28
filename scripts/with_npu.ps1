param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Cmd
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Source QNN env and enforce ARM64 venv (activate by default)
$NpuScript = Join-Path $RepoRoot "activate_npu.ps1"
if (Test-Path $NpuScript) { . $NpuScript } else {
  $Ensure = Join-Path $RepoRoot 'ensure_arm64_venv.ps1'
  if (Test-Path $Ensure) { . $Ensure; $null = Ensure-Arm64Venv -RepoRoot $RepoRoot -Activate }
}

# Execute the provided command in this prepared session
if ($Cmd -and $Cmd.Count -gt 0) {
  if ($Cmd.Count -eq 1) {
    & $Cmd[0]
  } else {
    & $Cmd[0] @($Cmd[1..($Cmd.Count-1)])
  }
} else {
  Write-Host "[with_npu] Environment prepared (QNN + venv). No command provided." -ForegroundColor Yellow
}

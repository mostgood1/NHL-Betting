param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Cmd
)
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

# Source QNN env if available
$NpuScript = Join-Path $RepoRoot "activate_npu.ps1"
if (Test-Path $NpuScript) { . $NpuScript }

# Activate venv if available
$Venv = Join-Path $RepoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $Venv) { . $Venv }

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

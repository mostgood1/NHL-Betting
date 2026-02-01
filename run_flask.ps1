Param(
  [int]$Port = 8080,
  [string]$BindHost = '127.0.0.1',
  [switch]$UseLegacyFlask
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

# Prefer FastAPI app (uvicorn) to match the desired version on 8080.
if (-not $UseLegacyFlask) {
  # Ensure NPU/venv environment
  $NpuScript = Join-Path $RepoRoot 'activate_npu.ps1'
  if (Test-Path $NpuScript) { . $NpuScript }
  else {
    $Ensure = Join-Path $RepoRoot 'ensure_arm64_venv.ps1'
    if (Test-Path $Ensure) { . $Ensure; $null = Ensure-Arm64Venv -RepoRoot $RepoRoot -Activate }
  }
  $PythonExe = Join-Path $RepoRoot '.venv/Scripts/python.exe'
  if (-not (Test-Path $PythonExe)) { $PythonExe = 'python' }
  Write-Host ("[fastapi] Starting on http://{0}:{1} …" -f $BindHost, $Port) -ForegroundColor Cyan
  & $PythonExe -m uvicorn nhl_betting.web.app:app --host $BindHost --port $Port
} else {
  # Legacy Flask app path
  $AppPath = Join-Path $RepoRoot 'web_flask/app.py'
  . (Join-Path $RepoRoot '.venv/Scripts/Activate.ps1')
  $env:FLASK_APP = $AppPath
  $env:FLASK_ENV = 'development'
  Write-Host ("[flask] Starting on http://{0}:{1} …" -f $BindHost, $Port) -ForegroundColor Cyan
  python -m flask run --host $BindHost --port $Port
}

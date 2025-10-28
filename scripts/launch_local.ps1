Param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"

# Resolve repo root (this script is in scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

Write-Host ("Repo root: {0}" -f $RepoRoot)

# Ensure QNN env (enforces ARM64 venv) so ONNX Runtime can find QNN EP
$NpuScript = Join-Path $RepoRoot "activate_npu.ps1"
if (Test-Path $NpuScript) { . $NpuScript } else {
    $Ensure = Join-Path $RepoRoot 'ensure_arm64_venv.ps1'
    if (Test-Path $Ensure) { . $Ensure; $null = Ensure-Arm64Venv -RepoRoot $RepoRoot -Activate }
}

# Install deps if needed
$Req = Join-Path $RepoRoot "requirements.txt"
if (Test-Path $Req) {
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
    pip install -q -r $Req
}

# Export app envs (optional)
${env:PYTHONPATH} = $RepoRoot

# Build the uvicorn args
$UvicornArgs = @('nhl_betting.web.app:app', '--host', $HostAddress, '--port', "$Port")
if (-not $NoReload) {
    $UvicornArgs += "--reload"
}

# Use venv python to run uvicorn for better Windows reliability
$PythonExe = Join-Path $VenvPath "Scripts/python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host ("Starting server at http://{0}:{1}" -f $HostAddress, $Port) -ForegroundColor Green
& $PythonExe -m uvicorn @UvicornArgs

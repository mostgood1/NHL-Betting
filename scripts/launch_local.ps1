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

# Ensure venv exists
$VenvPath = Join-Path $RepoRoot ".venv"
$Activate = Join-Path $VenvPath "Scripts/Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Host "Creating virtual environment at $VenvPath" -ForegroundColor Cyan
    python -m venv $VenvPath
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
. $Activate

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

Param(
  [int]$Port = 8000,
  [string]$BindHost = '127.0.0.1'
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppPath = Join-Path $RepoRoot 'web_flask/app.py'

. (Join-Path $RepoRoot '.venv/Scripts/Activate.ps1')
$env:FLASK_APP = $AppPath
$env:FLASK_ENV = 'development'
Write-Host ("[flask] Starting on http://{0}:{1} …" -f $BindHost, $Port) -ForegroundColor Cyan
python -m flask run --host $BindHost --port $Port

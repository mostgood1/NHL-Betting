param(
  [int]$Port = 8010,
  [string]$BindIP = '127.0.0.1'
)

Write-Host ('Starting dev server on {0}:{1} (will try next ports if busy)...' -f $BindIP, $Port) -ForegroundColor Cyan

function Test-PortFree($p) {
  try {
    $l = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $p)
    $l.Start(); $l.Stop(); return $true
  } catch { return $false }
}

$maxTries = 10
$chosen = $Port
for ($i=0; $i -lt $maxTries; $i++) {
  if (Test-PortFree $chosen) { break }
  $chosen++
}
if (-not (Test-PortFree $chosen)) {
  Write-Error "No free port found in range starting $Port"
  exit 1
}

Write-Host "Using port $chosen" -ForegroundColor Green

if (-not (Test-Path .\.venv\Scripts\Activate.ps1)) {
  Write-Host "Virtual environment not found (.venv). Create it first." -ForegroundColor Red
  exit 1
}

. .\.venv\Scripts\Activate.ps1

$env:UVICORN_WORKERS = 1
$env:PYTHONUNBUFFERED = 1

Write-Host ('Launching uvicorn on {0}:{1} (Ctrl+C to stop)...' -f $BindIP, $chosen) -ForegroundColor Cyan
uvicorn nhl_betting.web.app:app --host $BindIP --port $chosen --log-level debug
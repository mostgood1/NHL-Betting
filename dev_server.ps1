param(
  [int]$Port = 8000,
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

# Ensure NPU env is configured and ARM64 venv is enforced/activated
if (Test-Path .\activate_npu.ps1) { . .\activate_npu.ps1 }
else {
  # Fallback to enforce ARM64 venv directly
  $Ensure = Join-Path (Get-Location) 'ensure_arm64_venv.ps1'
  if (Test-Path $Ensure) { . $Ensure; $null = Ensure-Arm64Venv -RepoRoot (Get-Location) -Activate }
}

$env:UVICORN_WORKERS = 1
$env:PYTHONUNBUFFERED = 1

Write-Host ('Launching uvicorn on {0}:{1} (Ctrl+C to stop)...' -f $BindIP, $chosen) -ForegroundColor Cyan
uvicorn nhl_betting.web.app:app --host $BindIP --port $chosen --log-level debug
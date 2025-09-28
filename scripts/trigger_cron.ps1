# Sets TLS 1.2 and triggers cron endpoints with a provided token
param(
  [Parameter(Mandatory=$true)][string]$Token,
  [string]$BaseUrl = "https://nhl-betting.onrender.com",
  [switch]$Closers
)

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Headers = @{ Authorization = "Bearer $Token" }

Write-Host "Health:" -ForegroundColor Cyan
Invoke-WebRequest -Uri "$BaseUrl/health" -Method GET | Select-Object -ExpandProperty Content

Write-Host "\nRefresh Bovada (backfill):" -ForegroundColor Cyan
Invoke-WebRequest -Uri "$BaseUrl/api/cron/refresh-bovada" -Method POST -Headers $Headers | Select-Object -ExpandProperty Content

if ($Closers) {
  Write-Host "\nCapture Closers (today):" -ForegroundColor Cyan
  Invoke-WebRequest -Uri "$BaseUrl/api/cron/capture-closing?date=today" -Method POST -Headers $Headers | Select-Object -ExpandProperty Content
}

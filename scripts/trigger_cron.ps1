# Sets TLS 1.2 and triggers cron endpoints with a provided token
param(
  [Parameter(Mandatory=$true)][string]$Token,
  [string]$BaseUrl = "https://nhl-betting.onrender.com",
  [switch]$Closers,
  [string]$Date,
  [switch]$Yesterday,
  [string]$StartDate,
  [string]$EndDate
)

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$Headers = @{ Authorization = "Bearer $Token" }

Write-Host "Health:" -ForegroundColor Cyan
Invoke-WebRequest -Uri "$BaseUrl/health" -Method GET | Select-Object -ExpandProperty Content

# Compute the list of dates to process
$dates = @()
if ($StartDate -and $EndDate) {
  try {
    $start = [DateTime]::Parse($StartDate)
    $end = [DateTime]::Parse($EndDate)
    if ($end -lt $start) { throw "EndDate must be >= StartDate" }
    for ($d = $start; $d -le $end; $d = $d.AddDays(1)) {
      $dates += $d.ToString('yyyy-MM-dd')
    }
  } catch {
    Write-Error "Invalid StartDate/EndDate. Please use YYYY-MM-DD. $_"
    exit 1
  }
} else {
  if ($Yesterday) {
    $Date = (Get-Date).AddDays(-1).ToString('yyyy-MM-dd')
  }
  if (-not $Date) { $Date = 'today' }
  $dates = @($Date)
}

foreach ($d in $dates) {
  Write-Host "\nRefresh Bovada (backfill) for ${d}:" -ForegroundColor Cyan
  $refreshUrl = "$BaseUrl/api/cron/refresh-bovada?date=$d"
  Invoke-WebRequest -Uri $refreshUrl -Method POST -Headers $Headers | Select-Object -ExpandProperty Content

  if ($Closers) {
  Write-Host "\nCapture Closers for ${d}:" -ForegroundColor Cyan
    $closersUrl = "$BaseUrl/api/cron/capture-closing?date=$d"
    Invoke-WebRequest -Uri $closersUrl -Method POST -Headers $Headers | Select-Object -ExpandProperty Content
  }
}

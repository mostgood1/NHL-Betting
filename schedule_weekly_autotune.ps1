# Registers a Windows Scheduled Task to run weekly auto-tune on Mondays at 08:30
$repoRoot = $PSScriptRoot
$runScript = Join-Path $repoRoot 'run_weekly_autotune.ps1'
$taskName = 'NHLBetting_WeeklyAutoTune'
$startTime = '08:30'

Write-Host "[schedule] Registering task '$taskName' to run $runScript every Monday at $startTime" -ForegroundColor Cyan
# Remove existing task if present
try {
  schtasks /Query /TN $taskName | Out-Null 2>$null
  if ($LASTEXITCODE -eq 0) {
    schtasks /Delete /TN $taskName /F | Out-Null
    Write-Host "[schedule] Removed existing task '$taskName'" -ForegroundColor Yellow
  }
} catch {}

# Create the new task (runs under current user)
schtasks /Create /SC WEEKLY /D MON /TN $taskName /TR "powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File `"$runScript`"" /ST $startTime | Write-Host
Write-Host "[schedule] Task created. Verify in Task Scheduler under Task Scheduler Library." -ForegroundColor Green

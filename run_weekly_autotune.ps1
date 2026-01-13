param(
  [int]$DaysAhead = 2,
  [int]$PropsTop = 400
)
# Ensure working directory is the repo root
Push-Location $PSScriptRoot
try { . .\activate_npu.ps1 } catch {}
try { . .\.venv\Scripts\Activate.ps1 } catch {}
# Run daily update with sim-based props so weekly auto-tune logic executes
.\daily_update.ps1 -DaysAhead $DaysAhead -Quiet -PropsRecs -PropsUseSim -PropsTop $PropsTop -PropsIncludeGoalies
Pop-Location

param(
  [Parameter(Mandatory=$false)][string]$Start = "2026-01-20",
  [Parameter(Mandatory=$false)][string]$End = "2026-02-18",
  [Parameter(Mandatory=$false)][int]$NSims = 1000,
  [Parameter(Mandatory=$false)][int]$Seed = 42,
  [Parameter(Mandatory=$false)][double]$FatigueBeta = 0.03,
  [Parameter(Mandatory=$false)][double]$TravelBeta = 0.01,
  [Parameter(Mandatory=$false)][string]$ToiMode = "auto",
  [Parameter(Mandatory=$false)][string]$StarterSource = "auto",
  [Parameter(Mandatory=$false)][string]$RestSource = "schedule",
  [Parameter(Mandatory=$false)][string]$FromShotsPrefix = "goal_fromshots",
  [Parameter(Mandatory=$false)][string]$IndependentPrefix = "goal_ind"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Environment
. .\activate_npu.ps1
. .\.venv\Scripts\Activate.ps1

function Invoke-BoxscoreSim {
  param(
    [Parameter(Mandatory=$true)][string]$Date,
    [Parameter(Mandatory=$true)][string]$OutPrefix,
    [Parameter(Mandatory=$true)][string]$GoalModel
  )

  $histPath = Join-Path $RepoRoot ("data\\processed\\{0}_props_boxscores_sim_hist_{1}.csv" -f $OutPrefix, $Date)
  if (Test-Path $histPath) {
    Write-Host ("[ab-goal-model] skip existing {0}" -f $histPath)
    return
  }

  Write-Host ("[ab-goal-model] sim {0} goal_model={1} prefix={2} n={3} rest={4}" -f $Date, $GoalModel, $OutPrefix, $NSims, $RestSource)
  python -m nhl_betting.cli props-simulate-boxscores `
    --date $Date `
    --n-sims $NSims `
    --seed $Seed `
    --samples-mode hist `
    --rest-source $RestSource `
    --fatigue-beta $FatigueBeta `
    --travel-beta $TravelBeta `
    --toi-mode $ToiMode `
    --starter-source $StarterSource `
    --goal-model $GoalModel `
    --out-prefix $OutPrefix
}

$startDt = [datetime]::ParseExact($Start, "yyyy-MM-dd", $null)
$endDt = [datetime]::ParseExact($End, "yyyy-MM-dd", $null)

for ($d = $startDt; $d -le $endDt; $d = $d.AddDays(1)) {
  $ds = $d.ToString("yyyy-MM-dd")

  try {
    Invoke-BoxscoreSim -Date $ds -OutPrefix $FromShotsPrefix -GoalModel "from_shots"
  } catch {
    Write-Warning ("[ab-goal-model] from_shots sim failed for {0}: {1}" -f $ds, $_.Exception.Message)
  }

  try {
    Invoke-BoxscoreSim -Date $ds -OutPrefix $IndependentPrefix -GoalModel "independent"
  } catch {
    Write-Warning ("[ab-goal-model] independent sim failed for {0}: {1}" -f $ds, $_.Exception.Message)
  }
}

Write-Host "[ab-goal-model] running backtests..."
python -m nhl_betting.cli props-backtest-from-boxscores --start $Start --end $End --read-prefix $FromShotsPrefix --out-prefix ("{0}_ab" -f $FromShotsPrefix)
python -m nhl_betting.cli props-backtest-from-boxscores --start $Start --end $End --read-prefix $IndependentPrefix --out-prefix ("{0}_ab" -f $IndependentPrefix)

Write-Host "[ab-goal-model] summarizing A/B deltas..."
$deltaOut = Join-Path $RepoRoot ("data\\processed\\ab_props_backtest_delta_{0}_vs_{1}_{2}_to_{3}.json" -f $FromShotsPrefix, $IndependentPrefix, $Start, $End)
python .\scripts\summarize_ab_props_backtest.py --start $Start --end $End --a-prefix $FromShotsPrefix --b-prefix $IndependentPrefix --out $deltaOut

Write-Host "[ab-goal-model] done"

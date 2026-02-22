param(
  [Parameter(Mandatory=$false)][string]$Start = "2026-01-20",
  [Parameter(Mandatory=$false)][string]$End = "2026-02-18",
  [Parameter(Mandatory=$false)][int]$NSims = 1000,
  [Parameter(Mandatory=$false)][int]$Seed = 42,
  [Parameter(Mandatory=$false)][double]$FatigueBeta = 0.03,
  [Parameter(Mandatory=$false)][double]$TravelBeta = 0.01,
  [Parameter(Mandatory=$false)][string]$ToiMode = "auto",
  [Parameter(Mandatory=$false)][string]$StarterSource = "auto",
  [Parameter(Mandatory=$false)][string]$SchedPrefix = "sched",
  [Parameter(Mandatory=$false)][string]$LegacyPrefix = "legacy"
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
    [Parameter(Mandatory=$true)][string]$RestSource,
    [Parameter(Mandatory=$true)][string]$OutPrefix
  )

  $histPath = Join-Path $RepoRoot ("data\\processed\\{0}_props_boxscores_sim_hist_{1}.csv" -f $OutPrefix, $Date)
  if (Test-Path $histPath) {
    Write-Host ("[ab] skip existing {0}" -f $histPath)
    return
  }

  Write-Host ("[ab] sim {0} rest={1} prefix={2} n={3}" -f $Date, $RestSource, $OutPrefix, $NSims)
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
    --out-prefix $OutPrefix
}

$startDt = [datetime]::ParseExact($Start, "yyyy-MM-dd", $null)
$endDt = [datetime]::ParseExact($End, "yyyy-MM-dd", $null)

for ($d = $startDt; $d -le $endDt; $d = $d.AddDays(1)) {
  $ds = $d.ToString("yyyy-MM-dd")

  try {
    Invoke-BoxscoreSim -Date $ds -RestSource "schedule" -OutPrefix $SchedPrefix
  } catch {
    Write-Warning ("[ab] schedule sim failed for {0}: {1}" -f $ds, $_.Exception.Message)
  }

  try {
    Invoke-BoxscoreSim -Date $ds -RestSource "team_games" -OutPrefix $LegacyPrefix
  } catch {
    Write-Warning ("[ab] legacy sim failed for {0}: {1}" -f $ds, $_.Exception.Message)
  }
}

Write-Host "[ab] running backtests..."
python -m nhl_betting.cli props-backtest-from-boxscores --start $Start --end $End --read-prefix $SchedPrefix --out-prefix ("{0}_ab" -f $SchedPrefix)
python -m nhl_betting.cli props-backtest-from-boxscores --start $Start --end $End --read-prefix $LegacyPrefix --out-prefix ("{0}_ab" -f $LegacyPrefix)

Write-Host "[ab] summarizing A/B deltas..."
$deltaOut = Join-Path $RepoRoot ("data\\processed\\ab_props_backtest_delta_{0}_vs_{1}_{2}_to_{3}.json" -f $SchedPrefix, $LegacyPrefix, $Start, $End)
python .\scripts\summarize_ab_props_backtest.py --start $Start --end $End --a-prefix $SchedPrefix --b-prefix $LegacyPrefix --out $deltaOut

Write-Host "[ab] done"

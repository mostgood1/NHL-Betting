param(
  [Parameter(Mandatory=$true)][string]$Start,
  [Parameter(Mandatory=$true)][string]$End,
  [int]$NSims = 800,
  [int]$Seed = 42,
  [double]$MinEvDelta = 0.02,
  [string]$UsageA = 'stochastic',
  [string]$UsageB = 'deterministic',
  [double]$NoisySigmaA = 0.18,
  [double]$NoisySigmaB = 0.18,
  [string]$PrefixA = 'usage_stoch',
  [string]$PrefixB = 'usage_det',
  [string]$DeltaOut = '',
  [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Test-NonEmptyFile([string]$Path) {
  try {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $item = Get-Item -LiteralPath $Path
    return ($item.Length -gt 0)
  } catch { return $false }
}

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$logPath = Join-Path $repo ("data\processed\usage_ab_eval_{0}_to_{1}.log" -f $Start, $End)
try {
  Start-Transcript -Path $logPath -Append | Out-Null
} catch {
  # Best-effort logging; continue even if transcript fails.
}

$py = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
  throw "Python venv not found at $py"
}

function Invoke-BoxscoreSim([string]$DateYmd, [string]$VariantPrefix, [string]$UsageModel, [double]$NoisySigma) {
  $hist = Join-Path $repo ("data\processed\{0}_props_boxscores_sim_hist_{1}.csv" -f $VariantPrefix, $DateYmd)
  $simOut = Join-Path $repo ("data\processed\{0}_props_boxscores_sim_{1}.csv" -f $VariantPrefix, $DateYmd)

  if (-not $Force -and (Test-NonEmptyFile $hist) -and (Test-NonEmptyFile $simOut)) {
    # Default to skipping when outputs exist; only re-run when we can confirm n_sims mismatch.
    $nsOk = $true
    try {
      $first = (Get-Content -LiteralPath $hist -TotalCount 2 | ConvertFrom-Csv | Select-Object -First 1)
      if ($null -ne $first) {
        $ns = [int]($first.n_sims)
        if ($ns -ne $NSims) { $nsOk = $false }
      }
    } catch {
      $nsOk = $true
    }
    if ($nsOk) {
      Write-Output "[skip] $VariantPrefix $DateYmd (already exists; n_sims=$NSims)"
      return
    }
  }

  Write-Output "[sim]  $VariantPrefix $DateYmd usage=$UsageModel n=$NSims"
  & $py -m nhl_betting.cli props-simulate-boxscores --date $DateYmd --n-sims $NSims --seed $Seed --out-prefix $VariantPrefix --usage-model $UsageModel --usage-noisy-sigma $NoisySigma
}

$dt0 = [datetime]::ParseExact($Start, 'yyyy-MM-dd', $null)
$dt1 = [datetime]::ParseExact($End, 'yyyy-MM-dd', $null)

for ($dt = $dt0; $dt -le $dt1; $dt = $dt.AddDays(1)) {
  $d = $dt.ToString('yyyy-MM-dd')
  Invoke-BoxscoreSim -DateYmd $d -VariantPrefix $PrefixA -UsageModel $UsageA -NoisySigma $NoisySigmaA
  Invoke-BoxscoreSim -DateYmd $d -VariantPrefix $PrefixB -UsageModel $UsageB -NoisySigma $NoisySigmaB
}

Write-Output "[backtest] $PrefixA ($UsageA) vs $PrefixB ($UsageB)  $Start..$End"
& $py -m nhl_betting.cli props-backtest-from-boxscores --start $Start --end $End --out-prefix ("{0}_ab" -f $PrefixA) --read-prefix $PrefixA --markets SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS --min-ev -1
& $py -m nhl_betting.cli props-backtest-from-boxscores --start $Start --end $End --out-prefix ("{0}_ab" -f $PrefixB) --read-prefix $PrefixB --markets SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS --min-ev -1

Write-Output "[delta] min_ev=$MinEvDelta"
$deltaPath = $DeltaOut
if (-not $deltaPath -or $deltaPath.Trim() -eq '') {
  $deltaPath = ("data/processed/usage_model_delta_{0}_to_{1}.json" -f $Start, $End)
}
& $py scripts\summarize_ab_props_backtest.py --start $Start --end $End --a-prefix $PrefixA --b-prefix $PrefixB --min-ev $MinEvDelta --out $deltaPath

Write-Output "Done."

try {
  Stop-Transcript | Out-Null
} catch {
}

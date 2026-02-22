param(
  [Parameter(Mandatory=$true)][string]$Start,
  [Parameter(Mandatory=$true)][string]$End,
  [int]$NSims = 800,
  [int]$Seed = 42,
  [double]$MinEvDelta = 0.02,
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

$logPath = Join-Path $repo ("data\processed\assist_ab_eval_{0}_to_{1}.log" -f $Start, $End)
try {
  Start-Transcript -Path $logPath -Append | Out-Null
} catch {
  # Best-effort logging; continue even if transcript fails.
}

$py = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $py)) {
  throw "Python venv not found at $py"
}

function Invoke-BoxscoreSim([string]$DateYmd, [string]$VariantPrefix, [string]$AssistModel) {
  $hist = Join-Path $repo ("data\processed\{0}_props_boxscores_sim_hist_{1}.csv" -f $VariantPrefix, $DateYmd)
  $simOut = Join-Path $repo ("data\processed\{0}_props_boxscores_sim_{1}.csv" -f $VariantPrefix, $DateYmd)

  if (-not $Force -and (Test-NonEmptyFile $hist) -and (Test-NonEmptyFile $simOut)) {
    $nsOk = $false
    try {
      $first = (Get-Content -LiteralPath $hist -TotalCount 2 | ConvertFrom-Csv | Select-Object -First 1)
      if ($null -ne $first) {
        $ns = [int]($first.n_sims)
        if ($ns -eq $NSims) { $nsOk = $true }
      }
    } catch {
      $nsOk = $false
    }
    if ($nsOk) {
      Write-Output "[skip] $VariantPrefix $DateYmd (already exists; n_sims=$NSims)"
      return
    }
  }

  Write-Output "[sim]  $VariantPrefix $DateYmd assists=$AssistModel n=$NSims"
  & $py -m nhl_betting.cli props-simulate-boxscores --date $DateYmd --n-sims $NSims --seed $Seed --out-prefix $VariantPrefix --assist-model $AssistModel
}

$dt0 = [datetime]::ParseExact($Start, 'yyyy-MM-dd', $null)
$dt1 = [datetime]::ParseExact($End, 'yyyy-MM-dd', $null)

for ($dt = $dt0; $dt -le $dt1; $dt = $dt.AddDays(1)) {
  $d = $dt.ToString('yyyy-MM-dd')
  Invoke-BoxscoreSim -DateYmd $d -VariantPrefix 'assist_onice' -AssistModel 'onice'
  Invoke-BoxscoreSim -DateYmd $d -VariantPrefix 'assist_legacy' -AssistModel 'legacy'
}

Write-Output "[backtest] assist_onice vs assist_legacy  $Start..$End"
& $py -m nhl_betting.cli props-backtest-from-boxscores --start $Start --end $End --out-prefix assist_onice_ab --read-prefix assist_onice --markets SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS --min-ev -1
& $py -m nhl_betting.cli props-backtest-from-boxscores --start $Start --end $End --out-prefix assist_legacy_ab --read-prefix assist_legacy --markets SOG,SAVES,GOALS,ASSISTS,POINTS,BLOCKS --min-ev -1

Write-Output "[delta] min_ev=$MinEvDelta"
& $py scripts\summarize_ab_props_backtest.py --start $Start --end $End --a-prefix assist_onice --b-prefix assist_legacy --min-ev $MinEvDelta --out ("data/processed/assist_model_delta_{0}_to_{1}.json" -f $Start, $End)

Write-Output "Done."

try {
  Stop-Transcript | Out-Null
} catch {
}
param(
  [int]$Days = 30,
  [switch]$IncludeMultiplierSweep,
  [switch]$RunMoneylineSelection = $true,
  [switch]$MoneylineOnly,
  [int]$MoneylineHoldoutDays = 30,
  [string]$MoneylineCandidatePath = "data/processed/model_calibration_weekly_candidate.json",
  [string]$MoneylineCandidateWeights = "0.4,0.5,0.6,0.7,0.8,1.0",
  [switch]$MoneylineAutoFineGrid = $true,
  [double]$MoneylineFineRadius = 0.10,
  [double]$MoneylineFineStep = 0.05,
  [string]$MoneylineComparePath = "data/processed/compare_moneyline_calibration_candidates_weekly.json"
)

$ErrorActionPreference = 'Stop'

function Get-MoneylineCandidateWeights {
  param(
    [string]$WeightsCsv,
    [double]$FallbackWeight
  )

  $weights = New-Object System.Collections.Generic.List[double]
  foreach ($part in ($WeightsCsv -split ',')) {
    $trimmed = $part.Trim()
    if (-not $trimmed) {
      continue
    }
    $parsed = 0.0
    if ([double]::TryParse($trimmed, [System.Globalization.NumberStyles]::Float, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$parsed)) {
      if ($parsed -ge 0.0 -and $parsed -le 1.0) {
        $weights.Add([Math]::Round($parsed, 4))
      }
    }
  }
  if ($weights.Count -eq 0) {
    $weights.Add([Math]::Round($FallbackWeight, 4))
  }
  return $weights | Sort-Object -Unique
}

function Add-MoneylineFineGridWeights {
  param(
    [double[]]$BaseWeights,
    [double[]]$Centers,
    [double]$Radius,
    [double]$Step
  )

  $weights = New-Object System.Collections.Generic.List[double]
  foreach ($weight in $BaseWeights) {
    $weights.Add([Math]::Round([double]$weight, 4))
  }

  if ($Step -le 0.0 -or $Radius -le 0.0) {
    return $weights | Sort-Object -Unique
  }

  $radius = [Math]::Abs($Radius)
  $step = [Math]::Abs($Step)
  foreach ($center in $Centers) {
    $centerValue = [double]$center
    for ($offset = -1.0 * $radius; $offset -le ($radius + 1e-9); $offset += $step) {
      $candidate = [Math]::Round($centerValue + $offset, 4)
      if ($candidate -ge 0.0 -and $candidate -le 1.0) {
        $weights.Add($candidate)
      }
    }
  }

  return $weights | Sort-Object -Unique
}

function Write-MoneylineAnchorOnlyVariant {
  param(
    [string]$TemplatePath,
    [string]$OutPath,
    [double]$AnchorWeight,
    [string]$VariantLabel
  )

  $calibration = Get-Content $TemplatePath -Raw | ConvertFrom-Json
  $anchorWeight = [Math]::Round($AnchorWeight, 4)

  $calibration.market_anchor_w = $anchorWeight
  $calibration.market_anchor_w_ml = $anchorWeight
  $calibration.ml_temp = 1.0
  $calibration.ml_bias = 0.0
  $calibration | Add-Member -Force -NotePropertyName ml_post_calibration_policy -NotePropertyValue 'anchor_only'
  if (-not $calibration.moneyline) {
    $calibration | Add-Member -NotePropertyName moneyline -NotePropertyValue ([pscustomobject]@{})
  }
  $calibration.moneyline.t = 1.0
  $calibration.moneyline.b = 0.0
  if ($calibration.PSObject.Properties.Name -contains 'ml_validation') {
    $calibration.ml_validation | Add-Member -Force -NotePropertyName forced_anchor_only -NotePropertyValue $true
  }
  $calibration | Add-Member -Force -NotePropertyName moneyline_candidate_variant -NotePropertyValue $VariantLabel
  $calibration | Add-Member -Force -NotePropertyName last_calibrated_utc -NotePropertyValue ([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ'))

  $outDir = Split-Path $OutPath -Parent
  if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Path $outDir | Out-Null
  }
  $calibration | ConvertTo-Json -Depth 10 | Set-Content -Path $OutPath
}

# Activate environment
. .\activate_npu.ps1
. .\.venv\Scripts\Activate.ps1

# Date range (ET dates)
$end = (Get-Date).ToString('yyyy-MM-dd')
$start = (Get-Date).AddDays(-$Days).ToString('yyyy-MM-dd')

Write-Host "[weekly] Modes: MoneylineOnly=$MoneylineOnly RunMoneylineSelection=$RunMoneylineSelection Days=$Days HoldoutDays=$MoneylineHoldoutDays" -ForegroundColor DarkCyan

if (-not $MoneylineOnly) {
  Write-Host "[weekly] Calibrating simulations for $start..$end" -ForegroundColor Cyan
  # Global calibration (moneyline, totals, puckline)
  python -m nhl_betting.cli game-calibrate-sim --start $start --end $end

  # Per-total-line calibration for totals
  python -m nhl_betting.cli game-calibrate-sim-per-line --start $start --end $end

  # Precompute possession sim events over the window (ensures events exist)
  try {
    $dates = @(for ($d = [DateTime]::Parse($start); $d -le [DateTime]::Parse($end); $d = $d.AddDays(1)) { $d.ToString('yyyy-MM-dd') })
    foreach ($dt in $dates) {
      # Skip dates with no scheduled games
      try {
        $hasGames = (python -c "from nhl_betting.data.nhl_api_web import NHLWebClient; import sys; print(1 if NHLWebClient().schedule_day('$dt') else 0)").Trim()
        if ($hasGames -ne '1') {
          Write-Host "[weekly] Skipping ${dt} (no games)" -ForegroundColor DarkGray
          continue
        }
      } catch {
        Write-Warning "[weekly] Failed schedule check for ${dt}: $($_.Exception.Message)"; 
      }
      $eventsPath = Join-Path 'data/processed' "sim_events_pos_${dt}.csv"
      if (-not (Test-Path $eventsPath)) {
        Write-Host "[weekly] Preparing lineup/shifts and simulating possession for ${dt} …" -ForegroundColor DarkYellow
        try { python -m nhl_betting.cli lineup-update --date $dt } catch { Write-Warning "[weekly] lineup-update failed for ${dt}: $($_.Exception.Message)" }
        try { python -m nhl_betting.cli shifts-update --date $dt } catch { Write-Warning "[weekly] shifts-update failed for ${dt}: $($_.Exception.Message)" }
        try { python -m nhl_betting.cli game-simulate-possession --date $dt } catch { Write-Warning "[weekly] game-simulate-possession failed for ${dt}: $($_.Exception.Message)" }
      }
    }
  } catch {
    Write-Warning "[weekly] Failed to precompute possession events: $($_.Exception.Message)"
  }

  # Special teams (PP/PK) calibration from possession sim events
  try {
    Write-Host "[weekly] Calibrating special teams (PP/PK) …" -ForegroundColor Yellow
    python -m nhl_betting.cli game-calibrate-special-teams --start $start --end $end
  } catch {
    Write-Warning "[weekly] Special teams calibration failed: $($_.Exception.Message)"
  }

  # Optional: sweep totals multipliers (can be slow)
  if ($IncludeMultiplierSweep) {
    Write-Host "[weekly] Running totals multipliers sweep..." -ForegroundColor Yellow
    python scripts\calibrate_totals_multipliers.py
  }
} else {
  Write-Host "[weekly] MoneylineOnly enabled; skipping simulation calibration, possession prep, and totals sweep." -ForegroundColor DarkGray
}

if ($RunMoneylineSelection) {
  try {
    $holdoutDays = [Math]::Max(7, [int]$MoneylineHoldoutDays)
    $holdoutEndDt = [DateTime]::Parse($end)
    $holdoutStartDt = $holdoutEndDt.AddDays(-1 * ($holdoutDays - 1))
    $trainEndDt = $holdoutStartDt.AddDays(-1)
    $seasonStartYear = if ($holdoutEndDt.Month -ge 7) { $holdoutEndDt.Year } else { $holdoutEndDt.Year - 1 }
    $trainStartDt = Get-Date -Date ("{0}-09-01" -f $seasonStartYear)

    $trainStart = $trainStartDt.ToString('yyyy-MM-dd')
    $trainEnd = $trainEndDt.ToString('yyyy-MM-dd')
    $holdoutStart = $holdoutStartDt.ToString('yyyy-MM-dd')
    $holdoutEnd = $holdoutEndDt.ToString('yyyy-MM-dd')

    if ($trainEndDt -lt $trainStartDt) {
      Write-Warning "[weekly] Skipping moneyline selection: insufficient history before holdout window."
    } else {
      Write-Host "[weekly] Fitting moneyline calibration candidate on $trainStart..$trainEnd" -ForegroundColor Cyan
      python -m nhl_betting.cli game-auto-calibrate --start $trainStart --end $trainEnd --out-json $MoneylineCandidatePath

      $compareCandidates = New-Object System.Collections.Generic.List[string]
      $compareCandidates.Add('data/processed/model_calibration.json')
      $compareCandidates.Add($MoneylineCandidatePath)

      $candidateCalibration = Get-Content $MoneylineCandidatePath -Raw | ConvertFrom-Json
      $liveCalibration = Get-Content 'data/processed/model_calibration.json' -Raw | ConvertFrom-Json
      $fallbackWeight = if ($candidateCalibration.market_anchor_w_ml -ne $null) { [double]$candidateCalibration.market_anchor_w_ml } else { 1.0 }
      $liveWeight = if ($liveCalibration.market_anchor_w_ml -ne $null) { [double]$liveCalibration.market_anchor_w_ml } else { 1.0 }
      $candidateWeights = Get-MoneylineCandidateWeights -WeightsCsv $MoneylineCandidateWeights -FallbackWeight $fallbackWeight
      if ($MoneylineAutoFineGrid) {
        $candidateWeights = Add-MoneylineFineGridWeights -BaseWeights $candidateWeights -Centers @($liveWeight, $fallbackWeight) -Radius $MoneylineFineRadius -Step $MoneylineFineStep
      }

      foreach ($weight in $candidateWeights) {
        if ([Math]::Abs($weight - $liveWeight) -lt 0.0001 -and $liveCalibration.ml_post_calibration_policy -eq 'anchor_only') {
          continue
        }
        $weightSuffix = [int][Math]::Round($weight * 100.0)
        $variantPath = [System.IO.Path]::Combine(
          (Split-Path $MoneylineCandidatePath -Parent),
          ('{0}_anchor_w{1:D3}{2}' -f [System.IO.Path]::GetFileNameWithoutExtension($MoneylineCandidatePath), $weightSuffix, [System.IO.Path]::GetExtension($MoneylineCandidatePath))
        )
        Write-MoneylineAnchorOnlyVariant -TemplatePath $MoneylineCandidatePath -OutPath $variantPath -AnchorWeight $weight -VariantLabel ("anchor_only_w={0}" -f $weight)
        $compareCandidates.Add($variantPath)
      }

      Write-Host ("[weekly] Comparing {0} moneyline candidates on {1}..{2}" -f $compareCandidates.Count, $holdoutStart, $holdoutEnd) -ForegroundColor Cyan
      $compareArg = [string]::Join(',', ($compareCandidates | Select-Object -Unique))

      python -m nhl_betting.cli compare-moneyline-calibration-candidates $holdoutStart $holdoutEnd --candidates $compareArg --out-json $MoneylineComparePath

      Write-Host "[weekly] Promoting moneyline calibration winner if needed" -ForegroundColor Cyan
      python -m nhl_betting.cli promote-moneyline-calibration-winner $MoneylineComparePath
    }
  } catch {
    Write-Warning "[weekly] Moneyline selection failed: $($_.Exception.Message)"
  }
} else {
  Write-Host "[weekly] Moneyline selection skipped." -ForegroundColor DarkGray
}

Write-Host "[weekly] Calibration complete" -ForegroundColor Green

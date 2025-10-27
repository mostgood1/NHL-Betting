# Ensures .venv is created with an ARM64 Python interpreter and activates it (if requested by caller)
# Usage (typical):
#   . .\ensure_arm64_venv.ps1    # dot-source to define functions, then call Ensure-Arm64Venv
#   Ensure-Arm64Venv -RepoRoot $PSScriptRoot -Activate

param(
  [string]$RepoRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
  [switch]$Activate
)

$ErrorActionPreference = 'Stop'

function Get-PythonArch {
  param([string]$PythonExe)
  try {
    $code = 'import platform,struct; print(platform.machine()); print(struct.calcsize("P")*8)'
    $out = & $PythonExe -c $code 2>$null
    $lines = @($out -split "`n") | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    $arch = if ($lines.Length -ge 1) { $lines[0] } else { '' }
    $bits = if ($lines.Length -ge 2) { $lines[1] } else { '' }
    return @{ arch=$arch; bits=$bits }
  } catch {
    return @{ arch=''; bits='' }
  }
}

function Find-Arm64Python {
  # 1) Respect explicit env var
  if ($env:ARM64_PYTHON -and (Test-Path $env:ARM64_PYTHON)) {
    $info = Get-PythonArch -PythonExe $env:ARM64_PYTHON
    if ($info.arch -match 'ARM' -or $info.bits -eq '64') {
      return $env:ARM64_PYTHON
    }
  }
  # 2) Try py launcher to enumerate installations
  try {
    $candidates = & py -0p 2>$null
    if ($LASTEXITCODE -eq 0 -and $candidates) {
      foreach ($line in ($candidates -split "`n")) {
        $exe = $line.Trim()
        if (-not $exe) { continue }
        if (-not (Test-Path $exe)) { continue }
        $info = Get-PythonArch -PythonExe $exe
        if ($info.arch -match 'ARM') { return $exe }
      }
    }
  } catch {}
  # 3) Try common install paths
  $paths = @(
    "$env:ProgramFiles\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "C:\\Program Files\\WindowsApps\\PythonSoftwareFoundation.Python.3.11*_x64__*\\python.exe",
    "C:\\Program Files\\WindowsApps\\PythonSoftwareFoundation.Python.3.11*_arm64__*\\python.exe"
  )
  foreach ($p in $paths) {
    foreach ($cand in (Get-Item $p -ErrorAction SilentlyContinue)) {
      $exe = $cand.FullName
      $info = Get-PythonArch -PythonExe $exe
      if ($info.arch -match 'ARM') { return $exe }
    }
  }
  return $null
}

function Ensure-Arm64Venv {
  param(
    [string]$RepoRoot,
    [switch]$Activate
  )
  $venv = Join-Path $RepoRoot '.venv'
  $venvPy = Join-Path $venv 'Scripts/python.exe'
  $needCreate = $true
  if (Test-Path $venvPy) {
    $info = Get-PythonArch -PythonExe $venvPy
    if ($info.arch -match 'ARM') { $needCreate = $false }
  }
  if ($needCreate) {
    $armPy = Find-Arm64Python
    if (-not $armPy) {
      Write-Warning '[ARM64] Could not find an ARM64 Python installation. Install Python 3.11 ARM64 and set ARM64_PYTHON if needed.'
      return $false
    }
    if (Test-Path $venv) { Write-Host '[ARM64] Recreating .venv with ARM64 Python…' -ForegroundColor Yellow; Remove-Item -Recurse -Force $venv }
    & $armPy -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create ARM64 venv' }
  }
  if ($Activate) {
    $act = Join-Path $venv 'Scripts/Activate.ps1'
    . $act
    # Ensure base dependencies
    $req = Join-Path $RepoRoot 'requirements.txt'
    if (Test-Path $req) { pip install -q -r $req }
    # Ensure onnxruntime ARM64 if wheel present
    $wheel = Join-Path $RepoRoot 'onnxruntime-1.23.1-cp311-cp311-win_arm64.whl'
    if (Test-Path $wheel) {
      pip uninstall -y onnxruntime onnxruntime-gpu 2>$null | Out-Null
      pip install -q "$wheel"
    }
  }
  return $true
}

# If called directly (not dot-sourced), enforce and optionally activate
if ($MyInvocation.InvocationName -ne '.') {
  $ok = Ensure-Arm64Venv -RepoRoot $RepoRoot -Activate:$Activate
  if (-not $ok) { exit 1 }
}

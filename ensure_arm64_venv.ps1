# Ensures .venv is created with an ARM64 Python interpreter and activates it (if requested by caller)
# Usage (typical):
#   . .\ensure_arm64_venv.ps1    # dot-source to define functions, then call Ensure-Arm64Venv
#   Ensure-Arm64Venv -RepoRoot $PSScriptRoot -Activate

param(
  [string]$RepoRoot = (Split-Path -Parent $MyInvocation.MyCommand.Path),
  [switch]$Activate,
  [switch]$AutoInstall  # Attempt to auto-install an ARM64 Python via winget if not found
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
  # 3) Try common install paths and WindowsApps package folders (wildcards via Get-ChildItem)
  $paths = @(
    (Join-Path $env:ProgramFiles 'Python311/python.exe'),
    (Join-Path $env:LOCALAPPDATA 'Programs/Python/Python311/python.exe')
  )
  $cands = @()
  foreach ($p in $paths) {
    if (Test-Path $p) { $cands += $p }
  }
  # WindowsApps store installs
  $wa = 'C:\Program Files\WindowsApps'
  if (Test-Path $wa) {
    $pkgDirs = Get-ChildItem -Path $wa -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like 'PythonSoftwareFoundation.Python.3.11*_arm64__*' }
    foreach ($d in $pkgDirs) {
      $exe = Join-Path $d.FullName 'python.exe'
      if (Test-Path $exe) { $cands += $exe }
    }
  }
  foreach ($exe in $cands) {
    $info = Get-PythonArch -PythonExe $exe
    if ($info.arch -match 'ARM') { return $exe }
  }
  return $null
}

function Install-Arm64Python {
  Write-Host '[ARM64] Attempting to install Python 3.11 (arm64) via winget…' -ForegroundColor Yellow
  try {
    $wing = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $wing) { Write-Warning '[ARM64] winget not available; cannot auto-install Python.'; return $false }
    # Prefer the PythonSoftwareFoundation package id; specify architecture arm64 and silent flags
    $args = @('install','--id','PythonSoftwareFoundation.Python.3.11','--accept-package-agreements','--accept-source-agreements','--silent','--architecture','arm64')
  & winget @args *> $null
    Start-Sleep -Seconds 3
    $found = Find-Arm64Python
    if ($found) { Write-Host "[ARM64] Installed ARM64 Python at $found" -ForegroundColor Green; return $true }
    # Fallback id
    $args2 = @('install','--id','Python.Python.3.11','--accept-package-agreements','--accept-source-agreements','--silent','--architecture','arm64')
  & winget @args2 *> $null
    Start-Sleep -Seconds 3
    $found2 = Find-Arm64Python
    if ($found2) { Write-Host "[ARM64] Installed ARM64 Python at $found2" -ForegroundColor Green; return $true }
    Write-Warning '[ARM64] winget install did not yield an ARM64 Python. Install manually from Microsoft Store or python.org.'
    return $false
  } catch {
    Write-Warning "[ARM64] Auto-install failed: $($_.Exception.Message)"
    return $false
  }
}

function Ensure-Arm64Venv {
  param(
    [string]$RepoRoot,
    [switch]$Activate,
    [switch]$AutoInstall
  )
  $venv = Join-Path $RepoRoot '.venv'
  $venvPy = Join-Path $venv 'Scripts/python.exe'
  $needCreate = $true
  # Fast-path: if venv folder exists, assume reuse (we'll verify arch on activation)
  if (Test-Path $venv) {
    $needCreate = $false
  }
  if (Test-Path $venvPy) {
    $info = Get-PythonArch -PythonExe $venvPy
    if ($info.arch -match 'ARM') { $needCreate = $false }
  }
  if ($needCreate) {
    $armPy = Find-Arm64Python
    if (-not $armPy) {
      if ($AutoInstall) {
        $okInstall = Install-Arm64Python
        if ($okInstall) { $armPy = Find-Arm64Python }
      }
      if (-not $armPy) {
        Write-Warning '[ARM64] Could not find an ARM64 Python installation. Install Python 3.11 ARM64 and set ARM64_PYTHON if needed.'
        return $false
      }
    }
    if (Test-Path $venv) { Write-Host '[ARM64] Recreating .venv with ARM64 Python…' -ForegroundColor Yellow; Remove-Item -Recurse -Force $venv }
  & $armPy -m venv $venv *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create ARM64 venv' }
  }
  if ($Activate) {
    $act = Join-Path $venv 'Scripts/Activate.ps1'
    . $act
    # Optionally install base dependencies (opt-in via env to reduce activation noise)
    $req = Join-Path $RepoRoot 'requirements.txt'
    if ($env:NPU_INSTALL_DEPS -eq '1' -and (Test-Path $req)) {
      pip install -q -r $req
    }
    # Optionally install onnxruntime ARM64 wheel if present (opt-in)
    $wheel = Join-Path $RepoRoot 'onnxruntime-1.23.1-cp311-cp311-win_arm64.whl'
    if ($env:NPU_INSTALL_ONNX -eq '1' -and (Test-Path $wheel)) {
      pip uninstall -y onnxruntime onnxruntime-gpu *> $null
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

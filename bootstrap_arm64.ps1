# Bootstrap an ARM64 Python virtual environment for this repo
# - Installs Python 3.11 ARM64 via winget (if missing)
# - Creates .venv using ARM64 interpreter
# - Installs requirements and the ARM64 onnxruntime wheel (if present)
# - Prints interpreter path and architecture

$ErrorActionPreference = 'Stop'
try { Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force | Out-Null } catch {}

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $RepoRoot

# Ensure script exists
$ensure = Join-Path $RepoRoot 'ensure_arm64_venv.ps1'
if (-not (Test-Path $ensure)) { Write-Error "ensure_arm64_venv.ps1 not found"; exit 1 }
. $ensure

# Create and activate ARM64 venv (auto-install Python if needed)
$ok = Ensure-Arm64Venv -RepoRoot $RepoRoot -Activate -AutoInstall
if (-not $ok) { Write-Error 'Failed to create/activate ARM64 venv. Install ARM64 Python manually and retry.'; exit 1 }

# Show interpreter details
$py = "import platform, sys; print('[python]', sys.executable); print('[arch]', platform.machine())"
python -c $py

# Install requirements
$req = Join-Path $RepoRoot 'requirements.txt'
if (Test-Path $req) {
  pip install -q -r $req
}

# Install onnxruntime ARM64 wheel if present
$wheel = Join-Path $RepoRoot 'onnxruntime-1.23.1-cp311-cp311-win_arm64.whl'
if (Test-Path $wheel) {
  # Suppress any output from uninstall attempts (package may not be installed)
  pip uninstall -y onnxruntime onnxruntime-gpu *> $null
  pip install -q "$wheel"
}

# Quick provider sanity check (optional)
$pycheck = "import onnxruntime as ort, sys; print('onnxruntime', ort.__version__); print('providers', ort.get_available_providers())"
python -c $pycheck

Write-Host "[OK] ARM64 venv bootstrap complete" -ForegroundColor Green

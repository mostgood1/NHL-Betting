# NPU Activation Script
# Use this to activate the QNN SDK paths before running NPU-accelerated commands

$env:QNN_SDK_ROOT = "C:\Qualcomm\QNN_SDK"
$env:PATH = "C:\Qualcomm\QNN_SDK\lib\arm64x-windows-msvc;$env:PATH"

Write-Host "[OK] QNN SDK paths configured" -ForegroundColor Green
Write-Host "  SDK Root: $env:QNN_SDK_ROOT" -ForegroundColor Yellow
Write-Host "  NPU Libraries: C:\Qualcomm\QNN_SDK\lib\arm64x-windows-msvc" -ForegroundColor Yellow
Write-Host ""
Write-Host "Ready to use NPU!" -ForegroundColor Cyan
Write-Host "  Use system Python for NPU operations:" -ForegroundColor Yellow
Write-Host "  python -m nhl_betting.scripts.train_nn_props train-all --epochs 50" -ForegroundColor White
Write-Host ""

# Ensure .venv is ARM64 so ONNX QNN EP can load (optional best-effort)
try {
	$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
	$Ensure = Join-Path $RepoRoot 'ensure_arm64_venv.ps1'
	if (Test-Path $Ensure) {
		. $Ensure
		$ok = Ensure-Arm64Venv -RepoRoot $RepoRoot
		if (-not $ok) {
			Write-Warning "[ARM64] .venv is not ARM64; QNN EP may be unavailable in this session."
			Write-Warning "         Install ARM64 Python and set ARM64_PYTHON env var if auto-detect fails."
		}
	}
} catch {
	Write-Warning "[ARM64] ensure_arm64_venv.ps1 failed: $($_.Exception.Message)"
}

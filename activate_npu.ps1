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

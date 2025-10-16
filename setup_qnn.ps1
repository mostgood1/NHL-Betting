# Configure Qualcomm QNN SDK Environment for NPU Acceleration
# Run this script once to set up paths for ONNX Runtime QNN execution provider

$QNN_SDK_ROOT = "C:\Qualcomm\QNN_SDK"
$QNN_LIB_PATH = "$QNN_SDK_ROOT\lib\arm64x-windows-msvc"

Write-Host "Configuring Qualcomm QNN SDK for NPU acceleration..." -ForegroundColor Cyan
Write-Host "SDK Root: $QNN_SDK_ROOT" -ForegroundColor Yellow

# Check if SDK exists
if (-not (Test-Path $QNN_SDK_ROOT)) {
    Write-Host "[ERROR] QNN SDK not found at $QNN_SDK_ROOT" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $QNN_LIB_PATH)) {
    Write-Host "[ERROR] ARM64 libraries not found at $QNN_LIB_PATH" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] SDK found" -ForegroundColor Green

# Set environment variables for current session
$env:QNN_SDK_ROOT = $QNN_SDK_ROOT
$env:PATH = "$QNN_LIB_PATH;$env:PATH"

Write-Host ""
Write-Host "Environment variables set:" -ForegroundColor Cyan
Write-Host "  QNN_SDK_ROOT = $env:QNN_SDK_ROOT" -ForegroundColor Yellow
Write-Host "  PATH includes = $QNN_LIB_PATH" -ForegroundColor Yellow

# Verify key DLLs
Write-Host ""
Write-Host "Verifying QNN libraries..." -ForegroundColor Cyan
$required_dlls = @("QnnHtp.dll", "QnnSystem.dll", "QnnCpu.dll")
$all_present = $true

foreach ($dll in $required_dlls) {
    $dll_path = Join-Path $QNN_LIB_PATH $dll
    if (Test-Path $dll_path) {
        Write-Host "  [OK] $dll" -ForegroundColor Green
    } else {
        Write-Host "  [MISSING] $dll" -ForegroundColor Red
        $all_present = $false
    }
}

if ($all_present) {
    Write-Host ""
    Write-Host "All QNN libraries found!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[ERROR] Some QNN libraries are missing" -ForegroundColor Red
    exit 1
}

# Test ONNX Runtime NPU availability
Write-Host ""
Write-Host "Testing ONNX Runtime QNN provider..." -ForegroundColor Cyan

# Activate venv
$venv_activate = ".\..venv\Scripts\Activate.ps1"
if (Test-Path $venv_activate) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & $venv_activate
} else {
    Write-Host "[WARN] Virtual environment not found, using system Python" -ForegroundColor Yellow
}

# Check ONNX Runtime providers
$check_script = @"
import sys
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print('\nONNX Runtime version:', ort.__version__)
    print('Available providers:', providers)
    
    if 'QNNExecutionProvider' in providers:
        print('\n[SUCCESS] QNN Execution Provider is available!')
        print('  Your Qualcomm NPU is ready for inference.')
        sys.exit(0)
    else:
        print('\n[WARN] QNNExecutionProvider not found.')
        print('  You may need to install onnxruntime-qnn.')
        print('  Available providers:', ', '.join(providers))
        sys.exit(1)
except ImportError as e:
    print('\n[ERROR] onnxruntime not installed')
    print('  Run: pip install onnxruntime')
    sys.exit(1)
except Exception as e:
    print(f'\n[ERROR] {e}')
    sys.exit(1)
"@

python -c $check_script

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "===================================================" -ForegroundColor Green
    Write-Host "[SUCCESS] NPU CONFIGURATION COMPLETE!" -ForegroundColor Green
    Write-Host "===================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Train models:" -ForegroundColor Yellow
    Write-Host "     python -m nhl_betting.scripts.train_nn_props train-all --epochs 50" -ForegroundColor White
    Write-Host ""
    Write-Host "  2. Benchmark NPU vs CPU:" -ForegroundColor Yellow
    Write-Host "     python -m nhl_betting.scripts.train_nn_props benchmark --market SOG" -ForegroundColor White
    Write-Host ""
    Write-Host "  3. Use NPU in production:" -ForegroundColor Yellow
    Write-Host "     Set environment: USE_NPU=1" -ForegroundColor White
    Write-Host "     python -m nhl_betting.cli props-recommendations --date 2025-10-16" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "===================================================" -ForegroundColor Yellow
    Write-Host "[WARN] QNN PROVIDER NOT AVAILABLE" -ForegroundColor Yellow
    Write-Host "===================================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "The QNN SDK libraries are present, but ONNX Runtime" -ForegroundColor White
    Write-Host "cannot access the QNN execution provider." -ForegroundColor White
    Write-Host ""
    Write-Host "Possible solutions:" -ForegroundColor Cyan
    Write-Host "  1. Install onnxruntime-qnn (if available on PyPI):" -ForegroundColor Yellow
    Write-Host "     pip uninstall onnxruntime" -ForegroundColor White
    Write-Host "     pip install onnxruntime-qnn" -ForegroundColor White
    Write-Host ""
    Write-Host "  2. Check Qualcomm documentation for ONNX Runtime integration:" -ForegroundColor Yellow
    Write-Host "     C:\Qualcomm\QNN_SDK\docs\" -ForegroundColor White
    Write-Host ""
    Write-Host "  3. Your models will still work with CPU fallback" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Optionally: Set user environment variables permanently
$set_permanent = Read-Host "Set QNN_SDK_ROOT permanently for your user account? (y/n)"
if ($set_permanent -eq 'y' -or $set_permanent -eq 'Y') {
    try {
        [Environment]::SetEnvironmentVariable("QNN_SDK_ROOT", $QNN_SDK_ROOT, "User")
        
        # Add to user PATH if not already present
        $user_path = [Environment]::GetEnvironmentVariable("PATH", "User")
        if ($user_path -notlike "*$QNN_LIB_PATH*") {
            [Environment]::SetEnvironmentVariable("PATH", "$QNN_LIB_PATH;$user_path", "User")
            Write-Host ""
            Write-Host "[OK] Environment variables set permanently" -ForegroundColor Green
            Write-Host "     Restart your terminal for changes to take effect" -ForegroundColor Yellow
        } else {
            Write-Host ""
            Write-Host "[OK] PATH already includes QNN libraries" -ForegroundColor Green
        }
    } catch {
        Write-Host ""
        Write-Host "[WARN] Could not set permanent environment variables: $_" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green

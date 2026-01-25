# NPU Activation Script
# Use this to activate the QNN SDK paths before running NPU-accelerated commands

$env:QNN_SDK_ROOT = "C:\Qualcomm\QNN_SDK"
$env:PATH = "C:\Qualcomm\QNN_SDK\lib\arm64x-windows-msvc;$env:PATH"

Write-Host "[OK] QNN SDK paths configured" -ForegroundColor Green
Write-Host "  SDK Root: $env:QNN_SDK_ROOT" -ForegroundColor Yellow
Write-Host "  NPU Libraries: C:\Qualcomm\QNN_SDK\lib\arm64x-windows-msvc" -ForegroundColor Yellow
Write-Host ""
Write-Host "Ready to use NPU!" -ForegroundColor Cyan
Write-Host "  Python will be provided by the repo's ARM64 .venv (auto-activated below)." -ForegroundColor Yellow
Write-Host ""

# Enforce and activate ARM64 venv so it becomes the DEFAULT Python for this session
try {
	$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
	$Ensure = Join-Path $RepoRoot 'ensure_arm64_venv.ps1'
	if (Test-Path $Ensure) {
		. $Ensure
	$ok = Ensure-Arm64Venv -RepoRoot $RepoRoot -Activate
		if (-not $ok) {
			# Fallback: try activating existing .venv and verify arch
			$act = Join-Path $RepoRoot '.venv/Scripts/Activate.ps1'
			if (Test-Path $act) { . $act }
			$archCheck = "import platform, sys; print(sys.executable); print(platform.machine())"
			$lines = python -c $archCheck 2>$null
			if ($LASTEXITCODE -eq 0 -and $lines -match 'ARM') {
				$ok = $true
			} else {
				Write-Warning "[ARM64] .venv is not ARM64; QNN EP may be unavailable in this session."
				Write-Warning "         Install ARM64 Python and set ARM64_PYTHON env var if auto-detect fails."
			}
		}
		if ($ok) {
			# Show interpreter info for visibility
			Write-Host "[venv] Activated: $((Join-Path $RepoRoot '.venv'))" -ForegroundColor Green
			$py = "import platform, sys; print('[python]', sys.executable); print('[arch]', platform.machine())"
			python -c $py | ForEach-Object { Write-Host $_ }
		}
	}
} catch {
	Write-Warning "[ARM64] ensure_arm64_venv.ps1 failed: $($_.Exception.Message)"
}

# Load environment variables from .env (if present), e.g. ODDS_API_KEY
try {
    $EnvPath = Join-Path $RepoRoot '.env'
    if (Test-Path $EnvPath) {
        Get-Content $EnvPath | ForEach-Object {
            $line = $_.Trim()
            if ($line -eq '' -or $line.StartsWith('#')) { return }
            if ($line -match '^[A-Za-z_][A-Za-z0-9_]*=') {
                $parts = $line.Split('=', 2)
                $name = $parts[0].Trim()
                $val = $parts[1].Trim()
                # Strip optional surrounding double quotes
                if ($val.StartsWith('"') -and $val.EndsWith('"')) { $val = $val.Trim('"') }
                Set-Item -Path "Env:$name" -Value $val -ErrorAction SilentlyContinue
            }
        }
        Write-Host "[env] Loaded .env from $EnvPath" -ForegroundColor Green
    }
} catch {
    Write-Warning "[env] Failed to load .env: $($_.Exception.Message)"
}

@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%daily_update.ps1"

if not exist "%PS1%" (
  echo [ERROR] daily_update.ps1 not found beside this file.
  echo Looked for: %PS1%
  pause
  exit /b 1
)

REM Forward any arguments to the PowerShell script (e.g., -DaysAhead 2 -NoReconcile)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS1%" %*

REM Keep the window open so you can read output when double-clicking
echo.
echo [Done] Press any key to close...
pause >nul

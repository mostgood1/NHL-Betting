Param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8010,
    [switch]$NoReload
)

$ErrorActionPreference = "Stop"

$Inner = Join-Path $PSScriptRoot "scripts/launch_local.ps1"
if (-not (Test-Path $Inner)) {
    Write-Error ("Inner launcher not found: {0}" -f $Inner)
    exit 1
}

# Build argument list for inner script
$ArgsList = @("-HostAddress", $HostAddress, "-Port", "$Port")
if ($NoReload) { $ArgsList += "-NoReload" }

# Bypass execution policy for convenience
powershell -ExecutionPolicy Bypass -File $Inner @ArgsList

# Compatibility wrapper.
# Canonical implementation lives at scripts/daily_update.ps1.

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$CanonicalScript = Join-Path $RepoRoot "scripts/daily_update.ps1"

if (-not (Test-Path $CanonicalScript)) {
  throw "[daily_update] Canonical script not found: $CanonicalScript"
}

Push-Location $RepoRoot
try {
  $forwardArgs = @($args)
  & $CanonicalScript @forwardArgs
  exit $LASTEXITCODE
} finally {
  Pop-Location
}